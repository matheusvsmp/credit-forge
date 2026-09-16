"""Testes de resiliência (Circuit Breaker + Retry/Fallback) -- Etapa 29.

Nenhum destes testes chama a API real: usamos um "relógio falso" para o
Circuit Breaker (controlamos o tempo sem precisar de time.sleep de verdade)
e esperas minúsculas (0.01s) para o retry, e simulamos os erros da Anthropic
construindo os objetos de exceção diretamente (sem rede).
"""

import httpx2
import pytest
from anthropic import APIConnectionError, RateLimitError

from agents.base import LLMAgent
from observability.tracking import VerificadorOrcamento
from orchestration.circuit_breaker import CircuitBreaker, CircuitoAbertoError, EstadoCircuito
from orchestration.retry import com_retry_anthropic, executar_com_fallback


class _RelogioFalso:
    """Substitui time.monotonic() -- os testes avançam o tempo manualmente,
    sem esperar de verdade."""

    def __init__(self):
        self.agora = 0.0

    def __call__(self) -> float:
        return self.agora

    def avancar(self, segundos: float) -> None:
        self.agora += segundos


def _falhar():
    raise RuntimeError("boom")


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------


def test_circuit_breaker_comeca_fechado():
    cb = CircuitBreaker(failure_threshold=3, reset_timeout_s=30)
    assert cb.state == EstadoCircuito.CLOSED
    assert cb.call(lambda: "ok") == "ok"


def test_circuit_breaker_abre_apos_atingir_o_threshold_de_falhas():
    cb = CircuitBreaker(failure_threshold=3, reset_timeout_s=30, relogio=_RelogioFalso())

    for _ in range(3):
        with pytest.raises(RuntimeError):
            cb.call(_falhar)

    assert cb.state == EstadoCircuito.OPEN
    assert cb.failure_count == 3


def test_circuit_breaker_rejeita_sem_nem_tentar_quando_aberto():
    relogio = _RelogioFalso()
    cb = CircuitBreaker(failure_threshold=1, reset_timeout_s=30, relogio=relogio)

    with pytest.raises(RuntimeError):
        cb.call(_falhar)
    assert cb.state == EstadoCircuito.OPEN

    chamadas: list[int] = []

    def nunca_deveria_rodar():
        chamadas.append(1)
        return "nao deveria chegar aqui"

    with pytest.raises(CircuitoAbertoError):
        cb.call(nunca_deveria_rodar)

    assert chamadas == []  # a função nem foi chamada


def test_circuit_breaker_fecha_apos_sucesso_em_half_open():
    relogio = _RelogioFalso()
    cb = CircuitBreaker(failure_threshold=1, reset_timeout_s=30, relogio=relogio)

    with pytest.raises(RuntimeError):
        cb.call(_falhar)
    assert cb.state == EstadoCircuito.OPEN

    relogio.avancar(31)  # passou do tempo de espera -> próxima chamada é a "sonda"
    resultado = cb.call(lambda: "sucesso")

    assert resultado == "sucesso"
    assert cb.state == EstadoCircuito.CLOSED
    assert cb.failure_count == 0


def test_circuit_breaker_reabre_se_a_sonda_em_half_open_falhar():
    relogio = _RelogioFalso()
    cb = CircuitBreaker(failure_threshold=1, reset_timeout_s=30, relogio=relogio)

    with pytest.raises(RuntimeError):
        cb.call(_falhar)

    relogio.avancar(31)
    with pytest.raises(RuntimeError):
        cb.call(_falhar)  # a "sonda" também falha

    assert cb.state == EstadoCircuito.OPEN


# ---------------------------------------------------------------------------
# Retry com backoff
# ---------------------------------------------------------------------------


def test_retry_tenta_de_novo_e_sucede_apos_falhas_transitorias():
    tentativas = {"n": 0}

    @com_retry_anthropic(max_tentativas=3, espera_minima_s=0.01, espera_maxima_s=0.02)
    def flaky():
        tentativas["n"] += 1
        if tentativas["n"] < 3:
            raise APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com"))
        return "sucesso"

    assert flaky() == "sucesso"
    assert tentativas["n"] == 3


def test_retry_esgota_tentativas_e_relanca_erro_original():
    tentativas = {"n": 0}

    @com_retry_anthropic(max_tentativas=3, espera_minima_s=0.01, espera_maxima_s=0.02)
    def sempre_falha():
        tentativas["n"] += 1
        raise APIConnectionError(request=httpx2.Request("POST", "https://api.anthropic.com"))

    with pytest.raises(APIConnectionError):
        sempre_falha()
    assert tentativas["n"] == 3


def test_retry_nao_reintenta_erro_nao_transitorio():
    tentativas = {"n": 0}

    @com_retry_anthropic(max_tentativas=3, espera_minima_s=0.01, espera_maxima_s=0.02)
    def erro_de_validacao():
        tentativas["n"] += 1
        raise ValueError("payload inválido")

    with pytest.raises(ValueError):
        erro_de_validacao()
    assert tentativas["n"] == 1  # ValueError não é transitório -- não reinsiste


def test_retry_funciona_tambem_para_rate_limit():
    tentativas = {"n": 0}
    req = httpx2.Request("POST", "https://api.anthropic.com")
    resp = httpx2.Response(429, request=req)

    @com_retry_anthropic(max_tentativas=2, espera_minima_s=0.01, espera_maxima_s=0.02)
    def rate_limited_uma_vez():
        tentativas["n"] += 1
        if tentativas["n"] < 2:
            raise RateLimitError("rate limited", response=resp, body=None)
        return "ok"

    assert rate_limited_uma_vez() == "ok"
    assert tentativas["n"] == 2


# ---------------------------------------------------------------------------
# Fallback de modelo
# ---------------------------------------------------------------------------


class _AgenteFake(LLMAgent):
    """Agente de teste: falha se chamado com `model_id_que_falha`, senão
    devolve o próprio model_id usado -- permite verificar qual modelo
    efetivamente respondeu, sem chamar a API de verdade."""

    system_prompt = "irrelevante para este teste"
    result_model = None  # não usado -- sobrescrevemos executar()

    def __init__(self, model_id_que_falha: str):
        self.model_id_que_falha = model_id_que_falha

    def montar_conteudo(self, *args, **kwargs) -> str:
        return "n/a"

    def executar(self, *args, model_id: str, **kwargs):
        if model_id == self.model_id_que_falha:
            raise RuntimeError(f"modelo {model_id} indisponível")
        return model_id, {"tokens_input": 0, "tokens_output": 0}


def test_fallback_usa_modelo_alternativo_quando_principal_falha():
    agente = _AgenteFake(model_id_que_falha="claude-haiku-4-5")

    resultado, _, model_id_usado = executar_com_fallback(
        agente,
        model_id_principal="claude-haiku-4-5",
        model_id_fallback="claude-sonnet-5",
    )

    assert resultado == "claude-sonnet-5"
    assert model_id_usado == "claude-sonnet-5"


def test_fallback_nao_e_usado_quando_principal_funciona():
    agente = _AgenteFake(model_id_que_falha="claude-sonnet-5")

    resultado, _, model_id_usado = executar_com_fallback(
        agente,
        model_id_principal="claude-haiku-4-5",
        model_id_fallback="claude-sonnet-5",
    )
    assert model_id_usado == "claude-haiku-4-5"

    assert resultado == "claude-haiku-4-5"
