"""Testes do cache de prompt (Etapa 37).

Dois níveis, como já fizemos com persistence/db.py (Postgres) na Etapa 34:
1. Backend em memória -- rápido, sem Docker.
2. Redis real (docker-compose.yml) -- pulado automaticamente se indisponível.
"""

import json

import pytest

from agents.extrator import agente_extrator_real
from persistence.cache import (
    InMemoryCacheBackend,
    calcular_chave_cache,
    criar_backend_padrao,
)
from observability.tracking import VerificadorOrcamento


class _FakeUsage:
    def __init__(self, input_tokens=100, output_tokens=50):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeTextBlock:
    type = "text"

    def __init__(self, text):
        self.text = text


class _FakeResponse:
    def __init__(self, text, input_tokens=100, output_tokens=50, stop_reason="end_turn"):
        self.content = [_FakeTextBlock(text)]
        self.usage = _FakeUsage(input_tokens, output_tokens)
        self.stop_reason = stop_reason


class _FakeMessages:
    def __init__(self, respostas):
        self._respostas = list(respostas)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._respostas.pop(0)


class _FakeClient:
    def __init__(self, *respostas):
        self.messages = _FakeMessages(respostas)


EXTRACAO_JSON = json.dumps(
    {
        "empresa": "Empresa Cache",
        "faturamento_anual": 1_000_000,
        "anos_mercado": 5,
        "score_historico_pagamento": 0.8,
        "divida_total": 100_000,
        "capital_social": 300_000,
        "confianca": 0.9,
    }
)


# ---------------------------------------------------------------------------
# Backend em memória
# ---------------------------------------------------------------------------


def test_in_memory_cache_hit_e_miss():
    cache = InMemoryCacheBackend()

    assert cache.get("chave-x") is None  # miss

    cache.set("chave-x", {"valor": 42}, ttl_segundos=60)
    assert cache.get("chave-x") == {"valor": 42}  # hit


def test_in_memory_cache_expira_apos_ttl():
    """Relógio falso (mesmo princípio do CircuitBreaker, Etapa 29) -- sem
    time.sleep() real, e sem a instabilidade de timing que isso introduziria."""
    relogio = {"agora": 1000.0}
    cache = InMemoryCacheBackend(relogio=lambda: relogio["agora"])

    cache.set("chave-y", {"valor": 1}, ttl_segundos=10)
    assert cache.get("chave-y") == {"valor": 1}  # ainda dentro do TTL

    relogio["agora"] += 11  # avança o "tempo" além do TTL de 10s
    assert cache.get("chave-y") is None


def test_calcular_chave_cache_e_deterministica():
    chave1 = calcular_chave_cache("system", "conteudo", "claude-haiku-4-5")
    chave2 = calcular_chave_cache("system", "conteudo", "claude-haiku-4-5")
    assert chave1 == chave2


def test_calcular_chave_cache_muda_com_qualquer_diferenca():
    base = calcular_chave_cache("system", "conteudo", "claude-haiku-4-5")
    assert base != calcular_chave_cache("system diferente", "conteudo", "claude-haiku-4-5")
    assert base != calcular_chave_cache("system", "conteudo diferente", "claude-haiku-4-5")
    assert base != calcular_chave_cache("system", "conteudo", "claude-sonnet-5")


# ---------------------------------------------------------------------------
# Integração: LLMAgent com cache habilitado
# ---------------------------------------------------------------------------


def test_segunda_chamada_identica_e_cache_hit_e_nao_chama_a_api():
    client = _FakeClient(_FakeResponse(EXTRACAO_JSON))  # só 1 resposta enfileirada!
    verificador = VerificadorOrcamento(limite_usd=1.0)
    cache = InMemoryCacheBackend()

    # 1a chamada: cache miss -- chama a API de verdade
    resultado1, uso1 = agente_extrator_real("documento identico", client, verificador, cache=cache)
    assert uso1["cache_hit"] is False
    assert len(client.messages.calls) == 1
    custo_apos_primeira_chamada = verificador.gasto_acumulado_usd

    # 2a chamada: MESMO documento -- deveria ser cache hit (nem tenta chamar
    # a API; se tentasse, quebraria aqui porque só há 1 resposta na fila)
    resultado2, uso2 = agente_extrator_real("documento identico", client, verificador, cache=cache)
    assert uso2["cache_hit"] is True
    assert uso2["tokens_input"] == 0 and uso2["tokens_output"] == 0
    assert len(client.messages.calls) == 1  # não subiu -- não chamou a API de novo
    assert verificador.gasto_acumulado_usd == custo_apos_primeira_chamada  # custo não mudou

    assert resultado1.empresa == resultado2.empresa == "Empresa Cache"


def test_documento_diferente_e_sempre_cache_miss():
    client = _FakeClient(_FakeResponse(EXTRACAO_JSON), _FakeResponse(EXTRACAO_JSON))
    verificador = VerificadorOrcamento(limite_usd=1.0)
    cache = InMemoryCacheBackend()

    agente_extrator_real("documento A", client, verificador, cache=cache)
    agente_extrator_real("documento B", client, verificador, cache=cache)

    assert len(client.messages.calls) == 2  # prompts diferentes -> 2 chamadas de verdade


def test_sem_cache_passado_comportamento_e_identico_a_antes():
    """Regressão: model_id=None (padrão) preserva o comportamento pré-Etapa 37."""
    client = _FakeClient(_FakeResponse(EXTRACAO_JSON), _FakeResponse(EXTRACAO_JSON))
    verificador = VerificadorOrcamento(limite_usd=1.0)

    agente_extrator_real("documento identico", client, verificador)
    agente_extrator_real("documento identico", client, verificador)

    assert len(client.messages.calls) == 2  # sem cache, chama a API toda vez


# ---------------------------------------------------------------------------
# Integração real contra Redis (docker-compose.yml)
# ---------------------------------------------------------------------------


def _redis_disponivel() -> bool:
    import os

    try:
        import redis

        cliente = redis.from_url(
            os.environ.get("REDIS_URL", "redis://localhost:6380/0"), socket_connect_timeout=1
        )
        return cliente.ping()
    except Exception:
        return False


requer_redis = pytest.mark.skipif(
    not _redis_disponivel(), reason="Redis não está acessível (rode: docker compose up -d)"
)


@requer_redis
def test_redis_cache_hit_e_miss_de_verdade():
    import os
    import uuid

    os.environ["REDIS_URL"] = os.environ.get("REDIS_URL", "redis://localhost:6380/0")
    backend = criar_backend_padrao()

    # Conteúdo único por execução (em vez de tentar "limpar" um resquício de
    # uma rodada anterior) -- garante miss de verdade mesmo rodando os
    # testes várias vezes seguidas dentro do TTL.
    conteudo_unico = f"conteudo-teste-redis-{uuid.uuid4().hex}"
    chave = calcular_chave_cache("sistema-teste", conteudo_unico, "claude-haiku-4-5")

    assert backend.get(chave) is None
    backend.set(chave, {"empresa": "Teste Redis"}, ttl_segundos=30)
    assert backend.get(chave) == {"empresa": "Teste Redis"}
