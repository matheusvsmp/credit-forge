"""Testes do Template Method (agents/base.py) com um cliente Anthropic FALSO.

Nenhum destes testes gasta créditos de API -- simulamos a resposta da
Anthropic para testar exclusivamente a lógica do nosso código: montagem de
request, parsing defensivo de JSON, checagem de truncamento e registro de
custo. Isso também funciona como teste de regressão dos 3 bugs reais
encontrados nas Etapas 13/15/20/21 (cercas markdown, aspas escapadas
inválidas em JSON, resposta cortada por max_tokens).
"""

import json

import httpx2
import pytest
from anthropic import APIConnectionError

from agents.decision import agente_decisao_com_reflexao_real, agente_decisao_real
from agents.extrator import agente_extrator_real
from agents.validator import agente_validador_real
from data.schemas import ExtractionResult, ValidationResult
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
    """Substitui `client.messages`: devolve as respostas na ordem configurada
    e grava cada chamada em `self.calls` para inspeção nos testes."""

    def __init__(self, respostas):
        self._respostas = list(respostas)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        proxima = self._respostas.pop(0)
        if isinstance(proxima, Exception):
            raise proxima
        return proxima


class _FakeClient:
    def __init__(self, *respostas):
        self.messages = _FakeMessages(respostas)


EXTRACAO_JSON_VALIDO = json.dumps(
    {
        "empresa": "Empresa Teste",
        "faturamento_anual": 1_000_000,
        "anos_mercado": 5,
        "score_historico_pagamento": 0.8,
        "divida_total": 100_000,
        "capital_social": 300_000,
        "confianca": 0.9,
    }
)


def test_extrator_agent_faz_parsing_e_registra_custo():
    client = _FakeClient(_FakeResponse(EXTRACAO_JSON_VALIDO))
    verificador = VerificadorOrcamento(limite_usd=1.0)

    resultado, uso = agente_extrator_real("qualquer documento", client, verificador)

    assert isinstance(resultado, ExtractionResult)
    assert resultado.empresa == "Empresa Teste"
    assert uso["tokens_input"] == 100
    assert uso["tokens_output"] == 50
    assert uso["cache_hit"] is False
    assert verificador.gasto_acumulado_usd > 0
    assert client.messages.calls[0]["model"] == "claude-haiku-4-5"


def test_extrator_agent_remove_cercas_de_markdown():
    """Regressão da Etapa 13: o modelo às vezes envolve o JSON em ```json ... ```."""
    texto_com_cercas = f"```json\n{EXTRACAO_JSON_VALIDO}\n```"
    client = _FakeClient(_FakeResponse(texto_com_cercas))
    verificador = VerificadorOrcamento(limite_usd=1.0)

    resultado, _ = agente_extrator_real("doc", client, verificador)

    assert resultado.empresa == "Empresa Teste"


def test_agent_corrige_aspas_simples_escapadas_invalidas():
    """Regressão da Etapa 20: \\' é inválido em JSON puro, mas o modelo já
    produziu isso ao citar um valor entre aspas simples num campo de texto."""
    validacao_json_com_bug = (
        '{"coerente": true, '
        '"flags": ["classificado como \\\'baixo risco\\\' incorretamente"], '
        '"confianca": 0.8}'
    )
    client = _FakeClient(_FakeResponse(validacao_json_com_bug))
    verificador = VerificadorOrcamento(limite_usd=1.0)
    extracao = ExtractionResult(**json.loads(EXTRACAO_JSON_VALIDO))

    resultado, _ = agente_validador_real(extracao, client, verificador)

    assert isinstance(resultado, ValidationResult)
    assert "baixo risco" in resultado.flags[0]


def test_agent_levanta_erro_claro_quando_resposta_truncada():
    """Regressão da Etapa 20/21: nunca tentar parsear um JSON cortado por max_tokens."""
    client = _FakeClient(_FakeResponse('{"empresa": "incompl', stop_reason="max_tokens"))
    verificador = VerificadorOrcamento(limite_usd=1.0)

    with pytest.raises(RuntimeError, match="max_tokens"):
        agente_extrator_real("doc", client, verificador)


def test_decisor_agent_inclui_feedback_de_revisao_na_mensagem():
    decisao_json = json.dumps(
        {"decisao_final": "Risco Médio", "motivo": "motivo qualquer", "confianca": 0.8}
    )
    client = _FakeClient(_FakeResponse(decisao_json))
    verificador = VerificadorOrcamento(limite_usd=1.0)
    extracao = ExtractionResult(**json.loads(EXTRACAO_JSON_VALIDO))
    validacao = ValidationResult(coerente=True, flags=[], confianca=0.9)

    agente_decisao_real(
        extracao,
        validacao,
        client,
        verificador,
        feedback_revisao="considere o endividamento real",
    )

    mensagem_enviada = client.messages.calls[0]["messages"][0]["content"]
    assert "considere o endividamento real" in mensagem_enviada


def test_self_reflection_revisa_decisao_quando_recomendado():
    """Simula: 1a decisão -> reflexão pede revisão -> 2a decisão (diferente)."""
    decisao_inicial_json = json.dumps(
        {"decisao_final": "Risco Baixo", "motivo": "parece ok", "confianca": 0.7}
    )
    reflexao_json = json.dumps(
        {
            "confianca": 0.4,
            "achados_contraditorios": ["alavancagem alta ignorada"],
            "recomendacao": "revisar",
        }
    )
    decisao_revisada_json = json.dumps(
        {"decisao_final": "Risco Médio", "motivo": "corrigido após revisão", "confianca": 0.85}
    )
    client = _FakeClient(
        _FakeResponse(decisao_inicial_json),
        _FakeResponse(reflexao_json),
        _FakeResponse(decisao_revisada_json),
    )
    verificador = VerificadorOrcamento(limite_usd=1.0)
    extracao = ExtractionResult(**json.loads(EXTRACAO_JSON_VALIDO))
    validacao = ValidationResult(coerente=True, flags=[], confianca=0.9)

    decisao_final, reflexao, uso_total = agente_decisao_com_reflexao_real(
        extracao, validacao, client, verificador
    )

    assert reflexao.recomendacao == "revisar"
    assert decisao_final.decisao_final == "Risco Médio"
    assert len(client.messages.calls) == 3
    # 3 chamadas de 100+50 tokens cada (conforme _FakeResponse default)
    assert uso_total == {"tokens_input": 300, "tokens_output": 150}


def test_self_reflection_nao_revisa_quando_aceita():
    decisao_json = json.dumps(
        {"decisao_final": "Risco Baixo", "motivo": "ok", "confianca": 0.9}
    )
    reflexao_json = json.dumps(
        {"confianca": 0.95, "achados_contraditorios": [], "recomendacao": "aceitar"}
    )
    client = _FakeClient(_FakeResponse(decisao_json), _FakeResponse(reflexao_json))
    verificador = VerificadorOrcamento(limite_usd=1.0)
    extracao = ExtractionResult(**json.loads(EXTRACAO_JSON_VALIDO))
    validacao = ValidationResult(coerente=True, flags=[], confianca=0.9)

    decisao_final, reflexao, _ = agente_decisao_com_reflexao_real(
        extracao, validacao, client, verificador
    )

    assert reflexao.recomendacao == "aceitar"
    assert decisao_final.decisao_final == "Risco Baixo"
    assert len(client.messages.calls) == 2  # não chamou uma 3a vez


def test_agent_se_recupera_sozinho_de_erro_de_conexao_transitorio():
    """Prova a integração do retry (Etapa 29) dentro do Template Method:
    a 1a chamada falha com erro de rede, a 2a (automática) funciona."""
    erro_de_rede = APIConnectionError(
        request=httpx2.Request("POST", "https://api.anthropic.com")
    )
    client = _FakeClient(erro_de_rede, _FakeResponse(EXTRACAO_JSON_VALIDO))
    verificador = VerificadorOrcamento(limite_usd=1.0)

    resultado, _ = agente_extrator_real("qualquer documento", client, verificador)

    assert resultado.empresa == "Empresa Teste"
    assert len(client.messages.calls) == 2  # 1a falhou, 2a (retry) funcionou
