"""Testes de regressão dos agentes mock (deterministicos, sem custo de API).

Estes testes travam o comportamento ATUAL como "esperado" -- se um futuro
refactor (Fase 0 em diante) mudar silenciosamente a lógica de negócio dos
agentes mock, estes testes quebram e avisam antes que o erro chegue longe.
"""

from agents.decision import agente_decisao_mock
from agents.extrator import agente_extrator_mock
from agents.validator import agente_validador_mock

# Decisão mock esperada por caso -- igual ao ground_truth do dataset, exceto
# o caso 14, onde o modelo de pontos mock erra (documentado desde a Etapa 6).
DECISOES_MOCK_ESPERADAS = {
    1: "Risco Baixo",
    2: "Risco Alto",
    3: "Risco Médio",
    4: "Risco Baixo",
    5: "Risco Alto",
    6: "Risco Médio",
    7: "Risco Baixo",
    8: "Risco Alto",
    9: "Risco Médio",
    10: "Risco Baixo",
    11: "Risco Alto",
    12: "Risco Médio",
    13: "Risco Baixo",
    14: "Risco Médio",  # único caso em que o mock diverge do ground_truth
    15: "Risco Médio",
}


def _processar_mock(documento: str):
    extracao = agente_extrator_mock(documento)
    validacao = agente_validador_mock(extracao)
    decisao = agente_decisao_mock(extracao, validacao)
    return extracao, validacao, decisao


def test_extrator_mock_caso_do_roteiro_original(casos_teste):
    """Caso 3 é o exemplo literal do roteiro.md -- confere os números batem."""
    caso3 = next(c for c in casos_teste if c["id"] == 3)
    extracao = agente_extrator_mock(caso3["documento"])
    assert extracao.faturamento_anual == 2_500_000
    assert extracao.divida_total == 150_000
    assert extracao.capital_social == 500_000
    assert extracao.anos_mercado == 3
    assert extracao.score_historico_pagamento == 0.7


def test_pipeline_mock_bate_com_decisoes_esperadas(casos_teste):
    """Roda o pipeline mock completo nos 15 casos e compara caso a caso."""
    for caso in casos_teste:
        _, _, decisao = _processar_mock(caso["documento"])
        esperado = DECISOES_MOCK_ESPERADAS[caso["id"]]
        assert decisao.decisao_final == esperado, (
            f"caso {caso['id']}: esperado {esperado!r}, obtido {decisao.decisao_final!r}"
        )


def test_pipeline_mock_acerta_14_de_15_contra_gabarito(casos_teste):
    """Accuracy do mock contra o ground_truth deve continuar em 14/15."""
    acertos = 0
    for caso in casos_teste:
        _, _, decisao = _processar_mock(caso["documento"])
        if decisao.decisao_final == caso["ground_truth_decisao"]:
            acertos += 1
    assert acertos == 14
