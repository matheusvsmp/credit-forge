"""Testes de evaluation/metrics.py contra valores calculados à mão (Etapa 9)."""

import pytest

from evaluation.metrics import calcular_metricas


def test_calcular_metricas_exemplo_conferivel_na_mao():
    # 4 casos, 3 acertos -> accuracy = 0.75 (conferido manualmente na Etapa 9)
    predicoes = ["Risco Baixo", "Risco Alto", "Risco Médio", "Risco Alto"]
    ground_truth = ["Risco Baixo", "Risco Alto", "Risco Alto", "Risco Alto"]

    metricas = calcular_metricas(predicoes, ground_truth)

    assert metricas["accuracy"] == pytest.approx(0.75)
    assert metricas["recall"] == pytest.approx(0.75)
    # "Risco Médio" nunca aparece no ground_truth, então não pesa na média
    # ponderada de precision -- por isso ela fica em 1.0 mesmo havendo erro.
    assert metricas["precision"] == pytest.approx(1.0)


def test_calcular_metricas_previsao_perfeita_da_1_em_tudo():
    predicoes = ["Risco Baixo", "Risco Alto", "Risco Médio"]
    ground_truth = ["Risco Baixo", "Risco Alto", "Risco Médio"]

    metricas = calcular_metricas(predicoes, ground_truth)

    for valor in metricas.values():
        assert valor == pytest.approx(1.0)
