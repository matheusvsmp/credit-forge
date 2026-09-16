"""Testes dos contratos Pydantic (data/schemas.py)."""

import pytest
from pydantic import ValidationError

from data.schemas import ExtractionResult, TestCase


def test_extraction_result_aceita_dados_validos():
    resultado = ExtractionResult(
        empresa="Tech Startup XYZ",
        faturamento_anual=2_500_000,
        anos_mercado=3,
        score_historico_pagamento=0.7,
        divida_total=150_000,
        capital_social=500_000,
        confianca=0.92,
    )
    assert resultado.empresa == "Tech Startup XYZ"
    assert resultado.faturamento_anual == 2_500_000


def test_extraction_result_rejeita_confianca_acima_de_1():
    with pytest.raises(ValidationError):
        ExtractionResult(
            empresa="Empresa Ruim",
            faturamento_anual=1000,
            anos_mercado=1,
            score_historico_pagamento=0.5,
            divida_total=0,
            capital_social=0,
            confianca=1.5,
        )


def test_extraction_result_rejeita_anos_mercado_negativo():
    with pytest.raises(ValidationError):
        ExtractionResult(
            empresa="Empresa X",
            faturamento_anual=1000,
            anos_mercado=-1,
            score_historico_pagamento=0.5,
            divida_total=0,
            capital_social=0,
            confianca=0.5,
        )


def test_test_case_rejeita_decisao_fora_das_3_opcoes():
    with pytest.raises(ValidationError):
        TestCase(
            id=1,
            documento="qualquer coisa",
            ground_truth_decisao="Risco Extremo",
            motivo_esperado="motivo",
        )


def test_dataset_completo_valida_contra_test_case(casos_teste):
    """Os 15 casos do dataset oficial devem validar sem erro contra o schema."""
    validados = [TestCase(**c) for c in casos_teste]
    assert len(validados) == 15
