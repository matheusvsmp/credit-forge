"""Contratos de dados (schemas) trocados entre os agentes do pipeline.

Cada agente recebe e devolve um destes modelos. Se um agente tentar criar
um objeto com um campo faltando, de tipo errado, ou fora do intervalo
permitido (ex: confianca > 1.0), o Pydantic lança um erro imediatamente.
"""

from typing import Literal

from pydantic import BaseModel, Field

# Decisões possíveis do Agente Decisão. Usar Literal (em vez de `str`) trava
# o valor a essas 3 opções exatas — qualquer outro texto é rejeitado.
RiscoDecisao = Literal["Risco Baixo", "Risco Médio", "Risco Alto"]


class ExtractionResult(BaseModel):
    """Saída do Agente Extrator: campos estruturados extraídos do documento."""

    empresa: str
    faturamento_anual: float
    anos_mercado: int = Field(ge=0)
    score_historico_pagamento: float = Field(ge=0.0, le=1.0)
    divida_total: float = Field(ge=0.0)
    capital_social: float = Field(ge=0.0)
    confianca: float = Field(ge=0.0, le=1.0)


class ValidationResult(BaseModel):
    """Saída do Agente Validador: coerência dos dados extraídos + flags de risco."""

    coerente: bool
    flags: list[str] = Field(default_factory=list)
    confianca: float = Field(ge=0.0, le=1.0)


class DecisionResult(BaseModel):
    """Saída do Agente Decisão: veredito final de risco de crédito."""

    decisao_final: RiscoDecisao
    motivo: str
    confianca: float = Field(ge=0.0, le=1.0)


class ReflexaoResult(BaseModel):
    """Saída do self-reflection loop: crítica de outro agente sobre uma decisão já tomada."""

    confianca: float = Field(ge=0.0, le=1.0)
    achados_contraditorios: list[str] = Field(default_factory=list)
    recomendacao: Literal["aceitar", "rejeitar", "revisar"]


class AgentTrace(BaseModel):
    """Metadados de observabilidade de UM agente numa execução."""

    tokens_input: int = Field(ge=0)
    tokens_output: int = Field(ge=0)
    latencia_ms: float = Field(ge=0.0)
    model_id: str = "claude-haiku-4-5"

    @property
    def tokens(self) -> int:
        return self.tokens_input + self.tokens_output


class MetricasGlobais(BaseModel):
    """Métricas agregadas de observabilidade de UM caso processado."""

    total_tokens: int = Field(ge=0)
    custo_estimado: float = Field(ge=0.0)
    latencia_total_ms: float = Field(ge=0.0)


class ConsolidatedResult(BaseModel):
    """Struct final de UM caso: junta extração + validação + decisão + observabilidade."""

    empresa: str
    decisao_final: RiscoDecisao
    motivo: str
    score_confianca: float = Field(ge=0.0, le=1.0)
    extracao: ExtractionResult
    validacao: ValidationResult
    decisao: DecisionResult
    reflexao: ReflexaoResult | None = None
    traces: dict[str, AgentTrace]
    metricas_globais: MetricasGlobais


class TestCase(BaseModel):
    """UM caso do dataset de teste (data/test_cases.json)."""

    id: int
    documento: str
    ground_truth_decisao: RiscoDecisao
    motivo_esperado: str
