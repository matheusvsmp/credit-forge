"""Contratos HTTP da API (request/response) -- distintos dos contratos de
domínio em data/schemas.py. A API fala com o mundo externo; data/schemas.py
fala entre os agentes. Misturar os dois acopla a interface HTTP aos detalhes
internos do pipeline (ex: se um dia mudarmos ExtractionResult, a API não
deveria precisar mudar junto)."""

from pydantic import BaseModel, Field


class AnaliseRequest(BaseModel):
    """Corpo esperado por POST /api/v1/analise."""

    documento: str = Field(min_length=10, description="Texto do documento de crédito B2B")
    cliente: str | None = Field(default=None, description="Identificador do cliente/parceiro")


class AnaliseResponse(BaseModel):
    """Corpo devolvido por POST /api/v1/analise em caso de sucesso (HTTP 200)."""

    trace_id: str
    decisao_final: str
    motivo: str
    confianca: float


class ErroResponse(BaseModel):
    """Corpo devolvido em erros (400/500) -- sempre carrega o trace_id, mesmo
    em falha, para permitir rastrear o incidente nos logs/observabilidade."""

    trace_id: str
    detalhe: str


class AuditoriaLinhaResponse(BaseModel):
    """1 linha do log de auditoria (1 agente de 1 análise) -- GET /audit/trace/{trace_id}."""

    agente: str
    modelo_utilizado: str
    latencia_ms: int
    tokens_input: int
    tokens_output: int
    custo_usd: float
    status: str
    error_message: str | None = None


class CustoClienteResponse(BaseModel):
    """1 linha do relatório de custo agregado -- GET /costs/by-client."""

    cliente: str
    custo_total: float
    chamadas: int
