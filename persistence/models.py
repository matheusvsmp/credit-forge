"""Schema SQLAlchemy: auditoria, billing e cache de prompt (Camada 5 do
roteiro_final.md).

Usa o tipo JSON "com variante": SQLite (testes rápidos, sem Docker) recebe
um JSON genérico; PostgreSQL recebe JSONB de verdade (indexável, mais
eficiente) -- o mesmo código de modelo funciona nos dois motores.
"""

import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import JSON, DateTime, Float, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

JSONField = JSON().with_variant(JSONB(), "postgresql")
# timezone=True: sempre timestamps com timezone (evita ambiguidade sobre
# "em qual fuso está esse horário" -- comum em auditoria/compliance).
DataHora = DateTime(timezone=True)


class Base(DeclarativeBase):
    pass


def _novo_id() -> str:
    return uuid.uuid4().hex


def _agora_utc() -> datetime:
    return datetime.now(timezone.utc)


class AgentTraceDB(Base):
    """Log de auditoria: 1 linha por execução de UM agente (Camada 5.1).

    Existe para: auditoria regulatória (banco/Serasa), debugging (reexecutar
    uma análise via trace_id), e compliance LGPD (expires_at + limpeza).
    """

    __tablename__ = "agent_traces"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_novo_id)
    trace_id: Mapped[str] = mapped_column(String(50), index=True)
    cliente: Mapped[str] = mapped_column(String(100), index=True)
    agente: Mapped[str] = mapped_column(String(50))  # extrator/validador/decisao
    timestamp: Mapped[datetime] = mapped_column(DataHora, index=True, default=_agora_utc)

    input_data: Mapped[dict] = mapped_column(JSONField)
    output_data: Mapped[dict] = mapped_column(JSONField)

    modelo_utilizado: Mapped[str] = mapped_column(String(100))
    latencia_ms: Mapped[int] = mapped_column(Integer)
    tokens_input: Mapped[int] = mapped_column(Integer)
    tokens_output: Mapped[int] = mapped_column(Integer)
    custo_usd: Mapped[float] = mapped_column(Float)

    status: Mapped[str] = mapped_column(String(20))  # success/error
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    ip_address: Mapped[str | None] = mapped_column(String(50), nullable=True)
    user_agent: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DataHora, default=_agora_utc)
    # LGPD: expira em 1 ano por padrão; um cron job (Etapa futura) apaga
    # registros vencidos.
    expires_at: Mapped[datetime] = mapped_column(
        DataHora, default=lambda: _agora_utc() + timedelta(days=365)
    )


class TokenUsage(Base):
    """Billing: 1 linha por chamada de API, para relatório de custo por
    cliente (Camada 5.2)."""

    __tablename__ = "token_usage"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_novo_id)
    trace_id: Mapped[str] = mapped_column(String(50), index=True)
    modelo: Mapped[str] = mapped_column(String(100))
    input_tokens: Mapped[int] = mapped_column(Integer)
    output_tokens: Mapped[int] = mapped_column(Integer)
    custo_usd: Mapped[float] = mapped_column(Float)
    cliente: Mapped[str] = mapped_column(String(100), index=True)
    timestamp: Mapped[datetime] = mapped_column(DataHora, index=True, default=_agora_utc)


class PromptCache(Base):
    """Cache de prompts (Camada 5.3 / 6a) -- schema pronto aqui; a lógica de
    leitura/escrita (hit/miss) é implementada na Etapa 37."""

    __tablename__ = "prompt_cache"

    hash: Mapped[str] = mapped_column(String(64), primary_key=True)  # SHA-256 do prompt
    prompt_text: Mapped[str] = mapped_column(Text)
    cached_response: Mapped[str] = mapped_column(Text)
    modelo: Mapped[str] = mapped_column(String(100))
    ttl_minutes: Mapped[int] = mapped_column(Integer, default=10080)  # 7 dias
    created_at: Mapped[datetime] = mapped_column(DataHora, default=_agora_utc)
    last_hit: Mapped[datetime | None] = mapped_column(DataHora, nullable=True)
    hit_count: Mapped[int] = mapped_column(Integer, default=0)
