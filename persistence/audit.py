"""Write-path e read-path de auditoria (Camada 5 do roteiro_final.md).

Fica deliberadamente FORA da API e da orquestração: `orchestration/graph.py`
não sabe (nem precisa saber) que existe um banco de dados -- ele só produz
um `ConsolidatedResult`. É a camada de API (que já tem trace_id, cliente,
IP, user-agent) quem decide gravar isso, chamando as funções daqui.
"""

from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from data.schemas import ConsolidatedResult
from observability.tracking import estimar_custo
from persistence.models import AgentTraceDB, TokenUsage

# Cada agente real tem uma saída de domínio correspondente dentro do
# ConsolidatedResult -- usada como `output_data` do log de auditoria.
_CAMPO_DE_SAIDA_POR_AGENTE = {
    "extrator": "extracao",
    "validador": "validacao",
    "decisao": "decisao",
}


def registrar_resultado(
    session: Session,
    trace_id: str,
    cliente: str,
    resultado: ConsolidatedResult,
    ip_address: str | None = None,
    user_agent: str | None = None,
) -> None:
    """Grava 1 linha de auditoria + 1 linha de billing POR AGENTE envolvido
    numa análise bem-sucedida."""
    for agente, trace in resultado.traces.items():
        campo_saida = _CAMPO_DE_SAIDA_POR_AGENTE.get(agente)
        output_data = getattr(resultado, campo_saida).model_dump() if campo_saida else {}
        custo_usd = estimar_custo(trace.tokens_input, trace.tokens_output, model_id=trace.model_id)

        session.add(
            AgentTraceDB(
                trace_id=trace_id,
                cliente=cliente,
                agente=agente,
                input_data={"empresa": resultado.empresa},
                output_data=output_data,
                modelo_utilizado=trace.model_id,
                latencia_ms=int(trace.latencia_ms),
                tokens_input=trace.tokens_input,
                tokens_output=trace.tokens_output,
                custo_usd=custo_usd,
                status="success",
                ip_address=ip_address,
                user_agent=user_agent,
            )
        )
        session.add(
            TokenUsage(
                trace_id=trace_id,
                modelo=trace.model_id,
                input_tokens=trace.tokens_input,
                output_tokens=trace.tokens_output,
                custo_usd=custo_usd,
                cliente=cliente,
            )
        )
    session.commit()


def registrar_erro(
    session: Session,
    trace_id: str,
    cliente: str,
    agente: str,
    mensagem_erro: str,
    ip_address: str | None = None,
    user_agent: str | None = None,
) -> None:
    """Grava 1 linha de auditoria para uma análise que FALHOU."""
    session.add(
        AgentTraceDB(
            trace_id=trace_id,
            cliente=cliente,
            agente=agente,
            input_data={},
            output_data={},
            modelo_utilizado="n/a",
            latencia_ms=0,
            tokens_input=0,
            tokens_output=0,
            custo_usd=0.0,
            status="error",
            error_message=mensagem_erro,
            ip_address=ip_address,
            user_agent=user_agent,
        )
    )
    session.commit()


def buscar_por_trace_id(session: Session, trace_id: str) -> list[AgentTraceDB]:
    """Reconstrói o fluxo completo de UMA análise (1 linha por agente)."""
    return list(
        session.scalars(
            select(AgentTraceDB)
            .where(AgentTraceDB.trace_id == trace_id)
            .order_by(AgentTraceDB.timestamp)
        )
    )


def custos_por_cliente(session: Session, dias: int = 30) -> list[dict]:
    """Agrega custo total e nº de chamadas por cliente, nos últimos `dias`."""
    desde = datetime.now(timezone.utc) - timedelta(days=dias)
    linhas = session.execute(
        select(
            TokenUsage.cliente,
            func.sum(TokenUsage.custo_usd).label("custo_total"),
            func.count().label("chamadas"),
        )
        .where(TokenUsage.timestamp >= desde)
        .group_by(TokenUsage.cliente)
        .order_by(func.sum(TokenUsage.custo_usd).desc())
    ).all()
    return [
        {"cliente": linha.cliente, "custo_total": float(linha.custo_total), "chamadas": linha.chamadas}
        for linha in linhas
    ]
