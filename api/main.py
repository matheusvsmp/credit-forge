"""Camada 1 do Credit Forge: API Gateway (FastAPI).

Recebe POST /api/v1/analise, valida o payload, gera um trace_id único por
requisição, delega ao pipeline (orchestration/graph.py) e devolve a decisão.

Rodar localmente:
    .venv\\Scripts\\python.exe -m uvicorn api.main:app --reload
"""

import os
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from sqlalchemy.orm import Session

from api.deps import (
    get_circuit_breaker,
    get_db_session,
    get_pipeline_app,
    obter_chave_de_rate_limit,
    verificar_api_key,
)
from api.schemas import (
    AnaliseRequest,
    AnaliseResponse,
    AuditoriaLinhaResponse,
    CustoClienteResponse,
    ErroResponse,
)
from observability.logger import configurar_logger
from observability.metrics import configurar_metrics_global, get_metricas_globais
from observability.tracing import configurar_tracing_global, span_raiz
from orchestration.circuit_breaker import CircuitoAbertoError
from orchestration.graph import processar_documento
from persistence.audit import buscar_por_trace_id, custos_por_cliente, registrar_erro, registrar_resultado

logger = configurar_logger("credit_forge_api")

LIMITE_REQUISICOES = os.environ.get("RATE_LIMIT_ANALISE", "100/minute")

limiter = Limiter(key_func=obter_chave_de_rate_limit)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Telemetria é ligada aqui (startup real), não no import do módulo --
    isso evita threads de fundo (do exportador de métricas/spans) nascerem
    só por importar `api.main` em testes que nunca de fato "sobem" o app
    (o TestClient sem `with` não dispara este evento)."""
    configurar_tracing_global()  # console por enquanto -- OTLP/Jaeger na Etapa 33
    configurar_metrics_global()
    yield


app = FastAPI(title="Credit Forge API", version="1.0.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)


def _novo_trace_id() -> str:
    return f"tr-{uuid.uuid4().hex[:8]}"


@app.exception_handler(RequestValidationError)
async def payload_invalido(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Payload que falha na validação Pydantic vira HTTP 400 (o roteiro_final.md
    especifica 400; o padrão do FastAPI seria 422) -- sempre com um trace_id,
    mesmo em erro, para permitir rastrear o incidente depois."""
    trace_id = _novo_trace_id()
    logger.info("payload_invalido", extra={"trace_id": trace_id, "erros": exc.errors()})
    return JSONResponse(
        status_code=400,
        content=ErroResponse(trace_id=trace_id, detalhe=str(exc.errors())).model_dump(),
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/v1/analise", response_model=AnaliseResponse)
@limiter.limit(LIMITE_REQUISICOES)
def analise(
    request: Request,
    req: AnaliseRequest,
    pipeline_app=Depends(get_pipeline_app),
    api_key: str = Depends(verificar_api_key),
    circuito=Depends(get_circuit_breaker),
    session: Session = Depends(get_db_session),
):
    trace_id = _novo_trace_id()
    cliente = req.cliente or "anonimo"
    ip_address = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")
    logger.info("analise_iniciada", extra={"trace_id": trace_id, "cliente": cliente})

    try:
        with span_raiz(trace_id, cliente):
            resultado = circuito.call(processar_documento, pipeline_app, req.documento)
    except CircuitoAbertoError as exc:
        logger.error("circuito_aberto", extra={"trace_id": trace_id, "erro": str(exc)})
        get_metricas_globais().registrar_erro("pipeline", "circuito_aberto")
        registrar_erro(session, trace_id, cliente, "pipeline", str(exc), ip_address, user_agent)
        return JSONResponse(
            status_code=503,
            content=ErroResponse(trace_id=trace_id, detalhe=str(exc)).model_dump(),
        )
    except Exception as exc:  # noqa: BLE001 -- erro de agente/API externa vira 500 controlado
        logger.error("analise_falhou", extra={"trace_id": trace_id, "erro": str(exc)})
        get_metricas_globais().registrar_erro("pipeline", "erro_interno")
        registrar_erro(session, trace_id, cliente, "pipeline", str(exc), ip_address, user_agent)
        return JSONResponse(
            status_code=500,
            content=ErroResponse(trace_id=trace_id, detalhe=str(exc)).model_dump(),
        )

    final = resultado["resultado_final"]
    logger.info(
        "analise_concluida",
        extra={
            "trace_id": trace_id,
            "decisao": final.decisao_final,
            "custo": final.metricas_globais.custo_estimado,
        },
    )
    registrar_resultado(session, trace_id, cliente, final, ip_address, user_agent)
    return AnaliseResponse(
        trace_id=trace_id,
        decisao_final=final.decisao_final,
        motivo=final.motivo,
        confianca=final.score_confianca,
    )


@app.get("/audit/trace/{trace_id}", response_model=list[AuditoriaLinhaResponse])
def audit_trace(trace_id: str, session: Session = Depends(get_db_session)):
    """Reconstrói o fluxo completo de UMA análise -- 1 item por agente."""
    linhas = buscar_por_trace_id(session, trace_id)
    if not linhas:
        raise HTTPException(status_code=404, detail=f"Nenhum registro para trace_id={trace_id!r}")
    return [
        AuditoriaLinhaResponse(
            agente=linha.agente,
            modelo_utilizado=linha.modelo_utilizado,
            latencia_ms=linha.latencia_ms,
            tokens_input=linha.tokens_input,
            tokens_output=linha.tokens_output,
            custo_usd=linha.custo_usd,
            status=linha.status,
            error_message=linha.error_message,
        )
        for linha in linhas
    ]


@app.get("/costs/by-client", response_model=list[CustoClienteResponse])
def costs_by_client(dias: int = 30, session: Session = Depends(get_db_session)):
    """Agrega custo total e nº de chamadas por cliente, nos últimos `dias`."""
    return [CustoClienteResponse(**linha) for linha in custos_por_cliente(session, dias=dias)]
