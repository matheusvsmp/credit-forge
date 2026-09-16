"""OpenTelemetry: tracer, spans aninhados por agente, e configuração do
exportador -- console/memória nesta etapa, OTLP para o Jaeger na Etapa 33.

O `tracer` é injetável em `span_raiz`/`traced_agent` (em vez de sempre ler o
tracer provider GLOBAL do OpenTelemetry) porque o SDK só permite configurar
o provider global UMA VEZ por processo -- um problema real para testes, que
precisam de um exportador isolado e limpo a cada teste. Em produção,
`configurar_tracing_global()` configura o provider global uma vez, no
startup da aplicação (ver api/main.py).
"""

import atexit
import os
from contextlib import contextmanager

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)
from opentelemetry.trace import Status, StatusCode, Tracer

NOME_SERVICO = "credit-forge"

_provider_global: TracerProvider | None = None


def _exportador_padrao():
    """Console por padrão; troca para OTLP/Jaeger se OTEL_EXPORTER_OTLP_ENDPOINT
    estiver definida no ambiente (ver docker-compose.yml, Etapa 33)."""
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if endpoint:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        return OTLPSpanExporter(endpoint=endpoint, insecure=True)
    return ConsoleSpanExporter()


def criar_tracer_provider(exporter=None, sincrono: bool = False) -> TracerProvider:
    """Monta um TracerProvider ISOLADO (não mexe no global do processo).

    `sincrono=True` usa `SimpleSpanProcessor` (exporta cada span assim que
    termina -- necessário em testes, para os asserts rodarem logo depois).
    Em produção usamos `BatchSpanProcessor` (mais eficiente, mas assíncrono).
    """
    resource = Resource.create({"service.name": NOME_SERVICO})
    provider = TracerProvider(resource=resource)
    processor_cls = SimpleSpanProcessor if sincrono else BatchSpanProcessor
    provider.add_span_processor(processor_cls(exporter or ConsoleSpanExporter()))
    return provider


def configurar_tracing_global(exporter=None, sincrono: bool | None = None) -> TracerProvider:
    """Configura o tracer provider GLOBAL do processo, uma única vez.
    Chamado no startup da API (api/main.py, via lifespan) -- chamadas
    seguintes são no-op e devolvem o provider já existente.

    Sem `exporter` explícito, usa Console OU OTLP/Jaeger conforme
    `OTEL_EXPORTER_OTLP_ENDPOINT` (ver `_exportador_padrao`).

    `sincrono` (opcional): se não informado, decide sozinho -- Console é
    instantâneo (síncrono, sem ganho em rodar em thread de fundo); um
    exportador de REDE (OTLP) se beneficia de `BatchSpanProcessor`
    (assíncrono, agrupa spans em vez de 1 chamada de rede por span).
    """
    global _provider_global
    if _provider_global is None:
        exporter = exporter or _exportador_padrao()
        if sincrono is None:
            sincrono = isinstance(exporter, ConsoleSpanExporter)
        _provider_global = criar_tracer_provider(exporter, sincrono=sincrono)
        trace.set_tracer_provider(_provider_global)
        atexit.register(_provider_global.shutdown)
    return _provider_global


def get_tracer_global() -> Tracer:
    return trace.get_tracer(NOME_SERVICO)


@contextmanager
def span_raiz(trace_id: str, cliente: str, tracer: Tracer | None = None):
    """Span raiz de UMA requisição -- todo span aberto dentro deste bloco
    `with` (inclusive os de `traced_agent` chamados por dentro dos nós do
    LangGraph) vira automaticamente um filho dele."""
    tracer = tracer or get_tracer_global()
    with tracer.start_as_current_span("analise-completa") as span:
        span.set_attribute("trace.id", trace_id)
        span.set_attribute("cliente", cliente)
        try:
            yield span
        except Exception as exc:
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise
        else:
            span.set_status(Status(StatusCode.OK))


@contextmanager
def traced_agent(nome_agente: str, tracer: Tracer | None = None):
    """Span de UM agente. Quem usa o context manager recebe o `span` e
    preenche os atributos específicos (tokens, custo, modelo) antes do bloco
    terminar -- ver orchestration/graph.py."""
    tracer = tracer or get_tracer_global()
    with tracer.start_as_current_span(nome_agente) as span:
        span.set_attribute("agent", nome_agente)
        try:
            yield span
        except Exception as exc:
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise
        else:
            span.set_status(Status(StatusCode.OK))
