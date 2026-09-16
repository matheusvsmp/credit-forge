"""OpenTelemetry Metrics: histogramas e counters agregados entre requisições
(complementam os spans de observability/tracing.py, que mostram UMA
requisição em detalhe -- métricas mostram o comportamento agregado ao
longo de muitas).

Mesmo princípio de injeção de dependência do resto do projeto: `MetricasAgente`
aceita um `meter` explícito, para os testes usarem um leitor em memória
isolado em vez do MeterProvider GLOBAL (que só pode ser configurado uma vez
por processo).
"""

import atexit

from opentelemetry import metrics
from opentelemetry.metrics import Meter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource

NOME_SERVICO = "credit-forge"

_provider_global: MeterProvider | None = None
_metricas_globais: "MetricasAgente | None" = None


def criar_meter_provider(reader=None) -> MeterProvider:
    """Monta um MeterProvider ISOLADO (não mexe no global do processo) --
    útil para testes, que passam um `InMemoryMetricReader`."""
    resource = Resource.create({"service.name": NOME_SERVICO})
    leitor = reader or PeriodicExportingMetricReader(
        ConsoleMetricExporter(), export_interval_millis=60_000
    )
    return MeterProvider(resource=resource, metric_readers=[leitor])


def configurar_metrics_global(reader=None) -> MeterProvider:
    """Configura o MeterProvider GLOBAL do processo, uma única vez.

    O `PeriodicExportingMetricReader` roda numa thread de fundo (ao
    contrário do exportador de spans, métricas não têm uma opção 100%
    síncrona) -- `atexit` garante um `shutdown()` limpo (flush final +
    parar a thread) antes do processo encerrar, evitando uma tentativa de
    exportação depois que o stdout já foi fechado.
    """
    global _provider_global
    if _provider_global is None:
        _provider_global = criar_meter_provider(reader)
        metrics.set_meter_provider(_provider_global)
        atexit.register(_provider_global.shutdown)
    return _provider_global


def get_meter_global() -> Meter:
    return metrics.get_meter(NOME_SERVICO)


class MetricasAgente:
    """Agrupa os instrumentos (histograma + counters) usados pelo pipeline."""

    def __init__(self, meter: Meter | None = None):
        meter = meter or get_meter_global()
        self.latencia_histograma = meter.create_histogram(
            "agent.latency", unit="ms", description="Latência por execução de agente"
        )
        self.tokens_counter = meter.create_counter(
            "agent.tokens", unit="tokens", description="Tokens consumidos por agente"
        )
        self.custo_counter = meter.create_counter(
            "agent.cost", unit="usd", description="Custo estimado (USD) por agente"
        )
        self.erros_counter = meter.create_counter(
            "agent.errors", unit="1", description="Erros por agente/tipo"
        )

    def registrar_execucao(
        self, agente: str, latencia_ms: float, tokens: int, custo_usd: float
    ) -> None:
        atributos = {"agent": agente}
        self.latencia_histograma.record(latencia_ms, atributos)
        self.tokens_counter.add(tokens, atributos)
        self.custo_counter.add(custo_usd, atributos)

    def registrar_erro(self, agente: str, tipo_erro: str) -> None:
        self.erros_counter.add(1, {"agent": agente, "error_type": tipo_erro})


def get_metricas_globais() -> MetricasAgente:
    """Instância única por processo, usando o meter GLOBAL -- é isso que
    orchestration/graph.py e api/main.py usam em produção."""
    global _metricas_globais
    if _metricas_globais is None:
        _metricas_globais = MetricasAgente()
    return _metricas_globais
