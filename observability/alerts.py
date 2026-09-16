"""Alertas (Camada 6b do roteiro_final.md): monitora a saúde do serviço a
partir do que já está gravado em `agent_traces` e notifica quando algo sai
do esperado.

`Notifier` é uma interface abstrata (Dependency Inversion, mesmo princípio
de `PromptCacheBackend`) -- `ConsoleNotifier` funciona hoje (loga); um
`SlackNotifier` real seria só mais uma implementação da mesma interface,
plugável quando houver um webhook -- não bloqueia o resto do sistema.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from persistence.models import AgentTraceDB

LIMIAR_TAXA_ERRO_PADRAO = 0.05  # >5% de análises com erro -> CRITICAL
FATOR_LATENCIA_SPIKE_PADRAO = 1.5  # 1.5x a latência-base -> WARNING
FRACOES_ALERTA_BUDGET = (0.7, 0.9)  # 70%/90% do orçamento consumido -> WARNING


@dataclass(frozen=True)
class Alerta:
    tipo: str  # "taxa_erro" | "latencia_spike" | "budget"
    nivel: str  # "warning" | "critical"
    titulo: str
    mensagem: str


class Notifier(ABC):
    @abstractmethod
    def enviar(self, alerta: Alerta) -> None:
        raise NotImplementedError


class ConsoleNotifier(Notifier):
    """Implementação padrão: loga o alerta em JSON estruturado (mesmo logger
    de observability/logger.py)."""

    def __init__(self, logger=None):
        from observability.logger import configurar_logger

        self._logger = logger or configurar_logger("credit_forge_alerts")

    def enviar(self, alerta: Alerta) -> None:
        log = self._logger.error if alerta.nivel == "critical" else self._logger.warning
        log(alerta.titulo, extra={"tipo": alerta.tipo, "nivel": alerta.nivel, "mensagem": alerta.mensagem})


@dataclass(frozen=True)
class MetricasJanela:
    total_analises: int
    taxa_erro: float
    latencia_media_ms: float
    custo_total_usd: float


def coletar_metricas_janela(session: Session, janela_horas: int = 1) -> MetricasJanela:
    """Agrega `agent_traces` das últimas `janela_horas` -- 1 "análise" é o
    conjunto de linhas com o mesmo trace_id (normalmente 3, uma por agente;
    1 em caso de falha no nível do pipeline)."""
    desde = datetime.now(timezone.utc) - timedelta(hours=janela_horas)
    linhas = list(session.scalars(select(AgentTraceDB).where(AgentTraceDB.timestamp >= desde)))

    if not linhas:
        return MetricasJanela(total_analises=0, taxa_erro=0.0, latencia_media_ms=0.0, custo_total_usd=0.0)

    por_trace: dict[str, list[AgentTraceDB]] = {}
    for linha in linhas:
        por_trace.setdefault(linha.trace_id, []).append(linha)

    total_analises = len(por_trace)
    analises_com_erro = sum(
        1 for linhas_do_trace in por_trace.values() if any(l.status == "error" for l in linhas_do_trace)
    )
    taxa_erro = analises_com_erro / total_analises

    latencias_por_trace = [
        sum(l.latencia_ms for l in linhas_do_trace) for linhas_do_trace in por_trace.values()
    ]
    latencia_media_ms = sum(latencias_por_trace) / len(latencias_por_trace)

    custo_total_usd = sum(linha.custo_usd for linha in linhas)

    return MetricasJanela(total_analises, taxa_erro, latencia_media_ms, custo_total_usd)


def avaliar_saude(
    metricas: MetricasJanela,
    orcamento_usd: float,
    latencia_baseline_ms: float | None = None,
    limiar_taxa_erro: float = LIMIAR_TAXA_ERRO_PADRAO,
    fator_latencia_spike: float = FATOR_LATENCIA_SPIKE_PADRAO,
) -> list[Alerta]:
    """Regras puras (sem I/O) -- decide QUAIS alertas disparar a partir de
    métricas já coletadas. Separado de `coletar_metricas_janela` de propósito:
    testar as regras não deveria exigir um banco de dados."""
    if metricas.total_analises == 0:
        return []

    alertas: list[Alerta] = []

    if metricas.taxa_erro > limiar_taxa_erro:
        alertas.append(
            Alerta(
                tipo="taxa_erro",
                nivel="critical",
                titulo="Taxa de erro alta",
                mensagem=(
                    f"{metricas.taxa_erro * 100:.1f}% das análises falhando "
                    f"(limiar: {limiar_taxa_erro * 100:.0f}%)"
                ),
            )
        )

    if latencia_baseline_ms and metricas.latencia_media_ms > latencia_baseline_ms * fator_latencia_spike:
        alertas.append(
            Alerta(
                tipo="latencia_spike",
                nivel="warning",
                titulo="Latência elevada",
                mensagem=(
                    f"Atual: {metricas.latencia_media_ms:.0f}ms (normal: {latencia_baseline_ms:.0f}ms) -- "
                    f"{metricas.latencia_media_ms / latencia_baseline_ms:.1f}x mais lento"
                ),
            )
        )

    if orcamento_usd > 0:
        fracao_consumida = metricas.custo_total_usd / orcamento_usd
        # só o limiar mais alto já atingido -- evita 70% e 90% juntos no mesmo ciclo.
        for fracao_limiar in sorted(FRACOES_ALERTA_BUDGET, reverse=True):
            if fracao_consumida >= fracao_limiar:
                alertas.append(
                    Alerta(
                        tipo="budget",
                        nivel="warning",
                        titulo="Budget alert",
                        mensagem=(
                            f"Gasto: ${metricas.custo_total_usd:.2f} "
                            f"({fracao_consumida * 100:.0f}% de ${orcamento_usd:.2f})"
                        ),
                    )
                )
                break

    return alertas


def monitorar_e_notificar(
    session: Session,
    notifier: Notifier,
    janela_horas: int = 1,
    orcamento_usd: float = 5.0,
    latencia_baseline_ms: float | None = None,
) -> list[Alerta]:
    """1 ciclo de checagem: coleta -> avalia -> notifica. Devolve os alertas
    disparados (permite testar sem precisar inspecionar o notifier)."""
    metricas = coletar_metricas_janela(session, janela_horas)
    alertas = avaliar_saude(metricas, orcamento_usd, latencia_baseline_ms)
    for alerta in alertas:
        notifier.enviar(alerta)
    return alertas
