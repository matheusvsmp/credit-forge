"""Testes de alertas (Etapa 38).

Três camadas testadas separadamente:
1. `avaliar_saude`: regras puras, sem banco -- a maioria dos testes.
2. `coletar_metricas_janela`: agregação real contra SQLite em memória.
3. `monitorar_e_notificar`: integração dos dois + um Notifier falso.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from observability.alerts import (
    Alerta,
    MetricasJanela,
    Notifier,
    avaliar_saude,
    coletar_metricas_janela,
    monitorar_e_notificar,
)
from persistence.models import AgentTraceDB, Base


class _NotifierFalso(Notifier):
    def __init__(self):
        self.enviados: list[Alerta] = []

    def enviar(self, alerta: Alerta) -> None:
        self.enviados.append(alerta)


# ---------------------------------------------------------------------------
# avaliar_saude -- regras puras
# ---------------------------------------------------------------------------


def test_sem_analises_nao_gera_alertas():
    metricas = MetricasJanela(total_analises=0, taxa_erro=0.0, latencia_media_ms=0.0, custo_total_usd=0.0)
    assert avaliar_saude(metricas, orcamento_usd=5.0) == []


def test_tudo_saudavel_nao_gera_alertas():
    metricas = MetricasJanela(total_analises=100, taxa_erro=0.01, latencia_media_ms=2000, custo_total_usd=0.5)
    assert avaliar_saude(metricas, orcamento_usd=5.0, latencia_baseline_ms=2000) == []


def test_taxa_de_erro_acima_do_limiar_dispara_critical():
    metricas = MetricasJanela(total_analises=100, taxa_erro=0.07, latencia_media_ms=1000, custo_total_usd=0.1)
    alertas = avaliar_saude(metricas, orcamento_usd=5.0)

    assert len(alertas) == 1
    assert alertas[0].tipo == "taxa_erro"
    assert alertas[0].nivel == "critical"
    assert "7.0%" in alertas[0].mensagem


def test_taxa_de_erro_no_limiar_exato_nao_dispara():
    metricas = MetricasJanela(total_analises=100, taxa_erro=0.05, latencia_media_ms=1000, custo_total_usd=0.1)
    assert avaliar_saude(metricas, orcamento_usd=5.0) == []  # > (estrito), não >=


def test_latencia_spike_dispara_warning_quando_ha_baseline():
    metricas = MetricasJanela(total_analises=50, taxa_erro=0.0, latencia_media_ms=6200, custo_total_usd=0.1)
    alertas = avaliar_saude(metricas, orcamento_usd=5.0, latencia_baseline_ms=1800)

    assert len(alertas) == 1
    assert alertas[0].tipo == "latencia_spike"
    assert alertas[0].nivel == "warning"


def test_latencia_spike_nao_dispara_sem_baseline_configurada():
    metricas = MetricasJanela(total_analises=50, taxa_erro=0.0, latencia_media_ms=999_999, custo_total_usd=0.1)
    assert avaliar_saude(metricas, orcamento_usd=5.0, latencia_baseline_ms=None) == []


def test_budget_alert_dispara_apenas_o_limiar_mais_alto_atingido():
    metricas = MetricasJanela(total_analises=10, taxa_erro=0.0, latencia_media_ms=1000, custo_total_usd=4.6)
    alertas = avaliar_saude(metricas, orcamento_usd=5.0)  # 92% consumido

    assert len(alertas) == 1
    assert alertas[0].tipo == "budget"
    assert "92%" in alertas[0].mensagem  # não deveria também disparar o de 70%


def test_budget_alert_em_70_porcento():
    metricas = MetricasJanela(total_analises=10, taxa_erro=0.0, latencia_media_ms=1000, custo_total_usd=3.6)
    alertas = avaliar_saude(metricas, orcamento_usd=5.0)  # 72%

    assert len(alertas) == 1
    assert alertas[0].tipo == "budget"


def test_multiplos_alertas_podem_disparar_juntos():
    metricas = MetricasJanela(total_analises=10, taxa_erro=0.5, latencia_media_ms=5000, custo_total_usd=4.8)
    alertas = avaliar_saude(metricas, orcamento_usd=5.0, latencia_baseline_ms=1000)

    tipos = {a.tipo for a in alertas}
    assert tipos == {"taxa_erro", "latencia_spike", "budget"}


# ---------------------------------------------------------------------------
# coletar_metricas_janela -- agregação real (SQLite em memória)
# ---------------------------------------------------------------------------


@pytest.fixture()
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Sessao = sessionmaker(bind=engine)
    with Sessao() as s:
        yield s


def _inserir_trace(session, trace_id, agente, status="success", latencia_ms=100, custo_usd=0.001):
    session.add(
        AgentTraceDB(
            trace_id=trace_id,
            cliente="cliente-teste",
            agente=agente,
            input_data={},
            output_data={},
            modelo_utilizado="claude-haiku-4-5",
            latencia_ms=latencia_ms,
            tokens_input=10,
            tokens_output=5,
            custo_usd=custo_usd,
            status=status,
        )
    )


def test_coletar_metricas_conta_analises_por_trace_id_nao_por_linha(session):
    for agente in ("extrator", "validador", "decisao"):
        _inserir_trace(session, "tr-1", agente)
    session.commit()

    metricas = coletar_metricas_janela(session, janela_horas=1)

    assert metricas.total_analises == 1  # 3 linhas, mas é 1 análise só
    assert metricas.taxa_erro == 0.0


def test_coletar_metricas_calcula_taxa_de_erro_corretamente(session):
    for agente in ("extrator", "validador", "decisao"):
        _inserir_trace(session, "tr-ok", agente)
    _inserir_trace(session, "tr-falhou", "pipeline", status="error")
    session.commit()

    metricas = coletar_metricas_janela(session, janela_horas=1)

    assert metricas.total_analises == 2
    assert metricas.taxa_erro == 0.5  # 1 de 2 análises falhou


def test_coletar_metricas_soma_latencia_dos_agentes_do_mesmo_trace(session):
    _inserir_trace(session, "tr-1", "extrator", latencia_ms=1000)
    _inserir_trace(session, "tr-1", "validador", latencia_ms=800)
    _inserir_trace(session, "tr-1", "decisao", latencia_ms=1200)
    session.commit()

    metricas = coletar_metricas_janela(session, janela_horas=1)

    assert metricas.latencia_media_ms == 3000  # soma dos 3 agentes desse trace


# ---------------------------------------------------------------------------
# monitorar_e_notificar -- integração
# ---------------------------------------------------------------------------


def test_monitorar_e_notificar_dispara_quando_saude_ruim(session):
    for agente in ("extrator", "validador", "decisao"):
        _inserir_trace(session, "tr-ok", agente)
    for i in range(5):
        _inserir_trace(session, f"tr-falhou-{i}", "pipeline", status="error")
    session.commit()

    notifier = _NotifierFalso()
    alertas = monitorar_e_notificar(session, notifier, janela_horas=1, orcamento_usd=5.0)

    assert len(alertas) == 1
    assert alertas[0].tipo == "taxa_erro"
    assert notifier.enviados == alertas  # o notifier realmente recebeu o alerta


def test_monitorar_e_notificar_nao_dispara_nada_quando_saudavel(session):
    for agente in ("extrator", "validador", "decisao"):
        _inserir_trace(session, "tr-ok", agente)
    session.commit()

    notifier = _NotifierFalso()
    alertas = monitorar_e_notificar(session, notifier, janela_horas=1, orcamento_usd=5.0)

    assert alertas == []
    assert notifier.enviados == []
