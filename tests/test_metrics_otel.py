"""Testes das métricas OpenTelemetry (Etapa 32) com um MeterProvider ISOLADO
e leitor em memória -- nenhuma métrica real é exportada. Também cruza os
valores agregados com o que evaluation/evaluator.py já calcula por fora,
como uma checagem dupla de que os dois caminhos concordam.
"""

import pytest
from opentelemetry.sdk.metrics.export import InMemoryMetricReader

from observability.metrics import MetricasAgente, criar_meter_provider


def _metricas_de_teste():
    reader = InMemoryMetricReader()
    provider = criar_meter_provider(reader=reader)
    meter = provider.get_meter("teste")
    return MetricasAgente(meter=meter), reader


def _pontos_da_metrica(reader, nome_metrica: str):
    dados = reader.get_metrics_data()
    pontos = []
    for rm in dados.resource_metrics:
        for sm in rm.scope_metrics:
            for m in sm.metrics:
                if m.name == nome_metrica:
                    pontos.extend(m.data.data_points)
    return pontos


def test_registrar_execucao_alimenta_histograma_e_counters():
    metricas, reader = _metricas_de_teste()

    metricas.registrar_execucao("extrator", latencia_ms=120.0, tokens=402, custo_usd=0.0008)
    metricas.registrar_execucao("extrator", latencia_ms=80.0, tokens=300, custo_usd=0.0006)

    pontos_latencia = _pontos_da_metrica(reader, "agent.latency")
    assert len(pontos_latencia) == 1  # 1 série (mesmo atributo agent=extrator)
    assert pontos_latencia[0].count == 2
    assert pontos_latencia[0].sum == 200.0
    assert pontos_latencia[0].min == 80.0
    assert pontos_latencia[0].max == 120.0

    pontos_tokens = _pontos_da_metrica(reader, "agent.tokens")
    assert pontos_tokens[0].value == 702

    pontos_custo = _pontos_da_metrica(reader, "agent.cost")
    assert pontos_custo[0].value == pytest.approx(0.0014)


def test_metricas_sao_separadas_por_agente():
    metricas, reader = _metricas_de_teste()

    metricas.registrar_execucao("extrator", latencia_ms=100.0, tokens=400, custo_usd=0.001)
    metricas.registrar_execucao("decisao", latencia_ms=200.0, tokens=500, custo_usd=0.005)

    pontos_tokens = {p.attributes["agent"]: p.value for p in _pontos_da_metrica(reader, "agent.tokens")}
    assert pontos_tokens == {"extrator": 400, "decisao": 500}


def test_registrar_erro_conta_por_tipo():
    metricas, reader = _metricas_de_teste()

    metricas.registrar_erro("decisao", "timeout")
    metricas.registrar_erro("decisao", "timeout")
    metricas.registrar_erro("extrator", "parse_error")

    pontos = _pontos_da_metrica(reader, "agent.errors")
    contagens = {(p.attributes["agent"], p.attributes["error_type"]): p.value for p in pontos}
    assert contagens == {("decisao", "timeout"): 2, ("extrator", "parse_error"): 1}


def test_metricas_batem_com_relatorio_do_evaluator_no_pipeline_mock(casos_teste):
    """Cruza o total de EXECUÇÕES registradas nas métricas (3 agentes x N
    casos) com o número de casos processados pelo evaluator -- os dois
    caminhos (métricas OTel vs. relatório manual) devem concordar."""
    from evaluation.evaluator import avaliar_pipeline
    from orchestration.graph import construir_grafo, processar_documento

    metricas, reader = _metricas_de_teste()

    # Reaproveita o mesmo padrão de _logar_execucao, mas isolado aqui pra não
    # depender de mexer no orchestration/graph.py pra este teste específico.
    app = construir_grafo()
    casos_amostra = casos_teste[:5]
    for caso in casos_amostra:
        resultado = processar_documento(app, caso["documento"])
        for agente, trace in resultado["resultado_final"].traces.items():
            metricas.registrar_execucao(
                agente, trace.latencia_ms, trace.tokens, resultado["resultado_final"].metricas_globais.custo_estimado
            )

    pontos_latencia = _pontos_da_metrica(reader, "agent.latency")
    total_execucoes_metricas = sum(p.count for p in pontos_latencia)

    relatorio = avaliar_pipeline(construir_grafo(), casos_amostra)
    total_execucoes_evaluator = len(relatorio["resultados_por_caso"]) * 3  # 3 agentes/caso

    assert total_execucoes_metricas == total_execucoes_evaluator == 15
