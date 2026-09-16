"""Testa o pipeline mock ORQUESTRADO pelo LangGraph (orchestration/graph.py),
não só as funções de agente isoladas -- cobre a fiação do grafo (Etapa 7/8)."""

from orchestration.graph import construir_grafo, processar_documento


def test_grafo_mock_acerta_14_de_15(casos_teste):
    app = construir_grafo()

    acertos = 0
    for caso in casos_teste:
        resultado = processar_documento(app, caso["documento"])
        final = resultado["resultado_final"]
        if final.decisao_final == caso["ground_truth_decisao"]:
            acertos += 1

    assert acertos == 14


def test_grafo_mock_produz_resultado_consolidado_completo(casos_teste):
    app = construir_grafo()
    caso = casos_teste[0]

    resultado = processar_documento(app, caso["documento"])
    final = resultado["resultado_final"]

    assert final.empresa
    assert final.decisao_final in {"Risco Baixo", "Risco Médio", "Risco Alto"}
    assert set(final.traces.keys()) == {"extrator", "validador", "decisao"}
    assert final.metricas_globais.custo_estimado == 0.0  # mock não gasta nada
