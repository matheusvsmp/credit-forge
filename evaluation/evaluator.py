"""Roda o pipeline sobre o dataset de teste e monta o relatório de avaliação."""

import json
from collections import Counter

from evaluation.metrics import calcular_metricas
from main import construir_grafo, processar_documento
from observability.tracking import estimar_custo


def avaliar_pipeline(app, casos: list[dict]) -> dict:
    """Executa `app` sobre todos os `casos`, compara com o ground truth e agrega
    métricas de qualidade (accuracy/F1) e de observabilidade (custo/latência)."""

    predicoes: list[str] = []
    ground_truth: list[str] = []
    confiancas_por_agente = {"extrator": [], "validador": [], "decisao": []}
    tokens_input_por_agente = {"extrator": 0, "validador": 0, "decisao": 0}
    tokens_output_por_agente = {"extrator": 0, "validador": 0, "decisao": 0}
    latencias_totais_ms: list[float] = []
    resultados_por_caso = []
    modelos_extrator: Counter = Counter()
    revisoes_disparadas = 0

    for caso in casos:
        resultado = processar_documento(app, caso["documento"])
        final = resultado["resultado_final"]

        predicoes.append(final.decisao_final)
        ground_truth.append(caso["ground_truth_decisao"])

        confiancas_por_agente["extrator"].append(final.extracao.confianca)
        confiancas_por_agente["validador"].append(final.validacao.confianca)
        confiancas_por_agente["decisao"].append(final.decisao.confianca)

        for agente, trace in final.traces.items():
            tokens_input_por_agente[agente] += trace.tokens_input
            tokens_output_por_agente[agente] += trace.tokens_output

        modelos_extrator[final.traces["extrator"].model_id] += 1
        if final.reflexao is not None and final.reflexao.recomendacao == "revisar":
            revisoes_disparadas += 1

        latencias_totais_ms.append(final.metricas_globais.latencia_total_ms)

        resultados_por_caso.append(
            {
                "id": caso["id"],
                "empresa": final.empresa,
                "previsto": final.decisao_final,
                "gabarito": caso["ground_truth_decisao"],
                "acertou": final.decisao_final == caso["ground_truth_decisao"],
            }
        )

    metricas = calcular_metricas(predicoes, ground_truth)

    confianca_media_por_agente = {
        agente: sum(valores) / len(valores)
        for agente, valores in confiancas_por_agente.items()
    }
    agente_mais_confiavel = max(
        confianca_media_por_agente, key=confianca_media_por_agente.get
    )

    custo_por_agente = {
        agente: estimar_custo(
            tokens_input=tokens_input_por_agente[agente],
            tokens_output=tokens_output_por_agente[agente],
        )
        for agente in tokens_input_por_agente
    }
    agente_com_maior_custo = max(custo_por_agente, key=custo_por_agente.get)

    return {
        "metricas": metricas,
        "resultados_por_caso": resultados_por_caso,
        "confianca_media_por_agente": confianca_media_por_agente,
        "agente_mais_confiavel": agente_mais_confiavel,
        "custo_por_agente": custo_por_agente,
        "custo_total": sum(custo_por_agente.values()),
        "agente_com_maior_custo": agente_com_maior_custo,
        "latencia_media_por_caso_ms": sum(latencias_totais_ms) / len(latencias_totais_ms),
        "modelos_extrator": dict(modelos_extrator),
        "revisoes_disparadas": revisoes_disparadas,
    }


if __name__ == "__main__":
    with open("data/test_cases.json", encoding="utf-8") as f:
        casos = json.load(f)["test_cases"]

    app = construir_grafo()
    relatorio = avaliar_pipeline(app, casos)

    print("=== Métricas globais ===")
    for nome, valor in relatorio["metricas"].items():
        print(f"  {nome}: {valor:.3f}")

    print()
    print("=== Confiança média por agente ===")
    for agente, valor in relatorio["confianca_media_por_agente"].items():
        print(f"  {agente}: {valor:.3f}")
    print(f"  -> Agente mais confiável: {relatorio['agente_mais_confiavel']}")

    print()
    print("=== Observabilidade (custo e latência) ===")
    for agente, valor in relatorio["custo_por_agente"].items():
        print(f"  custo {agente}: ${valor:.4f}")
    print(f"  -> Custo total estimado: ${relatorio['custo_total']:.4f}")
    print(f"  -> Agente com maior custo: {relatorio['agente_com_maior_custo']}")
    print(f"  -> Latência média por caso: {relatorio['latencia_media_por_caso_ms']:.3f} ms")

    erros = [r for r in relatorio["resultados_por_caso"] if not r["acertou"]]
    print()
    print(f"=== Casos errados ({len(erros)}) ===")
    for erro in erros:
        print(
            f"  caso {erro['id']:2d} ({erro['empresa']}): "
            f"previsto={erro['previsto']!r} gabarito={erro['gabarito']!r}"
        )
