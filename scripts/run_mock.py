"""Roda o pipeline 100% mock sobre os 15 casos de teste (grátis, sem API).

Uso (rodar como módulo, não como arquivo direto -- ver README):
    .venv\\Scripts\\python.exe -m scripts.run_mock
"""

import json

from orchestration.graph import construir_grafo, processar_documento

if __name__ == "__main__":
    app = construir_grafo()

    with open("data/test_cases.json", encoding="utf-8") as f:
        casos = json.load(f)["test_cases"]

    acertos = 0
    for caso in casos:
        resultado = processar_documento(app, caso["documento"])
        final = resultado["resultado_final"]
        gt = caso["ground_truth_decisao"]
        ok = final.decisao_final == gt
        acertos += ok
        marca = "OK  " if ok else "ERRO"
        print(
            f"caso {caso['id']:2d} [{marca}] previsto={final.decisao_final:12s} "
            f"gabarito={gt:12s} confianca={final.score_confianca:.2f}"
        )

    print()
    print(f"Acertos: {acertos}/{len(casos)}")
