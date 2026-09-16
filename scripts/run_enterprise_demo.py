"""Etapa 41 -- fluxo end-to-end completo contra a pilha containerizada real:
API Gateway -> orquestração resiliente -> agentes -> tracing -> persistência.

Pré-requisito: `docker compose up -d` já rodando (API em localhost:8000).

Uso:
    .venv\\Scripts\\python.exe -m scripts.run_enterprise_demo
"""

import json
import statistics
import time

import httpx

API_URL = "http://localhost:8000"
API_KEY = "dev-key-local"  # ver API_KEYS no docker-compose.yml
CLIENTE = "demo-enterprise-final"

if __name__ == "__main__":
    with open("data/test_cases.json", encoding="utf-8") as f:
        casos = json.load(f)["test_cases"]

    headers = {"X-API-Key": API_KEY, "Content-Type": "application/json"}
    resultados = []
    latencias_wallclock_ms = []

    print(f"Rodando {len(casos)} casos contra {API_URL} (pilha containerizada completa)...")
    print()

    with httpx.Client(timeout=30.0) as client:
        for caso in casos:
            inicio = time.perf_counter()
            resp = client.post(
                f"{API_URL}/api/v1/analise",
                headers=headers,
                json={"documento": caso["documento"], "cliente": CLIENTE},
            )
            latencias_wallclock_ms.append((time.perf_counter() - inicio) * 1000)

            corpo = resp.json()
            acertou = resp.status_code == 200 and corpo.get("decisao_final") == caso["ground_truth_decisao"]
            resultados.append(
                {
                    "id": caso["id"],
                    "status_code": resp.status_code,
                    "trace_id": corpo.get("trace_id"),
                    "previsto": corpo.get("decisao_final"),
                    "gabarito": caso["ground_truth_decisao"],
                    "acertou": acertou,
                }
            )
            marca = "OK  " if acertou else "ERRO"
            print(
                f"caso {caso['id']:2d} [{marca}] http={resp.status_code} "
                f"previsto={corpo.get('decisao_final', '?'):12s} gabarito={caso['ground_truth_decisao']:12s}"
            )

        acertos = sum(1 for r in resultados if r["acertou"])
        accuracy = acertos / len(resultados)

        custos = client.get(f"{API_URL}/costs/by-client").json()
        custo_do_cliente = next((c for c in custos if c["cliente"] == CLIENTE), {"custo_total": 0.0, "chamadas": 0})

    latencias_ordenadas = sorted(latencias_wallclock_ms)
    p50 = statistics.median(latencias_ordenadas)
    p95 = latencias_ordenadas[int(len(latencias_ordenadas) * 0.95) - 1]

    print()
    print("=== Dashboard final (Etapa 41) ===")
    print(f"  Análises processadas: {len(resultados)}")
    print(f"  Accuracy: {accuracy:.3f} ({acertos}/{len(resultados)})")
    print(f"  Latência (wall-clock, via HTTP): P50={p50:.0f}ms  P95={p95:.0f}ms")
    print(f"  Custo total (via /costs/by-client): ${custo_do_cliente['custo_total']:.4f}")
    print(f"  Chamadas de agente registradas: {custo_do_cliente['chamadas']}")

    erros = [r for r in resultados if not r["acertou"]]
    if erros:
        print()
        print(f"=== Casos divergentes do gabarito ({len(erros)}) ===")
        for erro in erros:
            print(f"  caso {erro['id']}: previsto={erro['previsto']!r} gabarito={erro['gabarito']!r}")
