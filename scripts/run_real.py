"""Roda o pipeline completo (Extrator+Validador+Decisão) usando a API real
da Anthropic sobre os 15 casos de teste, com um teto de orçamento de segurança.

Uso (rodar como módulo, não como arquivo direto -- ver README):
    .venv\\Scripts\\python.exe -m scripts.run_real
    .venv\\Scripts\\python.exe -m scripts.run_real --limite 0.50
"""

import argparse
import json

import anthropic
from dotenv import load_dotenv

from evaluation.evaluator import avaliar_pipeline
from orchestration.graph import construir_grafo_real
from observability.tracking import OrcamentoExcedidoError, VerificadorOrcamento

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limite",
        type=float,
        default=0.20,
        help="Teto de orçamento em USD para esta execução (padrão: 0.20)",
    )
    args = parser.parse_args()

    load_dotenv()

    with open("data/test_cases.json", encoding="utf-8") as f:
        casos = json.load(f)["test_cases"]

    client = anthropic.Anthropic()
    verificador = VerificadorOrcamento(limite_usd=args.limite)
    app_real = construir_grafo_real(client, verificador)

    print(f"Rodando {len(casos)} casos com a API real (teto de orçamento: ${args.limite:.2f})...")
    print()

    try:
        relatorio = avaliar_pipeline(app_real, casos)
    except OrcamentoExcedidoError as e:
        print(f"EXECUÇÃO INTERROMPIDA: {e}")
        raise SystemExit(1)

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
    print(f"  -> Custo real acumulado (verificador): ${verificador.gasto_acumulado_usd:.4f}")
    print(f"  -> Agente com maior custo: {relatorio['agente_com_maior_custo']}")
    print(f"  -> Latência média por caso: {relatorio['latencia_media_por_caso_ms']:.0f} ms")

    erros = [r for r in relatorio["resultados_por_caso"] if not r["acertou"]]
    print()
    print(f"=== Casos errados ({len(erros)}) ===")
    for erro in erros:
        print(
            f"  caso {erro['id']:2d} ({erro['empresa']}): "
            f"previsto={erro['previsto']!r} gabarito={erro['gabarito']!r}"
        )
