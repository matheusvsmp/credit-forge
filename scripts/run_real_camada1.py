"""Roda o pipeline completo com as melhorias da Camada 1 (roteiro2.md):
memória de sessão + self-reflection na decisão + model routing na extração.
Sobre os 15 casos de teste, com teto de orçamento de segurança.

Uso (rodar como módulo, não como arquivo direto -- ver README):
    .venv\\Scripts\\python.exe -m scripts.run_real_camada1
    .venv\\Scripts\\python.exe -m scripts.run_real_camada1 --limite 0.50
"""

import argparse
import json

import anthropic
from dotenv import load_dotenv

from agents.memory import AgentMemory
from evaluation.evaluator import avaliar_pipeline
from orchestration.graph import construir_grafo_real_avancado
from observability.tracking import OrcamentoExcedidoError, VerificadorOrcamento

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limite", type=float, default=0.50)
    args = parser.parse_args()

    load_dotenv()

    with open("data/test_cases.json", encoding="utf-8") as f:
        casos = json.load(f)["test_cases"]

    client = anthropic.Anthropic()
    verificador = VerificadorOrcamento(limite_usd=args.limite)
    memoria = AgentMemory(client=client, verificador=verificador, max_turns=6, compress_after=12)
    app = construir_grafo_real_avancado(client, verificador, memoria)

    print(f"Rodando {len(casos)} casos (Camada 1: memória+reflexão+routing), teto: ${args.limite:.2f}")
    print()

    try:
        relatorio = avaliar_pipeline(app, casos)
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

    print()
    print("=== Observabilidade ===")
    for agente, valor in relatorio["custo_por_agente"].items():
        print(f"  custo {agente}: ${valor:.4f}")
    print(f"  -> Custo total (relatorio): ${relatorio['custo_total']:.4f}")
    print(f"  -> Custo real acumulado (verificador, inclui compressao de memoria): ${verificador.gasto_acumulado_usd:.4f}")
    print(f"  -> Latencia media por caso: {relatorio['latencia_media_por_caso_ms']:.0f} ms")

    print()
    print("=== Camada 1: efeitos observados ===")
    print(f"  Modelos usados no Extrator: {relatorio['modelos_extrator']}")
    print(f"  Revisoes disparadas pelo self-reflection: {relatorio['revisoes_disparadas']} / {len(casos)}")
    print(f"  Resumo final da memoria: {memoria.compression_summary}")

    erros = [r for r in relatorio["resultados_por_caso"] if not r["acertou"]]
    print()
    print(f"=== Casos errados ({len(erros)}) ===")
    for erro in erros:
        print(
            f"  caso {erro['id']:2d} ({erro['empresa']}): "
            f"previsto={erro['previsto']!r} gabarito={erro['gabarito']!r}"
        )
