"""Instrumentação: medir latência, estimar custo em USD e travar o orçamento."""

import time
from contextlib import contextmanager

# Pricing público do Claude Haiku 4.5 (USD por 1 milhão de tokens) -- o modelo
# mais barato da Anthropic, escolhido de propósito para este projeto de estudo
# com orçamento de créditos limitado. Ajuste aqui se trocar de modelo.
PRECO_INPUT_POR_1M_TOKENS = 1.00
PRECO_OUTPUT_POR_1M_TOKENS = 5.00


@contextmanager
def medir_latencia():
    """Context manager que mede o tempo decorrido (em ms) do bloco `with`.

    Uso:
        with medir_latencia() as tempo:
            fazer_algo()
        print(tempo["latencia_ms"])
    """
    inicio = time.perf_counter()
    dados = {"latencia_ms": 0.0}
    try:
        yield dados
    finally:
        dados["latencia_ms"] = (time.perf_counter() - inicio) * 1000


def estimar_custo(tokens_input: int, tokens_output: int) -> float:
    """Estima custo em USD a partir de tokens de entrada/saída de UMA chamada."""
    custo_input = (tokens_input / 1_000_000) * PRECO_INPUT_POR_1M_TOKENS
    custo_output = (tokens_output / 1_000_000) * PRECO_OUTPUT_POR_1M_TOKENS
    return custo_input + custo_output


class OrcamentoExcedidoError(Exception):
    """Levantado quando o gasto acumulado ultrapassa o limite configurado."""


class VerificadorOrcamento:
    """Trava de segurança: acumula o custo estimado de cada chamada real à API
    e interrompe a execução (levantando OrcamentoExcedidoError) assim que o
    total ultrapassa `limite_usd`.

    Isso protege contra: bugs em loop chamando a API mais vezes que o
    esperado, dataset maior que o previsto, ou qualquer situação em que o
    código tentaria gastar mais créditos do que você autorizou.
    """

    def __init__(self, limite_usd: float):
        self.limite_usd = limite_usd
        self.gasto_acumulado_usd = 0.0

    def registrar(self, custo_usd: float) -> None:
        self.gasto_acumulado_usd += custo_usd
        if self.gasto_acumulado_usd > self.limite_usd:
            raise OrcamentoExcedidoError(
                f"Orçamento de ${self.limite_usd:.4f} excedido: "
                f"gasto acumulado ${self.gasto_acumulado_usd:.4f}. "
                "Execução interrompida para proteger seus créditos."
            )
