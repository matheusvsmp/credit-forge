"""Instrumentação: medir latência, estimar custo em USD e travar o orçamento."""

import time
from contextlib import contextmanager

# Pricing público da Anthropic (USD por 1 milhão de tokens), por modelo.
# Haiku 4.5 é o padrão deste projeto (barato); Sonnet 5 entra em cena só
# quando o Model Router (agents/routing.py) decide que o caso é complexo
# o suficiente para justificar o custo maior.
PRECOS_POR_MODELO = {
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "claude-sonnet-5": {"input": 2.00, "output": 10.00},
}
MODELO_PADRAO = "claude-haiku-4-5"


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


def estimar_custo(
    tokens_input: int, tokens_output: int, model_id: str = MODELO_PADRAO
) -> float:
    """Estima custo em USD de UMA chamada, usando o preço do `model_id` informado."""
    precos = PRECOS_POR_MODELO.get(model_id)
    if precos is None:
        raise ValueError(f"Preço não configurado para o modelo {model_id!r}")
    custo_input = (tokens_input / 1_000_000) * precos["input"]
    custo_output = (tokens_output / 1_000_000) * precos["output"]
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
