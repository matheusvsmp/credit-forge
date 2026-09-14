"""Model routing: escolhe Haiku (barato) ou Sonnet 5 (mais capaz) por documento.

Adaptação do roteiro2.md -- lá o routing era entre Sonnet/Opus; aqui usamos
Haiku/Sonnet 5 para caber no orçamento de créditos deste projeto de estudo.
"""

# Palavras que indicam que o documento tem informação incerta/ambígua --
# esses casos se beneficiam de um modelo com mais capacidade de raciocínio.
MARCADORES_AMBIGUIDADE = [
    "aproximadamente",
    "cerca de",
    "estima-se",
    "em torno de",
    "não informado",
    "indefinido",
    "talvez",
]

# Acima deste tamanho, um documento típico já é considerado "longo" (os
# documentos do nosso dataset têm ~30-40 palavras).
PALAVRAS_PARA_COMPLEXIDADE_MAXIMA = 80


class ModelRouter:
    """Seleciona o modelo Claude adequado com base na complexidade do documento."""

    @staticmethod
    def estimate_complexity(documento: str) -> float:
        """Estima complexidade 0.0-1.0 combinando tamanho do texto e presença
        de linguagem ambígua (múltiplos valores possíveis, incerteza)."""
        palavras = len(documento.split())
        complexidade_tamanho = min(1.0, palavras / PALAVRAS_PARA_COMPLEXIDADE_MAXIMA)

        texto_lower = documento.lower()
        contagem_marcadores = sum(
            1 for marcador in MARCADORES_AMBIGUIDADE if marcador in texto_lower
        )
        complexidade_ambiguidade = min(1.0, contagem_marcadores / 3)

        complexidade = 0.6 * complexidade_tamanho + 0.4 * complexidade_ambiguidade
        return round(complexidade, 2)

    @staticmethod
    def select_model(complexidade: float) -> str:
        """
        complexidade < 0.5 -> Haiku 4.5 (rápido e barato, suficiente para
                               documentos curtos e diretos)
        complexidade >= 0.5 -> Sonnet 5 (mais capaz, para textos longos
                                e/ou com informação ambígua)
        """
        if complexidade < 0.5:
            return "claude-haiku-4-5"
        return "claude-sonnet-5"
