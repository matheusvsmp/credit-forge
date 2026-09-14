"""Utilidades compartilhadas pelos agentes reais (via API Anthropic)."""

# Modelo mais barato da Anthropic -- escolhido de propósito pelo orçamento
# limitado de créditos deste projeto de estudo.
MODEL_ID = "claude-haiku-4-5"


def limpar_json_da_resposta(texto: str) -> str:
    """Remove cercas de markdown (```json ... ```) que o modelo às vezes usa,
    mesmo quando instruído a responder em JSON puro."""
    texto = texto.strip()
    if texto.startswith("```"):
        texto = texto.split("\n", 1)[1] if "\n" in texto else texto
        if texto.endswith("```"):
            texto = texto.rsplit("```", 1)[0]
    return texto.strip()
