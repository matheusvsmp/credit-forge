"""Agente Extrator: recebe o texto do documento e devolve campos estruturados.

Duas versões:
- agente_extrator_mock: regex sobre o texto, sem LLM (Etapa 4).
- agente_extrator_real: chamada real à API da Anthropic (Etapa 13).
"""

import re

import anthropic

from agents.base import LLMAgent
from agents.llm_utils import MODEL_ID
from data.schemas import ExtractionResult
from observability.tracking import VerificadorOrcamento

SYSTEM_PROMPT_EXTRATOR = """Você é um agente extrator especializado em análise de crédito B2B.
Extraia os campos estruturados do documento fornecido pelo usuário.
Responda APENAS em JSON puro (sem texto antes ou depois, sem markdown), neste formato exato:
{
  "empresa": "<nome da empresa>",
  "faturamento_anual": <número float, em reais>,
  "anos_mercado": <número inteiro>,
  "score_historico_pagamento": <float entre 0.0 e 1.0; quanto mais atrasos de pagamento, menor o score>,
  "divida_total": <número float, em reais>,
  "capital_social": <número float, em reais>,
  "confianca": <float entre 0.0 e 1.0, sua confiança nesta extração>
}"""


def _parse_valor_monetario(texto: str) -> float:
    """Converte strings como '5M', '150k', '0.5M' em float (ex: 5_000_000.0)."""
    match = re.match(r"([\d.]+)\s*(M|k)?", texto.strip())
    if not match:
        raise ValueError(f"Não foi possível interpretar valor monetário: {texto!r}")

    valor = float(match.group(1))
    sufixo = match.group(2)
    if sufixo == "M":
        valor *= 1_000_000
    elif sufixo == "k":
        valor *= 1_000
    return valor


def agente_extrator_mock(documento: str) -> ExtractionResult:
    """Extrai campos estruturados de `documento` usando regras de texto simples."""

    empresa = re.search(r"Empresa:\s*(.+?)\.", documento).group(1)

    faturamento_str = re.search(
        r"Faturamento anual:\s*R\$\s*([\d.]+\s*[Mk]?)", documento
    ).group(1)
    faturamento_anual = _parse_valor_monetario(faturamento_str)

    anos_mercado = int(re.search(r"Anos no mercado:\s*(\d+)", documento).group(1))

    atrasos = int(re.search(r"(\d+)\s*atrasos", documento).group(1))
    # Regra simples: cada atraso reduz o score de pagamento em 0.1, até o mínimo de 0.
    score_historico_pagamento = max(0.0, 1.0 - atrasos / 10)

    divida_str = re.search(
        r"Dívida atual:\s*R\$\s*([\d.]+\s*[Mk]?)", documento
    ).group(1)
    divida_total = _parse_valor_monetario(divida_str)

    capital_str = re.search(
        r"Capital social:\s*R\$\s*([\d.]+\s*[Mk]?)", documento
    ).group(1)
    capital_social = _parse_valor_monetario(capital_str)

    # Confiança fixa e alta: o regex é determinístico, então "confia" sempre
    # que encontra todos os campos (se algum não fosse encontrado, o re.search
    # acima já teria lançado AttributeError antes de chegar aqui).
    confianca = 0.95

    return ExtractionResult(
        empresa=empresa,
        faturamento_anual=faturamento_anual,
        anos_mercado=anos_mercado,
        score_historico_pagamento=score_historico_pagamento,
        divida_total=divida_total,
        capital_social=capital_social,
        confianca=confianca,
    )


class ExtratorAgent(LLMAgent):
    """Agente real de extração -- infraestrutura de chamada vem de LLMAgent."""

    system_prompt = SYSTEM_PROMPT_EXTRATOR
    result_model = ExtractionResult
    # Sonnet 5 tende a ser mais verboso (ex: nomes completos, mais cuidado
    # com casos ambíguos) do que Haiku -- teto maior evita truncamento.
    max_tokens_outros_modelos = 600

    def montar_conteudo(self, documento: str, contexto_memoria: str = "") -> str:
        if contexto_memoria:
            return (
                f"[Contexto de casos já analisados nesta sessão]:\n{contexto_memoria}\n\n"
                f"[Documento atual a analisar]:\n{documento}"
            )
        return f"Analise:\n{documento}"


def agente_extrator_real(
    documento: str,
    client: anthropic.Anthropic,
    verificador: VerificadorOrcamento,
    contexto_memoria: str = "",
    model_id: str = MODEL_ID,
    cache=None,
) -> tuple[ExtractionResult, dict]:
    """Extrai campos estruturados chamando o Claude de verdade.

    Devolve (resultado, info_uso). `info_uso` traz tokens_input/tokens_output
    reais da resposta da API -- usado pelo consolidador para montar o
    AgentTrace de observabilidade com números reais (não mais 0).

    `verificador` registra o custo desta chamada e interrompe a execução
    (OrcamentoExcedidoError) se o gasto acumulado no processo ultrapassar o
    limite configurado.

    `contexto_memoria` (opcional): resumo de casos já processados nesta
    mesma sessão (ver agents/memory.py) -- injetado como contexto extra.

    `model_id` (opcional): por padrão usa Haiku 4.5; o Model Router
    (agents/routing.py) pode passar "claude-sonnet-5" para casos complexos.

    `cache` (opcional): um PromptCacheBackend (Etapa 37) -- por padrão
    `None` (cache desligado), para preservar o comportamento exato dos
    grafos já validados.
    """
    agente = ExtratorAgent(client, verificador, cache=cache)
    return agente.executar(documento, contexto_memoria=contexto_memoria, model_id=model_id)
