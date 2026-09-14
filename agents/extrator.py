"""Agente Extrator: recebe o texto do documento e devolve campos estruturados.

Duas versões:
- agente_extrator_mock: regex sobre o texto, sem LLM (Etapa 4).
- agente_extrator_real: chamada real à API da Anthropic (Etapa 13).
"""

import json
import re

import anthropic

from agents.llm_utils import MODEL_ID, limpar_json_da_resposta
from data.schemas import ExtractionResult
from observability.tracking import VerificadorOrcamento, estimar_custo

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


def agente_extrator_real(
    documento: str,
    client: anthropic.Anthropic,
    verificador: VerificadorOrcamento,
) -> tuple[ExtractionResult, dict]:
    """Extrai campos estruturados chamando o Claude (Haiku 4.5) de verdade.

    Devolve (resultado, info_uso). `info_uso` traz tokens_input/tokens_output
    reais da resposta da API -- usado pelo consolidador para montar o
    AgentTrace de observabilidade com números reais (não mais 0).

    `verificador` registra o custo desta chamada e interrompe a execução
    (OrcamentoExcedidoError) se o gasto acumulado no processo ultrapassar o
    limite configurado.
    """
    response = client.messages.create(
        model=MODEL_ID,
        max_tokens=300,  # saída é um JSON curto (~7 campos); evita gasto desnecessário
        system=SYSTEM_PROMPT_EXTRATOR,
        messages=[{"role": "user", "content": f"Analise:\n{documento}"}],
    )

    custo = estimar_custo(
        tokens_input=response.usage.input_tokens,
        tokens_output=response.usage.output_tokens,
    )
    verificador.registrar(custo)

    texto = next(b.text for b in response.content if b.type == "text")
    dados = json.loads(limpar_json_da_resposta(texto))
    resultado = ExtractionResult(**dados)

    info_uso = {
        "tokens_input": response.usage.input_tokens,
        "tokens_output": response.usage.output_tokens,
    }
    return resultado, info_uso
