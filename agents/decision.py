"""Agente Decisão: combina extração + validação num veredito final de risco.

Duas versões:
- agente_decisao_mock: modelo de pontos determinístico, sem LLM (Etapa 6).
- agente_decisao_real: chamada real à API da Anthropic (Etapa 14).
"""

import json

import anthropic

from agents.llm_utils import MODEL_ID, limpar_json_da_resposta
from data.schemas import DecisionResult, ExtractionResult, ValidationResult
from observability.tracking import VerificadorOrcamento, estimar_custo

SYSTEM_PROMPT_DECISAO = """Você é um agente de decisão de crédito B2B, responsável pelo
veredito final de risco de uma empresa, a partir dos dados extraídos e da validação
de coerência feitos por outros dois agentes.
Responda APENAS em JSON puro (sem texto antes ou depois, sem markdown), neste formato exato:
{
  "decisao_final": "Risco Baixo" | "Risco Médio" | "Risco Alto",
  "motivo": "<explicação curta e objetiva da decisão, em 1-2 frases>",
  "confianca": <float entre 0.0 e 1.0, sua confiança nesta decisão>
}
O valor de "decisao_final" deve ser exatamente uma destas 3 strings, sem variações."""


def _calcular_pontos(extracao: ExtractionResult, validacao: ValidationResult) -> int:
    pontos = 0

    alavancagem = extracao.divida_total / extracao.faturamento_anual
    if alavancagem > 0.3:
        pontos += 2
    elif alavancagem > 0.15:
        pontos += 1

    if extracao.score_historico_pagamento < 0.5:
        pontos += 2
    elif extracao.score_historico_pagamento < 0.8:
        pontos += 1

    if extracao.anos_mercado < 2:
        pontos += 2
    elif extracao.anos_mercado < 5:
        pontos += 1

    if extracao.faturamento_anual < 500_000:
        pontos += 1

    if not validacao.coerente:
        pontos += 1

    return pontos


def agente_decisao_mock(
    extracao: ExtractionResult, validacao: ValidationResult
) -> DecisionResult:
    """Calcula a decisão final de risco a partir dos sinais de extração e validação."""

    pontos = _calcular_pontos(extracao, validacao)

    if pontos <= 1:
        decisao_final = "Risco Baixo"
        motivo = "Baixa alavancagem, bom histórico de pagamento e empresa consolidada."
        confianca = 0.90
    elif pontos <= 4:
        decisao_final = "Risco Médio"
        motivo = "Alguns sinais de atenção presentes (alavancagem e/ou histórico de pagamento moderados)."
        confianca = 0.75
    else:
        decisao_final = "Risco Alto"
        motivo = "Múltiplos sinais de risco combinados: alta alavancagem, histórico de pagamento fraco e/ou empresa jovem."
        confianca = 0.90

    if validacao.flags:
        motivo += f" Flags identificadas: {', '.join(validacao.flags)}."

    return DecisionResult(
        decisao_final=decisao_final, motivo=motivo, confianca=confianca
    )


def agente_decisao_real(
    extracao: ExtractionResult,
    validacao: ValidationResult,
    client: anthropic.Anthropic,
    verificador: VerificadorOrcamento,
) -> tuple[DecisionResult, dict]:
    """Calcula a decisão final de risco chamando o Claude (Haiku 4.5) de verdade."""

    contexto = json.dumps(
        {
            "extracao": extracao.model_dump(),
            "validacao": validacao.model_dump(),
        },
        ensure_ascii=False,
        indent=2,
    )

    response = client.messages.create(
        model=MODEL_ID,
        max_tokens=300,
        system=SYSTEM_PROMPT_DECISAO,
        messages=[{"role": "user", "content": contexto}],
    )

    custo = estimar_custo(
        tokens_input=response.usage.input_tokens,
        tokens_output=response.usage.output_tokens,
    )
    verificador.registrar(custo)

    texto = next(b.text for b in response.content if b.type == "text")
    dados = json.loads(limpar_json_da_resposta(texto))
    resultado = DecisionResult(**dados)

    info_uso = {
        "tokens_input": response.usage.input_tokens,
        "tokens_output": response.usage.output_tokens,
    }
    return resultado, info_uso
