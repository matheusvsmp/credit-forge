"""Agente Validador: cruza os campos extraídos e sinaliza inconsistências/riscos.

Duas versões:
- agente_validador_mock: regras de negócio determinísticas, sem LLM (Etapa 5).
- agente_validador_real: chamada real à API da Anthropic (Etapa 14).
"""

import json

import anthropic

from agents.llm_utils import MODEL_ID, limpar_json_da_resposta
from data.schemas import ExtractionResult, ValidationResult
from observability.tracking import VerificadorOrcamento, estimar_custo

# Limiar de alavancagem (dívida / faturamento) acima do qual sinalizamos risco.
LIMIAR_ALAVANCAGEM = 0.3

SYSTEM_PROMPT_VALIDADOR = """Você é um agente validador especializado em análise de crédito B2B.
Você recebe dados JÁ EXTRAÍDOS (estruturados) de uma empresa e precisa checar
se eles são coerentes entre si e sinalizar riscos, cruzando os campos --
por exemplo: a dívida é muito alta frente ao faturamento ou ao capital social?
A empresa é jovem e já tem histórico de pagamento ruim?
Responda APENAS em JSON puro (sem texto antes ou depois, sem markdown), neste formato exato:
{
  "coerente": <true ou false -- false se o perfil combinado for muito problemático/contraditório>,
  "flags": [<lista de strings curtas em snake_case descrevendo cada risco encontrado, ou lista vazia>],
  "confianca": <float entre 0.0 e 1.0, sua confiança nesta validação>
}"""


def agente_validador_mock(extracao: ExtractionResult) -> ValidationResult:
    """Verifica coerência entre os campos extraídos e gera flags de risco."""

    flags: list[str] = []

    alavancagem = extracao.divida_total / extracao.faturamento_anual
    if alavancagem > LIMIAR_ALAVANCAGEM:
        flags.append("alavancagem_alta")

    if extracao.divida_total > extracao.capital_social:
        flags.append("divida_maior_que_capital")

    if extracao.anos_mercado < 2 and extracao.score_historico_pagamento < 0.7:
        flags.append("empresa_jovem_com_atrasos")

    if extracao.score_historico_pagamento < 0.5:
        flags.append("historico_pagamento_ruim")

    if extracao.faturamento_anual < 500_000:
        flags.append("faturamento_baixo")

    # Regra mock: consideramos os dados "incoerentes" (perfil contraditório
    # demais para confiar) quando 3 ou mais sinais de risco aparecem juntos.
    coerente = len(flags) < 3

    # Confiança cai 0.1 a cada flag encontrada, com piso de 0.6.
    confianca = max(0.6, 1.0 - 0.1 * len(flags))

    return ValidationResult(coerente=coerente, flags=flags, confianca=confianca)


def agente_validador_real(
    extracao: ExtractionResult,
    client: anthropic.Anthropic,
    verificador: VerificadorOrcamento,
) -> tuple[ValidationResult, dict]:
    """Valida coerência dos dados extraídos chamando o Claude (Haiku 4.5) de verdade."""

    dados_extraidos = extracao.model_dump_json(indent=2)

    response = client.messages.create(
        model=MODEL_ID,
        max_tokens=300,
        system=SYSTEM_PROMPT_VALIDADOR,
        messages=[
            {"role": "user", "content": f"Dados extraídos:\n{dados_extraidos}"}
        ],
    )

    custo = estimar_custo(
        tokens_input=response.usage.input_tokens,
        tokens_output=response.usage.output_tokens,
    )
    verificador.registrar(custo)

    texto = next(b.text for b in response.content if b.type == "text")
    dados = json.loads(limpar_json_da_resposta(texto))
    resultado = ValidationResult(**dados)

    info_uso = {
        "tokens_input": response.usage.input_tokens,
        "tokens_output": response.usage.output_tokens,
    }
    return resultado, info_uso
