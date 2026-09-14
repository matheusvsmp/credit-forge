"""Agente Decisão: combina extração + validação num veredito final de risco.

Duas versões:
- agente_decisao_mock: modelo de pontos determinístico, sem LLM (Etapa 6).
- agente_decisao_real: chamada real à API da Anthropic (Etapa 14).
"""

import json

import anthropic

from agents.llm_utils import MODEL_ID, limpar_json_da_resposta
from data.schemas import DecisionResult, ExtractionResult, ReflexaoResult, ValidationResult
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

SYSTEM_PROMPT_REFLEXAO = """Você é um agente crítico/revisor de decisões de crédito B2B.
Você recebe os dados de um caso e uma decisão JÁ TOMADA por outro agente, e deve avaliar
criticamente se essa decisão está bem fundamentada nos dados -- procure contradições,
inconsistências ou sinais ignorados.
Responda APENAS em JSON puro (sem texto antes ou depois, sem markdown), neste formato exato:
{
  "confianca": <float entre 0.0 e 1.0, sua confiança de que a decisão original está CORRETA>,
  "achados_contraditorios": [<lista de strings curtas com problemas encontrados, ou lista vazia>],
  "recomendacao": "aceitar" | "rejeitar" | "revisar"
}
Use "revisar" quando encontrar inconsistências relevantes entre os dados e a decisão;
"aceitar" quando a decisão parece bem fundamentada; "rejeitar" apenas em casos graves onde
a decisão parece completamente equivocada."""


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
    contexto_memoria: str = "",
    feedback_revisao: str = "",
) -> tuple[DecisionResult, dict]:
    """Calcula a decisão final de risco chamando o Claude (Haiku 4.5) de verdade.

    `contexto_memoria` (opcional): resumo de casos já processados nesta
    mesma sessão (ver agents/memory.py) -- injetado como contexto extra.

    `feedback_revisao` (opcional): crítica de uma reflexão anterior (ver
    agente_decisao_com_reflexao_real) -- pede para reconsiderar a decisão.
    """

    dados_caso = json.dumps(
        {
            "extracao": extracao.model_dump(),
            "validacao": validacao.model_dump(),
        },
        ensure_ascii=False,
        indent=2,
    )
    conteudo = dados_caso
    if contexto_memoria:
        conteudo = (
            f"[Contexto de casos já analisados nesta sessão]:\n{contexto_memoria}\n\n"
            f"[Dados do caso atual]:\n{dados_caso}"
        )
    if feedback_revisao:
        conteudo += f"\n\n[Revisão crítica da sua decisão anterior]:\n{feedback_revisao}"

    response = client.messages.create(
        model=MODEL_ID,
        max_tokens=300,
        system=SYSTEM_PROMPT_DECISAO,
        messages=[{"role": "user", "content": conteudo}],
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


def agente_decisao_com_reflexao_real(
    extracao: ExtractionResult,
    validacao: ValidationResult,
    client: anthropic.Anthropic,
    verificador: VerificadorOrcamento,
    contexto_memoria: str = "",
) -> tuple[DecisionResult, ReflexaoResult, dict]:
    """Decide, depois critica a própria decisão (self-reflection loop).

    Se a reflexão recomendar "revisar", faz UMA nova chamada de decisão com
    o feedback anexado (loop limitado a 1 correção -- nunca infinito) e usa
    esse resultado como final.

    Devolve (decisao_final, reflexao, info_uso_total) -- info_uso_total soma
    os tokens de TODAS as chamadas envolvidas (1 a 3, dependendo do caso).
    """
    decisao_inicial, uso1 = agente_decisao_real(
        extracao, validacao, client, verificador, contexto_memoria=contexto_memoria
    )

    contexto_reflexao = json.dumps(
        {
            "extracao": extracao.model_dump(),
            "validacao": validacao.model_dump(),
            "decisao_a_revisar": decisao_inicial.model_dump(),
        },
        ensure_ascii=False,
        indent=2,
    )
    response2 = client.messages.create(
        model=MODEL_ID,
        max_tokens=400,
        system=SYSTEM_PROMPT_REFLEXAO,
        messages=[{"role": "user", "content": contexto_reflexao}],
    )
    custo2 = estimar_custo(
        tokens_input=response2.usage.input_tokens,
        tokens_output=response2.usage.output_tokens,
    )
    verificador.registrar(custo2)
    if response2.stop_reason == "max_tokens":
        raise RuntimeError(
            "Resposta da reflexão foi cortada por atingir max_tokens -- "
            "aumente o limite em vez de tentar parsear um JSON incompleto."
        )
    texto2 = next(b.text for b in response2.content if b.type == "text")
    reflexao = ReflexaoResult(**json.loads(limpar_json_da_resposta(texto2)))

    uso2 = {
        "tokens_input": response2.usage.input_tokens,
        "tokens_output": response2.usage.output_tokens,
    }
    uso3 = {"tokens_input": 0, "tokens_output": 0}

    decisao_final = decisao_inicial
    if reflexao.recomendacao == "revisar":
        feedback = (
            f"Sua decisão anterior foi '{decisao_inicial.decisao_final}' com o motivo: "
            f"'{decisao_inicial.motivo}'. Uma revisão crítica encontrou os seguintes "
            f"problemas: {reflexao.achados_contraditorios}. Reconsidere os dados e "
            f"responda com uma nova decisão final."
        )
        decisao_final, uso3 = agente_decisao_real(
            extracao,
            validacao,
            client,
            verificador,
            contexto_memoria=contexto_memoria,
            feedback_revisao=feedback,
        )

    info_uso_total = {
        "tokens_input": uso1["tokens_input"] + uso2["tokens_input"] + uso3["tokens_input"],
        "tokens_output": uso1["tokens_output"] + uso2["tokens_output"] + uso3["tokens_output"],
    }
    return decisao_final, reflexao, info_uso_total
