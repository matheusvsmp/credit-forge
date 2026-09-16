"""Agente Decisão: combina extração + validação num veredito final de risco.

Duas versões:
- agente_decisao_mock: modelo de pontos determinístico, sem LLM (Etapa 6).
- agente_decisao_real: chamada real à API da Anthropic (Etapa 14).
"""

import json

import anthropic

from agents.base import LLMAgent
from agents.llm_utils import MODEL_ID
from data.schemas import DecisionResult, ExtractionResult, ReflexaoResult, ValidationResult
from observability.tracking import VerificadorOrcamento

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


class DecisorAgent(LLMAgent):
    """Agente real de decisão -- usado tanto para a 1ª decisão quanto para a
    decisão revisada após self-reflection (mesma classe, chamada 2 vezes)."""

    system_prompt = SYSTEM_PROMPT_DECISAO
    result_model = DecisionResult

    def montar_conteudo(
        self,
        extracao: ExtractionResult,
        validacao: ValidationResult,
        contexto_memoria: str = "",
        feedback_revisao: str = "",
    ) -> str:
        dados_caso = json.dumps(
            {"extracao": extracao.model_dump(), "validacao": validacao.model_dump()},
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
        return conteudo


class ReflexorAgent(LLMAgent):
    """Agente crítico: revisa uma decisão já tomada (self-reflection loop)."""

    system_prompt = SYSTEM_PROMPT_REFLEXAO
    result_model = ReflexaoResult
    # A lista de achados_contraditorios pode ser longa -- teto maior mesmo
    # sempre rodando em Haiku (ver Etapa 20: 250 tokens já truncou antes).
    max_tokens_haiku = 400

    def montar_conteudo(
        self,
        extracao: ExtractionResult,
        validacao: ValidationResult,
        decisao: DecisionResult,
    ) -> str:
        return json.dumps(
            {
                "extracao": extracao.model_dump(),
                "validacao": validacao.model_dump(),
                "decisao_a_revisar": decisao.model_dump(),
            },
            ensure_ascii=False,
            indent=2,
        )


def agente_decisao_real(
    extracao: ExtractionResult,
    validacao: ValidationResult,
    client: anthropic.Anthropic,
    verificador: VerificadorOrcamento,
    contexto_memoria: str = "",
    feedback_revisao: str = "",
    model_id: str = MODEL_ID,
) -> tuple[DecisionResult, dict]:
    """Calcula a decisão final de risco chamando o Claude de verdade.

    `contexto_memoria` (opcional): resumo de casos já processados nesta
    mesma sessão (ver agents/memory.py) -- injetado como contexto extra.

    `feedback_revisao` (opcional): crítica de uma reflexão anterior (ver
    agente_decisao_com_reflexao_real) -- pede para reconsiderar a decisão.

    `model_id` (opcional): por padrão usa Haiku 4.5 (mesmo comportamento da
    Etapa 15). A Etapa 30 passa "claude-sonnet-5" explicitamente no grafo
    avançado -- decisões de crédito são consideradas críticas o bastante
    para justificar o modelo mais capaz sempre, não só por complexidade.
    """
    agente = DecisorAgent(client, verificador)
    return agente.executar(
        extracao,
        validacao,
        contexto_memoria=contexto_memoria,
        feedback_revisao=feedback_revisao,
        model_id=model_id,
    )


def agente_decisao_com_reflexao_real(
    extracao: ExtractionResult,
    validacao: ValidationResult,
    client: anthropic.Anthropic,
    verificador: VerificadorOrcamento,
    contexto_memoria: str = "",
    model_id: str = MODEL_ID,
) -> tuple[DecisionResult, ReflexaoResult, dict]:
    """Decide, depois critica a própria decisão (self-reflection loop).

    Se a reflexão recomendar "revisar", faz UMA nova chamada de decisão com
    o feedback anexado (loop limitado a 1 correção -- nunca infinito) e usa
    esse resultado como final.

    `model_id` (opcional): modelo usado nas 2 chamadas de DECISÃO (não na
    crítica do Reflexor, que fica sempre em Haiku -- criticar é mais barato
    que decidir). Ver docstring de agente_decisao_real.

    Devolve (decisao_final, reflexao, info_uso_total) -- info_uso_total soma
    os tokens de TODAS as chamadas envolvidas (1 a 3, dependendo do caso).
    """
    decisor = DecisorAgent(client, verificador)
    reflexor = ReflexorAgent(client, verificador)

    decisao_inicial, uso1 = decisor.executar(
        extracao, validacao, contexto_memoria=contexto_memoria, model_id=model_id
    )
    reflexao, uso2 = reflexor.executar(extracao, validacao, decisao_inicial)

    uso3 = {"tokens_input": 0, "tokens_output": 0}
    decisao_final = decisao_inicial
    if reflexao.recomendacao == "revisar":
        feedback = (
            f"Sua decisão anterior foi '{decisao_inicial.decisao_final}' com o motivo: "
            f"'{decisao_inicial.motivo}'. Uma revisão crítica encontrou os seguintes "
            f"problemas: {reflexao.achados_contraditorios}. Reconsidere os dados e "
            f"responda com uma nova decisão final."
        )
        decisao_final, uso3 = decisor.executar(
            extracao,
            validacao,
            contexto_memoria=contexto_memoria,
            feedback_revisao=feedback,
            model_id=model_id,
        )

    info_uso_total = {
        "tokens_input": uso1["tokens_input"] + uso2["tokens_input"] + uso3["tokens_input"],
        "tokens_output": uso1["tokens_output"] + uso2["tokens_output"] + uso3["tokens_output"],
    }
    return decisao_final, reflexao, info_uso_total
