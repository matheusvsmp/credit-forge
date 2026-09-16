"""Orquestração: conecta os 3 agentes num grafo de estados (LangGraph).

Fluxo linear: extrator -> validador -> decisao -> consolidador.

O "estado" (PipelineState) é um dicionário tipado que vai sendo preenchido
conforme o grafo avança. Cada nó recebe o estado atual e devolve só as
chaves que ele alterou -- o LangGraph cuida de mesclar isso no estado global.

Três variantes de grafo, todas com a MESMA topologia (ver `_compilar_workflow`):
- `construir_grafo`: 100% mock, sem custo (Etapa 7).
- `construir_grafo_real`: agentes reais simples (Etapa 15).
- `construir_grafo_real_avancado`: agentes reais + memória de sessão +
  self-reflection + model routing (Camada 1 do roteiro2, Etapa 22).
"""

from typing import Optional, TypedDict

import anthropic
from langgraph.graph import END, START, StateGraph

from agents.decision import (
    agente_decisao_com_reflexao_real,
    agente_decisao_mock,
    agente_decisao_real,
)
from agents.extrator import ExtratorAgent, agente_extrator_mock, agente_extrator_real
from agents.llm_utils import MODEL_ID
from agents.memory import AgentMemory
from agents.routing import ModelRouter
from agents.validator import ValidadorAgent, agente_validador_mock, agente_validador_real
from data.schemas import (
    AgentTrace,
    ConsolidatedResult,
    DecisionResult,
    ExtractionResult,
    MetricasGlobais,
    ReflexaoResult,
    ValidationResult,
)
from contextlib import contextmanager

from observability.logger import configurar_logger
from observability.metrics import get_metricas_globais
from observability.tracing import traced_agent
from observability.tracking import VerificadorOrcamento, estimar_custo, medir_latencia
from orchestration.retry import executar_com_fallback

MODEL_ID_SONNET = "claude-sonnet-5"


@contextmanager
def _instrumentar(nome_agente: str):
    """Combina span do OpenTelemetry + medição de latência num só bloco.
    Devolve (tempo, span): quem usa preenche `span` com os atributos
    específicos (tokens, modelo) assim que os souber, ainda dentro do `with`."""
    with traced_agent(nome_agente) as span:
        with medir_latencia() as tempo:
            yield tempo, span

logger = configurar_logger()


class PipelineState(TypedDict):
    documento: str
    extracao: Optional[ExtractionResult]
    validacao: Optional[ValidationResult]
    decisao: Optional[DecisionResult]
    trace_extrator: Optional[AgentTrace]
    trace_validador: Optional[AgentTrace]
    trace_decisao: Optional[AgentTrace]
    reflexao: Optional[ReflexaoResult]
    resultado_final: Optional[ConsolidatedResult]


def _logar_execucao(agente: str, trace: AgentTrace) -> None:
    logger.info(
        "agente_executado",
        extra={
            "agente": agente,
            "tokens_input": trace.tokens_input,
            "tokens_output": trace.tokens_output,
            "latencia_ms": trace.latencia_ms,
        },
    )
    custo = estimar_custo(trace.tokens_input, trace.tokens_output, model_id=trace.model_id)
    get_metricas_globais().registrar_execucao(
        agente, latencia_ms=trace.latencia_ms, tokens=trace.tokens, custo_usd=custo
    )


def _compilar_workflow(node_extrator, node_validador, node_decisao):
    """Monta o grafo com a topologia fixa (linear) compartilhada por todas
    as variantes -- só o que cada nó FAZ muda entre mock/real/avançado."""
    workflow = StateGraph(PipelineState)

    workflow.add_node("extrator", node_extrator)
    workflow.add_node("validador", node_validador)
    workflow.add_node("decisao", node_decisao)
    workflow.add_node("consolidador", node_consolidador)

    workflow.add_edge(START, "extrator")
    workflow.add_edge("extrator", "validador")
    workflow.add_edge("validador", "decisao")
    workflow.add_edge("decisao", "consolidador")
    workflow.add_edge("consolidador", END)

    return workflow.compile()


def node_extrator(state: PipelineState) -> dict:
    with medir_latencia() as tempo:
        extracao = agente_extrator_mock(state["documento"])
    # tokens=0: versão mock, sem LLM real -- ver fazer_node_extrator_real.
    trace = AgentTrace(tokens_input=0, tokens_output=0, latencia_ms=tempo["latencia_ms"])
    _logar_execucao("extrator", trace)
    return {"extracao": extracao, "trace_extrator": trace}


def node_validador(state: PipelineState) -> dict:
    with medir_latencia() as tempo:
        validacao = agente_validador_mock(state["extracao"])
    trace = AgentTrace(tokens_input=0, tokens_output=0, latencia_ms=tempo["latencia_ms"])
    _logar_execucao("validador", trace)
    return {"validacao": validacao, "trace_validador": trace}


def node_decisao(state: PipelineState) -> dict:
    with medir_latencia() as tempo:
        decisao = agente_decisao_mock(state["extracao"], state["validacao"])
    trace = AgentTrace(tokens_input=0, tokens_output=0, latencia_ms=tempo["latencia_ms"])
    _logar_execucao("decisao", trace)
    return {"decisao": decisao, "trace_decisao": trace}


def fazer_node_extrator_real(
    client: anthropic.Anthropic,
    verificador: VerificadorOrcamento,
    memoria: AgentMemory | None = None,
):
    def _node(state: PipelineState) -> dict:
        contexto = memoria.get_context() if memoria else ""
        with _instrumentar("extrator") as (tempo, span):
            extracao, uso = agente_extrator_real(
                state["documento"], client, verificador, contexto_memoria=contexto
            )
            span.set_attribute("tokens_input", uso["tokens_input"])
            span.set_attribute("tokens_output", uso["tokens_output"])
            span.set_attribute("modelo", MODEL_ID)
        trace = AgentTrace(
            tokens_input=uso["tokens_input"],
            tokens_output=uso["tokens_output"],
            latencia_ms=tempo["latencia_ms"],
        )
        _logar_execucao("extrator", trace)
        if memoria:
            memoria.add(
                "extrator",
                f"{extracao.empresa}: faturamento=R${extracao.faturamento_anual:.0f}, "
                f"anos={extracao.anos_mercado}, score_pgto={extracao.score_historico_pagamento:.2f}",
            )
        return {"extracao": extracao, "trace_extrator": trace}

    return _node


def fazer_node_validador_real(
    client: anthropic.Anthropic,
    verificador: VerificadorOrcamento,
    memoria: AgentMemory | None = None,
):
    def _node(state: PipelineState) -> dict:
        contexto = memoria.get_context() if memoria else ""
        with _instrumentar("validador") as (tempo, span):
            validacao, uso = agente_validador_real(
                state["extracao"], client, verificador, contexto_memoria=contexto
            )
            span.set_attribute("tokens_input", uso["tokens_input"])
            span.set_attribute("tokens_output", uso["tokens_output"])
            span.set_attribute("modelo", MODEL_ID)
        trace = AgentTrace(
            tokens_input=uso["tokens_input"],
            tokens_output=uso["tokens_output"],
            latencia_ms=tempo["latencia_ms"],
        )
        _logar_execucao("validador", trace)
        if memoria:
            memoria.add(
                "validador",
                f"{state['extracao'].empresa}: coerente={validacao.coerente}, "
                f"flags={validacao.flags}",
            )
        return {"validacao": validacao, "trace_validador": trace}

    return _node


def fazer_node_decisao_real(
    client: anthropic.Anthropic,
    verificador: VerificadorOrcamento,
    memoria: AgentMemory | None = None,
):
    def _node(state: PipelineState) -> dict:
        contexto = memoria.get_context() if memoria else ""
        with _instrumentar("decisao") as (tempo, span):
            decisao, uso = agente_decisao_real(
                state["extracao"],
                state["validacao"],
                client,
                verificador,
                contexto_memoria=contexto,
            )
            span.set_attribute("tokens_input", uso["tokens_input"])
            span.set_attribute("tokens_output", uso["tokens_output"])
            span.set_attribute("modelo", MODEL_ID)
        trace = AgentTrace(
            tokens_input=uso["tokens_input"],
            tokens_output=uso["tokens_output"],
            latencia_ms=tempo["latencia_ms"],
        )
        _logar_execucao("decisao", trace)
        if memoria:
            memoria.add(
                "decisao",
                f"{state['extracao'].empresa}: decisao={decisao.decisao_final}",
            )
        return {"decisao": decisao, "trace_decisao": trace}

    return _node


def fazer_node_extrator_com_routing_real(
    client: anthropic.Anthropic,
    verificador: VerificadorOrcamento,
    memoria: AgentMemory | None = None,
):
    """Como fazer_node_extrator_real, mas escolhe Haiku ou Sonnet 5 por
    documento via ModelRouter (Etapa 21). Quando o roteador escolhe Haiku,
    ainda há uma rede de segurança: se Haiku falhar (mesmo após os retries
    automáticos do agente), tenta uma vez em Sonnet antes de desistir
    (fallback de modelo, Etapa 29/30)."""

    def _node(state: PipelineState) -> dict:
        documento = state["documento"]
        complexidade = ModelRouter.estimate_complexity(documento)
        model_id_escolhido = ModelRouter.select_model(complexidade)

        contexto = memoria.get_context() if memoria else ""
        agente = ExtratorAgent(client, verificador)

        with _instrumentar("extrator") as (tempo, span):
            if model_id_escolhido == MODEL_ID:
                extracao, uso, model_id_usado = executar_com_fallback(
                    agente,
                    documento,
                    contexto_memoria=contexto,
                    model_id_principal=model_id_escolhido,
                    model_id_fallback=MODEL_ID_SONNET,
                )
            else:
                extracao, uso = agente.executar(
                    documento, contexto_memoria=contexto, model_id=model_id_escolhido
                )
                model_id_usado = model_id_escolhido
            span.set_attribute("tokens_input", uso["tokens_input"])
            span.set_attribute("tokens_output", uso["tokens_output"])
            span.set_attribute("modelo", model_id_usado)
            span.set_attribute("complexidade", complexidade)

        trace = AgentTrace(
            tokens_input=uso["tokens_input"],
            tokens_output=uso["tokens_output"],
            latencia_ms=tempo["latencia_ms"],
            model_id=model_id_usado,
        )
        _logar_execucao("extrator", trace)
        if memoria:
            memoria.add(
                "extrator",
                f"{extracao.empresa}: faturamento=R${extracao.faturamento_anual:.0f}, "
                f"anos={extracao.anos_mercado}, score_pgto={extracao.score_historico_pagamento:.2f}",
            )
        return {"extracao": extracao, "trace_extrator": trace}

    return _node


def fazer_node_validador_com_routing_real(
    client: anthropic.Anthropic,
    verificador: VerificadorOrcamento,
    memoria: AgentMemory | None = None,
):
    """Como fazer_node_validador_real, mas segue a mesma decisão de
    complexidade do documento (Etapa 30) e tem o mesmo fallback Haiku->Sonnet."""

    def _node(state: PipelineState) -> dict:
        complexidade = ModelRouter.estimate_complexity(state["documento"])
        model_id_escolhido = ModelRouter.select_model(complexidade)

        contexto = memoria.get_context() if memoria else ""
        agente = ValidadorAgent(client, verificador)

        with _instrumentar("validador") as (tempo, span):
            if model_id_escolhido == MODEL_ID:
                validacao, uso, model_id_usado = executar_com_fallback(
                    agente,
                    state["extracao"],
                    contexto_memoria=contexto,
                    model_id_principal=model_id_escolhido,
                    model_id_fallback=MODEL_ID_SONNET,
                )
            else:
                validacao, uso = agente.executar(
                    state["extracao"], contexto_memoria=contexto, model_id=model_id_escolhido
                )
                model_id_usado = model_id_escolhido
            span.set_attribute("tokens_input", uso["tokens_input"])
            span.set_attribute("tokens_output", uso["tokens_output"])
            span.set_attribute("modelo", model_id_usado)
            span.set_attribute("complexidade", complexidade)

        trace = AgentTrace(
            tokens_input=uso["tokens_input"],
            tokens_output=uso["tokens_output"],
            latencia_ms=tempo["latencia_ms"],
            model_id=model_id_usado,
        )
        _logar_execucao("validador", trace)
        if memoria:
            memoria.add(
                "validador",
                f"{state['extracao'].empresa}: coerente={validacao.coerente}, "
                f"flags={validacao.flags}",
            )
        return {"validacao": validacao, "trace_validador": trace}

    return _node


def fazer_node_decisao_com_reflexao_real(
    client: anthropic.Anthropic,
    verificador: VerificadorOrcamento,
    memoria: AgentMemory | None = None,
):
    """Como fazer_node_decisao_real, mas com self-reflection loop (Etapa 20)
    e sempre em Sonnet 5 (Etapa 30) -- decisões de crédito são críticas o
    bastante para justificar o modelo mais capaz sempre, não só por rota de
    complexidade como o Extrator/Validador."""

    def _node(state: PipelineState) -> dict:
        contexto = memoria.get_context() if memoria else ""
        with _instrumentar("decisao") as (tempo, span):
            decisao, reflexao, uso = agente_decisao_com_reflexao_real(
                state["extracao"],
                state["validacao"],
                client,
                verificador,
                contexto_memoria=contexto,
                model_id=MODEL_ID_SONNET,
            )
            span.set_attribute("tokens_input", uso["tokens_input"])
            span.set_attribute("tokens_output", uso["tokens_output"])
            span.set_attribute("modelo", MODEL_ID_SONNET)
            span.set_attribute("reflexao_recomendacao", reflexao.recomendacao)
        # NOTA: uso["tokens_*"] soma chamadas de Sonnet (decisão) + Haiku
        # (crítica do Reflexor, sempre mais barata) -- rotular o trace como
        # Sonnet é uma aproximação no relatório de custo POR AGENTE (superestima
        # levemente). O gasto REAL fica correto de qualquer forma: cada
        # chamada já registrou seu próprio custo, com o preço certo, no
        # VerificadorOrcamento no momento em que aconteceu.
        trace = AgentTrace(
            tokens_input=uso["tokens_input"],
            tokens_output=uso["tokens_output"],
            latencia_ms=tempo["latencia_ms"],
            model_id=MODEL_ID_SONNET,
        )
        _logar_execucao("decisao", trace)
        if memoria:
            memoria.add(
                "decisao",
                f"{state['extracao'].empresa}: decisao={decisao.decisao_final} "
                f"(reflexao={reflexao.recomendacao})",
            )
        return {"decisao": decisao, "trace_decisao": trace, "reflexao": reflexao}

    return _node


def node_consolidador(state: PipelineState) -> dict:
    """Junta extracao + validacao + decisao + traces no struct final de um caso."""
    extracao = state["extracao"]
    validacao = state["validacao"]
    decisao = state["decisao"]
    traces = {
        "extrator": state["trace_extrator"],
        "validador": state["trace_validador"],
        "decisao": state["trace_decisao"],
    }

    total_tokens = sum(t.tokens for t in traces.values())
    latencia_total_ms = sum(t.latencia_ms for t in traces.values())
    custo_estimado = sum(
        estimar_custo(t.tokens_input, t.tokens_output, model_id=t.model_id)
        for t in traces.values()
    )

    resultado_final = ConsolidatedResult(
        empresa=extracao.empresa,
        decisao_final=decisao.decisao_final,
        motivo=decisao.motivo,
        score_confianca=decisao.confianca,
        extracao=extracao,
        validacao=validacao,
        decisao=decisao,
        reflexao=state.get("reflexao"),
        traces=traces,
        metricas_globais=MetricasGlobais(
            total_tokens=total_tokens,
            custo_estimado=custo_estimado,
            latencia_total_ms=latencia_total_ms,
        ),
    )
    return {"resultado_final": resultado_final}


def construir_grafo():
    return _compilar_workflow(node_extrator, node_validador, node_decisao)


def construir_grafo_real(
    client: anthropic.Anthropic,
    verificador: VerificadorOrcamento,
    memoria: AgentMemory | None = None,
):
    """Mesma estrutura de grafo do mock, trocando os 3 nós pelas versões que
    chamam a API real da Anthropic.

    `memoria` (opcional): AgentMemory compartilhada entre TODOS os casos
    processados por este grafo na mesma execução (ver Etapa 19/agents/memory.py).
    """
    return _compilar_workflow(
        fazer_node_extrator_real(client, verificador, memoria),
        fazer_node_validador_real(client, verificador, memoria),
        fazer_node_decisao_real(client, verificador, memoria),
    )


def construir_grafo_real_avancado(
    client: anthropic.Anthropic,
    verificador: VerificadorOrcamento,
    memoria: AgentMemory | None = None,
):
    """Grafo real com as 3 melhorias do roteiro2.md (Camada 1) combinadas:
    memória de sessão, self-reflection na decisão e model routing na extração.
    `construir_grafo_real` (Etapa 15) continua existindo intacto, sem essas
    melhorias, para manter aquele resultado reproduzível."""
    return _compilar_workflow(
        fazer_node_extrator_com_routing_real(client, verificador, memoria),
        fazer_node_validador_com_routing_real(client, verificador, memoria),
        fazer_node_decisao_com_reflexao_real(client, verificador, memoria),
    )


def processar_documento(app, documento: str) -> PipelineState:
    estado_inicial: PipelineState = {
        "documento": documento,
        "extracao": None,
        "validacao": None,
        "decisao": None,
        "trace_extrator": None,
        "trace_validador": None,
        "trace_decisao": None,
        "reflexao": None,
        "resultado_final": None,
    }
    return app.invoke(estado_inicial)
