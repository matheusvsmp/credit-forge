"""Orquestrador principal: conecta os 3 agentes num grafo de estados (LangGraph).

Fluxo linear: extrator -> validador -> decisao.

O "estado" (PipelineState) é um dicionário tipado que vai sendo preenchido
conforme o grafo avança. Cada nó recebe o estado atual e devolve só as
chaves que ele alterou -- o LangGraph cuida de mesclar isso no estado global.
"""

import json
from typing import Optional, TypedDict

from langgraph.graph import END, START, StateGraph

import anthropic

from agents.decision import agente_decisao_mock, agente_decisao_real
from agents.extrator import agente_extrator_mock, agente_extrator_real
from agents.validator import agente_validador_mock, agente_validador_real
from data.schemas import (
    AgentTrace,
    ConsolidatedResult,
    DecisionResult,
    ExtractionResult,
    MetricasGlobais,
    ValidationResult,
)
from observability.logger import configurar_logger
from observability.tracking import VerificadorOrcamento, estimar_custo, medir_latencia

logger = configurar_logger()


class PipelineState(TypedDict):
    documento: str
    extracao: Optional[ExtractionResult]
    validacao: Optional[ValidationResult]
    decisao: Optional[DecisionResult]
    trace_extrator: Optional[AgentTrace]
    trace_validador: Optional[AgentTrace]
    trace_decisao: Optional[AgentTrace]
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


def node_extrator(state: PipelineState) -> dict:
    with medir_latencia() as tempo:
        extracao = agente_extrator_mock(state["documento"])
    # tokens=0: versão mock, sem LLM real -- ver node_extrator_real.
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


def fazer_node_extrator_real(client: anthropic.Anthropic, verificador: VerificadorOrcamento):
    def _node(state: PipelineState) -> dict:
        with medir_latencia() as tempo:
            extracao, uso = agente_extrator_real(state["documento"], client, verificador)
        trace = AgentTrace(
            tokens_input=uso["tokens_input"],
            tokens_output=uso["tokens_output"],
            latencia_ms=tempo["latencia_ms"],
        )
        _logar_execucao("extrator", trace)
        return {"extracao": extracao, "trace_extrator": trace}

    return _node


def fazer_node_validador_real(client: anthropic.Anthropic, verificador: VerificadorOrcamento):
    def _node(state: PipelineState) -> dict:
        with medir_latencia() as tempo:
            validacao, uso = agente_validador_real(state["extracao"], client, verificador)
        trace = AgentTrace(
            tokens_input=uso["tokens_input"],
            tokens_output=uso["tokens_output"],
            latencia_ms=tempo["latencia_ms"],
        )
        _logar_execucao("validador", trace)
        return {"validacao": validacao, "trace_validador": trace}

    return _node


def fazer_node_decisao_real(client: anthropic.Anthropic, verificador: VerificadorOrcamento):
    def _node(state: PipelineState) -> dict:
        with medir_latencia() as tempo:
            decisao, uso = agente_decisao_real(
                state["extracao"], state["validacao"], client, verificador
            )
        trace = AgentTrace(
            tokens_input=uso["tokens_input"],
            tokens_output=uso["tokens_output"],
            latencia_ms=tempo["latencia_ms"],
        )
        _logar_execucao("decisao", trace)
        return {"decisao": decisao, "trace_decisao": trace}

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
        estimar_custo(t.tokens_input, t.tokens_output) for t in traces.values()
    )

    resultado_final = ConsolidatedResult(
        empresa=extracao.empresa,
        decisao_final=decisao.decisao_final,
        motivo=decisao.motivo,
        score_confianca=decisao.confianca,
        extracao=extracao,
        validacao=validacao,
        decisao=decisao,
        traces=traces,
        metricas_globais=MetricasGlobais(
            total_tokens=total_tokens,
            custo_estimado=custo_estimado,
            latencia_total_ms=latencia_total_ms,
        ),
    )
    return {"resultado_final": resultado_final}


def construir_grafo():
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


def construir_grafo_real(client: anthropic.Anthropic, verificador: VerificadorOrcamento):
    """Mesma estrutura de grafo do mock, trocando os 3 nós pelas versões que
    chamam a API real da Anthropic. O consolidador é reaproveitado sem
    nenhuma mudança -- ele já era agnóstico a mock vs real."""
    workflow = StateGraph(PipelineState)

    workflow.add_node("extrator", fazer_node_extrator_real(client, verificador))
    workflow.add_node("validador", fazer_node_validador_real(client, verificador))
    workflow.add_node("decisao", fazer_node_decisao_real(client, verificador))
    workflow.add_node("consolidador", node_consolidador)

    workflow.add_edge(START, "extrator")
    workflow.add_edge("extrator", "validador")
    workflow.add_edge("validador", "decisao")
    workflow.add_edge("decisao", "consolidador")
    workflow.add_edge("consolidador", END)

    return workflow.compile()


def processar_documento(app, documento: str) -> PipelineState:
    estado_inicial: PipelineState = {
        "documento": documento,
        "extracao": None,
        "validacao": None,
        "decisao": None,
        "trace_extrator": None,
        "trace_validador": None,
        "trace_decisao": None,
        "resultado_final": None,
    }
    return app.invoke(estado_inicial)


if __name__ == "__main__":
    app = construir_grafo()

    with open("data/test_cases.json", encoding="utf-8") as f:
        casos = json.load(f)["test_cases"]

    acertos = 0
    for caso in casos:
        resultado = processar_documento(app, caso["documento"])
        final = resultado["resultado_final"]
        gt = caso["ground_truth_decisao"]
        ok = final.decisao_final == gt
        acertos += ok
        marca = "OK  " if ok else "ERRO"
        print(
            f"caso {caso['id']:2d} [{marca}] previsto={final.decisao_final:12s} "
            f"gabarito={gt:12s} confianca={final.score_confianca:.2f}"
        )

    print()
    print(f"Acertos: {acertos}/{len(casos)}")
