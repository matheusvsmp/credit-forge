"""Testes do OpenTelemetry (Etapa 31) usando um TracerProvider ISOLADO com
exportador em memória -- nenhum span real é enviado a lugar nenhum, e cada
teste tem seu próprio provider (evita o problema de o SDK só permitir
configurar o tracer provider GLOBAL uma vez por processo).
"""

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from observability.tracing import criar_tracer_provider, span_raiz, traced_agent


def _tracer_de_teste():
    """Devolve (tracer, exporter) isolados -- `sincrono=True` faz o span ser
    exportado assim que termina, para os asserts rodarem logo em seguida."""
    exporter = InMemorySpanExporter()
    provider = criar_tracer_provider(exporter=exporter, sincrono=True)
    tracer = provider.get_tracer("teste")
    return tracer, exporter


def test_span_raiz_carrega_trace_id_e_cliente():
    tracer, exporter = _tracer_de_teste()

    with span_raiz("tr-abc123", "itau-prod", tracer=tracer):
        pass

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "analise-completa"
    assert spans[0].attributes["trace.id"] == "tr-abc123"
    assert spans[0].attributes["cliente"] == "itau-prod"
    assert spans[0].status.status_code == StatusCode.OK


def test_span_raiz_marca_erro_e_relanca_excecao():
    tracer, exporter = _tracer_de_teste()

    try:
        with span_raiz("tr-erro", "cliente-x", tracer=tracer):
            raise ValueError("boom")
    except ValueError:
        pass
    else:
        assert False, "a exceção deveria ter sido relançada"

    spans = exporter.get_finished_spans()
    assert spans[0].status.status_code == StatusCode.ERROR


def test_traced_agent_aninha_como_filho_do_span_raiz():
    tracer, exporter = _tracer_de_teste()

    with span_raiz("tr-xyz", "cliente-y", tracer=tracer):
        with traced_agent("extrator", tracer=tracer) as span:
            span.set_attribute("tokens_input", 296)
            span.set_attribute("modelo", "claude-haiku-4-5")

    spans = exporter.get_finished_spans()
    assert len(spans) == 2

    span_extrator = next(s for s in spans if s.name == "extrator")
    span_raiz_exportado = next(s for s in spans if s.name == "analise-completa")

    assert span_extrator.parent.span_id == span_raiz_exportado.context.span_id
    assert span_extrator.attributes["tokens_input"] == 296
    assert span_extrator.attributes["modelo"] == "claude-haiku-4-5"
    assert span_extrator.attributes["agent"] == "extrator"


def test_pipeline_completo_gera_arvore_de_4_spans():
    """Simula os 3 agentes + o span raiz -- a forma que orchestration/graph.py
    vai gerar de verdade quando conectarmos isso ao pipeline."""
    tracer, exporter = _tracer_de_teste()

    with span_raiz("tr-completo", "cliente-z", tracer=tracer):
        for nome in ("extrator", "validador", "decisao"):
            with traced_agent(nome, tracer=tracer) as span:
                span.set_attribute("latencia_ms", 123.0)

    spans = exporter.get_finished_spans()
    nomes = {s.name for s in spans}
    assert nomes == {"analise-completa", "extrator", "validador", "decisao"}

    raiz = next(s for s in spans if s.name == "analise-completa")
    for nome_agente in ("extrator", "validador", "decisao"):
        filho = next(s for s in spans if s.name == nome_agente)
        assert filho.parent.span_id == raiz.context.span_id


def test_traced_agent_marca_erro_quando_agente_falha():
    tracer, exporter = _tracer_de_teste()

    with span_raiz("tr-erro-agente", "cliente-w", tracer=tracer):
        try:
            with traced_agent("decisao", tracer=tracer):
                raise RuntimeError("modelo indisponível")
        except RuntimeError:
            pass

    spans = exporter.get_finished_spans()
    span_decisao = next(s for s in spans if s.name == "decisao")
    span_raiz_exportado = next(s for s in spans if s.name == "analise-completa")

    assert span_decisao.status.status_code == StatusCode.ERROR
    # o erro do filho não propaga automaticamente pro status do pai --
    # cada span reporta seu próprio sucesso/falha.
    assert span_raiz_exportado.status.status_code == StatusCode.OK
