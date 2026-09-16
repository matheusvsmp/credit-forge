"""Testes do write-path/read-path de auditoria (Etapa 35), isolados da API
e do orquestrador -- SQLite em memória, sem precisar de Docker."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from agents.decision import agente_decisao_mock
from agents.extrator import agente_extrator_mock
from agents.validator import agente_validador_mock
from data.schemas import AgentTrace, ConsolidatedResult, MetricasGlobais
from persistence.audit import buscar_por_trace_id, custos_por_cliente, registrar_erro, registrar_resultado
from persistence.models import Base


@pytest.fixture()
def session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Sessao = sessionmaker(bind=engine)
    with Sessao() as s:
        yield s


def _resultado_consolidado_de_exemplo(caso_documento: str, tokens: int = 200) -> ConsolidatedResult:
    """Monta um ConsolidatedResult plausível reaproveitando os agentes mock
    (Etapa 6), só trocando os traces para simular custo/tokens reais."""
    extracao = agente_extrator_mock(caso_documento)
    validacao = agente_validador_mock(extracao)
    decisao = agente_decisao_mock(extracao, validacao)

    trace = AgentTrace(tokens_input=tokens, tokens_output=50, latencia_ms=1000.0)
    return ConsolidatedResult(
        empresa=extracao.empresa,
        decisao_final=decisao.decisao_final,
        motivo=decisao.motivo,
        score_confianca=decisao.confianca,
        extracao=extracao,
        validacao=validacao,
        decisao=decisao,
        traces={"extrator": trace, "validador": trace, "decisao": trace},
        metricas_globais=MetricasGlobais(total_tokens=750, custo_estimado=0.001, latencia_total_ms=3000.0),
    )


DOCUMENTO_EXEMPLO = (
    "Empresa: Comercial Nortesul. Faturamento anual: R$ 5M. Anos no mercado: 10. "
    "Histórico de pagamento: 0 atrasos em 24 meses. Dívida atual: R$ 200k. Capital social: R$ 1M."
)


def test_registrar_resultado_grava_uma_linha_por_agente(session):
    resultado = _resultado_consolidado_de_exemplo(DOCUMENTO_EXEMPLO)

    registrar_resultado(session, "tr-teste1", "cliente-a", resultado)

    linhas = buscar_por_trace_id(session, "tr-teste1")
    assert {linha.agente for linha in linhas} == {"extrator", "validador", "decisao"}
    assert all(linha.status == "success" for linha in linhas)
    assert all(linha.cliente == "cliente-a" for linha in linhas)


def test_registrar_resultado_grava_output_data_especifico_do_agente(session):
    resultado = _resultado_consolidado_de_exemplo(DOCUMENTO_EXEMPLO)

    registrar_resultado(session, "tr-teste2", "cliente-b", resultado)

    linhas = {linha.agente: linha for linha in buscar_por_trace_id(session, "tr-teste2")}
    assert linhas["extrator"].output_data["empresa"] == resultado.empresa
    assert linhas["decisao"].output_data["decisao_final"] == resultado.decisao_final


def test_buscar_por_trace_id_nao_mistura_traces_diferentes(session):
    r1 = _resultado_consolidado_de_exemplo(DOCUMENTO_EXEMPLO)
    registrar_resultado(session, "tr-um", "cliente-a", r1)
    registrar_resultado(session, "tr-dois", "cliente-a", r1)

    linhas_um = buscar_por_trace_id(session, "tr-um")
    assert len(linhas_um) == 3
    assert all(linha.trace_id == "tr-um" for linha in linhas_um)


def test_registrar_erro_grava_status_error_com_mensagem(session):
    registrar_erro(session, "tr-erro", "cliente-c", "decisao", "timeout na API")

    linhas = buscar_por_trace_id(session, "tr-erro")
    assert len(linhas) == 1
    assert linhas[0].status == "error"
    assert linhas[0].error_message == "timeout na API"


def test_custos_por_cliente_agrega_corretamente(session):
    resultado = _resultado_consolidado_de_exemplo(DOCUMENTO_EXEMPLO, tokens=1_000_000)
    registrar_resultado(session, "tr-a1", "cliente-caro", resultado)
    registrar_resultado(session, "tr-a2", "cliente-caro", resultado)

    resultado_barato = _resultado_consolidado_de_exemplo(DOCUMENTO_EXEMPLO, tokens=100)
    registrar_resultado(session, "tr-b1", "cliente-barato", resultado_barato)

    relatorio = {linha["cliente"]: linha for linha in custos_por_cliente(session, dias=30)}

    assert relatorio["cliente-caro"]["chamadas"] == 6  # 2 análises x 3 agentes
    assert relatorio["cliente-barato"]["chamadas"] == 3
    assert relatorio["cliente-caro"]["custo_total"] > relatorio["cliente-barato"]["custo_total"]


def test_custos_por_cliente_ignora_fora_da_janela_de_dias(session):
    resultado = _resultado_consolidado_de_exemplo(DOCUMENTO_EXEMPLO)
    registrar_resultado(session, "tr-x", "cliente-antigo", resultado)

    # janela de 0 dias -> nada deveria ter timestamp "desde" (agora mesmo,
    # sem margem) além do que acabou de ser inserido no mesmo instante;
    # usamos -1 pra garantir que fica no passado da janela.
    relatorio = custos_por_cliente(session, dias=-1)

    assert relatorio == []
