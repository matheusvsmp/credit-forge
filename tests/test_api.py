"""Testes da API (Camada 1) usando o TestClient do FastAPI.

O grafo REAL (que gasta créditos) nunca roda aqui -- substituímos
`get_pipeline_app` pelo grafo MOCK e `get_api_keys_validas` por um conjunto
de teste fixo, via `app.dependency_overrides`. É o mesmo mecanismo de
Dependency Injection que a própria API usa em produção. Correção da decisão
em si já é coberta por test_agents_mock.py / test_pipeline_mock_langgraph.py;
aqui testamos o CONTRATO HTTP (status codes, forma da resposta, auth, trace_id).

Rate limiting é testado à parte, em test_rate_limit.py, com uma mini-app
isolada -- testar o limite real de produção (100/min) aqui seria lento e
frágil (dependeria de tempo real passando).

O circuit breaker também é sobrescrito por um com threshold muito alto: como
ele normalmente é um singleton por processo (`lru_cache`), sem isso as
falhas simuladas por `_GrafoQuebrado` se acumulariam entre os testes e
"vazariam" para os testes seguintes (que passariam a receber 503 em vez do
esperado). Ver test_orchestration.py para os testes de verdade do circuito.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from api.deps import get_api_keys_validas, get_circuit_breaker, get_pipeline_app, get_session_factory
from api.main import app
from orchestration.circuit_breaker import CircuitBreaker
from orchestration.graph import construir_grafo
from persistence.models import Base

CHAVE_TESTE = "test-key-123"
HEADERS_VALIDOS = {"X-API-Key": CHAVE_TESTE}

# Um único banco SQLite em memória, compartilhado por toda a suíte deste
# arquivo (StaticPool garante que todas as "conexões" caem na MESMA base em
# memória -- sem isso, cada sessão veria um banco vazio diferente).
_engine_teste = create_engine(
    "sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool
)
Base.metadata.create_all(_engine_teste)
_fabrica_sessao_teste = sessionmaker(bind=_engine_teste)

app.dependency_overrides[get_pipeline_app] = construir_grafo
app.dependency_overrides[get_api_keys_validas] = lambda: frozenset({CHAVE_TESTE})
app.dependency_overrides[get_circuit_breaker] = lambda: CircuitBreaker(
    failure_threshold=1_000_000, reset_timeout_s=0.0
)
app.dependency_overrides[get_session_factory] = lambda: _fabrica_sessao_teste

client = TestClient(app)


class _GrafoQuebrado:
    """Substituto do grafo que sempre levanta erro -- simula uma falha
    interna do pipeline (ex: API da Anthropic fora do ar) sem precisar
    provocar isso de verdade."""

    def invoke(self, estado_inicial):
        raise RuntimeError("falha simulada no pipeline")


def _forcar_falha():
    raise RuntimeError("falha para abrir o circuito")


def test_health_retorna_ok():
    """/health não exige API key -- usado por load balancers."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_analise_com_payload_valido_retorna_200(casos_teste):
    caso = casos_teste[0]
    response = client.post(
        "/api/v1/analise",
        json={"documento": caso["documento"], "cliente": "teste"},
        headers=HEADERS_VALIDOS,
    )

    assert response.status_code == 200
    corpo = response.json()
    assert corpo["trace_id"].startswith("tr-")
    assert corpo["decisao_final"] in {"Risco Baixo", "Risco Médio", "Risco Alto"}
    assert 0.0 <= corpo["confianca"] <= 1.0
    assert corpo["motivo"]


def test_analise_sem_api_key_retorna_401(casos_teste):
    response = client.post(
        "/api/v1/analise", json={"documento": casos_teste[0]["documento"]}
    )
    assert response.status_code == 401


def test_analise_com_api_key_invalida_retorna_401(casos_teste):
    response = client.post(
        "/api/v1/analise",
        json={"documento": casos_teste[0]["documento"]},
        headers={"X-API-Key": "chave-errada"},
    )
    assert response.status_code == 401


def test_analise_documento_vazio_retorna_400():
    response = client.post(
        "/api/v1/analise", json={"documento": ""}, headers=HEADERS_VALIDOS
    )

    assert response.status_code == 400
    corpo = response.json()
    assert corpo["trace_id"].startswith("tr-")


def test_analise_sem_campo_documento_retorna_400():
    response = client.post("/api/v1/analise", json={}, headers=HEADERS_VALIDOS)
    assert response.status_code == 400


def test_analise_gera_trace_id_diferente_por_requisicao(casos_teste):
    documento = casos_teste[0]["documento"]

    r1 = client.post(
        "/api/v1/analise", json={"documento": documento}, headers=HEADERS_VALIDOS
    )
    r2 = client.post(
        "/api/v1/analise", json={"documento": documento}, headers=HEADERS_VALIDOS
    )

    assert r1.json()["trace_id"] != r2.json()["trace_id"]


def test_analise_erro_interno_retorna_500_com_trace_id(casos_teste):
    app.dependency_overrides[get_pipeline_app] = lambda: _GrafoQuebrado()
    try:
        response = client.post(
            "/api/v1/analise",
            json={"documento": casos_teste[0]["documento"]},
            headers=HEADERS_VALIDOS,
        )
        assert response.status_code == 500
        corpo = response.json()
        assert corpo["trace_id"].startswith("tr-")
        assert "falha simulada" in corpo["detalhe"]
    finally:
        app.dependency_overrides[get_pipeline_app] = construir_grafo


def test_analise_com_circuito_aberto_retorna_503(casos_teste):
    circuito_ja_aberto = CircuitBreaker(failure_threshold=1, reset_timeout_s=1_000_000_000)
    with pytest.raises(RuntimeError):
        circuito_ja_aberto.call(_forcar_falha)

    app.dependency_overrides[get_circuit_breaker] = lambda: circuito_ja_aberto
    try:
        response = client.post(
            "/api/v1/analise",
            json={"documento": casos_teste[0]["documento"]},
            headers=HEADERS_VALIDOS,
        )
        assert response.status_code == 503
        assert response.json()["trace_id"].startswith("tr-")
    finally:
        app.dependency_overrides[get_circuit_breaker] = lambda: CircuitBreaker(
            failure_threshold=1_000_000, reset_timeout_s=0.0
        )


def test_analise_grava_auditoria_consultavel_por_trace_id(casos_teste):
    resposta = client.post(
        "/api/v1/analise",
        json={"documento": casos_teste[0]["documento"], "cliente": "cliente-auditoria"},
        headers=HEADERS_VALIDOS,
    )
    trace_id = resposta.json()["trace_id"]

    auditoria = client.get(f"/audit/trace/{trace_id}")

    assert auditoria.status_code == 200
    linhas = auditoria.json()
    assert {linha["agente"] for linha in linhas} == {"extrator", "validador", "decisao"}
    assert all(linha["status"] == "success" for linha in linhas)


def test_audit_trace_404_quando_trace_id_nao_existe():
    resposta = client.get("/audit/trace/tr-nunca-existiu")
    assert resposta.status_code == 404


def test_analise_com_erro_tambem_grava_auditoria(casos_teste):
    app.dependency_overrides[get_pipeline_app] = lambda: _GrafoQuebrado()
    try:
        resposta = client.post(
            "/api/v1/analise",
            json={"documento": casos_teste[0]["documento"], "cliente": "cliente-com-erro"},
            headers=HEADERS_VALIDOS,
        )
        trace_id = resposta.json()["trace_id"]
    finally:
        app.dependency_overrides[get_pipeline_app] = construir_grafo

    auditoria = client.get(f"/audit/trace/{trace_id}")
    assert auditoria.status_code == 200
    linhas = auditoria.json()
    assert len(linhas) == 1
    assert linhas[0]["status"] == "error"


def test_costs_by_client_agrega_apos_multiplas_analises(casos_teste):
    for _ in range(2):
        client.post(
            "/api/v1/analise",
            json={"documento": casos_teste[0]["documento"], "cliente": "cliente-custos-teste"},
            headers=HEADERS_VALIDOS,
        )

    relatorio = client.get("/costs/by-client").json()
    linha_do_cliente = next(l for l in relatorio if l["cliente"] == "cliente-custos-teste")

    assert linha_do_cliente["chamadas"] == 6  # 2 análises x 3 agentes
    # mock: tokens=0, então custo é exatamente 0.0 -- confere que a agregação
    # não quebra nem gera valores estranhos com custo zerado.
    assert linha_do_cliente["custo_total"] == 0.0
