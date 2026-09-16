"""Testes de persistência (Etapa 34): schema SQLAlchemy.

Dois níveis:
1. Testes RÁPIDOS contra SQLite em memória -- não dependem do Docker estar
   rodando, cobrem que o schema/CRUD básico funciona.
2. Um teste de INTEGRAÇÃO contra o PostgreSQL real (docker-compose.yml) --
   pulado automaticamente se o Postgres não estiver acessível, em vez de
   quebrar a suíte inteira quando alguém rodar os testes sem Docker.
"""

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from persistence.db import DATABASE_URL_PADRAO, criar_engine
from persistence.models import AgentTraceDB, Base, PromptCache, TokenUsage


@pytest.fixture()
def sqlite_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Sessao = sessionmaker(bind=engine)
    with Sessao() as session:
        yield session


def test_agent_trace_db_insercao_e_leitura(sqlite_session):
    trace = AgentTraceDB(
        trace_id="tr-abc123",
        cliente="itau-prod",
        agente="extrator",
        input_data={"documento": "texto qualquer"},
        output_data={"empresa": "Empresa X", "faturamento_anual": 1000000},
        modelo_utilizado="claude-haiku-4-5",
        latencia_ms=1200,
        tokens_input=296,
        tokens_output=106,
        custo_usd=0.0008,
        status="success",
    )
    sqlite_session.add(trace)
    sqlite_session.commit()

    lido = sqlite_session.scalars(
        select(AgentTraceDB).where(AgentTraceDB.trace_id == "tr-abc123")
    ).one()
    assert lido.cliente == "itau-prod"
    assert lido.output_data["empresa"] == "Empresa X"
    assert lido.expires_at > lido.created_at


def test_agent_trace_db_permite_multiplos_agentes_do_mesmo_trace(sqlite_session):
    for agente in ("extrator", "validador", "decisao"):
        sqlite_session.add(
            AgentTraceDB(
                trace_id="tr-xyz",
                cliente="cliente-y",
                agente=agente,
                input_data={},
                output_data={},
                modelo_utilizado="claude-haiku-4-5",
                latencia_ms=100,
                tokens_input=10,
                tokens_output=5,
                custo_usd=0.0001,
                status="success",
            )
        )
    sqlite_session.commit()

    linhas = sqlite_session.scalars(
        select(AgentTraceDB).where(AgentTraceDB.trace_id == "tr-xyz")
    ).all()
    assert {linha.agente for linha in linhas} == {"extrator", "validador", "decisao"}


def test_agent_trace_db_registra_erro(sqlite_session):
    trace = AgentTraceDB(
        trace_id="tr-erro",
        cliente="cliente-z",
        agente="decisao",
        input_data={},
        output_data={},
        modelo_utilizado="claude-sonnet-5",
        latencia_ms=500,
        tokens_input=0,
        tokens_output=0,
        custo_usd=0.0,
        status="error",
        error_message="timeout",
    )
    sqlite_session.add(trace)
    sqlite_session.commit()

    lido = sqlite_session.scalars(
        select(AgentTraceDB).where(AgentTraceDB.trace_id == "tr-erro")
    ).one()
    assert lido.status == "error"
    assert lido.error_message == "timeout"


def test_token_usage_insercao(sqlite_session):
    sqlite_session.add(
        TokenUsage(
            trace_id="tr-abc123",
            modelo="claude-haiku-4-5",
            input_tokens=296,
            output_tokens=106,
            custo_usd=0.0008,
            cliente="itau-prod",
        )
    )
    sqlite_session.commit()

    linha = sqlite_session.scalars(
        select(TokenUsage).where(TokenUsage.trace_id == "tr-abc123")
    ).one()
    assert linha.input_tokens == 296


def test_prompt_cache_valores_padrao():
    hash_exemplo = "a" * 64
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Sessao = sessionmaker(bind=engine)
    with Sessao() as session:
        session.add(
            PromptCache(
                hash=hash_exemplo,
                prompt_text="prompt de exemplo",
                cached_response='{"ok": true}',
                modelo="claude-haiku-4-5",
            )
        )
        session.commit()

        linha = session.get(PromptCache, hash_exemplo)
        assert linha.hit_count == 0
        assert linha.ttl_minutes == 10080  # 7 dias


# ---------------------------------------------------------------------------
# Integração real contra PostgreSQL (docker-compose.yml)
# ---------------------------------------------------------------------------


def _postgres_disponivel() -> bool:
    try:
        engine = criar_engine(DATABASE_URL_PADRAO)
        with engine.connect():
            return True
    except Exception:
        return False


requer_postgres = pytest.mark.skipif(
    not _postgres_disponivel(),
    reason="PostgreSQL não está acessível (rode: docker compose up -d)",
)


@requer_postgres
def test_postgres_cria_tabelas_e_persiste_de_verdade():
    engine = criar_engine(DATABASE_URL_PADRAO)
    Base.metadata.create_all(engine)
    Sessao = sessionmaker(bind=engine)

    with Sessao() as session:
        session.add(
            AgentTraceDB(
                trace_id="tr-postgres-teste",
                cliente="teste-integracao",
                agente="extrator",
                input_data={"documento": "teste real"},
                output_data={"empresa": "Empresa Postgres"},
                modelo_utilizado="claude-haiku-4-5",
                latencia_ms=999,
                tokens_input=1,
                tokens_output=1,
                custo_usd=0.0001,
                status="success",
            )
        )
        session.commit()

    with Sessao() as session:
        lido = session.scalars(
            select(AgentTraceDB).where(AgentTraceDB.trace_id == "tr-postgres-teste")
        ).one()
        assert lido.cliente == "teste-integracao"
        assert lido.output_data["empresa"] == "Empresa Postgres"
        # limpeza: não deixar lixo acumulando no banco de dev a cada rodada
        session.delete(lido)
        session.commit()
