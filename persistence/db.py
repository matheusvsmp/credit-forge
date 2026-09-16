"""Conexão com o banco (engine/session do SQLAlchemy).

Por padrão aponta para o PostgreSQL do docker-compose.yml. Pode ser
sobrescrito via `DATABASE_URL` -- os testes rápidos usam
`sqlite:///:memory:` para não depender do Docker estar rodando; testes de
integração de verdade usam o Postgres real.
"""

import os

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

DATABASE_URL_PADRAO = "postgresql+psycopg://credit_forge:credit_forge@localhost:5433/credit_forge"


def criar_engine(database_url: str | None = None) -> Engine:
    url = database_url or os.environ.get("DATABASE_URL", DATABASE_URL_PADRAO)
    return create_engine(url, pool_pre_ping=True)


def criar_session_factory(engine: Engine | None = None) -> sessionmaker[Session]:
    engine = engine or criar_engine()
    return sessionmaker(bind=engine)
