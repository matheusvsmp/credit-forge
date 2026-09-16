"""Dependências injetáveis do FastAPI: client Anthropic, verificador de
orçamento e o grafo compilado do pipeline.

Tudo é construído uma única vez por processo (via `lru_cache`) e reaproveitado
entre requisições -- um client HTTP novo a cada request seria desperdício de
conexões, e um VerificadorOrcamento novo a cada request perderia o controle
de gasto acumulado do serviço inteiro.

Em testes, `api/main.py` troca `get_pipeline_app` por uma versão que devolve
o grafo MOCK via `app.dependency_overrides` -- nenhuma chamada real à API
acontece rodando a suíte de testes.
"""

import os
from functools import lru_cache
from typing import Iterator

import anthropic
from fastapi import Depends, Header, HTTPException
from sqlalchemy.orm import Session, sessionmaker

from observability.tracking import VerificadorOrcamento
from orchestration.circuit_breaker import CircuitBreaker
from orchestration.graph import construir_grafo_real
from persistence.db import criar_engine, criar_session_factory
from persistence.models import Base


@lru_cache
def get_anthropic_client() -> anthropic.Anthropic:
    return anthropic.Anthropic()


@lru_cache
def get_verificador() -> VerificadorOrcamento:
    """Orçamento do PROCESSO inteiro (todas as requisições), não por request --
    equivalente ao "budget alert" de nível de serviço do roteiro_final.md."""
    limite = float(os.environ.get("ORCAMENTO_MAXIMO_USD", "5.0"))
    return VerificadorOrcamento(limite_usd=limite)


@lru_cache
def get_pipeline_app():
    """Grafo LangGraph compilado (versão real), usado pelo endpoint de análise."""
    client = get_anthropic_client()
    verificador = get_verificador()
    return construir_grafo_real(client, verificador)


@lru_cache
def get_circuit_breaker() -> CircuitBreaker:
    """Circuit breaker do PROCESSO inteiro -- protege /api/v1/analise como um
    todo (não por agente), como descrito na Camada 2 do roteiro_final.md."""
    threshold = int(os.environ.get("CIRCUIT_BREAKER_THRESHOLD", "5"))
    timeout_s = float(os.environ.get("CIRCUIT_BREAKER_TIMEOUT_S", "30"))
    return CircuitBreaker(failure_threshold=threshold, reset_timeout_s=timeout_s)


@lru_cache
def get_api_keys_validas() -> frozenset[str]:
    """Chaves aceitas, vindas da variável de ambiente API_KEYS (separadas por
    vírgula). Ler por trás de uma função (em vez de uma constante de módulo)
    permite que os testes troquem isso via `dependency_overrides`, sem
    depender da ordem de import/variáveis de ambiente."""
    bruto = os.environ.get("API_KEYS", "")
    return frozenset(chave.strip() for chave in bruto.split(",") if chave.strip())


def verificar_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    chaves_validas: frozenset[str] = Depends(get_api_keys_validas),
) -> str:
    """Exige um header `X-API-Key` presente e cadastrado. Sem nenhuma chave
    configurada no ambiente, TODA requisição é rejeitada (falha segura --
    nunca "abre" a API por engano por falta de configuração)."""
    if not x_api_key or x_api_key not in chaves_validas:
        raise HTTPException(status_code=401, detail="API key ausente ou inválida")
    return x_api_key


@lru_cache
def get_session_factory() -> sessionmaker[Session]:
    """Cria o engine + as tabelas (se ainda não existirem) uma única vez por
    processo. Testes substituem esta dependência inteira por uma fábrica de
    sessão SQLite em memória via `dependency_overrides`."""
    engine = criar_engine()
    Base.metadata.create_all(engine)
    return criar_session_factory(engine)


def get_db_session(
    fabrica: sessionmaker[Session] = Depends(get_session_factory),
) -> Iterator[Session]:
    """Uma sessão POR REQUISIÇÃO -- aberta no início, fechada no fim (o
    `yield` é o que permite ao FastAPI rodar a limpeza depois da resposta)."""
    session = fabrica()
    try:
        yield session
    finally:
        session.close()


def obter_chave_de_rate_limit(request) -> str:
    """Chave usada pelo slowapi para agrupar requisições: por API key quando
    presente (rate limit por CLIENTE, como pede o roteiro_final.md), caindo
    para o IP de origem se o header não vier (ex: em /health)."""
    from slowapi.util import get_remote_address

    return request.headers.get("X-API-Key") or get_remote_address(request)
