"""Cache de prompt (Camada 6a do roteiro_final.md): evita rechamar a API
para o MESMO prompt + modelo dentro de um TTL.

Interface abstrata (Dependency Inversion) + 2 implementações:
- `RedisCacheBackend`: produção, via docker-compose.yml.
- `InMemoryCacheBackend`: fallback automático se o Redis não estiver
  acessível, e o que os testes rápidos usam (sem depender de Docker).
"""

import hashlib
import json
import os
import time
from abc import ABC, abstractmethod


class PromptCacheBackend(ABC):
    """Contrato mínimo: guardar e recuperar um valor JSON-serializável por chave."""

    @abstractmethod
    def get(self, chave: str) -> dict | None:
        raise NotImplementedError

    @abstractmethod
    def set(self, chave: str, valor: dict, ttl_segundos: int) -> None:
        raise NotImplementedError


class InMemoryCacheBackend(PromptCacheBackend):
    """Cache em memória do próprio processo -- fallback e uso em testes.

    `relogio` é injetável (mesmo princípio do CircuitBreaker, Etapa 29) --
    testes de expiração de TTL usam um relógio falso, sem `time.sleep()`
    real nem a instabilidade de timing que isso pode introduzir.
    """

    def __init__(self, relogio=time.monotonic):
        self._dados: dict[str, tuple[float, dict]] = {}
        self._relogio = relogio

    def get(self, chave: str) -> dict | None:
        item = self._dados.get(chave)
        if item is None:
            return None
        expira_em, valor = item
        if self._relogio() > expira_em:
            del self._dados[chave]
            return None
        return valor

    def set(self, chave: str, valor: dict, ttl_segundos: int) -> None:
        self._dados[chave] = (self._relogio() + ttl_segundos, valor)


class RedisCacheBackend(PromptCacheBackend):
    """Cache real via Redis -- compartilhado entre múltiplos processos/réplicas
    da API (ao contrário do InMemoryCacheBackend, que é por processo)."""

    def __init__(self, redis_client):
        self._redis = redis_client

    def get(self, chave: str) -> dict | None:
        bruto = self._redis.get(chave)
        return json.loads(bruto) if bruto else None

    def set(self, chave: str, valor: dict, ttl_segundos: int) -> None:
        self._redis.set(chave, json.dumps(valor), ex=ttl_segundos)


def calcular_chave_cache(system_prompt: str, conteudo: str, model_id: str) -> str:
    """SHA-256 de (modelo + prompt de sistema + conteúdo) -- muda 1 caractere,
    muda a chave inteira (não tem "quase cache hit")."""
    bruto = f"{model_id}:{system_prompt}:{conteudo}"
    return hashlib.sha256(bruto.encode("utf-8")).hexdigest()


def criar_backend_padrao() -> PromptCacheBackend:
    """Tenta conectar no Redis (REDIS_URL); cai para memória se não conseguir
    (Redis fora do ar não deveria derrubar a aplicação inteira -- só perde
    o benefício de custo do cache)."""
    redis_url = os.environ.get("REDIS_URL")
    if redis_url:
        try:
            import redis

            cliente = redis.from_url(redis_url, socket_connect_timeout=1)
            cliente.ping()
            return RedisCacheBackend(cliente)
        except Exception:
            pass
    return InMemoryCacheBackend()
