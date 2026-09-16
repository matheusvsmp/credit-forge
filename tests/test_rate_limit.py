"""Testa o MECANISMO de rate limiting (slowapi) de forma isolada e rápida.

Não testamos o limite de produção (100/min) diretamente no app real: isso
exigiria 101 requisições e não provaria nada que o mecanismo em si já não
prove com um limite pequeno. Aqui montamos uma mini-app com limite de
2/minuto e confirmamos que a 3ª requisição no mesmo minuto recebe 429 --
mesma biblioteca, mesma configuração, só um número menor para o teste
rodar em milissegundos.
"""

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware


def _chave_por_header(request: Request) -> str:
    """No app real usamos a API key; aqui um header simples de teste --
    o ponto é provar que a cota é por CHAVE, não global ao processo."""
    return request.headers.get("X-Client-ID", "sem-id")


def _montar_app_de_teste(limite: str) -> FastAPI:
    limiter = Limiter(key_func=_chave_por_header)
    app = FastAPI()
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)

    @app.get("/ping")
    @limiter.limit(limite)
    def ping(request: Request):
        return {"pong": True}

    return app


def test_rate_limit_permite_ate_o_limite_e_bloqueia_depois():
    app = _montar_app_de_teste("2/minute")
    client = TestClient(app)

    assert client.get("/ping").status_code == 200
    assert client.get("/ping").status_code == 200
    resposta_excedida = client.get("/ping")

    assert resposta_excedida.status_code == 429


def test_rate_limit_e_por_chave_nao_global():
    """Dois clientes diferentes não compartilham a mesma cota."""
    app = _montar_app_de_teste("1/minute")
    client = TestClient(app)

    r1 = client.get("/ping", headers={"X-Client-ID": "cliente-a"})
    r2 = client.get("/ping", headers={"X-Client-ID": "cliente-b"})
    r3_mesmo_cliente_a = client.get("/ping", headers={"X-Client-ID": "cliente-a"})

    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r3_mesmo_cliente_a.status_code == 429
