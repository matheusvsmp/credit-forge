"""Retry com backoff exponencial + fallback de modelo.

Dois mecanismos complementares:
- `com_retry_anthropic`: decorator que tenta de novo (com espera crescente
  entre tentativas) quando a Anthropic responde com erro de conexão ou de
  rate limit -- erros tipicamente transitórios, que sumem numa 2ª tentativa.
- `executar_com_fallback`: se mesmo com retry a chamada continuar falhando,
  tenta UMA vez com um modelo alternativo (ex: Haiku -> Sonnet) antes de
  desistir de vez.
"""

from typing import TYPE_CHECKING, TypeVar

import anthropic
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

if TYPE_CHECKING:  # evita import circular em runtime (agents.base importa este módulo)
    from agents.base import LLMAgent

T = TypeVar("T")

# Erros da Anthropic considerados transitórios (valem retry). Erros de
# validação (400) ou autenticação (401) NÃO entram aqui -- tentar de novo
# não resolveria, só gastaria mais uma chamada.
ERROS_TRANSITORIOS = (anthropic.APIConnectionError, anthropic.RateLimitError)


def com_retry_anthropic(
    max_tentativas: int = 3,
    espera_minima_s: float = 1.0,
    espera_maxima_s: float = 10.0,
):
    """Fábrica de decorator de retry -- parametrizável para os testes usarem
    tempos de espera minúsculos (evita testes lentos de verdade)."""
    return retry(
        stop=stop_after_attempt(max_tentativas),
        wait=wait_exponential(multiplier=1, min=espera_minima_s, max=espera_maxima_s),
        retry=retry_if_exception_type(ERROS_TRANSITORIOS),
        reraise=True,  # após esgotar as tentativas, relança o erro ORIGINAL
    )


def executar_com_fallback(
    agente: "LLMAgent",
    *args,
    model_id_principal: str,
    model_id_fallback: str,
    **kwargs,
) -> tuple:
    """Executa `agente.executar(...)` no modelo principal; se falhar (mesmo
    após os retries automáticos do agente), tenta UMA vez no modelo de
    fallback antes de propagar o erro.

    Devolve (resultado, info_uso, model_id_usado) -- o 3º elemento é
    necessário para a observabilidade saber qual modelo respondeu de fato,
    já que o fallback pode ter trocado de modelo no meio do caminho.
    """
    try:
        resultado, info_uso = agente.executar(*args, model_id=model_id_principal, **kwargs)
        return resultado, info_uso, model_id_principal
    except Exception:
        resultado, info_uso = agente.executar(*args, model_id=model_id_fallback, **kwargs)
        return resultado, info_uso, model_id_fallback
