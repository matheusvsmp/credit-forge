"""Circuit Breaker: protege o sistema de martelar uma API externa que já
está falhando muito, dando um "tempo de respiro" antes de tentar de novo.

Três estados:
- CLOSED: tudo normal, chamadas passam direto.
- OPEN: falhas recentes ultrapassaram o limite -- toda chamada é rejeitada
  IMEDIATAMENTE (sem nem tentar), até `reset_timeout_s` se passar.
- HALF_OPEN: o tempo de espera passou; a PRÓXIMA chamada é uma "sonda" --
  se der certo, volta a CLOSED; se falhar, volta a OPEN.
"""

import time
from enum import Enum
from typing import Callable, TypeVar

T = TypeVar("T")


class EstadoCircuito(str, Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitoAbertoError(Exception):
    """Levantado quando o circuito está OPEN e a chamada é rejeitada sem tentar."""


class CircuitBreaker:
    """Trava de resiliência de nível de serviço (não por chamada individual).

    `relogio` é injetável (por padrão `time.monotonic`) para os testes
    controlarem a passagem de tempo sem precisar de `time.sleep()` de verdade.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout_s: float = 30.0,
        relogio: Callable[[], float] = time.monotonic,
    ):
        self.failure_threshold = failure_threshold
        self.reset_timeout_s = reset_timeout_s
        self._relogio = relogio

        self.state = EstadoCircuito.CLOSED
        self.failure_count = 0
        self._instante_ultima_falha: float | None = None

    def call(self, func: Callable[..., T], *args, **kwargs) -> T:
        """Executa `func` protegida pelo circuito.

        Levanta `CircuitoAbertoError` sem chamar `func` se o circuito estiver
        OPEN e o tempo de espera ainda não tiver passado.
        """
        if self.state == EstadoCircuito.OPEN:
            tempo_desde_falha = self._relogio() - self._instante_ultima_falha
            if tempo_desde_falha < self.reset_timeout_s:
                raise CircuitoAbertoError(
                    f"Circuito aberto -- rejeitando chamada "
                    f"({tempo_desde_falha:.1f}s de {self.reset_timeout_s:.1f}s de espera)"
                )
            self.state = EstadoCircuito.HALF_OPEN

        try:
            resultado = func(*args, **kwargs)
        except Exception:
            self._registrar_falha()
            raise
        else:
            self._registrar_sucesso()
            return resultado

    def _registrar_falha(self) -> None:
        self.failure_count += 1
        self._instante_ultima_falha = self._relogio()
        # Em HALF_OPEN, qualquer falha na "sonda" já reabre o circuito
        # (não espera acumular `failure_threshold` de novo).
        if self.state == EstadoCircuito.HALF_OPEN or self.failure_count >= self.failure_threshold:
            self.state = EstadoCircuito.OPEN

    def _registrar_sucesso(self) -> None:
        self.failure_count = 0
        self.state = EstadoCircuito.CLOSED
