"""Template Method para agentes que fazem UMA chamada estruturada ao Claude.

Elimina a duplicação que existia entre agente_extrator_real, agente_validador_real
e agente_decisao_real: montar a mensagem, chamar a API, checar truncamento,
limpar o JSON da resposta, validar contra um schema Pydantic e registrar o
custo no VerificadorOrcamento -- tudo isso é idêntico entre os agentes.

O que MUDA de agente para agente vira responsabilidade da subclasse:
- `system_prompt`: instrução de sistema fixa.
- `result_model`: classe Pydantic da saída esperada.
- `montar_conteudo(...)`: como construir o texto da mensagem a partir das
  entradas específicas daquele agente (assinatura livre por subclasse).
"""

import json
from abc import ABC, abstractmethod
from typing import Type, TypeVar

import anthropic
from pydantic import BaseModel

from agents.llm_utils import MODEL_ID, limpar_json_da_resposta
from observability.tracking import VerificadorOrcamento, estimar_custo
from orchestration.retry import com_retry_anthropic
from persistence.cache import PromptCacheBackend, calcular_chave_cache

ResultadoT = TypeVar("ResultadoT", bound=BaseModel)


class LLMAgent(ABC):
    """Classe base de um agente que chama a API da Anthropic e valida a saída."""

    system_prompt: str
    result_model: Type[BaseModel]

    # Tetos de max_tokens: um agente pode sobrescrever se sua saída for
    # tipicamente maior (ex: o Reflexor, que lista vários achados).
    max_tokens_haiku: int = 300
    max_tokens_outros_modelos: int = 600

    # TTL padrão do cache de prompt (Etapa 37) -- 7 dias, como no roteiro_final.md.
    cache_ttl_segundos: int = 7 * 24 * 3600

    def __init__(
        self,
        client: anthropic.Anthropic,
        verificador: VerificadorOrcamento,
        cache: PromptCacheBackend | None = None,
    ):
        self.client = client
        self.verificador = verificador
        # `cache=None` por padrão: desliga o cache -- preserva o
        # comportamento exato dos grafos já validados (Etapa 15/22/30), que
        # nunca pedem cache explicitamente. Passe um PromptCacheBackend para
        # habilitar.
        self.cache = cache

    @abstractmethod
    def montar_conteudo(self, *args, **kwargs) -> str:
        """Constrói o texto da mensagem do usuário para esta chamada."""
        raise NotImplementedError

    @com_retry_anthropic()
    def _chamar_api(self, model_id: str, max_tokens: int, conteudo: str):
        """Isola a chamada de rede em si -- só ela é protegida por retry.
        Erros de parsing/validação (que acontecem DEPOIS desta chamada) nunca
        deveriam ser reexecutados contra a API, só erros de rede/rate-limit."""
        return self.client.messages.create(
            model=model_id,
            max_tokens=max_tokens,
            system=self.system_prompt,
            messages=[{"role": "user", "content": conteudo}],
        )

    def executar(self, *args, model_id: str = MODEL_ID, **kwargs) -> tuple[BaseModel, dict]:
        """Faz a chamada real à API e devolve (resultado_validado, info_uso).

        Com cache habilitado: um prompt+modelo IDÊNTICO a uma chamada
        anterior (dentro do TTL) devolve o resultado guardado, sem chamar a
        API -- `info_uso` reporta tokens=0 (nenhum token novo foi gasto) e
        `cache_hit=True`, para quem estiver observando saber a diferença.
        """
        conteudo = self.montar_conteudo(*args, **kwargs)

        chave_cache = None
        if self.cache is not None:
            chave_cache = calcular_chave_cache(self.system_prompt, conteudo, model_id)
            cacheado = self.cache.get(chave_cache)
            if cacheado is not None:
                resultado = self.result_model(**cacheado)
                return resultado, {"tokens_input": 0, "tokens_output": 0, "cache_hit": True}

        max_tokens = (
            self.max_tokens_haiku if model_id == MODEL_ID else self.max_tokens_outros_modelos
        )

        response = self._chamar_api(model_id, max_tokens, conteudo)

        custo = estimar_custo(
            tokens_input=response.usage.input_tokens,
            tokens_output=response.usage.output_tokens,
            model_id=model_id,
        )
        self.verificador.registrar(custo)

        if response.stop_reason == "max_tokens":
            raise RuntimeError(
                f"Resposta de {type(self).__name__} ({model_id}) foi cortada por "
                "atingir max_tokens -- aumente o limite em vez de parsear um "
                "JSON incompleto."
            )

        texto = next(b.text for b in response.content if b.type == "text")
        dados = json.loads(limpar_json_da_resposta(texto))
        resultado = self.result_model(**dados)

        if self.cache is not None:
            self.cache.set(chave_cache, dados, ttl_segundos=self.cache_ttl_segundos)

        info_uso = {
            "tokens_input": response.usage.input_tokens,
            "tokens_output": response.usage.output_tokens,
            "cache_hit": False,
        }
        return resultado, info_uso
