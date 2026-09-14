"""Memória de sessão compartilhada entre agentes ao longo de uma execução.

Diferente de "memória de conversa" (várias idas e vindas dentro de UM caso),
aqui a memória acumula o que cada agente concluiu em CADA caso processado
na mesma execução -- permite que um agente "lembre" de padrões vistos em
casos anteriores da mesma sessão, em vez de analisar cada caso isolado.

Quando o histórico cresce demais, mensagens antigas são resumidas por uma
chamada real ao Claude (compressão), para o contexto não crescer sem limite.
"""

from datetime import datetime

import anthropic

from agents.llm_utils import MODEL_ID, limpar_json_da_resposta
from observability.tracking import VerificadorOrcamento, estimar_custo


class AgentMemory:
    """Acumula histórico de execuções de agentes numa sessão, com compressão."""

    def __init__(
        self,
        client: anthropic.Anthropic | None = None,
        verificador: VerificadorOrcamento | None = None,
        max_turns: int = 10,
        compress_after: int = 20,
    ):
        self.messages: list[dict] = []
        self.max_turns = max_turns
        self.compress_after = compress_after
        self.compression_summary: str | None = None
        self.client = client
        self.verificador = verificador

    def add(self, role: str, content: str, metadata: dict | None = None) -> None:
        """Adiciona uma entrada ao histórico (ex: o que um agente concluiu num caso)."""
        self.messages.append(
            {
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat(),
                "metadata": metadata or {},
            }
        )
        if len(self.messages) > self.compress_after:
            self._compress()

    def _compress(self) -> None:
        """Resume as mensagens mais antigas (tudo exceto as `max_turns` mais
        recentes) numa chamada real ao Claude, e descarta o restante."""
        if self.client is None:
            raise RuntimeError(
                "AgentMemory precisa de um client Anthropic para comprimir "
                "o histórico (passe `client=...` no construtor)."
            )

        antigas = self.messages[: -self.max_turns]
        recentes = self.messages[-self.max_turns :]

        texto_antigas = "\n".join(f"{m['role']}: {m['content']}" for m in antigas)
        prompt_base = (
            f"[Resumo anterior]: {self.compression_summary}\n\n"
            if self.compression_summary
            else ""
        )

        response = self.client.messages.create(
            model=MODEL_ID,
            max_tokens=200,
            system=(
                "Resuma o histórico de execuções de agentes abaixo em 3-4 frases, "
                "preservando padrões e decisões importantes. Responda apenas com "
                "o texto do resumo, sem formatação."
            ),
            messages=[{"role": "user", "content": f"{prompt_base}{texto_antigas}"}],
        )

        if self.verificador is not None:
            custo = estimar_custo(
                tokens_input=response.usage.input_tokens,
                tokens_output=response.usage.output_tokens,
            )
            self.verificador.registrar(custo)

        texto = next(b.text for b in response.content if b.type == "text")
        self.compression_summary = limpar_json_da_resposta(texto)
        self.messages = recentes

    def get_context(self) -> str:
        """Contexto para injetar no prompt de um agente: resumo + mensagens recentes."""
        recentes = self.messages[-self.max_turns :]
        partes = []
        if self.compression_summary:
            partes.append(f"[Histórico anterior resumido]: {self.compression_summary}")
        partes.extend(f"{m['role']}: {m['content']}" for m in recentes)
        return "\n".join(partes)
