# Sistema de Análise de Crédito Automático com Avaliação de Qualidade

Sistema multiagente que analisa documentos de crédito B2B (simplificados), valida os dados
extraídos e toma uma decisão de risco (Baixo/Médio/Alto) — com orquestração via LangGraph,
avaliação formal contra um gabarito (accuracy/precision/recall/F1) e observabilidade
(tokens, custo, latência) por agente.

Duas implementações lado a lado, com a **mesma arquitetura**:
- **Mock**: regras determinísticas (regex + fórmulas), sem custo, sem chamadas de rede.
- **Real**: Claude (`claude-haiku-4-5`) via API da Anthropic.

## Arquitetura

```
documento
   │
   ▼
┌───────────┐    ┌────────────┐    ┌───────────┐    ┌──────────────┐
│ Extrator  │───▶│ Validador  │───▶│  Decisão  │───▶│ Consolidador │
└───────────┘    └────────────┘    └───────────┘    └──────────────┘
     │                  │                 │                  │
     ▼                  ▼                 ▼                  ▼
ExtractionResult   ValidationResult  DecisionResult   ConsolidatedResult
                                                       (struct final + traces)
```

Orquestrado por um grafo de estados do **LangGraph** (`main.py`): cada nó recebe o estado
atual do caso e devolve só as chaves que alterou; o LangGraph mescla tudo automaticamente.

## Estrutura do projeto

```
main.py                   # Orquestrador: grafo LangGraph (versões mock e real)
agents/
├── extrator.py            # Extração de campos estruturados do documento
├── validator.py           # Checagem de coerência entre os campos + flags de risco
├── decision.py             # Veredito final de risco (Baixo/Médio/Alto)
└── llm_utils.py            # Utilidades compartilhadas pelos agentes reais (model id, parsing)
data/
├── schemas.py              # Contratos Pydantic trocados entre os agentes
└── test_cases.json         # 15 casos de teste com ground truth (gabarito)
evaluation/
├── metrics.py              # accuracy / precision / recall / F1 (via scikit-learn)
└── evaluator.py             # Roda o pipeline sobre o dataset e monta o relatório final
observability/
├── tracking.py              # Latência real, estimativa de custo, guardião de orçamento
└── logger.py                 # Logging estruturado em JSON
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\pip install anthropic langgraph pydantic pytest scikit-learn python-json-logger python-dotenv
```

Crie um arquivo `.env` na raiz com sua chave da Anthropic (nunca commitado — está no `.gitignore`):
```
ANTHROPIC_API_KEY=sk-ant-...
```

## Como rodar

**Pipeline mock** (grátis, sem API):
```powershell
.venv\Scripts\python.exe main.py
```

**Relatório completo de avaliação (mock)** — accuracy/F1 + confiança por agente + custo/latência:
```powershell
.venv\Scripts\python.exe -m evaluation.evaluator
```

**Pipeline com API real** (gasta créditos — ver seção de custo abaixo):
```python
from dotenv import load_dotenv
load_dotenv()
import anthropic
from main import construir_grafo_real
from evaluation.evaluator import avaliar_pipeline
from observability.tracking import VerificadorOrcamento
import json

with open("data/test_cases.json", encoding="utf-8") as f:
    casos = json.load(f)["test_cases"]

client = anthropic.Anthropic()
verificador = VerificadorOrcamento(limite_usd=0.20)  # trava de segurança
app_real = construir_grafo_real(client, verificador)
relatorio = avaliar_pipeline(app_real, casos)
```

## Decisões de projeto e por quê

- **Mock antes de real**: cada peça (schemas, agentes, orquestração, avaliação,
  observabilidade) foi validada com regras determinísticas antes de gastar créditos de API,
  separando "a estrutura funciona?" de "o LLM se comporta bem?".
- **Modelo `claude-haiku-4-5`**: o mais barato da Anthropic no momento, escolhido pelo
  orçamento reduzido deste projeto de estudo ($5 de créditos).
- **`VerificadorOrcamento`** (`observability/tracking.py`): trava que interrompe a execução
  se o custo acumulado ultrapassar um limite configurável — proteção contra bugs de loop ou
  datasets maiores que o esperado.
- **`max_tokens=300`** nas chamadas reais: a saída é sempre um JSON curto; um teto baixo evita
  gasto desnecessário sem risco de truncar a resposta.
- **Parsing defensivo de JSON** (`agents/llm_utils.py`): o LLM às vezes envolve a resposta em
  ` ```json ... ``` ` mesmo quando instruído a não fazer isso — o parsing remove essas cercas
  antes de tentar `json.loads`.

## Resultados (15 casos de teste)

| | Mock | Real (Haiku 4.5) |
|---|---|---|
| Accuracy / Precision / Recall / F1 | 0.933 / 0.944 / 0.933 / 0.933 | 0.933 / 0.944 / 0.933 / 0.933 |
| Latência média por caso | ~0.07 ms | ~4.3 s |
| Custo total (15 casos) | $0.00 | ~$0.04 |

As duas versões acertam a mesma proporção de casos, mas **erram casos diferentes** — prova
de que o LLM real não é uma cópia do modelo de regras, e sim um "raciocinador" com um perfil
de erro próprio.

## Limitações conhecidas

- Dataset de apenas 15 casos — pequeno demais para conclusões estatísticas fortes, serve
  para fins didáticos e de demonstração.
- `ValidationResult.coerente` e as `flags` do validador mock são heurísticas simples, não
  substituem um motor de regras de crédito real.
- Sem persistência: cada execução do pipeline é stateless (não grava histórico em banco).
