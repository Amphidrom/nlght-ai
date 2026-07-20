# nlght-ai
The layer between your app and your LLM — workflows, tools, memory, any provider.

<!-- release-badges:start -->
[![Tests](https://raw.githubusercontent.com/Amphidrom/nlght-ai/v0.1.2/.github/badges/tests.svg)](https://github.com/Amphidrom/nlght-ai/blob/v0.1.2/.github/badges/tests.svg)
[![Coverage](https://raw.githubusercontent.com/Amphidrom/nlght-ai/v0.1.2/.github/badges/coverage.svg)](https://github.com/Amphidrom/nlght-ai/blob/v0.1.2/.github/badges/coverage.svg)
[![PyPI](https://raw.githubusercontent.com/Amphidrom/nlght-ai/v0.1.2/.github/badges/pypi.svg)](https://github.com/Amphidrom/nlght-ai/blob/v0.1.2/.github/badges/pypi.svg)
<!-- release-badges:end -->

> **E2E scope note.** The test badge covers the cloud model-client e2e suite
> (`tests/e2e`); each provider skips itself when its API key is absent. The
> **local Ollama** model-client suite (`tests/local_e2e`) is intentionally **not
> run in CI** — no Ollama daemon is available there — so its absence/skip is
> expected. It is verified locally against a running `ollama serve` with
> `pytest tests/local_e2e -m local_e2e`.

---

## What it does

```
Client (OpenAI / Ollama API)
        ↓
  Protocol Detection
        ↓
  Trigger Resolution
        ↓
  Workflow Executor  ──→  Steps  ──→  Tools / Playbooks / LLM
        ↓
  Signal Emitter (streaming or batch)
        ↓
Client Response
```

A **workflow** is a sequence of **steps**. Each step receives a context with the current messages, an LLM client, a tool catalog, a playbook catalog, and optionally a session store (Hive Mind). Steps can call the LLM, use tools, read and write session memory, and emit results back to the caller — all without touching HTTP or protocol specifics.

---

## Features

| Feature | Description |
|---|---|
| **OpenAI-compatible API** | Drop-in replacement endpoint — works with any OpenAI client |
| **Ollama API** | Native Ollama protocol support |
| **Multi-provider** | Ollama, Anthropic Claude, OpenAI, Google Gemini |
| **Workflow engine** | Stateless step machine with configurable routing |
| **Tool system** | Register custom tools, call them from steps |
| **Playbook catalog** | YAML-defined agent capabilities, loaded from filesystem |
| **Session memory** | Hive Mind: tiered stores for directives, conversation, facts, working memory |
| **Session routing** | Configurable `SessionKeyResolver` per protocol adapter |
| **PostgreSQL persistence** | Workflow metadata and resource registry via SQLAlchemy |
| **Ports & Adapters** | Clean hexagonal architecture — core has zero framework dependencies |

---

## Install

Install [uv](https://docs.astral.sh/uv/), then add nlght-ai to your project:

```bash
uv add nlght-ai
```

Optional extras:

```bash
uv add 'nlght-ai[anthropic]'   # Anthropic Claude
uv add 'nlght-ai[openai]'      # OpenAI / Azure OpenAI
uv add 'nlght-ai[google]'      # Google Gemini
uv add 'nlght-ai[hive-mind]'   # File-backed session memory
uv add 'nlght-ai[docker]'      # Docker OS runtime
uv add 'nlght-ai[web-search]'  # Web search tool
```

This declares nlght-ai in the project, so later `uv sync` and `uv run`
commands keep it installed. For a standalone CLI that is independent of any
project environment, use `uv tool install nlght-ai` instead.

---

## Quickstart

**1. Configure** — create `.config/platform.yaml`:

```yaml
licensing:
  license_key: YOUR-LICENSE-KEY   # free developer licenses available

protocol_adapters:
  - name: openai-http
    kind: openai
    enabled: true
    config:
      base_path: /v1
      model_provider: ollama
      session_key_resolver:
        resolver: header
        key: x-session-id

gateways:
  - name: http-gateway
    kind: http
    enabled: true
    config:
      host: 0.0.0.0
      port: 8000

integrations:
  persistence:
    workflows:
      backend: postgres
      url: postgresql+asyncpg://user:pass@localhost/nlght
  model_providers:
    - name: ollama
      kind: ollama-local
      config:
        base_url: http://localhost:11434
        default_model: llama3.2
```

**2. Start PostgreSQL** (matches the `url` above — for local development):

```bash
docker run -d --name nlght-postgres \
  -e POSTGRES_USER=user -e POSTGRES_PASSWORD=pass -e POSTGRES_DB=nlght \
  -p 5432:5432 postgres:16
```

**3. Migrate the database** (initial setup and after every upgrade):

```bash
uv run nlght-ai migrate
```

**4. Start**:

```bash
uv run nlght-ai serve
# NLGHT_CONFIG=./my-config.yaml uv run nlght-ai serve
```

**5. Call**:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3.2", "messages": [{"role": "user", "content": "Hello!"}]}'
```

---

## Writing a Step

```python
from typing import ClassVar
from nlght.core.workflow.step import StepBase, StepResult, WorkflowStepContext

class SummarizeStep(StepBase):
    TYPE: ClassVar[str] = "summarize"

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        if ctx.llm is None:
            return StepResult(ctx=ctx, verdict="failed")
        system = "Summarize the conversation in three bullet points."
        await ctx.llm.call(messages=[{"role": "system", "content": system}] + ctx.messages)
        return StepResult(ctx=ctx, verdict="done")
```

Register and wire it up in your entrypoint, then reference it in a workflow definition stored in the database.

---

## Session Memory (Hive Mind)

Hive Mind gives each session four persistent stores:

- **DirectiveStore** — behavioral rules, set by tools or steps
- **ConversationStore** — turn summaries across requests
- **SessionResultStore** — promoted, validated session facts and evaluation outcomes; never raw LLM responses (survives restarts)
- **WorkingMemory** — transient task-scoped atoms, cleared after each task

Two coordinators — `simple` (in-memory) and `filesystem` (file-backed):

```yaml
integrations:
  persistence:
    hive_mind:
      provider:
        kind: filesystem
        config:
          base_dir: ./.sessions
```

Use `SystemPromptBuilder` in your steps to inject session context automatically:

```python
from nlght.core.hive_mind.system_prompt import SystemPromptBuilder

builder = SystemPromptBuilder()
system  = builder.build(mental_model=model, ego="You are a helpful assistant.", scope="task")
```

---

## Demo

The [Demo Showcase](docs/documentation/getting-started/showcase.mdx) introduces six complete workflows across grounding, tools, routing, session memory, and isolated runtime execution. Setup, migration, and run requirements are maintained by the demo repository linked there.

---

## Model Providers

| Provider | Extra | Config `kind` |
|---|---|---|
| Ollama (local) | _(none)_ | `ollama-local` |
| Anthropic Claude | `[anthropic]` | `anthropic` |
| OpenAI | `[openai]` | `openai` |
| Google Gemini | `[google]` | `google` |

---

## Architecture

nlght-ai follows a strict hexagonal (ports & adapters) architecture:

- **`core/`** — domain logic, zero framework dependencies, no I/O
- **`ports/`** — Protocol interfaces for every external concern
- **`adapters/inbound/`** — HTTP (FastAPI), protocol detectors
- **`adapters/outbound/`** — Model clients, persistence, tools, Hive Mind, OS runtime
- **`application/`** — Use-case services wiring ports together
- **`bootstrap/`** — DI container, startup, config loading

See the [nlght-ai documentation](https://nlght.ai/documentation/) for the full user documentation.

---

## Requirements

- Python 3.11+
- PostgreSQL (for workflow metadata and resource registry)
- A model provider (Ollama recommended for local development)

## Development

Dependencies are pinned in `uv.lock` for a reproducible environment. With
[`mise`](https://mise.jdx.dev) (see `mise.toml` — pins Python 3.13 and `uv`):

```bash
mise install
uv sync
uv run pytest -q -m "not e2e and not integration"
uv run ruff check src tests
```

`uv sync --locked` (used in CI) fails instead of silently re-resolving if
`pyproject.toml` and `uv.lock` have drifted apart — run `uv lock` after
changing dependencies and commit the updated lockfile.
