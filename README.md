# nlght-ai
The runtime between your app and your LLM: workflows, tools, memory, and model providers.

<!-- release-badges:start -->
[![Tests](https://raw.githubusercontent.com/Amphidrom/nlght-ai/v0.1.4/.github/badges/tests.svg)](https://github.com/Amphidrom/nlght-ai/blob/v0.1.4/.github/badges/tests.svg)
[![Coverage](https://raw.githubusercontent.com/Amphidrom/nlght-ai/v0.1.4/.github/badges/coverage.svg)](https://github.com/Amphidrom/nlght-ai/blob/v0.1.4/.github/badges/coverage.svg)
[![PyPI](https://raw.githubusercontent.com/Amphidrom/nlght-ai/v0.1.4/.github/badges/pypi.svg)](https://github.com/Amphidrom/nlght-ai/blob/v0.1.4/.github/badges/pypi.svg)
<!-- release-badges:end -->

> **E2E scope note.** The test badge covers the cloud model-client e2e suite
> (`tests/e2e`); each provider skips itself when its API key is absent. The
> **local Ollama** model-client suite (`tests/local_e2e`) runs separately against
> a local Ollama daemon. Run it with `pytest tests/local_e2e -m local_e2e`.

---

## What it does

1. A protocol adapter accepts an OpenAI or Ollama API request and resolves its
   workflow trigger.
2. The workflow executor runs the configured steps. Steps receive request
   messages and the configured model, tool, playbook, and session ports.
3. Steps call models or tools, use session memory, and emit signals without
   handling HTTP.
4. The adapter returns the result as a batch response or stream.

---

## Features

| Feature | Description |
|---|---|
| **OpenAI-compatible API** | Chat Completions endpoint for supported OpenAI clients |
| **Ollama API** | Native Ollama protocol support |
| **Multi-provider** | Ollama, Anthropic Claude, OpenAI, Google Gemini |
| **Workflow engine** | Stateless step machine with configurable routing |
| **Tool system** | Register custom tools, call them from steps |
| **Playbook catalog** | YAML-defined agent capabilities, loaded from filesystem |
| **Session memory** | Hive Mind: conversation, session results, and working memory |
| **Session routing** | Configurable `SessionKeyResolver` per protocol adapter |
| **PostgreSQL persistence** | Workflow metadata and resource registry via SQLAlchemy |
| **Ports & Adapters** | Core contracts have no framework dependency |

---

## Install

Install [uv](https://docs.astral.sh/uv/), then add nlght-ai inside a Python project:

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

In the configuration examples, `YOUR-LICENSE-KEY` means the key from a
[free developer license](docs/licenses/free.mdx). Replace it with your own key.

**1. Configure:** Create `.config/platform.yaml`:

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

**2. Start PostgreSQL** with the local credentials above:

```bash
docker run -d --name nlght-postgres \
  -e POSTGRES_USER=user -e POSTGRES_PASSWORD=pass -e POSTGRES_DB=nlght \
  -p 5432:5432 postgres:16
```

**3. Migrate the database** (initial setup and after every upgrade):

```bash
uv run nlght-ai migrate
```

**4. Start Ollama and the runtime** (use separate terminals):

Terminal 1:

```bash
ollama serve
```

Terminal 2:

```bash
ollama pull llama3.2
uv run nlght-ai serve
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

from nlght.core.model.messages import TrustedInstructionMessage
from nlght.core.workflow.step import StepBase, StepResult, WorkflowStepContext


class SummarizeStep(StepBase):
    TYPE: ClassVar[str] = "summarize"

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        if ctx.llm is None:
            return StepResult(ctx=ctx, verdict="failed")
        instructions = TrustedInstructionMessage(
            "Summarize the conversation in three bullet points."
        )
        await ctx.llm.call(messages=[instructions, *ctx.messages])
        return StepResult(ctx=ctx, verdict="done")
```

Register and wire it up in your entrypoint, then reference it in a workflow definition stored in the database.

---

## Session Memory (Hive Mind)

Hive Mind persists conversation turns and session results. Working memory holds transient atoms
for the active task. Applications can store JSON-compatible values in namespaced coordinator
slots. The store has no directive API.

Configure a session backend and select model context with `PromptInputPolicy`. See
[Session Memory](docs/documentation/configuration/session-memory.mdx) for the current configuration,
store contracts, and copyable examples.

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

- **`core/`**: Domain logic and framework independent types
- **`ports/`**: Interfaces for external concerns
- **`adapters/inbound/`**: HTTP and protocol adapters
- **`adapters/outbound/`**: Model clients, persistence, tools, Hive Mind, and OS runtime
- **`application/`**: Services that coordinate ports
- **`bootstrap/`**: Container, startup, and configuration loading

See the [nlght-ai documentation](https://nlght.ai/documentation/) for the full user documentation.

---

## Requirements

- Python 3.11+
- PostgreSQL (for workflow metadata and resource registry)
- A model provider (Ollama recommended for local development)

## Development

Dependencies are pinned in `uv.lock` for a reproducible environment. With
[`mise`](https://mise.jdx.dev) (see `mise.toml`, which pins Python 3.13 and `uv`):

```bash
mise install
uv sync
uv run pytest -q -m "not e2e and not integration"
uv run ruff check src tests
```

The default `dev` group installs `nlght-ai[all]`, so local development and CI
exercise every portable runtime capability. Accelerator-specific
`embedding-cpu` and `embedding-cuda` remain explicit machine choices.

`uv sync --locked` (used in CI) fails instead of silently re-resolving if
`pyproject.toml` and `uv.lock` have drifted apart. Run `uv lock` after
changing dependencies and commit the updated lockfile.
