# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""E2E — cloud model clients against the real APIs.

Runs the shared :mod:`model_client_suite` (call, streaming, native tool loop in
both call() and stream() form, and the step-driven append_tool_turn() loop)
against every cloud ModelClient: Anthropic, OpenAI, Google, Ollama Cloud.

Each client skips itself when its environment is absent (missing key or SDK
extra), so the suite stays green without secrets. Env naming is uniform and
prefix-free — ``<PROVIDER>_API_KEY`` for the credential, ``<PROVIDER>_MODEL``
for the model override:

  Anthropic:    ANTHROPIC_API_KEY   [anthropic extra]   ANTHROPIC_MODEL (default: claude-haiku-4-5)
  OpenAI:       OPENAI_API_KEY      [openai extra]       OPENAI_MODEL    (default: gpt-4o-mini)
                OPENAI_BASE_URL     (optional, e.g. Azure)
  Google:       GOOGLE_API_KEY      [google extra]       GOOGLE_MODEL    (default: gemini-3.5-flash)
  Ollama Cloud: OLLAMA_API_KEY                            OLLAMA_MODEL    (default: gpt-oss:20b)

Ollama Cloud needs no URL: the client defaults to ``https://ollama.com``
(override with ``OLLAMA_BASE_URL`` if pointing at a self-hosted remote).

The local Ollama client is tested separately under ``tests/local_e2e`` (it needs
a running daemon and never runs in CI).
"""
from __future__ import annotations

import os

import pytest

# Imported so pytest collects them as this module's tests; parametrized by the
# `backend` fixture defined below.
from model_client_suite import (  # noqa: F401
    test_call_returns_assistant_result,
    test_native_tool_loop_call_executes_tool,
    test_native_tool_loop_stream_executes_tool,
    test_prompt_injection_corpus_is_observational,
    test_step_driven_tool_loop_round_trip,
    test_stream_yields_tokens_and_done,
    test_untrusted_context_is_readable_without_being_an_instruction,
)

pytestmark = pytest.mark.e2e


# ---------------------------------------------------------------------------
# Cloud client backends — each skips itself when not configured
# ---------------------------------------------------------------------------


def _anthropic_backend():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")
    pytest.importorskip("anthropic")
    from nlght.adapters.outbound.model.anthropic import AnthropicModelClient

    model = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5")
    return AnthropicModelClient(api_key="", default_model=model)


def _openai_backend():
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")
    pytest.importorskip("openai")
    from nlght.adapters.outbound.model.openai_cloud import OpenAICloudModelClient

    model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
    base_url = os.environ.get("OPENAI_BASE_URL")
    if base_url:
        return OpenAICloudModelClient(api_key="", default_model=model, base_url=base_url)
    return OpenAICloudModelClient(api_key="", default_model=model)


def _google_backend():
    if not os.environ.get("GOOGLE_API_KEY"):
        pytest.skip("GOOGLE_API_KEY not set")
    pytest.importorskip("google.genai")
    from nlght.adapters.outbound.model.google import GoogleModelClient

    model = os.environ.get("GOOGLE_MODEL", "gemini-3.5-flash")
    return GoogleModelClient(api_key="", default_model=model)


def _ollama_cloud_backend():
    key = os.environ.get("OLLAMA_API_KEY")
    if not key:
        pytest.skip("OLLAMA_API_KEY not set")
    import httpx2

    from nlght.adapters.outbound.model import OllamaCloudClient

    # Default to a free-tier (no-subscription) level-1 model — gpt-oss:20b is
    # tool-capable and light on the Ollama Cloud free quota. Subscription-gated
    # models (e.g. qwen3.5:cloud) return 403; override with OLLAMA_MODEL.
    model = os.environ.get("OLLAMA_MODEL", "gpt-oss:20b")
    # No URL required — the client defaults to https://ollama.com. Only a
    # self-hosted remote needs OLLAMA_BASE_URL.
    base_url = os.environ.get("OLLAMA_BASE_URL")
    kwargs = {"http_client": httpx2.AsyncClient(), "default_model": model, "api_key": key}
    if base_url:
        kwargs["base_url"] = base_url
    return OllamaCloudClient(**kwargs)


_BACKENDS = {
    "anthropic": _anthropic_backend,
    "openai": _openai_backend,
    "google": _google_backend,
    "ollama-cloud": _ollama_cloud_backend,
}


# Function-scoped: async SDK clients bind their transport to the event loop that
# is running when they are first used. pytest-asyncio uses a fresh function-scoped
# loop per test, so a module-scoped client would raise "Event loop is closed" from
# the second test onward (observed on the Google client). One client per test.
@pytest.fixture(params=sorted(_BACKENDS))
def backend(request):
    return _BACKENDS[request.param]()


@pytest.fixture
def provider_name(request) -> str:
    return str(request.node.callspec.params["backend"])
