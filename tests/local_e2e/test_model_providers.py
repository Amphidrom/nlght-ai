# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Local E2E — the ollama-local model client against a live `ollama serve`.

Runs the exact same shared :mod:`model_client_suite` as the cloud clients
(``tests/e2e``), but against a locally running Ollama daemon. This lives in its
own package and carries the ``local_e2e`` marker because it can only run where
Ollama is reachable — CI tests cloud clients only. It is deselected by default
(``-m "not local_e2e"`` in ``pyproject.toml``); run it explicitly with:

    pytest tests/local_e2e -m local_e2e

Environment (prefix-free, matching the cloud clients' scheme):

  OLLAMA_HOST   local daemon base URL   (default: http://localhost:11434)
  OLLAMA_MODEL  tool-capable model      (default: llama3.2)

The whole module skips when the daemon is unreachable or has no usable model.
"""
from __future__ import annotations

import os

import httpx2
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

pytestmark = [pytest.mark.e2e, pytest.mark.local_e2e]

OLLAMA_HOST  = os.environ.get("OLLAMA_HOST",  "http://localhost:11434")
PREFER_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")


def _reachable() -> bool:
    try:
        return httpx2.get(f"{OLLAMA_HOST}/api/version", timeout=3.0).status_code == 200
    except Exception:
        return False


def _pick_model() -> str | None:
    """Prefer OLLAMA_MODEL, else the first locally-pulled (non-remote) model."""
    try:
        resp = httpx2.get(f"{OLLAMA_HOST}/api/tags", timeout=5.0)
        if resp.status_code != 200:
            return None
        models = [m["name"] for m in resp.json().get("models", []) if not m.get("remote_model")]
    except Exception:
        return None
    for m in models:
        if m == PREFER_MODEL or m.startswith(f"{PREFER_MODEL}:"):
            return m
    return models[0] if models else None


def _ollama_local_backend():
    if not _reachable():
        pytest.skip(
            f"Ollama not reachable at {OLLAMA_HOST} — start it with `ollama serve` "
            "or set OLLAMA_HOST."
        )
    model = _pick_model()
    if model is None:
        pytest.skip(f"No usable model in Ollama at {OLLAMA_HOST} — pull one, e.g. `ollama pull {PREFER_MODEL}`.")

    from nlght.adapters.outbound.model import OllamaClient

    return OllamaClient(
        http_client=httpx2.AsyncClient(),
        base_url=OLLAMA_HOST,
        default_model=model,
    )


# Function-scoped for the same event-loop reason as the cloud suite: one client
# per test, bound to that test's loop.
@pytest.fixture
def backend(request):
    return _ollama_local_backend()


@pytest.fixture
def provider_name() -> str:
    return "ollama-local"
