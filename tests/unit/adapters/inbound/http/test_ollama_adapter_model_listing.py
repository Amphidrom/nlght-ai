# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Tests for OllamaHttpProtocolAdapter model listing and /api/ps capability.

Verifies:
  - GET /api/tags  returns Ollama-format model list from ModelProviderBackend
  - GET /api/ps    returns running models when backend is RunningModelsCapable
  - GET /api/ps    returns empty list when backend lacks the capability
  - GET /api/version raises 501 for non-NativeProxyCapable backend
"""
from __future__ import annotations

import pytest

pytest.importorskip("httpx2")

from fastapi.testclient import TestClient

from nlght.adapters.inbound.http.ollama_adapter import OllamaHttpProtocolAdapter
from nlght.core.model.model_info import ModelInfo, RunningModelInfo

# ---------------------------------------------------------------------------
# Fake backends
# ---------------------------------------------------------------------------


class _FakeBasicBackend:
    """ModelProviderBackend without running-models or native-proxy capability."""

    async def list_models(self) -> list[ModelInfo]:
        return [
            ModelInfo(name="llama3.2:latest", size=2_000_000, digest="abc"),
            ModelInfo(name="qwen3:7b"),
        ]

    def bind(self, *, model, emitter, stream, token_budget=None):
        pass

    async def token_budget(self, model=None, *, client_max_tokens=None):
        pass


class _FakeRunningCapableBackend(_FakeBasicBackend):
    """Also implements RunningModelsCapable."""

    async def list_running_models(self) -> list[RunningModelInfo]:
        return [
            RunningModelInfo(name="llama3.2:latest", size_vram=4_000_000_000),
        ]


# ---------------------------------------------------------------------------
# GET /api/tags
# ---------------------------------------------------------------------------


def test_tags_returns_ollama_format() -> None:
    adapter = OllamaHttpProtocolAdapter(model_backend=_FakeBasicBackend())
    app = _make_app(adapter)
    with TestClient(app) as client:
        resp = client.get("/api/tags")
    assert resp.status_code == 200
    body = resp.json()
    assert "models" in body
    names = [m["name"] for m in body["models"]]
    assert "llama3.2:latest" in names
    assert "qwen3:7b" in names


def test_tags_model_entry_has_required_fields() -> None:
    adapter = OllamaHttpProtocolAdapter(model_backend=_FakeBasicBackend())
    app = _make_app(adapter)
    with TestClient(app) as client:
        resp = client.get("/api/tags")
    entry = resp.json()["models"][0]
    assert "name" in entry
    assert "model" in entry
    assert "size" in entry
    assert "digest" in entry


def test_tags_returns_503_without_backend() -> None:
    adapter = OllamaHttpProtocolAdapter(model_backend=None)
    app = _make_app(adapter)
    with TestClient(app) as client:
        resp = client.get("/api/tags")
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# GET /api/ps
# ---------------------------------------------------------------------------


def test_ps_returns_running_models_when_capable() -> None:
    adapter = OllamaHttpProtocolAdapter(model_backend=_FakeRunningCapableBackend())
    app = _make_app(adapter)
    with TestClient(app) as client:
        resp = client.get("/api/ps")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["models"]) == 1
    assert body["models"][0]["name"] == "llama3.2:latest"
    assert body["models"][0]["size_vram"] == 4_000_000_000


def test_ps_returns_empty_list_when_not_capable() -> None:
    adapter = OllamaHttpProtocolAdapter(model_backend=_FakeBasicBackend())
    app = _make_app(adapter)
    with TestClient(app) as client:
        resp = client.get("/api/ps")
    assert resp.status_code == 200
    assert resp.json() == {"models": []}


def test_ps_returns_empty_list_without_backend() -> None:
    adapter = OllamaHttpProtocolAdapter(model_backend=None)
    app = _make_app(adapter)
    with TestClient(app) as client:
        resp = client.get("/api/ps")
    assert resp.status_code == 200
    assert resp.json() == {"models": []}


# ---------------------------------------------------------------------------
# Native-proxy endpoints — 501 without NativeProxyCapable
# ---------------------------------------------------------------------------


def test_version_returns_501_without_native_capable_backend() -> None:
    adapter = OllamaHttpProtocolAdapter(model_backend=_FakeBasicBackend())
    app = _make_app(adapter)
    with TestClient(app) as client:
        resp = client.get("/api/version")
    assert resp.status_code == 501


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_app(adapter: OllamaHttpProtocolAdapter):
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(adapter.build_router())
    return app
