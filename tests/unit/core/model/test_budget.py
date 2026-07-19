# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Tests for TokenBudget — core token-budget management type."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

import nlght.core.model.budget as budget_module
from nlght.core.model.budget import TokenBudget

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _budget(context_window: int, client_max_tokens: int | None = None) -> TokenBudget:
    return TokenBudget(context_window, client_max_tokens=client_max_tokens)


# ---------------------------------------------------------------------------
# Partition ratios
# ---------------------------------------------------------------------------


def test_store_budget_is_40_percent() -> None:
    b = _budget(1000)
    assert b.store_budget == 400


def test_prompt_budget_is_20_percent() -> None:
    b = _budget(1000)
    assert b.prompt_budget == 200


def test_output_budget_is_20_percent_by_default() -> None:
    b = _budget(1000)
    assert b.output_budget == 200


def test_total_matches_context_window() -> None:
    b = _budget(32768)
    assert b.total == 32768


def test_minimum_context_window_floor() -> None:
    b = _budget(0)
    assert b.total == 1024
    assert b.store_budget > 0


# ---------------------------------------------------------------------------
# client_max_tokens caps output_budget
# ---------------------------------------------------------------------------


def test_client_max_tokens_caps_output_when_smaller() -> None:
    b = _budget(1000, client_max_tokens=50)
    assert b.output_budget == 50


def test_client_max_tokens_ignored_when_larger_than_20_percent() -> None:
    b = _budget(1000, client_max_tokens=500)
    # 20% of 1000 = 200; client cap 500 > 200 so not applied
    assert b.output_budget == 200


def test_client_max_tokens_none_uses_full_output_budget() -> None:
    b = _budget(1000, client_max_tokens=None)
    assert b.output_budget == 200


# ---------------------------------------------------------------------------
# get_available_for_input
# ---------------------------------------------------------------------------


def test_available_for_input_equals_total_minus_output_and_safety() -> None:
    # M=1000: output=200, safety=200, reserved=400 → available=600
    b = _budget(1000)
    assert b.get_available_for_input() == 600


def test_available_for_input_subtracts_system_prompt_tokens() -> None:
    b = _budget(1000)
    # 600 - 100 system = 500
    assert b.get_available_for_input(system_prompt_tokens=100) == 500


def test_available_for_input_never_negative() -> None:
    b = _budget(1000)
    # passing enormous system tokens
    assert b.get_available_for_input(system_prompt_tokens=10000) == 0


# ---------------------------------------------------------------------------
# for_model — async factory
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clear_context_window_cache():
    """Isolate tests from the module-level per-process cache."""
    budget_module._context_window_cache.clear()
    yield
    budget_module._context_window_cache.clear()


async def test_for_model_uses_ollama_api_show_response() -> None:
    data = {
        "model_info": {
            "general.architecture": "llama",
            "llama.context_length": 131072,
        }
    }
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = data

    http_client = AsyncMock()
    http_client.post = AsyncMock(return_value=mock_resp)

    b = await TokenBudget.for_model("llama3.1:8b", base_url="http://ollama:11434", http_client=http_client)

    assert b.total == 131072


async def test_for_model_falls_back_to_static_table_on_api_failure() -> None:
    http_client = AsyncMock()
    http_client.post = AsyncMock(side_effect=Exception("connection refused"))

    b = await TokenBudget.for_model("llama3.1:8b", base_url="http://ollama:11434", http_client=http_client)

    # llama3.1:8b is in the fallback table with 131072
    assert b.total == 131072


async def test_for_model_falls_back_to_default_for_unknown_model() -> None:
    http_client = AsyncMock()
    http_client.post = AsyncMock(side_effect=Exception("timeout"))

    b = await TokenBudget.for_model("unknown-model:latest", base_url="http://ollama:11434", http_client=http_client)

    assert b.total == 32768  # _DEFAULT_CONTEXT_WINDOW


async def test_for_model_caches_per_process() -> None:
    data = {"model_info": {"general.architecture": "llama", "llama.context_length": 8192}}
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = data
    http_client = AsyncMock()
    http_client.post = AsyncMock(return_value=mock_resp)

    await TokenBudget.for_model("test-model", base_url="http://ollama:11434", http_client=http_client)
    await TokenBudget.for_model("test-model", base_url="http://ollama:11434", http_client=http_client)

    # /api/show called only once — second call served from cache
    assert http_client.post.await_count == 1


async def test_for_model_falls_back_when_architecture_key_missing() -> None:
    data = {"model_info": {}}  # no general.architecture
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = data
    http_client = AsyncMock()
    http_client.post = AsyncMock(return_value=mock_resp)

    b = await TokenBudget.for_model("some-model", base_url="http://ollama:11434", http_client=http_client)

    assert b.total == 32768  # default fallback


async def test_for_model_applies_client_max_tokens() -> None:
    http_client = AsyncMock()
    http_client.post = AsyncMock(side_effect=Exception("fail"))

    b = await TokenBudget.for_model(
        "llama3.1:8b",
        base_url="http://ollama:11434",
        http_client=http_client,
        client_max_tokens=100,
    )

    assert b.output_budget == 100
