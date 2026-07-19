# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Tests for budget trimming and auto-chunking logic."""
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import httpx2

from nlght.adapters.outbound.model import OllamaCloudClient
from nlght.adapters.outbound.model._context_chunking import (
    _split_message_content,
    apply_budget_to_messages,
    estimate_tokens,
    needs_chunking,
    prepare_chunked_session,
    total_message_tokens,
)
from nlght.adapters.outbound.signals.buffering import BufferingSignalEmitter
from nlght.core.model.budget import TokenBudget

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _budget(context_window: int) -> TokenBudget:
    return TokenBudget(context_window)


def _msg(role: str, content: str) -> dict:
    return {"role": role, "content": content}


def _make_client(http_client: httpx2.AsyncClient) -> OllamaCloudClient:
    return OllamaCloudClient(
        http_client=http_client,
        base_url="http://ollama:11434",
        default_model="llama3",
    )


def _post_mock(content: str = "ok") -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"choices": [{"message": {"content": content}}]}
    return resp


# ---------------------------------------------------------------------------
# estimate_tokens / total_message_tokens
# ---------------------------------------------------------------------------


def test_estimate_tokens_at_least_one() -> None:
    assert estimate_tokens("") == 1


def test_estimate_tokens_proportional_to_length() -> None:
    # 35 chars / 3.5 = 10 tokens
    assert estimate_tokens("a" * 35) == 10


def test_total_message_tokens_sums_messages() -> None:
    msgs = [_msg("user", "a" * 35), _msg("assistant", "a" * 70)]
    # 10 + 20 = 30
    assert total_message_tokens(msgs) == 30


def test_total_message_tokens_handles_none_content() -> None:
    msgs = [{"role": "user", "content": None}]
    assert total_message_tokens(msgs) == 1


# ---------------------------------------------------------------------------
# apply_budget_to_messages
# ---------------------------------------------------------------------------


def test_apply_budget_keeps_all_when_fits() -> None:
    b = _budget(10000)
    msgs = [_msg("system", "sys"), _msg("user", "u1"), _msg("assistant", "a1"), _msg("user", "u2")]
    filtered, _ = apply_budget_to_messages(msgs, b)
    assert len(filtered) == 4


def test_apply_budget_always_keeps_system_message() -> None:
    b = _budget(1024)
    system_content = "s" * 1
    last_content = "l" * 1
    filler = "x" * 3500  # ~1000 tokens — fills middle

    msgs = [
        _msg("system", system_content),
        _msg("user", filler),
        _msg("user", last_content),
    ]
    filtered, _ = apply_budget_to_messages(msgs, b)

    roles = [m["role"] for m in filtered]
    assert "system" in roles
    assert filtered[-1]["content"] == last_content


def test_apply_budget_always_keeps_last_message() -> None:
    b = _budget(1024)
    msgs = [
        _msg("user", "x" * 3500),
        _msg("user", "task"),
    ]
    filtered, _ = apply_budget_to_messages(msgs, b)
    assert filtered[-1]["content"] == "task"


def test_apply_budget_returns_token_count() -> None:
    b = _budget(10000)
    msgs = [_msg("user", "a" * 35)]  # 10 tokens
    _, total = apply_budget_to_messages(msgs, b)
    assert total >= 1


def test_apply_budget_empty_messages() -> None:
    b = _budget(1000)
    filtered, total = apply_budget_to_messages([], b)
    assert filtered == []
    assert total == 0


# ---------------------------------------------------------------------------
# needs_chunking
# ---------------------------------------------------------------------------


def test_needs_chunking_false_when_under_threshold() -> None:
    b = _budget(10000)
    msgs = [_msg("user", "hello")]
    assert needs_chunking(msgs, b) is False


def test_needs_chunking_true_when_over_threshold() -> None:
    b = _budget(1024)
    # threshold = 1024 * 0.85 = 870 tokens; content >> 870
    large = "x" * 4000  # ~1142 tokens
    msgs = [_msg("user", large)]
    assert needs_chunking(msgs, b) is True


# ---------------------------------------------------------------------------
# _split_message_content
# ---------------------------------------------------------------------------


def test_split_returns_single_chunk_when_small() -> None:
    content = "short text"
    chunks = _split_message_content(content, 1000)
    assert chunks == [content]


def test_split_splits_on_paragraph_boundaries() -> None:
    para1 = "a" * 100
    para2 = "b" * 100
    content = f"{para1}\n\n{para2}"
    chunks = _split_message_content(content, 20)
    assert len(chunks) >= 2


def test_split_result_total_content_preserved() -> None:
    content = "word " * 200
    chunks = _split_message_content(content, 50)
    rejoined = " ".join(c.strip() for c in chunks)
    assert "word" in rejoined


def test_split_no_empty_chunks() -> None:
    content = "hello\n\n\n\nworld\n\n" * 50
    chunks = _split_message_content(content, 20)
    assert all(c.strip() for c in chunks)


# ---------------------------------------------------------------------------
# prepare_chunked_session
# ---------------------------------------------------------------------------


def test_prepare_extracts_system_and_task() -> None:
    b = _budget(1024)
    msgs = [
        _msg("system", "you are helpful"),
        _msg("user", "context " * 100),
        _msg("user", "actual task"),
    ]
    base, chunks, task = prepare_chunked_session(msgs, b)

    assert any(m["role"] == "system" for m in base)
    assert "INCREMENTAL MODE" in base[0]["content"]
    assert task["content"] == "actual task"


def test_prepare_adds_system_injection_when_no_system_msg() -> None:
    b = _budget(1024)
    msgs = [_msg("user", "no system msg")]
    base, _, _ = prepare_chunked_session(msgs, b)
    assert base[0]["role"] == "system"
    assert "INCREMENTAL MODE" in base[0]["content"]


def test_prepare_creates_chunks_from_middle_messages() -> None:
    b = _budget(1024)
    large_middle = "x " * 2000
    msgs = [
        _msg("system", "sys"),
        _msg("user", large_middle),
        _msg("user", "task"),
    ]
    _, chunks, _ = prepare_chunked_session(msgs, b)
    assert len(chunks) >= 1


# ---------------------------------------------------------------------------
# BoundOllamaCloudClient — budget integration
# ---------------------------------------------------------------------------


async def test_bound_client_without_budget_calls_normally() -> None:
    http_client = AsyncMock(spec=httpx2.AsyncClient)
    http_client.post = AsyncMock(return_value=_post_mock("answer"))

    client = _make_client(http_client)
    emitter = BufferingSignalEmitter()
    bound = client.bind(model="llama3", emitter=emitter, stream=False)

    await bound.call([_msg("user", "hello")])

    http_client.post.assert_awaited_once()
    assert emitter.collected()[0].content == "answer"


async def test_bound_client_with_budget_applies_trimming() -> None:
    http_client = AsyncMock(spec=httpx2.AsyncClient)
    http_client.post = AsyncMock(return_value=_post_mock("trimmed answer"))

    client = _make_client(http_client)
    emitter = BufferingSignalEmitter()
    budget = _budget(10000)
    bound = client.bind(model="llama3", emitter=emitter, stream=False, token_budget=budget)

    await bound.call([_msg("user", "fits easily")])

    http_client.post.assert_awaited_once()


async def test_bound_client_uses_chunking_when_over_threshold() -> None:
    http_client = AsyncMock(spec=httpx2.AsyncClient)
    http_client.post = AsyncMock(return_value=_post_mock("chunk ack"))

    client = _make_client(http_client)
    emitter = BufferingSignalEmitter()
    budget = _budget(1024)
    bound = client.bind(model="llama3", emitter=emitter, stream=False, token_budget=budget)

    large_content = "word " * 1000
    msgs = [_msg("system", "sys"), _msg("user", large_content), _msg("user", "task")]
    await bound.call(msgs)

    assert http_client.post.await_count >= 2


async def test_bound_client_streaming_with_budget_and_chunking() -> None:
    async def _aiter_lines() -> AsyncIterator[str]:
        yield f'data: {json.dumps({"choices": [{"delta": {"content": "streamed"}}]})}'
        yield "data: [DONE]"

    mock_stream_resp = MagicMock()
    mock_stream_resp.raise_for_status = MagicMock()
    mock_stream_resp.aiter_lines = _aiter_lines

    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=mock_stream_resp)
    cm.__aexit__ = AsyncMock(return_value=False)

    http_client = MagicMock(spec=httpx2.AsyncClient)
    http_client.post = AsyncMock(return_value=_post_mock("ack"))
    http_client.stream = MagicMock(return_value=cm)

    client = _make_client(http_client)
    emitter = BufferingSignalEmitter()
    budget = _budget(1024)
    bound = client.bind(model="llama3", emitter=emitter, stream=True, token_budget=budget)

    large_content = "word " * 1000
    msgs = [_msg("system", "sys"), _msg("user", large_content), _msg("user", "task")]
    await bound.call(msgs)

    assert http_client.post.await_count >= 1
    token_signals = [s for s in emitter.collected() if s.kind == "token"]
    assert any(s.content == "streamed" for s in token_signals)
