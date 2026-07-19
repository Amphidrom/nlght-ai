# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Tests for OllamaCloudClient and BoundModelClient.

HTTP calls are intercepted via httpx2.AsyncClient(transport=...) using a
custom ASGI-style mock transport — no live server required.
"""
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import httpx2

from nlght.adapters.outbound.model import OllamaClient, OllamaCloudClient, _has_tool_protocol_messages
from nlght.adapters.outbound.signals.buffering import BufferingSignalEmitter
from nlght.core.model.model_info import ModelInfo, RunningModelInfo

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(
    http_client: httpx2.AsyncClient,
    base_url: str = "http://ollama:11434",
    default_model: str = "llama3",
    api_key: str = "",
    extra_headers: dict[str, str] | None = None,
    request_timeout_s: float | None = None,
    stream_connect_timeout_s: float | None = None,
    stream_read_timeout_s: float | None = None,
) -> OllamaCloudClient:
    return OllamaCloudClient(
        http_client=http_client,
        base_url=base_url,
        default_model=default_model,
        api_key=api_key,
        extra_headers=extra_headers,
        request_timeout_s=request_timeout_s,
        stream_connect_timeout_s=stream_connect_timeout_s,
        stream_read_timeout_s=stream_read_timeout_s,
    )


# ---------------------------------------------------------------------------
# bind()
# ---------------------------------------------------------------------------


def test_bind_returns_bound_client_with_given_model() -> None:
    http_client = MagicMock(spec=httpx2.AsyncClient)
    client = _make_client(http_client)
    emitter = BufferingSignalEmitter()

    bound = client.bind(model="qwen3:7b", emitter=emitter, stream=False)

    assert hasattr(bound, "stream") and hasattr(bound, "call")
    assert bound._model == "qwen3:7b"
    assert bound._stream is False


def test_bind_falls_back_to_default_model_when_model_is_none() -> None:
    http_client = MagicMock(spec=httpx2.AsyncClient)
    client = _make_client(http_client, default_model="llama3")

    bound = client.bind(model=None, emitter=BufferingSignalEmitter(), stream=False)

    assert bound._model == "llama3"


def test_bind_falls_back_to_default_model_when_model_is_empty() -> None:
    http_client = MagicMock(spec=httpx2.AsyncClient)
    client = _make_client(http_client, default_model="llama3")

    bound = client.bind(model="", emitter=BufferingSignalEmitter(), stream=False)

    assert bound._model == "llama3"


# ---------------------------------------------------------------------------
# call() — non-streaming
# ---------------------------------------------------------------------------


async def test_call_non_stream_emits_result_signal() -> None:
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": "World"}}]
    }

    http_client = AsyncMock(spec=httpx2.AsyncClient)
    http_client.post.return_value = mock_response

    client = _make_client(http_client)
    emitter = BufferingSignalEmitter()
    bound = client.bind(model="llama3", emitter=emitter, stream=False)

    await bound.call([{"role": "user", "content": "Hello"}])

    signals = emitter.collected()
    assert len(signals) == 1
    assert signals[0].kind == "result"
    assert signals[0].content == "World"
    assert signals[0].role == "assistant"


async def test_call_non_stream_posts_to_correct_url() -> None:
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": ""}}]
    }

    http_client = AsyncMock(spec=httpx2.AsyncClient)
    http_client.post.return_value = mock_response

    client = _make_client(http_client, base_url="http://ollama:11434")
    bound = client.bind(model="llama3", emitter=BufferingSignalEmitter(), stream=False)

    messages = [{"role": "user", "content": "ping"}]
    await bound.call(messages)

    http_client.post.assert_called_once()
    call_kwargs = http_client.post.call_args
    assert "http://ollama:11434/v1/chat/completions" in call_kwargs[0]


async def test_call_non_stream_sends_bearer_auth_header() -> None:
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"role": "assistant", "content": ""}}]
    }

    http_client = AsyncMock(spec=httpx2.AsyncClient)
    http_client.post.return_value = mock_response

    client = _make_client(http_client, api_key="secret-token")
    bound = client.bind(model="llama3", emitter=BufferingSignalEmitter(), stream=False)

    await bound.call([{"role": "user", "content": "ping"}])

    _, kwargs = http_client.post.call_args
    assert kwargs["headers"]["Authorization"] == "Bearer secret-token"


async def test_call_non_stream_handles_empty_content() -> None:
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {"choices": [{"message": {"content": None}}]}

    http_client = AsyncMock(spec=httpx2.AsyncClient)
    http_client.post.return_value = mock_response

    client = _make_client(http_client)
    emitter = BufferingSignalEmitter()
    bound = client.bind(model="llama3", emitter=emitter, stream=False)

    await bound.call([])

    assert emitter.collected()[0].content == ""


# ---------------------------------------------------------------------------
# call() — streaming
# ---------------------------------------------------------------------------


async def _make_streaming_http_client(lines: list[str]) -> httpx2.AsyncClient:
    """Returns an AsyncClient mock whose .stream() yields the given SSE lines."""

    async def _aiter_lines() -> AsyncIterator[str]:
        for line in lines:
            yield line

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.aiter_lines = _aiter_lines

    # async context manager for httpx2.AsyncClient.stream(...)
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=mock_response)
    cm.__aexit__ = AsyncMock(return_value=False)

    http_client = MagicMock(spec=httpx2.AsyncClient)
    http_client.stream.return_value = cm
    return cast(httpx2.AsyncClient, http_client)


def _token_line(token: str) -> str:
    return f'data: {json.dumps({"choices": [{"delta": {"content": token}}]})}'


async def test_call_stream_emits_token_signals() -> None:
    lines = [_token_line("Hello"), _token_line(" world"), "data: [DONE]"]

    http_client = await _make_streaming_http_client(lines)
    client = _make_client(http_client)
    emitter = BufferingSignalEmitter()
    bound = client.bind(model="llama3", emitter=emitter, stream=True)

    await bound.call([{"role": "user", "content": "hi"}])

    signals = emitter.collected()
    token_signals = [s for s in signals if s.kind == "token"]
    assert [s.content for s in token_signals] == ["Hello", " world"]


async def test_call_stream_uses_configured_headers_and_timeout() -> None:
    lines = [_token_line("Hello"), "data: [DONE]"]
    http_client = await _make_streaming_http_client(lines)
    client = _make_client(
        http_client,
        api_key="secret-token",
        extra_headers={"X-Test": "1"},
        stream_connect_timeout_s=7.0,
        stream_read_timeout_s=33.0,
    )
    emitter = BufferingSignalEmitter()
    bound = client.bind(model="llama3", emitter=emitter, stream=True)

    await bound.call([{"role": "user", "content": "hi"}])

    _, kwargs = http_client.stream.call_args
    assert kwargs["headers"]["Authorization"] == "Bearer secret-token"
    assert kwargs["headers"]["X-Test"] == "1"
    assert isinstance(kwargs["timeout"], httpx2.Timeout)
    assert kwargs["timeout"].connect == 7.0
    assert kwargs["timeout"].read == 33.0


async def test_call_stream_uses_no_read_timeout_when_not_configured() -> None:
    lines = [_token_line("Hello"), "data: [DONE]"]
    http_client = await _make_streaming_http_client(lines)
    client = _make_client(http_client)
    emitter = BufferingSignalEmitter()
    bound = client.bind(model="llama3", emitter=emitter, stream=True)

    await bound.call([{"role": "user", "content": "hi"}])

    _, kwargs = http_client.stream.call_args
    assert isinstance(kwargs["timeout"], httpx2.Timeout)
    assert kwargs["timeout"].read is None
    assert kwargs["timeout"].connect is None


async def test_call_stream_emits_done_signal() -> None:
    lines = ["data: [DONE]"]

    http_client = await _make_streaming_http_client(lines)
    client = _make_client(http_client)
    emitter = BufferingSignalEmitter()
    bound = client.bind(model="llama3", emitter=emitter, stream=True)

    await bound.call([])

    done_signals = [s for s in emitter.collected() if s.kind == "done"]
    assert len(done_signals) == 1


async def test_call_stream_skips_non_data_lines() -> None:
    token_line = f'data: {json.dumps({"choices": [{"delta": {"content": "x"}}]})}'
    lines = ["", ": comment", token_line, "data: [DONE]"]

    http_client = await _make_streaming_http_client(lines)
    client = _make_client(http_client)
    emitter = BufferingSignalEmitter()
    bound = client.bind(model="llama3", emitter=emitter, stream=True)

    await bound.call([])

    token_signals = [s for s in emitter.collected() if s.kind == "token"]
    assert len(token_signals) == 1
    assert token_signals[0].content == "x"


async def test_call_stream_skips_empty_delta_content() -> None:
    empty_delta = f'data: {json.dumps({"choices": [{"delta": {"content": ""}}]})}'
    lines = [empty_delta, "data: [DONE]"]

    http_client = await _make_streaming_http_client(lines)
    client = _make_client(http_client)
    emitter = BufferingSignalEmitter()
    bound = client.bind(model="llama3", emitter=emitter, stream=True)

    await bound.call([])

    token_signals = [s for s in emitter.collected() if s.kind == "token"]
    assert token_signals == []


def test_has_tool_protocol_messages_detects_tool_history() -> None:
    assert _has_tool_protocol_messages([{"role": "tool", "content": "x"}]) is True
    assert _has_tool_protocol_messages([{"role": "assistant", "tool_calls": [{"id": "1"}]}]) is True
    assert _has_tool_protocol_messages([{"role": "user", "content": "x"}]) is False


async def test_stream_uses_api_chat_when_tool_history_exists_without_new_tools() -> None:
    lines = ['{"message": {"content": "ok"}, "done": true}']
    http_client = await _make_streaming_http_client(lines)
    local_client = OllamaClient(
        http_client=http_client,
        base_url="http://ollama:11434",
        default_model="llama3",
    )
    emitter = BufferingSignalEmitter()
    bound = local_client.bind(model="llama3", emitter=emitter, stream=True)

    assert hasattr(bound, "stream") and hasattr(bound, "call")

    messages = [
        {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "web_search", "arguments": {}}}]},
        {"role": "tool", "name": "web_search", "content": '{"phase":"collect_search","hits":[]}'},
        {"role": "user", "content": "[Phase: evaluate] Evaluate the collected results."},
    ]

    events = []
    async for event in bound.stream(messages, tools=None):
        events.append(event)

    stream_call = http_client.stream.call_args
    assert stream_call is not None
    assert stream_call.args[1] == "http://ollama:11434/api/chat"
    assert any(event.kind == "token" for event in events)


# ---------------------------------------------------------------------------
# list_models
# ---------------------------------------------------------------------------


def _mock_get_response(body: dict, status_code: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = body
    resp.status_code = status_code
    return resp


async def test_list_models_returns_model_info_list() -> None:
    http_client = AsyncMock(spec=httpx2.AsyncClient)
    http_client.get.return_value = _mock_get_response({
        "models": [
            {
                "name": "llama3.2:latest",
                "size": 2_019_393_189,
                "modified_at": "2024-12-19T11:00:24+00:00",
                "digest": "abc123",
            },
            {"name": "qwen3:7b", "size": 0},
        ]
    })
    client = _make_client(http_client)
    models = await client.list_models()
    assert len(models) == 2
    assert all(isinstance(m, ModelInfo) for m in models)
    assert models[0].name == "llama3.2:latest"
    assert models[0].size == 2_019_393_189
    assert models[0].digest == "abc123"
    assert models[1].name == "qwen3:7b"


async def test_list_models_returns_empty_on_request_error() -> None:
    http_client = AsyncMock(spec=httpx2.AsyncClient)
    http_client.get.side_effect = httpx2.RequestError("connection refused")
    client = _make_client(http_client)
    models = await client.list_models()
    assert models == []


# ---------------------------------------------------------------------------
# list_running_models
# ---------------------------------------------------------------------------


async def test_list_running_models_returns_running_model_info() -> None:
    http_client = AsyncMock(spec=httpx2.AsyncClient)
    http_client.get.return_value = _mock_get_response({
        "models": [
            {
                "name": "llama3.2:latest",
                "size_vram": 4_096_000_000,
                "expires_at": "2024-06-04T21:38:31+00:00",
                "digest": "def456",
                "size": 2_019_393_189,
            }
        ]
    })
    client = _make_client(http_client)
    running = await client.list_running_models()
    assert len(running) == 1
    assert all(isinstance(r, RunningModelInfo) for r in running)
    assert running[0].name == "llama3.2:latest"
    assert running[0].size_vram == 4_096_000_000


async def test_list_running_models_returns_empty_on_error() -> None:
    http_client = AsyncMock(spec=httpx2.AsyncClient)
    http_client.get.side_effect = httpx2.RequestError("timeout")
    client = _make_client(http_client)
    running = await client.list_running_models()
    assert running == []


def test_ollama_client_exposes_native_base_url() -> None:
    http_client = MagicMock(spec=httpx2.AsyncClient)
    client = _make_client(http_client, base_url="http://ollama:11434")
    assert client.native_base_url == "http://ollama:11434"
