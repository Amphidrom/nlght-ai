# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx2
import pytest

from nlght.adapters.outbound.model._ollama_http import _merge_streamed_fragment, _OllamaHttp
from nlght.adapters.outbound.signals.buffering import BufferingSignalEmitter


class _Response:
    def __init__(self, body: dict, *, status_error: bool = False) -> None:
        self._body = body
        self.status_error = status_error

    def json(self) -> dict:
        return self._body

    def raise_for_status(self) -> None:
        if self.status_error:
            raise httpx2.HTTPStatusError("bad", request=httpx2.Request("GET", "http://x"), response=httpx2.Response(500))


class _Stream:
    def __init__(self, lines: list[str]) -> None:
        self.lines = lines

    async def __aenter__(self) -> _Stream:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    async def aiter_lines(self):
        for line in self.lines:
            yield line


class _Http:
    def __init__(self) -> None:
        self.get_responses: list[_Response | Exception] = []
        self.post_responses: list[_Response] = []
        self.streams: list[_Stream | Exception] = []
        self.posts: list[dict] = []
        self.stream_calls: list[dict] = []

    async def get(self, *args: object, **kwargs: object) -> _Response:
        response = self.get_responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    async def post(self, *args: object, **kwargs: object) -> _Response:
        self.posts.append({"args": args, **kwargs})
        return self.post_responses.pop(0)

    def stream(self, *args: object, **kwargs: object) -> _Stream:
        self.stream_calls.append({"args": args, **kwargs})
        stream = self.streams.pop(0)
        if isinstance(stream, Exception):
            raise stream
        return stream


def _backend(http: _Http) -> _OllamaHttp:
    return _OllamaHttp(
        http_client=http,
        base_url="http://ollama/",
        api_key="secret",
        extra_headers={"X-Test": "1"},
        request_timeout_s=2,
        stream_connect_timeout_s=1,
        stream_read_timeout_s=3,
    )


def test_merge_streamed_fragment_removes_overlaps() -> None:
    assert _merge_streamed_fragment("", "abc") == "abc"
    assert _merge_streamed_fragment("abc", "") == "abc"
    assert _merge_streamed_fragment("abc", "abc") == "abc"
    assert _merge_streamed_fragment("abc", "abcdef") == "abcdef"
    assert _merge_streamed_fragment("abcdef", "defghi") == "abcdefghi"
    assert _merge_streamed_fragment("abc", "xyz") == "abcxyz"


async def test_ollama_lists_models_running_models_and_handles_errors() -> None:
    http = _Http()
    http.get_responses.extend([
        _Response({"models": [{"name": "m", "size": 1, "digest": "d", "modified_at": "2026-01-01T00:00:00"}]}),
        httpx2.RequestError("offline", request=httpx2.Request("GET", "http://x")),
        _Response({"models": [{"name": "m", "size_vram": 2, "expires_at": "invalid"}]}),
        httpx2.RequestError("offline", request=httpx2.Request("GET", "http://x")),
    ])
    backend = _backend(http)

    assert (await backend.list_models())[0].name == "m"
    assert await backend.list_models() == []
    assert (await backend.list_running_models())[0].size_vram == 2
    assert await backend.list_running_models() == []
    assert backend.request_headers() == {"X-Test": "1", "Authorization": "Bearer secret"}
    assert backend._stream_timeout().read == 3


async def test_ollama_complete_call_raw_and_metering() -> None:
    http = _Http()
    http.post_responses.extend([
        _Response({"choices": [{"message": {"content": "hello"}}], "usage": {"prompt_tokens": 4, "completion_tokens": 5}}),
        _Response({"choices": []}),
    ])
    backend = _backend(http)
    emitter = BufferingSignalEmitter()
    recorded: list[dict] = []
    metering = SimpleNamespace(
        port=SimpleNamespace(record_tokens=lambda **kwargs: _record(recorded, kwargs)),
        caller=object(),
        session_key="s",
        workflow="wf",
        step="step",
        provider="ollama",
    )

    await backend.complete("http://ollama/v1/chat/completions", {"model": "m"}, emitter, metering)
    assert emitter.collected()[0].content == "hello"
    assert recorded[0]["input_tokens"] == 4

    assert await backend.call_raw([{"role": "user", "content": "hi"}], "m", max_tokens=1) == ""
    assert http.posts[1]["json"]["max_tokens"] == 1


async def _record(recorded: list[dict], kwargs: dict) -> None:
    recorded.append(kwargs)


async def test_ollama_stream_events_sse_merges_tool_calls_and_usage() -> None:
    http = _Http()
    lines = [
        "ignored",
        "data: " + json.dumps({"choices": [{"delta": {"content": "tok"}}]}),
        "data: " + json.dumps({"usage": {"prompt_tokens": 2, "completion_tokens": 3}, "choices": [{"delta": {}}]}),
        "data: " + json.dumps({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "call-1", "function": {"name": "loo", "arguments": "{\"q\""}}]}}]}),
        "data: " + json.dumps({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "function": {"name": "lookup", "arguments": "{\"q\":\"x\"}"}}
        ]}, "finish_reason": "tool_calls"}]}),
        "data: [DONE]",
    ]
    http.streams.append(_Stream(lines))
    backend = _backend(http)

    events = [event async for event in backend.stream_events_sse("http://ollama/v1/chat/completions", {"model": "m"})]

    assert [event.kind for event in events] == ["token", "tool_call", "usage", "done"]
    assert events[1].raw == {"id": "call-1", "name": "lookup", "input": {"q": "x"}}
    assert events[2].raw == {"input_tokens": 2, "output_tokens": 3}
    assert http.stream_calls[0]["json"]["stream_options"] == {"include_usage": True}


async def test_ollama_stream_events_ndjson_handles_tokens_tools_and_bad_json() -> None:
    http = _Http()
    http.streams.append(
        _Stream([
            "",
            "{bad",
            json.dumps({"message": {"content": "tok", "tool_calls": [{"function": {"name": "lookup", "arguments": {"q": "x"}}}]}}),
            json.dumps({"done": True}),
        ])
    )
    backend = _backend(http)

    events = [event async for event in backend.stream_events_ndjson("http://ollama/api/chat", {"model": "m"}, tool_choice="auto")]

    assert [event.kind for event in events] == ["token", "tool_call", "usage", "done"]
    assert events[1].raw == {"id": "call_0", "name": "lookup", "input": {"q": "x"}}
    assert http.stream_calls[0]["json"]["tool_choice"] == "auto"


async def test_ollama_stream_via_emitter_and_chunk_feeding(monkeypatch: pytest.MonkeyPatch) -> None:
    http = _Http()
    backend = _backend(http)
    emitter = BufferingSignalEmitter()

    async def fake_events(target: str, body: dict):
        yield SimpleNamespace(kind="token", content="tok", raw=None)
        yield SimpleNamespace(kind="usage", content="", raw={"input_tokens": 1, "output_tokens": 2})
        yield SimpleNamespace(kind="done", content="", raw=None)

    recorded: list[dict] = []
    metering = SimpleNamespace(
        port=SimpleNamespace(record_tokens=lambda **kwargs: _record(recorded, kwargs)),
        caller=object(),
        session_key=None,
        workflow="wf",
        step="step",
        provider="ollama",
    )
    monkeypatch.setattr(backend, "stream_events_sse", fake_events)

    await backend.stream_via_emitter("target", {"model": "m"}, emitter, metering)

    assert [signal.kind for signal in emitter.collected()] == ["token", "done"]
    assert recorded[0]["output_tokens"] == 2
