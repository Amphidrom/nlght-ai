# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Check cloud model clients against installed provider SDK signatures.

The tests use real SDK clients with substituted HTTP transports."""
from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING, Any

import httpx2
import pytest

from nlght.adapters.outbound.signals.buffering import BufferingSignalEmitter
from nlght.core.model.messages import UserMessage

if TYPE_CHECKING:
    import httpx

# The real google-genai modules, captured at collection. Other tests in this
# package install MagicMock stand-ins under these names in sys.modules and do
# not remove them. The Google test below restores the real ones for its run.
try:
    import google.genai as _real_genai
    import google.genai.types as _real_genai_types
except ImportError:  # pragma: no cover - google extra not installed
    _real_genai = None
    _real_genai_types = None


def _text(emitter: BufferingSignalEmitter) -> str:
    return "".join(s.content for s in emitter.collected() if s.content)


async def test_anthropic_call_reaches_the_sdk_without_sampling_parameters() -> None:
    anthropic = pytest.importorskip("anthropic")
    from nlght.adapters.outbound.model.anthropic import AnthropicModelClient

    seen: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(json.loads(request.content))
        return httpx2.Response(200, json={
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-sonnet-4-6",
            "content": [{"type": "text", "text": "hello"}],
            "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 3, "output_tokens": 1},
        })

    client = AnthropicModelClient(api_key="test-key", default_model="claude-sonnet-4-6")
    client._client = anthropic.AsyncAnthropic(
        api_key="test-key",
        http_client=anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler)),
    )
    emitter = BufferingSignalEmitter()

    await client.bind(model=None, emitter=emitter, stream=False).call([UserMessage("hi")])

    assert _text(emitter) == "hello"
    assert seen[0]["model"] == "claude-sonnet-4-6"
    assert "temperature" not in seen[0]


async def test_anthropic_stream_reaches_the_sdk_without_sampling_parameters() -> None:
    anthropic = pytest.importorskip("anthropic")
    from nlght.adapters.outbound.model.anthropic import AnthropicModelClient

    events = [
        ("message_start", {"type": "message_start", "message": {
            "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-sonnet-4-6",
            "content": [], "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": 3, "output_tokens": 0},
        }}),
        ("content_block_start", {"type": "content_block_start", "index": 0,
                                 "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0,
                                 "delta": {"type": "text_delta", "text": "streamed"}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                           "usage": {"output_tokens": 1}}),
        ("message_stop", {"type": "message_stop"}),
    ]
    sse = "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events)
    seen: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(json.loads(request.content))
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=sse.encode())

    client = AnthropicModelClient(api_key="test-key", default_model="claude-sonnet-4-6")
    client._client = anthropic.AsyncAnthropic(
        api_key="test-key",
        http_client=anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler)),
    )
    emitter = BufferingSignalEmitter()

    await client.bind(model=None, emitter=emitter, stream=True).call([UserMessage("hi")])

    assert "streamed" in _text(emitter)
    assert seen[0]["stream"] is True
    assert "temperature" not in seen[0]


async def test_openai_call_sends_the_provider_temperature() -> None:
    openai = pytest.importorskip("openai")
    from nlght.adapters.outbound.model.openai_cloud import OpenAICloudModelClient

    seen: list[dict[str, Any]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(json.loads(request.content))
        return httpx2.Response(200, json={
            "id": "chatcmpl-1", "object": "chat.completion", "created": 0, "model": "gpt-4o",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "hello"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
        })

    client = OpenAICloudModelClient(api_key="test-key", default_model="gpt-4o", temperature=0.2)
    client._client = openai.AsyncOpenAI(
        api_key="test-key",
        base_url="https://api.openai.com/v1",
        http_client=openai.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler)),
    )
    emitter = BufferingSignalEmitter()

    await client.bind(model=None, emitter=emitter, stream=False).call([UserMessage("hi")])

    assert _text(emitter) == "hello"
    assert seen[0]["temperature"] == 0.2


async def test_google_call_sends_the_provider_temperature(monkeypatch: pytest.MonkeyPatch) -> None:
    if _real_genai is None:
        pytest.skip("google-genai not installed")
    monkeypatch.setitem(sys.modules, "google.genai", _real_genai)
    monkeypatch.setitem(sys.modules, "google.genai.types", _real_genai_types)
    genai = _real_genai
    httpx = pytest.importorskip("httpx")
    from nlght.adapters.outbound.model.google import GoogleModelClient

    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={
            "candidates": [{"content": {"role": "model", "parts": [{"text": "hello"}]}, "finishReason": "STOP"}],
            "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 1, "totalTokenCount": 4},
        })

    client = GoogleModelClient(api_key="test-key", default_model="gemini-3.6-flash", temperature=0.2)
    client._client = genai.Client(
        api_key="test-key",
        http_options=genai.types.HttpOptions(
            httpx_async_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        ),
    )
    emitter = BufferingSignalEmitter()

    await client.bind(model=None, emitter=emitter, stream=False).call([UserMessage("hi")])

    assert _text(emitter) == "hello"
    assert seen[0]["generationConfig"]["temperature"] == 0.2
