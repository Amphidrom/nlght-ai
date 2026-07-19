# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Tests for AnthropicModelClient, OpenAICloudModelClient, GoogleModelClient.

The SDK packages (anthropic, openai, google-genai) are optional extras and
are NOT installed in the test environment.  We therefore bypass __init__ via
__new__ and inject mocked clients directly.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nlght.adapters.outbound.model.anthropic import (
    AnthropicModelClient,
    BoundAnthropicModelClient,
)
from nlght.adapters.outbound.model.google import (
    BoundGoogleModelClient,
    GoogleModelClient,
    _to_google_contents,
)
from nlght.adapters.outbound.model.ollama_cloud import OllamaCloudClient
from nlght.adapters.outbound.model.openai_cloud import (
    DEFAULT_OPENAI_BASE_URL,
    BoundOpenAICloudModelClient,
    OpenAICloudModelClient,
)
from nlght.adapters.outbound.model.openai_cloud_debug import (
    OpenAICloudModelClient as DebugOpenAICloudModelClient,
)
from nlght.adapters.outbound.signals.buffering import BufferingSignalEmitter
from nlght.core.model.model_info import ModelInfo

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _async_iter(items):
    for item in items:
        yield item


# ---------------------------------------------------------------------------
# Anthropic
# ---------------------------------------------------------------------------

def _anthropic_client(default_model: str = "claude-sonnet-4-6") -> AnthropicModelClient:
    obj = AnthropicModelClient.__new__(AnthropicModelClient)
    obj._client = MagicMock()
    obj._default_model = default_model
    return obj


def test_anthropic_bind_returns_bound_client() -> None:
    client  = _anthropic_client()
    emitter = BufferingSignalEmitter()
    bound   = client.bind(model="claude-sonnet-4-6", emitter=emitter, stream=False)
    assert isinstance(bound, BoundAnthropicModelClient)


def test_anthropic_append_tool_turn_uses_canonical_ollama_shape() -> None:
    # append_tool_turn must return the canonical (Ollama-style) shape --
    # _convert_messages_for_anthropic lowers it to tool_use/tool_result
    # blocks on the next call, it does not accept Anthropic's own block
    # format as input.
    client  = _anthropic_client()
    emitter = BufferingSignalEmitter()
    bound   = client.bind(model="claude-sonnet-4-6", emitter=emitter, stream=False)

    updated = bound.append_tool_turn(
        [{"role": "user", "content": "weather?"}],
        [{"id": "toolu_01", "name": "get_weather", "input": {"city": "Berlin"}}],
        ["22C"],
        assistant_text="Let me check.",
    )

    assert updated[-2] == {
        "role": "assistant",
        "content": "Let me check.",
        "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Berlin"}}}],
    }
    assert updated[-1] == {"role": "tool", "content": "22C"}


def test_anthropic_append_tool_turn_round_trips_through_convert_messages() -> None:
    # Prove the two halves fit together: what append_tool_turn returns is
    # exactly what _convert_messages_for_anthropic expects as input.
    from nlght.adapters.outbound.model.anthropic import _convert_messages_for_anthropic

    client  = _anthropic_client()
    emitter = BufferingSignalEmitter()
    bound   = client.bind(model="claude-sonnet-4-6", emitter=emitter, stream=False)

    updated = bound.append_tool_turn(
        [{"role": "user", "content": "weather?"}],
        [{"id": "toolu_01", "name": "get_weather", "input": {"city": "Berlin"}}],
        ["22C"],
    )
    converted = _convert_messages_for_anthropic(updated)

    assistant_turn = converted[-2]
    assert assistant_turn["role"] == "assistant"
    tool_use = next(b for b in assistant_turn["content"] if b["type"] == "tool_use")
    assert tool_use["name"] == "get_weather"
    assert tool_use["input"] == {"city": "Berlin"}

    result_turn = converted[-1]
    assert result_turn["role"] == "user"
    tool_result = result_turn["content"][0]
    assert tool_result["type"] == "tool_result"
    assert tool_result["content"] == "22C"
    assert tool_result["tool_use_id"] == tool_use["id"]


def test_anthropic_bind_uses_default_model() -> None:
    client  = _anthropic_client(default_model="claude-opus-4-6")
    emitter = BufferingSignalEmitter()
    bound   = client.bind(model=None, emitter=emitter, stream=False)
    assert bound._model == "claude-opus-4-6"


async def test_anthropic_token_budget_known_model() -> None:
    client = _anthropic_client()
    budget = await client.token_budget("claude-sonnet-4-6")
    assert budget.total == 200_000


async def test_anthropic_token_budget_unknown_model() -> None:
    client = _anthropic_client()
    budget = await client.token_budget("unknown-model-xyz")
    assert budget.total == 200_000  # default


async def test_anthropic_complete_non_stream() -> None:
    client  = _anthropic_client()
    emitter = BufferingSignalEmitter()

    fake_response = MagicMock()
    fake_response.content = [SimpleNamespace(text="hello from claude")]
    client._client.messages.create = AsyncMock(return_value=fake_response)

    bound = client.bind(model="claude-sonnet-4-6", emitter=emitter, stream=False)
    await bound.call([{"role": "user", "content": "hi"}])

    signals = emitter.collected()
    assert len(signals) == 1
    assert signals[0].content == "hello from claude"
    assert signals[0].kind == "result"


async def test_anthropic_stream() -> None:
    client  = _anthropic_client()
    emitter = BufferingSignalEmitter()

    class _FakeStream:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *_):
            pass
        @property
        def text_stream(self):
            async def _gen():
                yield "chunk1"
                yield "chunk2"
            return _gen()

    client._client.messages.stream = MagicMock(return_value=_FakeStream())
    bound = client.bind(model="claude-sonnet-4-6", emitter=emitter, stream=True)
    await bound.call([{"role": "user", "content": "hi"}])

    signals = emitter.collected()
    tokens  = [s for s in signals if s.kind == "token"]
    done    = [s for s in signals if s.kind == "done"]
    assert len(tokens) == 2
    assert len(done) == 1


async def test_anthropic_stream_surfaces_tool_use_as_tool_call() -> None:
    # Regression: AsyncMessageStream.get_final_message() is async and MUST be
    # awaited. Without the await, `final` is a coroutine, the except swallows the
    # AttributeError, and tool_use blocks never reach the caller as tool_call
    # events — the streaming tool loop silently produces no call.
    client  = _anthropic_client()
    emitter = BufferingSignalEmitter()

    final_msg = SimpleNamespace(
        usage=SimpleNamespace(input_tokens=3, output_tokens=5),
        content=[SimpleNamespace(type="tool_use", id="toolu_1", name="get_magic_number", input={})],
    )

    class _FakeStream:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *_):
            pass
        @property
        def text_stream(self):
            async def _gen():
                if False:  # model went straight to a tool call — no text
                    yield ""
            return _gen()
        async def get_final_message(self):
            return final_msg

    client._client.messages.stream = MagicMock(return_value=_FakeStream())
    bound = client.bind(model="claude-sonnet-4-6", emitter=emitter, stream=True)

    tool_def = {"type": "function", "function": {"name": "get_magic_number", "parameters": {"type": "object", "properties": {}}}}
    events = [e async for e in bound.stream([{"role": "user", "content": "call it"}], tools=[tool_def])]

    tool_calls = [e for e in events if e.kind == "tool_call"]
    assert len(tool_calls) == 1
    assert tool_calls[0].raw == {"id": "toolu_1", "name": "get_magic_number", "input": {}}
    assert events[-1].kind == "done"


async def test_anthropic_system_message_extracted() -> None:
    client  = _anthropic_client()
    emitter = BufferingSignalEmitter()

    fake_response = MagicMock()
    fake_response.content = [SimpleNamespace(text="ok")]
    client._client.messages.create = AsyncMock(return_value=fake_response)

    bound = client.bind(model="claude-sonnet-4-6", emitter=emitter, stream=False)
    await bound.call([
        {"role": "system", "content": "Be helpful"},
        {"role": "user", "content": "hi"},
    ])
    call_kwargs = client._client.messages.create.call_args.kwargs
    assert call_kwargs["system"] == "Be helpful"


async def test_anthropic_empty_response_handled() -> None:
    client  = _anthropic_client()
    emitter = BufferingSignalEmitter()

    fake_response = MagicMock()
    fake_response.content = []
    client._client.messages.create = AsyncMock(return_value=fake_response)

    bound = client.bind(model="claude-sonnet-4-6", emitter=emitter, stream=False)
    await bound.call([{"role": "user", "content": "hi"}])
    signals = emitter.collected()
    assert signals[0].content == ""


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("client_type", [OpenAICloudModelClient, DebugOpenAICloudModelClient])
def test_openai_defaults_to_official_endpoint(monkeypatch, client_type) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "")

    client = client_type(api_key="", default_model="gpt-4o-mini")

    assert str(client._client.base_url).rstrip("/") == DEFAULT_OPENAI_BASE_URL


@pytest.mark.parametrize("client_type", [OpenAICloudModelClient, DebugOpenAICloudModelClient])
def test_openai_base_url_override_wins(monkeypatch, client_type) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    client = client_type(
        api_key="",
        default_model="deployment-name",
        base_url="https://example.openai.azure.com/openai/deployments/example",
    )

    assert str(client._client.base_url).rstrip("/") == (
        "https://example.openai.azure.com/openai/deployments/example"
    )


def _openai_client(default_model: str = "gpt-4o") -> OpenAICloudModelClient:
    obj = OpenAICloudModelClient.__new__(OpenAICloudModelClient)
    obj._client = MagicMock()
    obj._default_model = default_model
    return obj


def test_openai_bind_uses_default_model() -> None:
    client  = _openai_client(default_model="gpt-4o-mini")
    emitter = BufferingSignalEmitter()
    bound   = client.bind(model=None, emitter=emitter, stream=False)
    assert bound._model == "gpt-4o-mini"


def test_openai_append_tool_turn_uses_openai_wire_format() -> None:
    client  = _openai_client()
    emitter = BufferingSignalEmitter()
    bound   = client.bind(model="gpt-4o", emitter=emitter, stream=False)

    updated = bound.append_tool_turn(
        [{"role": "user", "content": "weather?"}],
        [{"id": "call_01", "name": "get_weather", "input": {"city": "Berlin"}}],
        ["22C"],
        assistant_text="Let me check.",
    )

    assistant_turn = updated[-2]
    assert assistant_turn["role"] == "assistant"
    assert assistant_turn["tool_calls"][0]["id"] == "call_01"
    assert assistant_turn["tool_calls"][0]["type"] == "function"
    # OpenAI wants arguments as a JSON string, not a dict.
    assert assistant_turn["tool_calls"][0]["function"]["arguments"] == '{"city": "Berlin"}'

    tool_turn = updated[-1]
    assert tool_turn == {"role": "tool", "tool_call_id": "call_01", "content": "22C", "name": "get_weather"}


def test_openai_bind_returns_bound_client() -> None:
    client  = _openai_client()
    emitter = BufferingSignalEmitter()
    bound   = client.bind(model="gpt-4o", emitter=emitter, stream=False)
    assert isinstance(bound, BoundOpenAICloudModelClient)


async def test_openai_token_budget_known_model() -> None:
    client = _openai_client()
    budget = await client.token_budget("gpt-4o")
    assert budget.total == 128_000


async def test_openai_token_budget_o1() -> None:
    client = _openai_client()
    budget = await client.token_budget("o1")
    assert budget.total == 200_000


async def test_openai_token_budget_unknown_model() -> None:
    client = _openai_client()
    budget = await client.token_budget("future-model-xyz")
    assert budget.total == 128_000


async def test_openai_complete_non_stream() -> None:
    client  = _openai_client()
    emitter = BufferingSignalEmitter()

    fake = MagicMock()
    fake.choices = [SimpleNamespace(message=SimpleNamespace(content="hello openai"))]
    client._client.chat.completions.create = AsyncMock(return_value=fake)

    bound = client.bind(model="gpt-4o", emitter=emitter, stream=False)
    await bound.call([{"role": "user", "content": "hi"}])

    signals = emitter.collected()
    assert signals[0].content == "hello openai"
    assert signals[0].kind == "result"


async def test_openai_stream() -> None:
    client  = _openai_client()
    emitter = BufferingSignalEmitter()

    class _Chunk:
        def __init__(self, text):
            self.choices = [SimpleNamespace(delta=SimpleNamespace(content=text))]

    client._client.chat.completions.create = AsyncMock(
        return_value=_async_iter([_Chunk("tok1"), _Chunk("tok2")])
    )

    bound = client.bind(model="gpt-4o", emitter=emitter, stream=True)
    await bound.call([{"role": "user", "content": "hi"}])

    signals = emitter.collected()
    tokens  = [s for s in signals if s.kind == "token"]
    done    = [s for s in signals if s.kind == "done"]
    assert len(tokens) == 2
    assert len(done) == 1


async def test_openai_empty_response_handled() -> None:
    client  = _openai_client()
    emitter = BufferingSignalEmitter()

    fake = MagicMock()
    fake.choices = []
    client._client.chat.completions.create = AsyncMock(return_value=fake)

    bound = client.bind(model="gpt-4o", emitter=emitter, stream=False)
    await bound.call([{"role": "user", "content": "hi"}])
    assert emitter.collected()[0].content == ""


# ---------------------------------------------------------------------------
# Google
# ---------------------------------------------------------------------------

def _google_client(default_model: str = "gemini-3.5-flash") -> GoogleModelClient:
    obj = GoogleModelClient.__new__(GoogleModelClient)
    obj._client = MagicMock()

    # Stub out genai types so the inline imports inside _complete/_stream work
    fake_genai = MagicMock()
    fake_types  = MagicMock()
    fake_types.GenerateContentConfig = MagicMock(return_value=MagicMock())
    fake_genai.types = fake_types
    sys.modules.setdefault("google", MagicMock())
    sys.modules["google.genai"] = fake_genai
    sys.modules["google.genai.types"] = fake_types
    obj._genai = fake_genai

    obj._default_model = default_model
    return obj


def test_google_bind_uses_default_model() -> None:
    client  = _google_client(default_model="gemini-2.5-pro")
    emitter = BufferingSignalEmitter()
    bound   = client.bind(model=None, emitter=emitter, stream=False)
    assert bound._model == "gemini-2.5-pro"


def test_google_append_tool_turn_uses_canonical_ollama_shape() -> None:
    client  = _google_client()
    emitter = BufferingSignalEmitter()
    bound   = client.bind(model="gemini-3.5-flash", emitter=emitter, stream=False)

    updated = bound.append_tool_turn(
        [{"role": "user", "content": "weather?"}],
        [{"id": "1", "name": "get_weather", "input": {"city": "Berlin"}}],
        ["22C"],
    )

    assert updated[-2] == {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Berlin"}}}],
    }
    assert updated[-1] == {"role": "tool", "content": "22C"}


def test_to_google_contents_lowers_tool_turn_to_function_call_and_response() -> None:
    # Prove the two halves fit together: what append_tool_turn returns is
    # exactly what _to_google_contents expects as input on the next call.
    messages = [
        {"role": "user", "content": "weather?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Berlin"}}}],
        },
        {"role": "tool", "content": "22C"},
    ]

    _system, contents = _to_google_contents(messages)

    model_turn = next(c for c in contents if c["role"] == "model" and any("function_call" in p for p in c["parts"]))
    fc_part = next(p for p in model_turn["parts"] if "function_call" in p)
    assert fc_part["function_call"] == {"name": "get_weather", "args": {"city": "Berlin"}}

    response_turn = contents[-1]
    assert response_turn["role"] == "user"
    fr_part = next(p for p in response_turn["parts"] if "function_response" in p)
    assert fr_part["function_response"] == {"name": "get_weather", "response": {"output": "22C"}}


def test_to_google_contents_carries_thought_signature_on_function_call() -> None:
    # Gemini 3.x requires the model's thought_signature echoed back on the
    # function_call part; it rides through the canonical tool-turn as base64 and
    # must be decoded back to bytes on the Google part (else 400 INVALID_ARGUMENT).
    import base64
    sig = b"\x00\x01thought-sig"
    messages = [
        {"role": "user", "content": "weather?"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{
                "function": {"name": "get_weather", "arguments": {"city": "Berlin"}},
                "thought_signature": base64.b64encode(sig).decode("ascii"),
            }],
        },
        {"role": "tool", "content": "22C"},
    ]

    _system, contents = _to_google_contents(messages)

    fc_part = next(p for c in contents if c["role"] == "model" for p in c["parts"] if "function_call" in p)
    assert fc_part["thought_signature"] == sig  # decoded back to the original bytes


def test_google_append_tool_turn_preserves_thought_signature() -> None:
    # The provider-opaque thought_signature survives the canonical tool-turn so
    # _to_google_contents can re-emit it on the next call.
    client  = _google_client()
    emitter = BufferingSignalEmitter()
    bound   = client.bind(model="gemini-3.5-flash", emitter=emitter, stream=False)

    updated = bound.append_tool_turn(
        [{"role": "user", "content": "weather?"}],
        [{"id": "call_x", "name": "get_weather", "input": {"city": "Berlin"}, "thought_signature": "YWJj"}],
        ["22C"],
    )
    assert updated[-2]["tool_calls"][0]["thought_signature"] == "YWJj"


def test_google_bind_returns_bound_client() -> None:
    client  = _google_client()
    emitter = BufferingSignalEmitter()
    bound   = client.bind(model="gemini-3.5-flash", emitter=emitter, stream=False)
    assert isinstance(bound, BoundGoogleModelClient)


async def test_google_token_budget_known_model() -> None:
    client = _google_client()
    budget = await client.token_budget("gemini-3.5-flash")
    assert budget.total == 1_048_576


async def test_google_token_budget_unknown_model() -> None:
    client = _google_client()
    budget = await client.token_budget("future-gemini-99")
    assert budget.total == 1_048_576  # default


async def test_google_complete_non_stream() -> None:
    client  = _google_client()
    emitter = BufferingSignalEmitter()

    fake = MagicMock()
    fake.text = "hello gemini"
    client._client.aio.models.generate_content = AsyncMock(return_value=fake)

    bound = client.bind(model="gemini-3.5-flash", emitter=emitter, stream=False)
    await bound.call([{"role": "user", "content": "hi"}])

    signals = emitter.collected()
    assert signals[0].content == "hello gemini"
    assert signals[0].kind == "result"


async def test_google_stream() -> None:
    client  = _google_client()
    emitter = BufferingSignalEmitter()

    class _Chunk:
        def __init__(self, text):
            self.text = text

    client._client.aio.models.generate_content_stream = AsyncMock(
        return_value=_async_iter([_Chunk("tok1"), _Chunk("tok2")])
    )

    bound = client.bind(model="gemini-3.5-flash", emitter=emitter, stream=True)
    await bound.call([{"role": "user", "content": "hi"}])

    signals = emitter.collected()
    tokens  = [s for s in signals if s.kind == "token"]
    done    = [s for s in signals if s.kind == "done"]
    assert len(tokens) == 2
    assert len(done) == 1


# ---------------------------------------------------------------------------
# _to_google_contents
# ---------------------------------------------------------------------------

def test_to_google_contents_extracts_system() -> None:
    system, contents = _to_google_contents([
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "hello"},
    ])
    assert system == "You are helpful."
    assert len(contents) == 1
    assert contents[0]["role"] == "user"


def test_to_google_contents_maps_assistant_to_model() -> None:
    _, contents = _to_google_contents([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ])
    assert contents[1]["role"] == "model"


def test_to_google_contents_merges_consecutive_same_role() -> None:
    _, contents = _to_google_contents([
        {"role": "user", "content": "msg1"},
        {"role": "user", "content": "msg2"},
    ])
    assert len(contents) == 1
    assert len(contents[0]["parts"]) == 2


def test_to_google_contents_multiple_system_messages_joined() -> None:
    system, _ = _to_google_contents([
        {"role": "system", "content": "Part1"},
        {"role": "system", "content": "Part2"},
        {"role": "user", "content": "hi"},
    ])
    assert "Part1" in system
    assert "Part2" in system


# ---------------------------------------------------------------------------
# list_models — Anthropic is static-only; OpenAI/Google query live, with a
# static fallback on error
# ---------------------------------------------------------------------------

async def test_anthropic_list_models_returns_static_list() -> None:
    client = _anthropic_client()
    models = await client.list_models()
    assert all(isinstance(m, ModelInfo) for m in models)
    names = [m.name for m in models]
    assert "claude-sonnet-4-6" in names
    assert "claude-opus-4-6" in names


async def test_openai_list_models_returns_static_fallback_on_error() -> None:
    """When the SDK raises, list_models falls back to the static catalogue."""
    client = _openai_client()
    client._client.models.list = AsyncMock(side_effect=Exception("network error"))
    models = await client.list_models()
    assert len(models) > 0
    assert all(isinstance(m, ModelInfo) for m in models)


async def test_openai_list_models_uses_api_when_available() -> None:
    client = _openai_client()
    fake_page = MagicMock()
    fake_page.data = [SimpleNamespace(id="gpt-4o"), SimpleNamespace(id="gpt-4o-mini")]
    client._client.models.list = AsyncMock(return_value=fake_page)
    models = await client.list_models()
    assert [m.name for m in models] == ["gpt-4o", "gpt-4o-mini"]


async def test_google_list_models_returns_static_fallback_on_error() -> None:
    """When the SDK raises, list_models falls back to the static catalogue."""
    client = _google_client()
    client._client.aio.models.list = AsyncMock(side_effect=Exception("network error"))
    models = await client.list_models()
    assert len(models) > 0
    assert all(isinstance(m, ModelInfo) for m in models)
    assert "gemini-3.5-flash" in [m.name for m in models]


async def test_google_list_models_uses_api_when_available() -> None:
    client = _google_client()
    fake_models = [
        SimpleNamespace(name="publishers/google/models/gemini-3.5-flash"),
        SimpleNamespace(name="publishers/google/models/gemini-3.5-pro"),
    ]
    client._client.aio.models.list = AsyncMock(return_value=_async_iter(fake_models))
    models = await client.list_models()
    assert [m.name for m in models] == ["gemini-3.5-flash", "gemini-3.5-pro"]


# ---------------------------------------------------------------------------
# Ollama Cloud — provider-specific static budget (no /api/show)
# ---------------------------------------------------------------------------

def _ollama_cloud_client(default_model: str = "qwen3-coder:480b") -> OllamaCloudClient:
    obj = OllamaCloudClient.__new__(OllamaCloudClient)
    obj._http = MagicMock()
    obj._default_model = default_model
    return obj


async def test_ollama_cloud_token_budget_known_model() -> None:
    client = _ollama_cloud_client()
    budget = await client.token_budget("qwen3-coder:480b")
    assert budget.total == 262_144


async def test_ollama_cloud_token_budget_prefix_match() -> None:
    # Suffixed cloud names resolve via prefix-match to the same window.
    client = _ollama_cloud_client()
    budget = await client.token_budget("qwen3-coder:480b-cloud")
    assert budget.total == 262_144


async def test_ollama_cloud_token_budget_unknown_model() -> None:
    client = _ollama_cloud_client()
    budget = await client.token_budget("future-cloud-model-xyz")
    assert budget.total == 131_072  # default


async def test_ollama_cloud_token_budget_does_not_query_api_show() -> None:
    # The cloud provider must not fall back to the local /api/show transport.
    client = _ollama_cloud_client()
    await client.token_budget("qwen3-coder:480b")
    client._http.token_budget.assert_not_called()


def test_ollama_cloud_defaults_to_public_endpoint() -> None:
    # A cloud provider needs only an API key — base_url defaults to ollama.com.
    client = OllamaCloudClient(http_client=MagicMock(), default_model="gpt-oss:20b")
    assert client.native_base_url == "https://ollama.com"


def test_ollama_cloud_base_url_override_wins() -> None:
    client = OllamaCloudClient(
        http_client=MagicMock(),
        base_url="https://remote.example/",
        default_model="gpt-oss:20b",
    )
    assert client.native_base_url == "https://remote.example"
