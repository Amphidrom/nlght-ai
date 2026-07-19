# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from nlght.adapters.outbound.model.anthropic import (
    _convert_messages_for_anthropic,
    _split_messages,
    _to_anthropic_tool,
    _to_anthropic_tool_choice,
)
from nlght.adapters.outbound.model.openai_cloud import OpenAICloudModelClient
from nlght.adapters.outbound.signals.buffering import BufferingSignalEmitter


async def _async_iter(items):
    for item in items:
        yield item


async def _record(recorded: list[dict], **kwargs: object) -> None:
    recorded.append(kwargs)


def test_anthropic_helpers_convert_tools_and_tool_history() -> None:
    system, chat = _split_messages([
        {"role": "system", "content": "A"},
        {"role": "system", "content": "B"},
        {"role": "user", "content": "hi"},
    ])
    assert system == "A\n\nB"
    assert chat == [{"role": "user", "content": "hi"}]

    tool = _to_anthropic_tool(
        {"type": "function", "function": {"name": "lookup", "description": "desc", "parameters": {"type": "object"}}}
    )
    assert tool == {"name": "lookup", "description": "desc", "input_schema": {"type": "object"}}
    assert _to_anthropic_tool_choice("required") == {"type": "any"}
    assert _to_anthropic_tool_choice("unknown") == {"type": "auto"}
    assert _to_anthropic_tool_choice({"type": "function", "function": {"name": "lookup"}}) == {
        "type": "tool",
        "name": "lookup",
    }

    converted = _convert_messages_for_anthropic([
        {"role": "system", "content": "sys"},
        {
            "role": "assistant",
            "content": "need tool",
            "tool_calls": [
                {"function": {"name": "lookup", "arguments": {"q": "x"}}},
                {"function": {"name": "fetch", "arguments": {"url": "u"}}},
            ],
        },
        {"role": "tool", "content": "lookup-result"},
        {"role": "tool", "content": "fetch-result"},
        {"role": "user", "content": "done"},
    ])

    assert converted[0]["role"] == "assistant"
    assert converted[0]["content"][1]["id"] == "toolu_0001"
    assert converted[1]["content"][0]["tool_use_id"] == "toolu_0001"
    assert converted[1]["content"][1]["content"] == "fetch-result"


async def test_openai_complete_and_stream_record_metering() -> None:
    client = OpenAICloudModelClient.__new__(OpenAICloudModelClient)
    client._default_model = "gpt-4o"
    client._client = MagicMock()
    recorded: list[dict] = []
    metering = SimpleNamespace(
        port=SimpleNamespace(record_tokens=lambda **kwargs: _record(recorded, **kwargs)),
        caller=object(),
        session_key="s",
        workflow="wf",
        step="step",
        provider="openai",
    )

    complete_response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="answer"))],
        usage=SimpleNamespace(prompt_tokens=9, completion_tokens=4),
    )
    client._client.chat.completions.create = AsyncMock(return_value=complete_response)
    emitter = BufferingSignalEmitter()

    await client._complete([{"role": "user", "content": "hi"}], "gpt-4o", emitter, metering)

    assert emitter.collected()[0].content == "answer"
    assert recorded[0]["input_tokens"] == 9

    chunk_with_token = SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content="tok"))],
        usage=None,
        model_dump=lambda: {"raw": True},
    )
    chunk_with_usage = SimpleNamespace(
        choices=[],
        usage=SimpleNamespace(prompt_tokens=1, completion_tokens=2),
    )
    client._client.chat.completions.create = AsyncMock(
        return_value=_async_iter([chunk_with_token, chunk_with_usage])
    )

    bound = client.bind(model="gpt-4o", emitter=BufferingSignalEmitter(), stream=True, metering=metering)
    events = [event async for event in bound.stream([{"role": "user", "content": "hi"}])]

    assert [event.kind for event in events] == ["token", "done"]
    assert recorded[-1]["output_tokens"] == 2


async def test_openai_metering_failures_do_not_break_completion() -> None:
    client = OpenAICloudModelClient.__new__(OpenAICloudModelClient)
    client._client = MagicMock()
    client._client.chat.completions.create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
        )
    )
    metering = SimpleNamespace(
        port=SimpleNamespace(record_tokens=AsyncMock(side_effect=RuntimeError("metering down"))),
        caller=object(),
        session_key="s",
        workflow="wf",
        step="step",
        provider="openai",
    )
    emitter = BufferingSignalEmitter()

    await client._complete([], "gpt-4o", emitter, metering)

    assert emitter.collected()[0].content == "ok"
