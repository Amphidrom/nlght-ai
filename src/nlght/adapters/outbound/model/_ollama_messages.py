# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Canonical message mapping for Ollama local and Ollama Cloud."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from nlght.adapters.outbound.model._messages import (
    serialize_caller_instruction,
    serialize_untrusted_context,
)
from nlght.core.model.messages import (
    AssistantMessage,
    CallerInstructionMessage,
    MessageLike,
    ToolResultMessage,
    TrustedInstructionMessage,
    UntrustedContextMessage,
    UserMessage,
)


def to_ollama_messages(messages: Sequence[MessageLike]) -> list[dict[str, Any]]:
    """Map every authority type to Ollama's supported chat roles."""

    lowered: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, dict):
            lowered.append(dict(message))
        elif isinstance(message, TrustedInstructionMessage):
            lowered.append({"role": "system", "content": message.content})
        elif isinstance(message, CallerInstructionMessage):
            lowered.append({"role": "user", "content": serialize_caller_instruction(message)})
        elif isinstance(message, UserMessage):
            lowered.append({"role": "user", "content": message.content, **dict(message.attributes)})
        elif isinstance(message, AssistantMessage):
            wire = {"role": "assistant", "content": message.content, **dict(message.attributes)}
            if message.tool_calls:
                wire["tool_calls"] = [
                    {
                        "function": {
                            "name": call.name,
                            "arguments": dict(call.arguments),
                        },
                        **dict(call.provider_metadata),
                    }
                    for call in message.tool_calls
                ]
            lowered.append(wire)
        elif isinstance(message, ToolResultMessage):
            wire = {"role": "tool", "content": message.content, **dict(message.attributes)}
            if message.tool_call_id:
                wire["tool_call_id"] = message.tool_call_id
            if message.name:
                wire["name"] = message.name
            lowered.append(wire)
        elif isinstance(message, UntrustedContextMessage):
            lowered.append({"role": "user", "content": serialize_untrusted_context(message)})
        else:
            raise TypeError(f"unsupported Ollama canonical message type '{type(message).__name__}'")
    return lowered
