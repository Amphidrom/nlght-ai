# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Provider-neutral messages with explicit authority.

Provider role names are transport details, not the platform's trust model.  In
particular, a caller-supplied ``{"role": "system"}`` is still caller content;
only installed application code can construct :class:`TrustedInstructionMessage`.
Knowledge is kept as typed records until a provider adapter deliberately lowers
it to that provider's safest available untrusted-data representation.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, TypeAlias


class ContextKind(StrEnum):
    """Why an untrusted record is present in the current model context."""

    CONVERSATION = "turn"
    SESSION_RESULT = "result"
    WORKING_MEMORY = "atom"
    RETRIEVAL = "passage"
    REQUEST_INTERPRETATION = "request_interpretation"
    SUMMARY = "summary"
    DELTA = "delta"
    PLAYBOOK = "playbook"
    TOOL_GUIDANCE = "tool_guidance"
    PROMPT_SLOT = "prompt_slot"
    OTHER = "other"

    @classmethod
    def _missing_(cls, value: object) -> ContextKind:
        """Keep extension element labels typed without a central registry."""

        obj = str.__new__(cls, str(value))
        obj._value_ = str(value)
        obj._name_ = str(value)
        return obj


@dataclass(frozen=True, slots=True)
class ContextSupport:
    """One source observation supporting a context record."""

    document_id: str = ""
    observed_document_revision: str = ""
    document_path: str = ""
    slot_id: str = ""


@dataclass(frozen=True, slots=True)
class ContextProvenance:
    """Typed source lineage retained until provider serialization.

    The fields intentionally cover retrieval's full citation identity as well
    as session/runtime sources. Empty fields mean "not applicable", not an
    invitation to collapse the object into a free-form metadata dictionary.
    """

    source: str
    store: str = ""
    item_id: str = ""
    turn_nr: int | None = None
    document_id: str = ""
    chunk_id: str = ""
    assertion_id: str = ""
    source_revision_id: str = ""
    processing_revision_id: str = ""
    knowledge_revision_id: str = ""
    position: int = 0
    start_offset: int = 0
    end_offset: int = 0
    path: str = ""
    source_name: str = ""
    external_id: str = ""
    support: tuple[ContextSupport, ...] = ()


@dataclass(frozen=True, slots=True)
class ContextRecord:
    """One inert piece of knowledge offered to a model as untrusted data."""

    record_id: str
    kind: ContextKind
    content: str
    provenance: ContextProvenance
    section: str = ""
    representation_level: str = "full"
    retention: str = "useful"
    relevance: float = 0.0
    estimated_tokens: int = 0


@dataclass(frozen=True, slots=True)
class PromptEnvelope:
    """Trusted instructions and untrusted context before message assembly."""

    trusted_instructions: str
    context: tuple[ContextRecord, ...] = ()


def compose_prompt(
    trusted_instructions: str,
    context: Sequence[ContextRecord],
) -> PromptEnvelope:
    """Join separately built authority domains without flattening either one."""

    return PromptEnvelope(
        trusted_instructions=trusted_instructions,
        context=tuple(context),
    )


@dataclass(frozen=True, slots=True)
class TrustedInstructionMessage:
    """Deployment-authored instructions allowed to reach a system channel."""

    content: str


@dataclass(frozen=True, slots=True)
class CallerInstructionMessage:
    """Caller content that claimed a privileged provider role on input."""

    content: Any
    claimed_role: str
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class UserMessage:
    content: Any
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolCall:
    """Provider-neutral tool invocation plus opaque continuation metadata."""

    call_id: str
    name: str
    arguments: Mapping[str, Any] = field(default_factory=dict)
    provider_metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AssistantMessage:
    content: Any = ""
    tool_calls: tuple[ToolCall, ...] = ()
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolResultMessage:
    content: Any
    tool_call_id: str = ""
    name: str = ""
    original_role: str = "tool"
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class UntrustedContextMessage:
    records: tuple[ContextRecord, ...]


CanonicalMessage: TypeAlias = (
    TrustedInstructionMessage
    | CallerInstructionMessage
    | UserMessage
    | AssistantMessage
    | ToolResultMessage
    | UntrustedContextMessage
)
MessageLike: TypeAlias = CanonicalMessage | Mapping[str, Any]


def normalize_workflow_messages(
    messages: Sequence[MessageLike],
) -> list[CanonicalMessage]:
    """Normalize inbound workflow messages without accepting claimed authority.

    Already-canonical messages are preserved for internal callers. Raw messages
    are the public compatibility edge and can never create a trusted message.
    """

    normalized: list[CanonicalMessage] = []
    canonical_types = (
        TrustedInstructionMessage,
        CallerInstructionMessage,
        UserMessage,
        AssistantMessage,
        ToolResultMessage,
        UntrustedContextMessage,
    )
    for message in messages:
        if isinstance(message, canonical_types):
            normalized.append(message)
            continue
        if not isinstance(message, Mapping):
            raise TypeError(f"workflow message must be a mapping or canonical message, got {type(message).__name__}")

        raw = dict(message)
        role = str(raw.pop("role", ""))
        content = raw.pop("content", "")
        if role in {"system", "developer"}:
            normalized.append(CallerInstructionMessage(
                content=content,
                claimed_role=role,
                attributes=raw,
            ))
        elif role == "user":
            normalized.append(UserMessage(content=content, attributes=raw))
        elif role == "assistant":
            tool_calls = tuple(
                _normalize_tool_call(call)
                for call in (raw.pop("tool_calls", ()) or ())
            )
            normalized.append(AssistantMessage(
                content=content,
                tool_calls=tool_calls,
                attributes=raw,
            ))
        elif role in {"tool", "function"}:
            normalized.append(ToolResultMessage(
                content=content,
                tool_call_id=str(raw.pop("tool_call_id", "")),
                name=str(raw.pop("name", "")),
                original_role=role,
                attributes=raw,
            ))
        else:
            raise ValueError(f"unsupported workflow message role '{role}'")
    return normalized


def append_canonical_tool_turn(
    messages: Sequence[MessageLike],
    tool_calls_raw: Sequence[Mapping[str, Any]],
    results: Sequence[str],
    assistant_text: str = "",
) -> list[CanonicalMessage]:
    """Append genuine typed tool calls/results without choosing a provider wire."""

    current = normalize_workflow_messages(messages)
    calls = tuple(
        ToolCall(
            call_id=str(call.get("id", "")),
            name=str(call.get("name", "")),
            arguments=dict(call.get("input") or {}) if isinstance(call.get("input"), Mapping) else {},
            provider_metadata={
                key: value
                for key, value in call.items()
                if key not in {"id", "name", "input"}
            },
        )
        for call in tool_calls_raw
    )
    current.append(AssistantMessage(content=assistant_text, tool_calls=calls))
    for call, result in zip(calls, results, strict=False):
        current.append(ToolResultMessage(
            content=result,
            tool_call_id=call.call_id,
            name=call.name,
        ))
    return current


def _normalize_tool_call(call: object) -> ToolCall:
    if isinstance(call, ToolCall):
        return call
    if not isinstance(call, Mapping):
        raise TypeError(f"tool call must be a mapping or ToolCall, got {type(call).__name__}")
    raw = dict(call)
    function = raw.pop("function", {}) or {}
    if not isinstance(function, Mapping):
        function = {}
    function_data = dict(function)
    arguments = function_data.pop("arguments", {}) or {}
    if isinstance(arguments, str):
        try:
            decoded = json.loads(arguments)
            arguments = decoded if isinstance(decoded, Mapping) else {}
        except json.JSONDecodeError:
            arguments = {}
    if not isinstance(arguments, Mapping):
        arguments = {}
    name = str(function_data.pop("name", raw.pop("name", "")))
    call_id = str(raw.pop("id", ""))
    raw.pop("type", None)
    return ToolCall(
        call_id=call_id,
        name=name,
        arguments=dict(arguments),
        provider_metadata={**raw, **function_data},
    )


def compose_messages(
    envelope: PromptEnvelope,
    conversation: Sequence[CanonicalMessage],
) -> list[CanonicalMessage]:
    """Assemble trusted framing, history, context, and the current request.

    Context is deliberately adjacent to, but before, the final real user
    request. This preserves the caller's instruction as the last user-level
    instruction and prevents the prompt builder from owning or duplicating it.
    """

    result = list(conversation)
    if envelope.context:
        insertion = next(
            (index for index in range(len(result) - 1, -1, -1) if isinstance(result[index], UserMessage)),
            None,
        )
        if insertion is None:
            raise ValueError("untrusted context requires a current user request")
        result.insert(insertion, UntrustedContextMessage(records=envelope.context))
    if envelope.trusted_instructions:
        result.insert(0, TrustedInstructionMessage(envelope.trusted_instructions))
    return result
