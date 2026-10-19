# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Define provider-neutral messages and explicit content authority."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, TypeAlias

from nlght.core.model.trust import (
    PLATFORM_TEXT,
    SOURCE_CALLER,
    SOURCE_CALLER_ATTACHMENT,
    SOURCE_CALLER_HISTORY,
    UNANNOTATED,
    ContentOrigin,
    ContentTrust,
    SourceRef,
)

if TYPE_CHECKING:
    from nlght.core.tools.tool import ToolOutput


def caller_origin(principal_id: str) -> ContentOrigin:
    """Attribute caller content without granting it authority.

    Args:
        principal_id (str): Caller principal ID, or an empty ID for anonymous callers.

    Returns:
        ContentOrigin: Untrusted caller origin.
    """
    return ContentOrigin.of(SourceRef(
        SOURCE_CALLER, principal_id or "anonymous", "", ContentTrust.UNTRUSTED,
    ))


def _caller_history_origin(principal_id: str) -> ContentOrigin:
    """Attribute unverified replayed history to a caller.

    Args:
        principal_id (str): Caller principal ID.

    Returns:
        ContentOrigin: Untrusted caller-history origin.
    """
    return ContentOrigin.of(SourceRef(
        SOURCE_CALLER_HISTORY, principal_id or "anonymous", "", ContentTrust.UNTRUSTED,
    ))


#: Separate attached content from caller text because foreign text cannot inherit
#: the caller's own origin or raise its trust.
ATTACHMENT_MARKER = "nlght_attachment"


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
    ATTACHMENT = "attachment"
    OTHER = "other"

    @classmethod
    def _missing_(cls, value: object) -> ContextKind:
        """Preserve extension labels as typed context kinds.

        Args:
            value (object): An unregistered kind value.

        Returns:
            ContextKind: A string-backed ContextKind member.
        """

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
    #: Keep provenance typed until serialization so providers cannot treat
    #: source labels as instructions (ADR-0074).
    origin: ContentOrigin = UNANNOTATED


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
    """Preserve trusted instructions and context as separate fields.

    Args:
        trusted_instructions (str): Instructions from a trusted caller.
        context (Sequence[ContextRecord]): Context records with their own origins.

    Returns:
        PromptEnvelope: Envelope that keeps instructions and context separate.
    """

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
    #: Reject authority claimed by the request. Only integration code can
    #: assign a different origin (ADR-0074).
    origin: ContentOrigin = field(default_factory=lambda: caller_origin(""))


@dataclass(frozen=True, slots=True)
class UserMessage:
    content: Any
    attributes: Mapping[str, Any] = field(default_factory=dict)
    #: Keep the origin explicit because a user message can carry foreign text
    #: whose trust the inbound integration must establish (ADR-0074).
    origin: ContentOrigin = field(default_factory=lambda: caller_origin(""))


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
    #: Retain read provenance through model output for later action decisions.
    origin: ContentOrigin = UNANNOTATED


@dataclass(frozen=True, slots=True)
class ToolResultMessage:
    content: Any
    tool_call_id: str = ""
    name: str = ""
    original_role: str = "tool"
    attributes: Mapping[str, Any] = field(default_factory=dict)
    #: Preserve tool-reported origin. Bare strings cannot establish provenance.
    origin: ContentOrigin = UNANNOTATED


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
    *,
    caller: str = "",
) -> list[CanonicalMessage]:
    """Normalize inbound messages without accepting claimed authority.

    Args:
        messages (Sequence[MessageLike]): Canonical or raw inbound messages.
        caller (str): Principal ID for raw caller content, if available.

    Returns:
        list[CanonicalMessage]: Canonical messages with untrusted origin for raw caller content.

    Raises:
        TypeError: A message has an unsupported shape. ValueError: A role is unsupported.
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
                origin=caller_origin(caller),
            ))
        elif role == "user":
            content, attachments = _split_attachments(content, caller, len(normalized))
            if attachments:
                normalized.append(UntrustedContextMessage(records=attachments))
            normalized.append(UserMessage(
                content=content, attributes=raw, origin=caller_origin(caller),
            ))
        elif role == "assistant":
            tool_calls = tuple(
                _normalize_tool_call(call)
                for call in (raw.pop("tool_calls", ()) or ())
            )
            normalized.append(AssistantMessage(
                content=content,
                tool_calls=tool_calls,
                attributes=raw,
                origin=_caller_history_origin(caller),
            ))
        elif role in {"tool", "function"}:
            normalized.append(ToolResultMessage(
                content=content,
                tool_call_id=str(raw.pop("tool_call_id", "")),
                name=str(raw.pop("name", "")),
                original_role=role,
                attributes=raw,
                origin=_caller_history_origin(caller),
            ))
        else:
            raise ValueError(f"unsupported workflow message role '{role}'")
    return normalized


def _split_attachments(
    content: object,
    caller: str,
    position: int,
) -> tuple[object, tuple[ContextRecord, ...]]:
    """Isolate attached foreign content from caller text.

    Args:
        content (object): User-message content to inspect.
        caller (str): Caller identity attached to discovered content.
        position (int): Position of the message in the conversation.

    Returns:
        tuple[object, tuple[ContextRecord, ...]]: Remaining user content and typed
            untrusted attachment records.
    """
    if not isinstance(content, list):
        return content, ()
    kept: list[Any] = []
    records: list[ContextRecord] = []
    for index, part in enumerate(content):
        if not isinstance(part, Mapping) or ATTACHMENT_MARKER not in part:
            kept.append(part)
            continue
        marker = part.get(ATTACHMENT_MARKER)
        label = str(marker.get("name", "")) if isinstance(marker, Mapping) else ""
        text = part.get("text")
        body = text if isinstance(text, str) else json.dumps(
            {key: value for key, value in part.items() if key != ATTACHMENT_MARKER},
            ensure_ascii=False, default=str,
        )
        record_id = f"attachment:{position}:{index}"
        records.append(ContextRecord(
            record_id=record_id,
            kind=ContextKind.ATTACHMENT,
            content=body,
            provenance=ContextProvenance(
                source=SOURCE_CALLER_ATTACHMENT,
                item_id=record_id,
                source_name=label,
                origin=ContentOrigin.of(SourceRef(
                    SOURCE_CALLER_ATTACHMENT,
                    caller or "anonymous",
                    label,
                    ContentTrust.UNTRUSTED,
                    label,
                )),
            ),
            section="Attached by the caller",
        ))
    return kept, tuple(records)


def message_origin(message: CanonicalMessage) -> ContentOrigin | None:
    """Read a message's contribution to model-context origin.

    Args:
        message (CanonicalMessage): A canonical message.

    Returns:
        ContentOrigin | None: Combined origin, or None for trusted instructions.
    """
    if isinstance(message, TrustedInstructionMessage):
        return None
    if isinstance(message, UntrustedContextMessage):
        return ContentOrigin.combine(*(record.provenance.origin for record in message.records))
    return message.origin


def origin_of(messages: Sequence[MessageLike], *, caller: str = "") -> ContentOrigin:
    """Combine origins of all content in a model context.

    Args:
        messages (Sequence[MessageLike]): Messages whose content origin is combined.
        caller (str): Principal ID for raw messages, if available.

    Returns:
        ContentOrigin: Combined content origin, empty when no content was read.
    """
    origins = [
        origin
        for message in normalize_workflow_messages(messages, caller=caller)
        if (origin := message_origin(message)) is not None
    ]
    return ContentOrigin.combine(*origins) if origins else ContentOrigin()


def append_canonical_tool_turn(
    messages: Sequence[MessageLike],
    tool_calls_raw: Sequence[Mapping[str, Any]],
    results: Sequence[str | ToolOutput],
    assistant_text: str = "",
) -> list[CanonicalMessage]:
    """Append typed tool calls and results to a conversation.

    Args:
        messages (Sequence[MessageLike]): Existing conversation messages.
        tool_calls_raw (Sequence[Mapping[str, Any]]): Model tool calls in raw or typed form.
        results (Sequence[str | ToolOutput]): Tool results, including any explicit origin labels.
        assistant_text (str): Assistant text accompanying the tool calls.

    Returns:
        list[CanonicalMessage]: Conversation with typed tool calls and results appended.

    Raises:
        TypeError: A tool call has an unsupported shape.
    """

    current = normalize_workflow_messages(messages)
    # Attribute an answer to platform text when no untrusted source was read.
    read_so_far = origin_of(current)
    if not read_so_far.sources:
        read_so_far = PLATFORM_TEXT
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
    current.append(AssistantMessage(
        content=assistant_text, tool_calls=calls, origin=read_so_far,
    ))
    for call, result in zip(calls, results, strict=False):
        if isinstance(result, str):
            content, origin = result, UNANNOTATED
        else:
            content, origin = result.content, result.origin
        current.append(ToolResultMessage(
            content=content,
            tool_call_id=call.call_id,
            name=call.name,
            origin=origin,
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
    """Assemble instructions, context, and conversation in authority order.

    Args:
        envelope (PromptEnvelope): Trusted instructions and labelled context.
        conversation (Sequence[CanonicalMessage]): Canonical conversation to extend.

    Returns:
        list[CanonicalMessage]: Messages with context inserted before the latest user request.

    Raises:
        ValueError: Untrusted context has no current user request.
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
