# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import json
from typing import Any

from nlght.core.entry.context import RequestContext
from nlght.core.errors.errors import TriggerResolutionError
from nlght.core.protocol.protocol import DetectedProtocol, ProtocolKind
from nlght.core.trigger.trigger import Trigger, TriggerKind
from nlght.ports.outbound.trigger_resolution import TriggerResolver


class DefaultTriggerResolver(TriggerResolver):
    """Resolves incoming requests into triggers.

    The ``session_key`` is resolved by the caller (inbound adapter) and
    passed in directly — this resolver does not manage its own resolver instances.
    """

    async def resolve(
        self,
        protocol: DetectedProtocol,
        context: RequestContext,
        raw_body: bytes,
        session_key: str | None = None,
    ) -> Trigger:
        payload = _parse_json_or_empty(raw_body)

        if protocol.kind == ProtocolKind.OPENAI_MODELS:
            return Trigger(
                kind=TriggerKind.MODELS_DISCOVERY,
                protocol=protocol.kind,
                operation="list_models",
                payload={},
                context=context,
                metadata={"protocol_reason": protocol.reason},
            )

        if protocol.kind == ProtocolKind.OPENAI_CHAT_COMPLETIONS:
            return Trigger(
                kind=TriggerKind.MODEL_REQUEST,
                protocol=protocol.kind,
                operation="chat_completions",
                payload=payload,
                context=context,
                stream=bool(payload.get("stream", False)),
                session_key=session_key,
                metadata={"protocol_reason": protocol.reason},
            )

        if protocol.kind == ProtocolKind.OLLAMA_MODELS:
            return Trigger(
                kind=TriggerKind.MODELS_DISCOVERY,
                protocol=protocol.kind,
                operation="list_models",
                payload={},
                context=context,
                metadata={"protocol_reason": protocol.reason},
            )

        if protocol.kind == ProtocolKind.OLLAMA_CHAT:
            return Trigger(
                kind=TriggerKind.MODEL_REQUEST,
                protocol=protocol.kind,
                operation="chat",
                payload=payload,
                context=context,
                stream=bool(payload.get("stream", True)),
                session_key=session_key,
                metadata={"protocol_reason": protocol.reason},
            )

        if protocol.kind == ProtocolKind.GENERIC_JSON:
            return Trigger(
                kind=TriggerKind.HTTP_REQUEST,
                protocol=protocol.kind,
                operation="generic_json_request",
                payload=payload,
                context=context,
                session_key=session_key,
                metadata={"protocol_reason": protocol.reason},
            )

        raise TriggerResolutionError("Could not resolve trigger for detected protocol.")


def _parse_json_or_empty(raw_body: bytes) -> dict[str, Any]:
    if not raw_body:
        return {}

    try:
        parsed = json.loads(raw_body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise TriggerResolutionError("Invalid JSON body.") from exc

    if not isinstance(parsed, dict):
        raise TriggerResolutionError("JSON body must be an object.")

    return parsed
