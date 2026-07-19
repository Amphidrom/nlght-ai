# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

import asyncio
from datetime import datetime

import pytest

from nlght.adapters.outbound.trigger.default import DefaultTriggerResolver, _parse_json_or_empty
from nlght.core.entry.context import RequestContext
from nlght.core.errors.errors import TriggerResolutionError
from nlght.core.protocol.protocol import DetectedProtocol, ProtocolKind
from nlght.core.trigger.trigger import TriggerKind


@pytest.fixture
def request_context() -> RequestContext:
    return RequestContext(
        correlation_id="cid",
        request_id="rid",
        received_at=datetime(2026, 1, 1),
        path="/path",
        method="POST",
        headers={},
        query_params={},
        client_host="127.0.0.1",
    )


def test_resolve_returns_models_discovery_trigger(request_context: RequestContext) -> None:
    resolver = DefaultTriggerResolver()

    trigger = asyncio.run(
        resolver.resolve(
            protocol=DetectedProtocol(
                kind=ProtocolKind.OPENAI_MODELS,
                confidence=1.0,
                reason="matched",
                version="v1",
            ),
            context=request_context,
            raw_body=b"",
        )
    )

    assert trigger.kind == TriggerKind.MODELS_DISCOVERY
    assert trigger.operation == "list_models"
    assert trigger.payload == {}


def test_resolve_returns_model_request_trigger(request_context: RequestContext) -> None:
    resolver = DefaultTriggerResolver()

    trigger = asyncio.run(
        resolver.resolve(
            protocol=DetectedProtocol(
                kind=ProtocolKind.OPENAI_CHAT_COMPLETIONS,
                confidence=1.0,
                reason="matched",
                version="v1",
            ),
            context=request_context,
            raw_body=b'{"messages":[]}',
        )
    )

    assert trigger.kind == TriggerKind.MODEL_REQUEST
    assert trigger.operation == "chat_completions"
    assert trigger.payload == {"messages": []}
    assert trigger.stream is False


def test_resolve_sets_stream_true_when_requested(request_context: RequestContext) -> None:
    resolver = DefaultTriggerResolver()

    trigger = asyncio.run(
        resolver.resolve(
            protocol=DetectedProtocol(
                kind=ProtocolKind.OPENAI_CHAT_COMPLETIONS,
                confidence=1.0,
                reason="matched",
                version="v1",
            ),
            context=request_context,
            raw_body=b'{"messages":[], "stream": true}',
        )
    )

    assert trigger.stream is True


def test_resolve_models_discovery_never_streams(request_context: RequestContext) -> None:
    resolver = DefaultTriggerResolver()

    trigger = asyncio.run(
        resolver.resolve(
            protocol=DetectedProtocol(
                kind=ProtocolKind.OPENAI_MODELS,
                confidence=1.0,
                reason="matched",
                version="v1",
            ),
            context=request_context,
            raw_body=b"",
        )
    )

    assert trigger.stream is False


def test_resolve_raises_for_unknown_protocol(request_context: RequestContext) -> None:
    resolver = DefaultTriggerResolver()

    with pytest.raises(TriggerResolutionError):
        asyncio.run(
            resolver.resolve(
                protocol=DetectedProtocol(
                    kind=ProtocolKind.UNKNOWN,
                    confidence=0.0,
                    reason="none",
                ),
                context=request_context,
                raw_body=b"",
            )
        )


def test_parse_json_or_empty_raises_for_non_object_json() -> None:
    with pytest.raises(TriggerResolutionError):
        _parse_json_or_empty(b'["not-an-object"]')
