# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

import asyncio

from nlght.adapters.outbound.protocol.composite import CompositeProtocolDetector
from nlght.adapters.outbound.protocol.generic_json import GenericJsonProtocolDetector
from nlght.adapters.outbound.protocol.openai import OpenAIProtocolDetector
from nlght.core.protocol.protocol import ProtocolKind


def test_openai_detector_detects_models_endpoint() -> None:
    detector = OpenAIProtocolDetector(base_path="/v1")

    detected = asyncio.run(
        detector.detect(
            path="/v1/models",
            method="GET",
            headers={},
            raw_body=b"",
        )
    )

    assert detected.kind == ProtocolKind.OPENAI_MODELS
    assert detected.confidence == 1.0


def test_openai_detector_detects_chat_completions_endpoint() -> None:
    detector = OpenAIProtocolDetector(base_path="/v1")

    detected = asyncio.run(
        detector.detect(
            path="/v1/chat/completions",
            method="POST",
            headers={"content-type": "application/json"},
            raw_body=b"{}",
        )
    )

    assert detected.kind == ProtocolKind.OPENAI_CHAT_COMPLETIONS


def test_generic_json_detector_returns_generic_kind_for_json_with_body() -> None:
    detector = GenericJsonProtocolDetector()

    detected = asyncio.run(
        detector.detect(
            path="/any",
            method="POST",
            headers={"content-type": "application/json"},
            raw_body=b'{"hello":"world"}',
        )
    )

    assert detected.kind == ProtocolKind.GENERIC_JSON
    assert detected.confidence > 0


def test_composite_protocol_detector_picks_highest_confidence() -> None:
    detector = CompositeProtocolDetector([GenericJsonProtocolDetector(), OpenAIProtocolDetector()])

    detected = asyncio.run(
        detector.detect(
            path="/v1/chat/completions",
            method="POST",
            headers={"content-type": "application/json"},
            raw_body=b'{"messages":[]}',
        )
    )

    assert detected.kind == ProtocolKind.OPENAI_CHAT_COMPLETIONS
    assert detected.confidence == 1.0
