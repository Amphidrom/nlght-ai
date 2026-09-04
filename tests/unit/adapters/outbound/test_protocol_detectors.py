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


def test_generic_json_detector_matches_bodyless_get() -> None:
    # A parameterless trigger — a reindex kicked off by hand — arrives as a GET
    # with no body and no content type. It must still resolve to a generic JSON
    # trigger rather than falling through to "unknown".
    detector = GenericJsonProtocolDetector()

    detected = asyncio.run(
        detector.detect(
            path="/hooks/reindex-docs",
            method="GET",
            headers={},
            raw_body=b"",
        )
    )

    assert detected.kind == ProtocolKind.GENERIC_JSON
    assert detected.confidence > 0


def test_generic_json_detector_never_reports_unknown() -> None:
    # It is the catch-all: whatever comes in, it claims (at low confidence).
    detector = GenericJsonProtocolDetector()

    detected = asyncio.run(
        detector.detect(path="/anything", method="DELETE", headers={}, raw_body=b"")
    )

    assert detected.kind == ProtocolKind.GENERIC_JSON


def test_composite_specific_detector_still_wins_over_bodyless_generic() -> None:
    # The generic detector now always matches, so this guards that a specific
    # protocol on its own endpoint is not shadowed by the catch-all.
    detector = CompositeProtocolDetector(
        [GenericJsonProtocolDetector(), OpenAIProtocolDetector()]
    )

    detected = asyncio.run(
        detector.detect(path="/v1/models", method="GET", headers={}, raw_body=b"")
    )

    assert detected.kind == ProtocolKind.OPENAI_MODELS
    assert detected.confidence == 1.0


def test_composite_falls_back_to_generic_for_bodyless_hook() -> None:
    detector = CompositeProtocolDetector(
        [GenericJsonProtocolDetector(), OpenAIProtocolDetector()]
    )

    detected = asyncio.run(
        detector.detect(
            path="/hooks/reindex-docs", method="GET", headers={}, raw_body=b""
        )
    )

    assert detected.kind == ProtocolKind.GENERIC_JSON


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
