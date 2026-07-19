# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Tests for OllamaProtocolDetector."""
from __future__ import annotations

from nlght.adapters.outbound.protocol.ollama import OllamaProtocolDetector
from nlght.core.protocol.protocol import ProtocolKind


async def _detect(path: str, method: str = "GET") -> ProtocolKind:
    d = OllamaProtocolDetector()
    result = await d.detect(path=path, method=method, headers={}, raw_body=b"")
    return result.kind


async def test_get_api_tags_detected() -> None:
    assert await _detect("/api/tags", "GET") == ProtocolKind.OLLAMA_MODELS


async def test_get_api_ps_detected() -> None:
    assert await _detect("/api/ps", "GET") == ProtocolKind.OLLAMA_MODELS


async def test_get_api_version_detected() -> None:
    assert await _detect("/api/version", "GET") == ProtocolKind.OLLAMA_MODELS


async def test_post_api_chat_detected() -> None:
    assert await _detect("/api/chat", "POST") == ProtocolKind.OLLAMA_CHAT


async def test_unknown_path_returns_unknown() -> None:
    assert await _detect("/other", "GET") == ProtocolKind.UNKNOWN


async def test_post_to_api_tags_returns_unknown() -> None:
    assert await _detect("/api/tags", "POST") == ProtocolKind.UNKNOWN


async def test_confidence_on_match() -> None:
    d = OllamaProtocolDetector()
    result = await d.detect(path="/api/chat", method="POST", headers={}, raw_body=b"")
    assert result.confidence == 1.0
