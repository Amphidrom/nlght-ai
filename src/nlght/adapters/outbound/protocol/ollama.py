# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from nlght.core.protocol.protocol import DetectedProtocol, ProtocolKind
from nlght.ports.outbound.protocol_detection import ProtocolDetector


class OllamaProtocolDetector(ProtocolDetector):
    """Detects Ollama native API requests (``/api/chat``, ``/api/tags``, etc.)."""

    async def detect(
        self,
        path: str,
        method: str,
        headers: dict[str, str],
        raw_body: bytes,
    ) -> DetectedProtocol:
        if method.upper() == "GET" and path in ("/api/tags", "/api/ps", "/api/version"):
            return DetectedProtocol(
                kind=ProtocolKind.OLLAMA_MODELS,
                confidence=1.0,
                reason=f"Matched GET {path}.",
            )

        if method.upper() == "POST" and path == "/api/chat":
            return DetectedProtocol(
                kind=ProtocolKind.OLLAMA_CHAT,
                confidence=1.0,
                reason="Matched POST /api/chat.",
            )

        return DetectedProtocol(
            kind=ProtocolKind.UNKNOWN,
            confidence=0.0,
            reason="Ollama detector did not match.",
        )
