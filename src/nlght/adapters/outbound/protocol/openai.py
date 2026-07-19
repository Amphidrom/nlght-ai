# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from nlght.core.protocol.protocol import DetectedProtocol, ProtocolKind
from nlght.ports.outbound.protocol_detection import ProtocolDetector


class OpenAIProtocolDetector(ProtocolDetector):
    def __init__(self, base_path: str = "/v1") -> None:
        self._base_path = base_path.rstrip("/")

    async def detect(
        self,
        path: str,
        method: str,
        headers: dict[str, str],
        raw_body: bytes,
    ) -> DetectedProtocol:
        content_type = headers.get("content-type", "")
        models_path = f"{self._base_path}/models"
        chat_completions_path = f"{self._base_path}/chat/completions"

        if method.upper() == "GET" and path == models_path:
            return DetectedProtocol(
                kind=ProtocolKind.OPENAI_MODELS,
                confidence=1.0,
                reason=f"Matched GET {models_path}.",
                version="v1",
            )

        if (
            method.upper() == "POST"
            and path == chat_completions_path
            and "application/json" in content_type
        ):
            return DetectedProtocol(
                kind=ProtocolKind.OPENAI_CHAT_COMPLETIONS,
                confidence=1.0,
                reason=f"Matched POST {chat_completions_path} with JSON body.",
                version="v1",
            )

        return DetectedProtocol(
            kind=ProtocolKind.UNKNOWN,
            confidence=0.0,
            reason="OpenAI detector did not match.",
        )