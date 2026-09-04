# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from nlght.core.protocol.protocol import DetectedProtocol, ProtocolKind
from nlght.ports.outbound.protocol_detection import ProtocolDetector


class GenericJsonProtocolDetector(ProtocolDetector):
    """The catch-all detector: any HTTP request maps to a generic JSON trigger.

    It puts no requirement on method, body, or content type. A workflow mapped
    to a path must be reachable by a plain GET or a bodyless POST, because many
    such workflows are parameterless — a reindex, a rebuild, a job kicked off by
    hand. The name describes the *response* shape, not a demand on the request:
    a body, if one is sent, is still expected to be JSON, but that is the
    trigger resolver's contract, not a precondition for detection here.

    Kept at a low confidence so any specific detector (OpenAI, Ollama), which
    reports ``1.0`` on its own endpoints, still wins in the composite.
    """

    async def detect(
        self,
        path: str,
        method: str,
        headers: dict[str, str],
        raw_body: bytes,
    ) -> DetectedProtocol:
        return DetectedProtocol(
            kind=ProtocolKind.GENERIC_JSON,
            confidence=0.2,
            reason="Generic catch-all: any HTTP request maps to a generic JSON trigger.",
        )