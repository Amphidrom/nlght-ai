# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from nlght.core.protocol.protocol import DetectedProtocol, ProtocolKind
from nlght.ports.outbound.protocol_detection import ProtocolDetector


class GenericJsonProtocolDetector(ProtocolDetector):
    async def detect(
        self,
        path: str,
        method: str,
        headers: dict[str, str],
        raw_body: bytes,
    ) -> DetectedProtocol:
        content_type = headers.get("content-type", "")

        if "application/json" in content_type and raw_body:
            return DetectedProtocol(
                kind=ProtocolKind.GENERIC_JSON,
                confidence=0.2,
                reason="Request looks like generic JSON.",
            )

        return DetectedProtocol(
            kind=ProtocolKind.UNKNOWN,
            confidence=0.0,
            reason="Generic JSON detector did not match.",
        )