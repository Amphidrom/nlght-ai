# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from nlght.core.protocol.protocol import DetectedProtocol, ProtocolKind
from nlght.ports.outbound.protocol_detection import ProtocolDetector


class CompositeProtocolDetector(ProtocolDetector):
    def __init__(self, detectors: list[ProtocolDetector]) -> None:
        self._detectors = detectors

    async def detect(
        self,
        path: str,
        method: str,
        headers: dict[str, str],
        raw_body: bytes,
    ) -> DetectedProtocol:
        best = DetectedProtocol(
            kind=ProtocolKind.UNKNOWN,
            confidence=0.0,
            reason="No detector matched.",
        )

        for detector in self._detectors:
            candidate = await detector.detect(
                path=path,
                method=method,
                headers=headers,
                raw_body=raw_body,
            )
            if candidate.confidence > best.confidence:
                best = candidate

        return best