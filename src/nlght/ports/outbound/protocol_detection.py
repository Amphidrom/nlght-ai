# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from typing import Protocol

from nlght.core.protocol.protocol import DetectedProtocol


class ProtocolDetector(Protocol):
    async def detect(self, path: str, method: str, headers: dict[str, str], raw_body: bytes) -> DetectedProtocol:
        raise NotImplementedError