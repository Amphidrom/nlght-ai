# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from typing import Any, Protocol


class ModelGateway(Protocol):
    async def complete(self, *, model: str, prompt: str, options: dict[str, Any] | None = None) -> str:
        raise NotImplementedError
