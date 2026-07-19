# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from typing import Protocol


class SessionKeyResolver(Protocol):
    async def resolve(
        self,
        path: str,
        method: str,
        headers: dict[str, str],
        query_params: dict[str, str],
        raw_body: bytes,
    ) -> str | None:
        raise NotImplementedError
