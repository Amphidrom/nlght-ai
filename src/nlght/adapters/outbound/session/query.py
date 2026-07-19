# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from nlght.ports.outbound.session_key_resolver import SessionKeyResolver


class QueryParamSessionKeyResolver(SessionKeyResolver):
    """Liest den Session-Key aus einem konfigurierbaren Query-Parameter."""

    def __init__(self, parameter_name: str) -> None:
        self._parameter_name = parameter_name

    async def resolve(
        self,
        path: str,
        method: str,
        headers: dict[str, str],
        query_params: dict[str, str],
        raw_body: bytes,
    ) -> str | None:
        return query_params.get(self._parameter_name) or None
