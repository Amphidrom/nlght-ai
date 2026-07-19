# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from nlght.ports.outbound.session_key_resolver import SessionKeyResolver


class CompositeSessionKeyResolver(SessionKeyResolver):
    """Tries multiple resolvers in order; returns the first hit."""

    def __init__(self, resolvers: list[SessionKeyResolver]) -> None:
        self._resolvers = resolvers

    async def resolve(
        self,
        path: str,
        method: str,
        headers: dict[str, str],
        query_params: dict[str, str],
        raw_body: bytes,
    ) -> str | None:
        for resolver in self._resolvers:
            value = await resolver.resolve(
                path=path,
                method=method,
                headers=headers,
                query_params=query_params,
                raw_body=raw_body,
            )
            if value:
                return value
        return None
