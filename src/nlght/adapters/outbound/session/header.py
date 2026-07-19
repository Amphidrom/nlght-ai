# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from nlght.ports.outbound.session_key_resolver import SessionKeyResolver


class HeaderSessionKeyResolver(SessionKeyResolver):
    """Liest den Session-Key aus einem konfigurierbaren HTTP-Header.

    Konfiguration in platform.yaml::

        protocol_adapters:
          - kind: openai
            config:
              session_key_header: x-session-id
    """

    def __init__(self, header_name: str) -> None:
        self._header_name = header_name.lower()

    async def resolve(
        self,
        path: str,
        method: str,
        headers: dict[str, str],
        query_params: dict[str, str],
        raw_body: bytes,
    ) -> str | None:
        return headers.get(self._header_name) or None
