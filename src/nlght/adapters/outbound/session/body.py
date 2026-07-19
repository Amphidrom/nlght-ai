# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from nlght.ports.outbound.session_key_resolver import SessionKeyResolver


class BodyParameterSessionKeyResolver(SessionKeyResolver):
    """Reads the session key from a configurable JSON body field.

    ``key`` supports dot notation for nested fields
    (e.g. ``"meta.session_id"`` reads ``payload["meta"]["session_id"]``).

    Returns ``None`` if:
    - ``content-type`` does not contain ``application/json``
    - the body is empty or not valid JSON
    - the key is missing or not a non-empty string
    """

    def __init__(self, key: str) -> None:
        self._key = key

    async def resolve(
        self,
        path: str,
        method: str,
        headers: dict[str, str],
        query_params: dict[str, str],
        raw_body: bytes,
    ) -> str | None:
        content_type = headers.get("content-type", "")
        if not raw_body or "application/json" not in content_type:
            return None

        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return None

        if not isinstance(payload, dict):
            return None

        value = _get_nested_value(payload, self._key)
        if isinstance(value, str) and value.strip():
            return value
        return None


def _get_nested_value(data: Mapping[str, Any], path: str) -> object:
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current
