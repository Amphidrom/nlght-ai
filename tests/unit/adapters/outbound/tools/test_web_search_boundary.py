# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""`web_search` is the only network operation the default policy allows.

It is allowed because its destination is fixed in operator configuration rather
than proposed by the model — `BOUNDED_DESTINATION` (ADR-0069). A followed
redirect would break exactly that: the effective destination would be one the
operator never named, and in langsearch mode the request carries an
`Authorization: Bearer` header that would go with it.
"""

from __future__ import annotations

from typing import Any

import httpx2
import pytest

from nlght.adapters.outbound.tools.builtin.web_search import WebSearchTool


class _Response:
    status_code = 200

    @staticmethod
    def json() -> dict[str, Any]:
        return {"results": []}


class _RecordingClient:
    kwargs: dict[str, object] = {}

    def __init__(self, **kwargs: object) -> None:
        _RecordingClient.kwargs = kwargs

    async def __aenter__(self) -> _RecordingClient:
        return self

    async def __aexit__(self, *_: object) -> None:
        return None

    async def get(self, *_: object, **__: object) -> _Response:
        return _Response()

    async def post(self, *_: object, **__: object) -> _Response:
        return _Response()


@pytest.mark.parametrize("mode", ["searxng", "langsearch"])
async def test_the_search_client_never_follows_a_redirect(monkeypatch, mode: str) -> None:  # noqa: ANN001
    monkeypatch.setattr(httpx2, "AsyncClient", _RecordingClient)
    tool = WebSearchTool(
        name="search",
        config={"endpoint": "https://search.example/api", "mode": mode, "api_key": "k"},
    )

    await tool._fetch("anything", 3)  # noqa: SLF001

    assert _RecordingClient.kwargs.get("follow_redirects") is False, (
        "a bounded-destination operation must not silently follow a redirect "
        "to a host the operator never configured"
    )
