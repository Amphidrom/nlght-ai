# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Keep web search within its configured network destination.

Redirects must not move a request or its credentials to a destination chosen
outside operator configuration (ADR-0069)."""

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
async def test_the_search_client_never_follows_a_redirect(monkeypatch, mode: str) -> None:
    monkeypatch.setattr(httpx2, "AsyncClient", _RecordingClient)
    tool = WebSearchTool(
        name="search",
        config={"endpoint": "https://search.example/api", "mode": mode, "api_key": "k"},
    )

    await tool._fetch("anything", 3)

    assert _RecordingClient.kwargs.get("follow_redirects") is False, (
        "a bounded-destination operation must not silently follow a redirect "
        "to a host the operator never configured"
    )
