# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Tests for WebSearchTool — the built-in web search tool."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

from nlght.adapters.outbound.tools.builtin.web_search import (
    WebSearchTool,
    _clean,
    _domain,
    _parse_response,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_tool(config: dict | None = None) -> WebSearchTool:
    return WebSearchTool(
        name="ws",
        config=config or {"endpoint": "http://search.local", "mode": "searxng"},
    )


def _searxng_response(results: list[dict]) -> dict:
    return {"results": results}


def _mock_http(status: int = 200, data: dict | None = None) -> MagicMock:
    """Return a mock that behaves as ``async with httpx2.AsyncClient() as client``."""
    response = MagicMock()
    response.status_code = status
    response.json.return_value = data or {}

    client = AsyncMock()
    client.get = AsyncMock(return_value=response)
    client.post = AsyncMock(return_value=response)
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)

    return client


# ---------------------------------------------------------------------------
# KIND / signatures
# ---------------------------------------------------------------------------


def test_kind_is_web_search() -> None:
    assert WebSearchTool.KIND == "web_search"


def test_signatures_count() -> None:
    assert len(WebSearchTool.signatures()) == 1


def test_signature_names() -> None:
    names = {s.name for s in WebSearchTool.signatures()}
    assert names == {"web_search"}


def test_each_signature_has_matching_method() -> None:
    tool = _make_tool()
    for sig in WebSearchTool.signatures():
        assert hasattr(tool, sig.method_name)


# ---------------------------------------------------------------------------
# helper functions
# ---------------------------------------------------------------------------


def test_clean_collapses_whitespace() -> None:
    assert _clean("  hello   world  ") == "hello world"


def test_domain_extracts_hostname() -> None:
    assert _domain("https://example.com/path?q=1") == "example.com"


def test_domain_returns_empty_on_invalid_url() -> None:
    assert _domain("not-a-url") == ""



def test_parse_response_searxng() -> None:
    data = {"results": [{"title": "A", "url": "http://a.com", "content": "aaa"}]}
    items = _parse_response(data, "searxng", 10)
    assert len(items) == 1
    assert items[0]["title"] == "A"


def test_parse_response_langsearch() -> None:
    data = {"data": {"webPages": {"value": [{"name": "B", "url": "http://b.com", "summary": "bbb"}]}}}
    items = _parse_response(data, "langsearch", 10)
    assert items[0]["title"] == "B"
    assert items[0]["content"] == "bbb"


def test_parse_response_generic_fallback() -> None:
    data = {"results": [{"title": "C", "url": "http://c.com", "content": "ccc"}]}
    items = _parse_response(data, "unknown_mode", 10)
    assert items[0]["title"] == "C"


# ---------------------------------------------------------------------------
# no endpoint → error JSON
# ---------------------------------------------------------------------------


async def test_search_returns_error_when_no_endpoint() -> None:
    tool = WebSearchTool(name="ws", config={})
    result = json.loads(await tool.search(query="test"))
    assert result["result_count"] == 0
    assert "error" in result


# ---------------------------------------------------------------------------
# search — happy path
# ---------------------------------------------------------------------------


async def test_search_returns_results_json() -> None:
    api_data = _searxng_response([
        {"title": "Foo", "url": "http://foo.com", "content": "foo content"},
        {"title": "Bar", "url": "http://bar.com", "content": "bar content"},
    ])
    mock_client = _mock_http(data=api_data)

    tool = _make_tool()
    with patch("nlght.adapters.outbound.tools.builtin.web_search.httpx2.AsyncClient", return_value=mock_client):
        result = json.loads(await tool.search(query="foo bar"))

    assert result["query"] == "foo bar"
    assert result["result_count"] >= 1
    assert "results" in result


async def test_search_result_has_expected_fields() -> None:
    api_data = _searxng_response([{"title": "X", "url": "http://x.com", "content": "xcontent"}])
    mock_client = _mock_http(data=api_data)

    tool = _make_tool()
    with patch("nlght.adapters.outbound.tools.builtin.web_search.httpx2.AsyncClient", return_value=mock_client):
        result = json.loads(await tool.search(query="x"))

    if result["result_count"] > 0:
        r = result["results"][0]
        assert "rank" in r
        assert r["kind"] == "source_candidate"
        assert "title" in r
        assert "url" in r
        assert "content" in r
        assert "domain" in r
        assert r["query"] == "x"
        assert "confidence" in r


async def test_search_returns_error_on_http_failure() -> None:
    mock_client = _mock_http(status=500)
    tool = _make_tool()

    with patch("nlght.adapters.outbound.tools.builtin.web_search.httpx2.AsyncClient", return_value=mock_client):
        result = json.loads(await tool.search(query="broken"))

    assert result["result_count"] == 0
    assert "error" in result


async def test_search_returns_error_on_network_exception() -> None:
    client = AsyncMock()
    client.__aenter__ = AsyncMock(return_value=client)
    client.__aexit__ = AsyncMock(return_value=None)
    client.get = AsyncMock(side_effect=Exception("network down"))
    tool = _make_tool()

    with patch("nlght.adapters.outbound.tools.builtin.web_search.httpx2.AsyncClient", return_value=client):
        result = json.loads(await tool.search(query="broken"))

    assert result["result_count"] == 0


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# langsearch mode
# ---------------------------------------------------------------------------


async def test_langsearch_mode_uses_post() -> None:
    api_data = {"data": {"webPages": {"value": []}}}
    mock_client = _mock_http(data=api_data)

    tool = WebSearchTool(
        name="ws",
        config={"endpoint": "http://lang.local/search", "mode": "langsearch", "api_key": "secret"},
    )
    with patch("nlght.adapters.outbound.tools.builtin.web_search.httpx2.AsyncClient", return_value=mock_client):
        await tool.search(query="python")

    mock_client.post.assert_awaited()


async def test_langsearch_sends_auth_header() -> None:
    api_data = {"data": {"webPages": {"value": []}}}
    mock_client = _mock_http(data=api_data)

    tool = WebSearchTool(
        name="ws",
        config={"endpoint": "http://lang.local/search", "mode": "langsearch", "api_key": "tok123"},
    )
    with patch("nlght.adapters.outbound.tools.builtin.web_search.httpx2.AsyncClient", return_value=mock_client):
        await tool.search(query="x")

    _, kwargs = mock_client.post.call_args
    assert kwargs["headers"].get("Authorization") == "Bearer tok123"


# ---------------------------------------------------------------------------
# top_k respected
# ---------------------------------------------------------------------------


async def test_search_top_k_limits_results() -> None:
    api_data = _searxng_response([
        {"title": f"R{i}", "url": f"http://r{i}.com", "content": "content"}
        for i in range(10)
    ])
    mock_client = _mock_http(data=api_data)

    tool = _make_tool()
    with patch("nlght.adapters.outbound.tools.builtin.web_search.httpx2.AsyncClient", return_value=mock_client):
        result = json.loads(await tool.search(query="test", top_k=3))

    assert result["result_count"] <= 3
