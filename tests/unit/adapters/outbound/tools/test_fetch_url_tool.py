# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx2

from nlght.adapters.outbound.tools.builtin import fetch_url as fetch_url_module
from nlght.adapters.outbound.tools.builtin.fetch_url import FetchUrlTool


def _make_tool(config: dict | None = None) -> FetchUrlTool:
    return FetchUrlTool(name="fetch", config=config or {})


def test_fetch_url_signature_accepts_candidate_id() -> None:
    signature = FetchUrlTool.signatures()[0]
    param_names = [param.name for param in signature.parameters]
    url_param = next(param for param in signature.parameters if param.name == "url")

    assert signature.name == "fetch_url"
    assert "candidate_id" in param_names
    assert "url" in param_names
    assert url_param.required is False


async def test_fetch_url_rejects_invalid_url_even_with_candidate_id() -> None:
    tool = _make_tool()

    result = json.loads(await tool.fetch(url="not-a-url", candidate_id="cand_1"))

    assert result["url"] == "not-a-url"
    assert "Invalid URL" in result["error"]


def test_browser_headers_include_referer_when_supplied() -> None:
    headers = fetch_url_module._browser_headers(referer="https://example.com/")

    assert headers["Referer"] == "https://example.com/"
    assert headers["Accept"].startswith("text/html")
    assert "Accept-Encoding" not in headers


def test_html_to_text_fallback_strips_noise_and_unescapes() -> None:
    text = fetch_url_module._html_to_text_fallback(
        "<html><script>bad()</script><nav>menu</nav><p>Hello&nbsp;<b>world</b></p><br>Next</html>"
    )

    assert "bad" not in text
    assert "menu" not in text
    assert text == "Hello world Next"


def test_timeout_uses_configured_value() -> None:
    assert _make_tool({"timeout_s": 2.5})._timeout() == 2.5


async def test_fetch_returns_content_when_page_text_is_available(monkeypatch) -> None:
    async def _fake_fetch_page_text(client: httpx2.AsyncClient, url: str) -> str:
        return f"content from {url}"

    monkeypatch.setattr(fetch_url_module, "_fetch_page_text", _fake_fetch_page_text)

    result = json.loads(await _make_tool().fetch(url="https://example.com/a"))

    assert result == {"url": "https://example.com/a", "content": "content from https://example.com/a"}


async def test_fetch_returns_empty_error_when_page_has_no_content(monkeypatch) -> None:
    async def _fake_fetch_page_text(client: httpx2.AsyncClient, url: str) -> str:
        return ""

    monkeypatch.setattr(fetch_url_module, "_fetch_page_text", _fake_fetch_page_text)

    result = json.loads(await _make_tool().fetch(url="https://example.com/empty"))

    assert result["url"] == "https://example.com/empty"
    assert "No content retrieved" in result["error"]
    assert result["content"] == ""


async def test_fetch_returns_exception_as_error(monkeypatch) -> None:
    async def _fake_fetch_page_text(client: httpx2.AsyncClient, url: str) -> str:
        raise RuntimeError("network exploded")

    monkeypatch.setattr(fetch_url_module, "_fetch_page_text", _fake_fetch_page_text)

    result = json.loads(await _make_tool().fetch(url="https://example.com/fail"))

    assert result["url"] == "https://example.com/fail"
    assert result["error"] == "network exploded"
    assert result["content"] == ""


async def test_robots_policy_reports_allow_deny_and_unavailable(monkeypatch) -> None:
    class _RobotParser:
        allowed = True
        raises = False

        def set_url(self, url: str) -> None:
            self.url = url

        def read(self) -> None:
            if self.raises:
                raise OSError("robots unavailable")

        def can_fetch(self, user_agent: str, url: str) -> bool:
            assert user_agent
            assert self.url == "https://example.com/robots.txt"
            assert url == "https://example.com/page"
            return self.allowed

    parser = _RobotParser()
    monkeypatch.setattr(fetch_url_module.urllib.robotparser, "RobotFileParser", lambda: parser)
    assert await fetch_url_module._robots_allows_url("https://example.com/page") is True
    parser.allowed = False
    assert await fetch_url_module._robots_allows_url("https://example.com/page") is False
    parser.raises = True
    assert await fetch_url_module._robots_allows_url("https://example.com/page") is None


async def test_fetch_page_text_honours_robots_and_http_status(monkeypatch) -> None:
    client = SimpleNamespace(get=AsyncMock())

    async def denied(_: str) -> bool:
        return False

    monkeypatch.setattr(fetch_url_module, "_robots_allows_url", denied)
    assert await fetch_url_module._fetch_page_text(client, "https://example.com/private") == ""
    client.get.assert_not_awaited()

    async def allowed(_: str) -> bool:
        return True

    monkeypatch.setattr(fetch_url_module, "_robots_allows_url", allowed)
    client.get.return_value = SimpleNamespace(status_code=503, text="unavailable")
    assert await fetch_url_module._fetch_page_text(client, "https://example.com/down") == ""
    assert client.get.await_args.kwargs["follow_redirects"] is True
    assert client.get.await_args.kwargs["headers"]["Referer"] == "https://example.com/"


async def test_fetch_page_text_cleans_html_and_recovers_from_request_error(monkeypatch) -> None:
    async def allowed(_: str) -> bool:
        return True

    monkeypatch.setattr(fetch_url_module, "_robots_allows_url", allowed)
    client = SimpleNamespace(get=AsyncMock(return_value=SimpleNamespace(
        status_code=200,
        text="<html><nav>menu</nav><p>Hello <b>world</b></p><script>bad()</script></html>",
    )))
    assert await fetch_url_module._fetch_page_text(client, "https://example.com") == "Hello world"

    client.get.side_effect = RuntimeError("connection reset")
    assert await fetch_url_module._fetch_page_text(client, "https://example.com") == ""
