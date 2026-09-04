# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import asyncio
import html
import json
import logging
import random
import re
import urllib.robotparser
from typing import ClassVar
from urllib.parse import urlparse

import httpx2

from nlght.adapters.outbound.tools.builtin.action_semantics import ARBITRARY_NETWORK_READ
from nlght.core.tools.tool import ToolBase, ToolParameter, ToolSignature

logger = logging.getLogger(__name__)

_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0",
]

_ACCEPT_LANGUAGES = [
    "en-US,en;q=0.9",
    "en-GB,en;q=0.9,en-US;q=0.8",
    "en-US,en;q=0.9,de;q=0.8",
    "de-DE,de;q=0.9,en;q=0.8",
    "fr-FR,fr;q=0.9,en;q=0.8",
]


def _browser_headers(referer: str | None = None) -> dict[str, str]:
    # Accept-Encoding is intentionally omitted — httpx manages decompression
    # automatically and only advertises encodings it can actually decode.
    # Manually setting it (e.g. "br") causes garbage output when brotli is not installed.
    headers: dict[str, str] = {
        "User-Agent": random.choice(_USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": random.choice(_ACCEPT_LANGUAGES),
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
    }
    if referer:
        headers["Referer"] = referer
    return headers


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _html_to_text_fallback(markup: str) -> str:
    text = str(markup or "")
    text = re.sub(r"(?is)<(script|style|noscript|header|footer|nav|aside)\b.*?>.*?</\1>", " ", text)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</p\s*>", "\n", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    return _clean(html.unescape(text))


async def _robots_allows_url(url: str) -> bool | None:
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    ua = random.choice(_USER_AGENTS)
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(robots_url)
    try:
        await asyncio.get_event_loop().run_in_executor(None, rp.read)
    except Exception:
        logger.debug("fetch_url.robots_unavailable url=%s", url, exc_info=True)
        return None
    try:
        return bool(rp.can_fetch(ua, url))
    except Exception:
        logger.debug("fetch_url.robots_check_failed url=%s", url, exc_info=True)
        return None


async def _fetch_page_text(client: httpx2.AsyncClient, url: str) -> str:
    try:
        from bs4 import BeautifulSoup  # noqa: PLC0415 (lazy import: optional extra or deliberate startup-cost/cycle avoidance)
    except ImportError:
        BeautifulSoup = None

    try:
        robots_allowed = await _robots_allows_url(url)
        if robots_allowed is False:
            return ""

        parsed = urlparse(url)
        referer = f"{parsed.scheme}://{parsed.netloc}/"
        r = await client.get(
            url,
            headers=_browser_headers(referer=referer),
            follow_redirects=True,
        )

        if r.status_code != 200:
            return ""

        if BeautifulSoup is None:
            return _html_to_text_fallback(r.text)[:20_000]

        soup = BeautifulSoup(r.text, "html.parser")

        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
            tag.decompose()

        text = _clean(soup.get_text(separator=" "))
        if text:
            return text[:20_000]
        return _html_to_text_fallback(r.text)[:20_000]

    except Exception:
        logger.debug("fetch_url.page_fetch_failed url=%s", url, exc_info=True)
        return ""


class FetchUrlTool(ToolBase):
    """Built-in tool for fetching the full content of a specific URL.

    Use when the LLM already has a concrete URL and needs the full page text,
    rather than running a search query.

    Configuration:
      ``timeout_s`` — HTTP timeout in seconds (default ``15``)
    """

    KIND: ClassVar[str]     = "fetch_url"
    PROVIDER: ClassVar[str] = "http"

    @classmethod
    def signatures(cls) -> list[ToolSignature]:
        return [
            ToolSignature(
                name="fetch_url",
                description=(
                    "Fetch the full text content of a specific URL. "
                    "Use when you have a concrete URL and need the complete page — "
                    "not for discovery; use web_search for that."
                ),
                method_name="fetch",
                parameters=[
                    ToolParameter(
                        name="candidate_id",
                        type="string",
                        required=False,
                        description="Optional stable source-candidate reference from the act runtime.",
                    ),
                    ToolParameter(
                        name="url",
                        type="string",
                        required=False,
                        description="The full URL to fetch (must start with http:// or https://).",
                    ),
                ],
                action=ARBITRARY_NETWORK_READ,
            )
        ]

    def _timeout(self) -> float:
        return float(self.config.get("timeout_s", 15))

    async def fetch(self, *, url: str = "", candidate_id: str | None = None) -> str:
        url = url.strip()
        if not url.startswith(("http://", "https://")):
            return json.dumps({"url": url, "error": "Invalid URL — must start with http:// or https://", "content": ""})

        try:
            async with httpx2.AsyncClient(timeout=self._timeout()) as client:
                content = await _fetch_page_text(client, url)
        except Exception as exc:
            logger.warning("fetch_url.failed | url=%s error=%s", url, exc)
            return json.dumps({"url": url, "error": str(exc), "content": ""})

        if not content:
            return json.dumps({"url": url, "error": "No content retrieved (blocked or empty page)", "content": ""})

        logger.info("fetch_url.ok | url=%s chars=%d", url, len(content))
        return json.dumps({"url": url, "content": content})
