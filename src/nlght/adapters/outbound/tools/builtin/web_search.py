# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import json
import logging
import re
from typing import Any, ClassVar
from urllib.parse import urlparse

import httpx2

from nlght.adapters.outbound.tools.builtin.action_semantics import CONFIGURED_NETWORK_READ
from nlght.core.tools.tool import ToolBase, ToolParameter, ToolSignature

logger = logging.getLogger(__name__)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _domain(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except Exception:
        return ""



def _parse_response(data: dict[str, Any], mode: str, limit: int) -> list[dict[str, Any]]:
    if mode == "searxng":
        raw = data.get("results", [])[:limit]
        return [
            {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", ""), "rank": i}
            for i, r in enumerate(raw)
        ]
    if mode == "langsearch":
        raw = data.get("data", {}).get("webPages", {}).get("value", [])[:limit]
        return [
            {"title": r.get("name", ""), "url": r.get("url", ""), "content": r.get("summary", ""), "rank": i}
            for i, r in enumerate(raw)
        ]
    # generic fallback
    raw = data.get("results", [])[:limit]
    return [
        {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", ""), "rank": i}
        for i, r in enumerate(raw)
    ]


def _source_candidate_entry(query: str, index: int, item: dict[str, Any]) -> dict[str, Any]:
    url = str(item.get("url", "")).strip()
    return {
        "kind": "source_candidate",
        "rank": index,
        "title": _clean(str(item.get("title", ""))),
        "url": url,
        "content": _clean(str(item.get("content", ""))),
        "domain": _domain(url),
        "query": query,
        "confidence": round(1 / (1 + index), 3),
    }


class WebSearchTool(ToolBase):
    """Built-in tool for searching the web.

    Supports SearXNG and LangSearch backends. Returns structured source-candidate
    objects directly from the search API — no page enrichment.

    Configuration:
      ``endpoint``  — search API URL (required)
      ``api_key``   — API key if required by the backend (optional)
      ``timeout_s`` — HTTP timeout in seconds (default ``10``)
      ``mode``      — ``searxng`` (default) or ``langsearch``
    """

    KIND: ClassVar[str]     = "web_search"
    PROVIDER: ClassVar[str] = "http"

    @classmethod
    def signatures(cls) -> list[ToolSignature]:
        params = [
            ToolParameter(name="query", type="string", description="Search query — natural language or keywords."),
            ToolParameter(name="top_k", type="number", description="Number of results to return.", required=False, default=5),
        ]
        return [
            ToolSignature(
                name="web_search",
                description=(
                    "Search the web for external information and documentation. "
                    "Returns source-candidate objects as a single JSON artifact."
                ),
                method_name="search",
                parameters=params,
                action=CONFIGURED_NETWORK_READ,
            ),
        ]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _fetch(self, query: str, top_k: int) -> dict[str, Any] | None:
        endpoint: str | None = self.config.get("endpoint")
        if not endpoint:
            return None

        api_key: str | None = self.config.get("api_key")
        timeout: float = float(self.config.get("timeout_s", 10))
        mode: str = str(self.config.get("mode", "searxng"))
        engines: str = str(self.config.get("engines", "google,bing,duckduckgo"))

        try:
            # `follow_redirects=False` is the whole reason this operation may be
            # classified BOUNDED_DESTINATION (ADR-0069): the destination is fixed
            # in operator config, and a 302 would move it somewhere the operator
            # never named. In langsearch mode the request carries an
            # `Authorization: Bearer` header, so a followed redirect would hand
            # the key to whatever host the redirect chose. Stated rather than
            # inherited from the client default, because a later "our endpoint
            # redirects now" fix must be a deliberate decision about the boundary.
            async with httpx2.AsyncClient(
                timeout=timeout, follow_redirects=False,
            ) as client:
                if mode == "searxng":
                    r = await client.get(
                        endpoint,
                        params={"q": query, "format": "json", "engines": engines, "pageno": 1},
                        headers={"Accept": "application/json"},
                    )
                elif mode == "langsearch":
                    headers: dict[str, str] = {"Content-Type": "application/json"}
                    if api_key:
                        headers["Authorization"] = f"Bearer {api_key}"
                    r = await client.post(endpoint, json={"query": query, "count": top_k}, headers=headers)
                else:
                    r = await client.get(endpoint, params={"q": query, "format": "json"})

                if r.status_code != 200:
                    logger.warning("web_search.http_error status=%s", r.status_code)
                    return None
                result = r.json()
                return result if isinstance(result, dict) else None
        except Exception:
            logger.exception("web_search.fetch_failed query=%s", query)
            return None

    def _error_json(self, query: str, reason: str) -> str:
        return json.dumps({"query": query, "result_count": 0, "error": reason, "results": []})

    def _check_endpoint(self, query: str) -> str | None:
        if not self.config.get("endpoint"):
            return self._error_json(query, "No endpoint configured")
        return None

    async def _resolve(self, query: str, top_k: int) -> list[dict[str, Any]] | None:
        mode = str(self.config.get("mode", "searxng"))
        data = await self._fetch(query, top_k)
        if data is None:
            return None
        return _parse_response(data, mode, top_k)

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    async def search(self, *, query: str, top_k: int = 5) -> str:
        err = self._check_endpoint(query)
        if err:
            return err

        items = await self._resolve(query, top_k)
        if items is None:
            return self._error_json(query, "Search failed or returned no data")

        results = [_source_candidate_entry(query, i, item) for i, item in enumerate(items)]

        for r in results:
            logger.info("web_search.hit rank=%d url=%s title=%s", r["rank"], r["url"], r["title"])
        return json.dumps(
            {
                "query": query,
                "result_count": len(results),
                "mode": self.config.get("mode", "searxng"),
                "results": results,
            }
        )
