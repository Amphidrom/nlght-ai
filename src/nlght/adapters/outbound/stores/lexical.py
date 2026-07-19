# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import dataclasses
import json
import logging
import re
from collections.abc import Iterator
from typing import Any, ClassVar

try:
    from opensearchpy import OpenSearch as _OpenSearch
    _HAS_OPENSEARCH = True
except ImportError:
    _HAS_OPENSEARCH = False
    _OpenSearch = None

from nlght.core.tools.tool import ToolBase, ToolParameter, ToolSignature
from nlght.ports.outbound.lexical_store import LexicalResult

logger = logging.getLogger(__name__)


def _serialize(items: list[LexicalResult]) -> str:
    return json.dumps([dataclasses.asdict(i) for i in items])


class LexicalStoreTool(ToolBase):
    KIND: ClassVar[str]     = "lexical_store"
    PROVIDER: ClassVar[str] = "opensearch"

    def __init__(self, *, name: str, config: dict[str, Any], **kwargs: Any) -> None:  # noqa: ANN401 (forwarded into ToolBase.__init__ typed kw surface)
        super().__init__(name=name, config=config, **kwargs)
        self._url          = config["url"]
        self._index        = config.get("collection", "default")
        self._username     = config.get("username")
        self._password     = config.get("password")
        self._verify_certs = config.get("verify_certs", False)
        self._timeout      = config.get("timeout", 30)
        if not _HAS_OPENSEARCH:
            raise ImportError(
                "opensearch-py is required for LexicalStoreTool. "
                "Install it with: pip install 'nlght-ai[lexical-store]'"
            )
        self.__client: _OpenSearch | None = None

    @classmethod
    def signatures(cls) -> list[ToolSignature]:
        p_query   = ToolParameter("query",   "string", "Suchanfrage")
        p_queries = ToolParameter("queries", "array",  "Mehrere Suchanfragen")
        p_limit   = ToolParameter("limit",   "number", "Max. Ergebnisse", required=False, default=10)
        return [
            ToolSignature("lexical-search",        "Lexikalische Code-/Inhaltssuche",          "search_content_tool",  [p_query,   p_limit]),
            ToolSignature("lexical-search-symbols", "Lexikalische Symbol-Suche",               "search_symbols_tool",  [p_query,   p_limit]),
            ToolSignature("lexical-search-multi",  "Mehrfach-Suche, ein Artifact pro Query",   "search_multi_tool",    [p_queries, p_limit]),
        ]

    # ── OpenSearch client (lazy, sync) ────────────────────────────────────────

    @property
    def _client(self) -> _OpenSearch:
        if self.__client is None:
            http_auth = None
            if self._username:
                http_auth = (self._username, self._password or "")
            self.__client = _OpenSearch(
                hosts=[self._url],
                http_auth=http_auth,
                verify_certs=self._verify_certs,
                timeout=self._timeout,
            )
        return self.__client

    # ── Port Protocol ─────────────────────────────────────────────────────────

    async def search(self, query: str, limit: int = 10) -> list[LexicalResult]:
        return list(self._search_content(query, limit))

    async def search_symbols(self, query: str, limit: int = 10) -> list[LexicalResult]:
        return list(self._search_symbols_impl(query, limit))

    async def search_multi(self, queries: list[str], limit: int = 10) -> list[LexicalResult]:
        seen: set[str] = set()
        results: list[LexicalResult] = []
        for q in queries:
            for r in self._search_content(q, limit):
                if r.id not in seen:
                    seen.add(r.id)
                    results.append(r)
        return results

    # ── Tool-Dispatch ─────────────────────────────────────────────────────────

    async def search_content_tool(self, *, query: str, limit: int = 10) -> str:
        return _serialize(await self.search(query=query, limit=limit))

    async def search_symbols_tool(self, *, query: str, limit: int = 10) -> str:
        return _serialize(await self.search_symbols(query=query, limit=limit))

    async def search_multi_tool(self, *, queries: list[str], limit: int = 10) -> str:
        return _serialize(await self.search_multi(queries=queries, limit=limit))

    # ── Internals ─────────────────────────────────────────────────────────────

    def _search_content(self, query: str, limit: int) -> Iterator[LexicalResult]:
        tokens = list(set(re.findall(r"[A-Za-z_][A-Za-z0-9_.]*", query)))
        should = []
        for t in tokens:
            should.append({"term": {"symbols": {"value": t, "boost": 20}}})
            should.append({"term": {"fqns":    {"value": t, "boost": 18}}})
        for t in tokens:
            if len(t) >= 3:
                should.append({"wildcard": {"symbols": {"value": f"*{t}*", "boost": 12}}})
                should.append({"wildcard": {"fqns":    {"value": f"*{t}*", "boost": 10}}})
        for t in tokens:
            should.append({"match": {"path": {"query": t, "boost": 8}}})
        should.append({"match": {"content": {"query": query, "boost": 5}}})

        body = {
            "size": limit,
            "query": {"bool": {"should": should, "minimum_should_match": 1}},
            "_source": True,
        }
        try:
            resp = self._client.search(index=self._index, body=body)
            for hit in resp.get("hits", {}).get("hits", []):
                src   = hit["_source"]
                score = hit.get("_score", 0.0)
                yield LexicalResult(
                    id       = hit.get("_id", src.get("id", "")),
                    score    = score,
                    content  = src.get("content", ""),
                    metadata = {k: v for k, v in src.items() if k != "content"},
                )
        except Exception:
            logger.exception("lexical._search_content query=%s", query)

    def _search_symbols_impl(self, query: str, limit: int) -> Iterator[LexicalResult]:
        body = {
            "size": limit,
            "query": {
                "bool": {
                    "should": [
                        {"term":     {"symbols": {"value": query,         "boost": 25}}},
                        {"term":     {"fqns":    {"value": query,         "boost": 20}}},
                        {"wildcard": {"symbols": {"value": f"*{query}*",  "boost": 15}}},
                        {"wildcard": {"fqns":    {"value": f"*{query}*",  "boost": 12}}},
                        {"match":    {"path":    {"query": query,         "boost": 5}}},
                    ],
                    "minimum_should_match": 1,
                }
            },
            "_source": True,
        }
        try:
            resp = self._client.search(index=self._index, body=body)
            for hit in resp.get("hits", {}).get("hits", []):
                src   = hit["_source"]
                score = hit.get("_score", 0.0)
                yield LexicalResult(
                    id       = hit.get("_id", src.get("id", "")),
                    score    = score,
                    content  = src.get("content", ""),
                    metadata = {k: v for k, v in src.items() if k != "content"},
                )
        except Exception:
            logger.exception("lexical._search_symbols query=%s", query)
