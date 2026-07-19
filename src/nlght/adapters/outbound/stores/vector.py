# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import dataclasses
import json
import logging
from collections.abc import Iterator
from typing import Any, ClassVar

try:
    from qdrant_client import QdrantClient as _QdrantClient
    _HAS_QDRANT = True
except ImportError:
    _HAS_QDRANT = False
    _QdrantClient = None

from nlght.core.tools.tool import ToolBase, ToolParameter, ToolSignature
from nlght.ports.outbound.vector_store import VectorResult

logger = logging.getLogger(__name__)


def _serialize(items: list[VectorResult]) -> str:
    return json.dumps([dataclasses.asdict(i) for i in items])


class VectorStoreTool(ToolBase):
    KIND: ClassVar[str]     = "vector_store"
    PROVIDER: ClassVar[str] = "qdrant"

    def __init__(self, *, name: str, config: dict[str, Any], **kwargs: Any) -> None:  # noqa: ANN401 (forwarded into ToolBase.__init__ typed kw surface)
        super().__init__(name=name, config=config, **kwargs)
        self._url        = config["url"]
        self._collection = config.get("collection", "default")
        self._api_key    = config.get("api_key")
        self._timeout    = config.get("timeout", 30)
        if not _HAS_QDRANT:
            raise ImportError(
                "qdrant-client is required for VectorStoreTool. "
                "Install it with: pip install 'nlght-ai[vector-store]'"
            )
        self.__client: _QdrantClient | None = None

    @classmethod
    def signatures(cls) -> list[ToolSignature]:
        p_vec  = ToolParameter("query_vector",  "array", "Embedding-Vektor")
        p_vecs = ToolParameter("query_vectors", "array", "Mehrere Embedding-Vektoren")
        p_lim  = ToolParameter("limit",         "number", "Max. Ergebnisse", required=False, default=10)
        return [
            ToolSignature("vector-search",        "Semantische Vektor-Suche",                  "search_content_tool",  [p_vec,  p_lim]),
            ToolSignature("vector-search-symbols", "Semantische Symbol-Suche",                 "search_symbols_tool",  [p_vec,  p_lim]),
            ToolSignature("vector-search-multi",  "Mehrfach-Vektorsuche, eine Liste gesamt",   "search_multi_tool",    [p_vecs, p_lim]),
        ]

    # ── Qdrant client (lazy, sync) ────────────────────────────────────────────

    @property
    def _client(self) -> _QdrantClient:
        if self.__client is None:
            self.__client = _QdrantClient(
                url=self._url,
                api_key=self._api_key,
                timeout=self._timeout,
            )
        return self.__client

    # ── Port Protocol ─────────────────────────────────────────────────────────

    async def search(self, query_vector: list[float], limit: int = 10) -> list[VectorResult]:
        return list(self._search_content(query_vector, limit))

    async def search_symbols(self, query_vector: list[float], limit: int = 10) -> list[VectorResult]:
        return list(self._search_symbols_impl(query_vector, limit))

    async def search_multi(self, query_vectors: list[list[float]], limit: int = 10) -> list[VectorResult]:
        seen: set[str] = set()
        results: list[VectorResult] = []
        for vec in query_vectors:
            for r in self._search_content(vec, limit):
                if r.id not in seen:
                    seen.add(r.id)
                    results.append(r)
        return results

    # ── Tool-Dispatch ─────────────────────────────────────────────────────────

    async def search_content_tool(self, *, query_vector: list[float], limit: int = 10) -> str:
        return _serialize(await self.search(query_vector=query_vector, limit=limit))

    async def search_symbols_tool(self, *, query_vector: list[float], limit: int = 10) -> str:
        return _serialize(await self.search_symbols(query_vector=query_vector, limit=limit))

    async def search_multi_tool(self, *, query_vectors: list[list[float]], limit: int = 10) -> str:
        return _serialize(await self.search_multi(query_vectors=query_vectors, limit=limit))

    # ── Internals ─────────────────────────────────────────────────────────────

    def _search_content(
        self,
        query_vector: list[float],
        limit: int,
        score_threshold: float = 0.7,
    ) -> Iterator[VectorResult]:
        try:
            hits = self._client.search(
                collection_name=self._collection,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
            )
            for hit in hits:
                payload = hit.payload or {}
                yield VectorResult(
                    id       = str(hit.id),
                    score    = hit.score,
                    content  = payload.get("content", ""),
                    metadata = {k: v for k, v in payload.items() if k != "content"},
                )
        except Exception:
            logger.exception("vector._search_content")

    def _search_symbols_impl(
        self,
        query_vector: list[float],
        limit: int,
        score_threshold: float = 0.75,
    ) -> Iterator[VectorResult]:
        try:
            hits = self._client.search(
                collection_name=self._collection,
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
            )
            for hit in hits:
                payload = hit.payload or {}
                if not (payload.get("symbols") or payload.get("fqn")):
                    continue
                yield VectorResult(
                    id       = str(hit.id),
                    score    = hit.score,
                    content  = payload.get("content", ""),
                    metadata = {k: v for k, v in payload.items() if k != "content"},
                )
        except Exception:
            logger.exception("vector._search_symbols")
