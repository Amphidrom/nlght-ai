# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Data store retrieval tool.

One resource activation spanning Qdrant, OpenSearch and the corpus in
PostgreSQL. Keyword and semantic discovery are two capabilities of one store
over one corpus rather than a store each (ADR-0063), a caller asks a question in
text, and the store decides per operation which backend answers it.

The indexes say *which* documents match; PostgreSQL says which revision is
published and what it says (ADR-0062). Nothing leaves here on a revision that is
no longer published, and nothing leaves here quoting an index payload where the
corpus can be read.

The text-to-vector step uses the embedding port, which is what makes this
usable at all — the previous ``vector-search`` signature asked the model itself
for a ``query_vector`` array.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from nlght.adapters.outbound.persistence.ingestion_repository import (
    SqlAlchemyIngestionRepository,
)
from nlght.adapters.outbound.stores.connections import StoreConnections
from nlght.adapters.outbound.tools.builtin.action_semantics import READ_REQUEST
from nlght.core.retrieval import (
    CHUNK,
    DOCUMENT,
    LEXICAL,
    VECTOR,
    Provenance,
    RetrievalHit,
    as_fields,
    fuse,
    rank_within_sources,
)
from nlght.core.tools.tool import ToolBase, ToolOption, ToolParameter, ToolSignature
from nlght.ports.outbound.embedding_client import EmbeddingClient
from nlght.ports.outbound.ingestion_repository import IngestionRepository

logger = logging.getLogger(__name__)

try:  # pragma: no cover - import guard
    from qdrant_client import QdrantClient as _QdrantClient

    _HAS_QDRANT = True
except ImportError:  # pragma: no cover - import guard
    _QdrantClient = None
    _HAS_QDRANT = False

try:  # pragma: no cover - import guard
    from opensearchpy import OpenSearch as _OpenSearch

    _HAS_OPENSEARCH = True
except ImportError:  # pragma: no cover - import guard
    _OpenSearch = None
    _HAS_OPENSEARCH = False

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")


def _number(payload: Mapping[str, Any], name: str) -> int:
    """One locator number from a payload, or zero where the store has none."""
    try:
        return int(payload.get(name) or 0)
    except (TypeError, ValueError):
        return 0

@dataclass(slots=True, frozen=True)
class DataHit:
    """One finding, in the shape the model-facing signatures have always had.

    ``content`` is the authoritative text once a finding has been resolved, and
    the payload's copy only where a caller asked for candidates and did not
    resolve them. The distinction is the point of ADR-0062, and it is why
    ``search()`` resolves before it serialises.
    """

    id: str
    score: float
    content: str
    metadata: dict[str, Any]
    sources: tuple[str, ...]


def _serialize(hits: list[DataHit]) -> str:
    return json.dumps(
        [
            {
                "id": hit.id,
                "score": round(hit.score, 6),
                "content": hit.content,
                "sources": list(hit.sources),
                **{k: v for k, v in hit.metadata.items() if k != "content"},
            }
            for hit in hits
        ],
        ensure_ascii=False,
        default=str,
    )


@dataclass(slots=True, frozen=True)
class ResolvedText:
    """What a finding actually says, read from the authority.

    ``text`` came out of PostgreSQL, not out of a payload. For a chunk it is the
    span its locator names; for a document it is the whole revision, because a
    document names no span and inventing one would answer with text the finding
    never identified (ADR-0062).
    """

    carrier_id: str
    document_id: str
    processing_revision_id: str
    text: str


class DataStoreTool(ToolBase):
    """Hybrid retrieval over indexed documents.

    Config: ``pg_url``, ``qdrant_url``, ``opensearch_url``, optional
    ``collection``, ``index``, ``qdrant_api_key``, ``opensearch_username``,
    ``opensearch_password``, ``timeout``, and ``score_threshold``.

    ``pg_url`` points at the knowledge database, which holds the corpus. It is
    required because the two indexes are discovery: they say *which* documents
    match, and PostgreSQL says which revision is published and what it says
    (ADR-0062).
    """

    KIND: ClassVar[str] = "data_store"
    PROVIDER: ClassVar[str] = "qdrant+opensearch"

    def __init__(
        self,
        *,
        name: str,
        config: dict[str, Any],
        embedding_client: EmbeddingClient | None = None,
        store_connections: StoreConnections | None = None,
        **kwargs: Any,  # noqa: ANN401 (forwarded into ToolBase.__init__ typed kw surface)
    ) -> None:
        super().__init__(name=name, config=config, **kwargs)
        if not _HAS_QDRANT or not _HAS_OPENSEARCH:
            raise ImportError(
                "DataStoreTool needs both backends. Install them with: "
                "pip install 'nlght-ai[vector-store,lexical-store]'"
            )
        # The authority. Required, and with no way around it: without it this
        # tool cannot tell a published revision from a superseded one, and would
        # answer with whatever the index happened to still hold (ADR-0062).
        pg_url = str(config.get("pg_url", "")).strip()
        if not pg_url:
            raise ValueError(
                f"data store '{name}' requires 'pg_url' in its resource config, "
                "pointing at the knowledge database that holds the corpus "
                "(run 'nlght-ai migrate-knowledge'). Without it a search cannot "
                "tell a current revision from a stale one."
            )
        # The runtime's pools, or a private set when constructed outside it.
        self._connections = store_connections or StoreConnections()
        self._repository: IngestionRepository = SqlAlchemyIngestionRepository(
            self._connections.engine(pg_url, pool_pre_ping=True)
        )
        self._embedding = embedding_client
        self._collection = str(config.get("collection", "default"))
        self._index = str(config.get("index", self._collection))
        self._timeout = int(config.get("timeout", 30))
        self._score_threshold = float(config.get("score_threshold", 0.0))
        self._qdrant_url = config["qdrant_url"]
        self._qdrant_api_key = config.get("qdrant_api_key")
        self._opensearch_url = config["opensearch_url"]
        self._opensearch_username = config.get("opensearch_username")
        self._opensearch_password = config.get("opensearch_password")

    # -- clients -------------------------------------------------------------

    @property
    def _vector(self) -> Any:  # noqa: ANN401 (qdrant-client is untyped)
        return self._connections.client(
            ("qdrant", self._qdrant_url, self._qdrant_api_key, self._timeout),
            lambda: _QdrantClient(
                url=self._qdrant_url, api_key=self._qdrant_api_key, timeout=self._timeout
            ),
            close=lambda client: client.close(),
        )

    @property
    def _lexical(self) -> Any:  # noqa: ANN401 (opensearch-py is untyped)
        auth = (
            (self._opensearch_username, self._opensearch_password or "")
            if self._opensearch_username
            else None
        )
        return self._connections.client(
            ("opensearch", self._opensearch_url, auth, self._timeout),
            lambda: _OpenSearch(
                hosts=[self._opensearch_url], http_auth=auth, timeout=self._timeout
            ),
            close=lambda client: client.close(),
        )

    @classmethod
    def options(cls) -> list[ToolOption]:
        return [
            ToolOption("pg_url", "string",
                       "Knowledge database URL; holds the corpus this store reads "
                       "its answers from and checks its hits against",
                       required=True, secret=True,
                       placeholder="postgresql+asyncpg://user:pass@host/knowledge"),
            ToolOption("qdrant_url", "string", "Qdrant URL", required=True,
                       placeholder="http://localhost:6333"),
            ToolOption("opensearch_url", "string", "OpenSearch URL", required=True,
                       placeholder="http://localhost:9200"),
            ToolOption("collection", "string", "Qdrant collection for chunk vectors",
                       default="default"),
            ToolOption("index", "string", "OpenSearch index; defaults to the collection name"),
            ToolOption("qdrant_api_key", "string", "Qdrant API key", secret=True),
            ToolOption("opensearch_username", "string", "OpenSearch user"),
            ToolOption("opensearch_password", "string", "OpenSearch password", secret=True),
            ToolOption("timeout", "integer", "Request timeout in seconds", default=30),
        ]

    @classmethod
    def signatures(cls) -> list[ToolSignature]:
        query = ToolParameter("query", "string", "What to search for, in natural language")
        limit = ToolParameter("limit", "number", "Max results", required=False, default=10)
        return [
            ToolSignature(
                "data-search",
                "Hybrid search over indexed documents (semantic + keyword).",
                "search_tool",
                [query, limit],
                action=READ_REQUEST,
            ),
            ToolSignature(
                "data-search-symbols",
                "Search code symbols and fully qualified names.",
                "search_symbols_tool",
                [query, limit],
                action=READ_REQUEST,
            ),
            ToolSignature(
                "data-search-keyword",
                "Keyword-only search, for exact strings a semantic search would blur.",
                "search_keyword_tool",
                [query, limit],
                action=READ_REQUEST,
            ),
        ]

    # -- discovery: candidates, already reduced to the published fassung -----

    async def search_keyword_candidates(
        self, *, query: str, limit: int = 10
    ) -> list[RetrievalHit]:
        """What the text index found, minus everything no longer published.

        Candidates, not answers: a caller ranks them, and the store's promise is
        that nothing stale reaches that ranking (ADR-0062). `content` on them is
        still the payload's copy — it is what a ranking reads, and it is not what
        a reader answers with.
        """
        hits, _ = await self._candidates(self._search_lexical(query, limit), LEXICAL)
        return hits

    async def search_semantic_candidates(
        self, *, query: str, limit: int = 10
    ) -> list[RetrievalHit]:
        """The same, from the vector index. Empty where no embedding is configured."""
        dense = await self._search_dense(query, limit)
        hits, _ = await self._candidates(dense or [], VECTOR)
        return hits

    async def resolve(self, hits: Sequence[RetrievalHit]) -> dict[str, ResolvedText]:
        """What these findings actually say, read from PostgreSQL.

        Called with what a ranking selected rather than with every candidate:
        currency needs identity only, and this reads whole documents.

        A chunk is answered with the span its locator names; a document with the
        whole revision, because a document names no span and inventing one would
        answer with text the finding never identified. A finding whose revision
        has no stored text is absent from the result — the caller then has a hit
        it cannot answer with, which is honest, where falling back to the payload
        would return a projection as though it were the source.
        """
        keys = [
            (hit.provenance.document_id, hit.provenance.processing_revision_id)
            for hit in hits
            if hit.provenance.document_id and hit.provenance.processing_revision_id
        ]
        contents = await self._repository.read_revision_contents(keys)

        resolved: dict[str, ResolvedText] = {}
        for hit in hits:
            key = (hit.provenance.document_id, hit.provenance.processing_revision_id)
            content = contents.get(key)
            if content is None:
                continue
            span = hit.carrier == CHUNK and hit.provenance.end_offset > hit.provenance.start_offset
            resolved[hit.carrier_id] = ResolvedText(
                carrier_id=hit.carrier_id,
                document_id=key[0],
                processing_revision_id=key[1],
                text=(
                    content[hit.provenance.start_offset : hit.provenance.end_offset]
                    if span
                    else content
                ),
            )
        return resolved

    # -- programmatic API ----------------------------------------------------

    async def search(self, *, query: str, limit: int = 10) -> list[DataHit]:
        """Hybrid, over the one fusion this platform has.

        Candidates from both capabilities, each already reduced to the published
        fassung, ranked by `core/retrieval` — the same reciprocal-rank fusion
        `retrieval.search` uses — and then resolved against PostgreSQL, so what
        comes back is the corpus rather than a payload.

        The local RRF this used to carry keyed lexical hits by document id and
        vector hits by the Qdrant *point* id — a UUID derived from the chunk
        hash, and not an identity anything else uses. Two incompatible id spaces
        with no stated reason for being incompatible.

        Replacing it with the platform's fusion does not make the two backends
        reinforce each other, and nothing here should imply that it does: they
        answer at different granularities on purpose (`core/retrieval/hit.py`).
        OpenSearch holds a document, Qdrant holds chunks of it, and a document
        and a chunk of it are two findings — relatedness is not identity. What
        changes is that this is now true for the stated reason rather than by
        accident of key shapes, that the ranking is the one the platform
        defines, and that a chunk's `id` is its own hash instead of a UUID
        derived from it.
        """
        limit = max(1, int(limit))
        lexical, lexical_meta = await self._candidates(
            self._search_lexical(query, limit), LEXICAL
        )
        dense = await self._search_dense(query, limit)
        semantic, semantic_meta = await self._candidates(dense or [], VECTOR)
        metadata = {**lexical_meta, **semantic_meta}

        fused = fuse(rank_within_sources(lexical + semantic))[:limit]
        resolved = await self.resolve([group.best for group in fused])
        return [
            DataHit(
                id=group.key[1],
                score=group.score,
                content=self._answer(group.key[1], group.best, resolved),
                metadata=metadata.get(group.key[1], {}),
                sources=group.sources,
            )
            for group in fused
        ]

    async def search_symbols(self, *, query: str, limit: int = 10) -> list[DataHit]:
        limit = max(1, int(limit))
        hits, metadata = await self._candidates(self._search_symbols(query, limit), LEXICAL)
        return await self._answered(hits, metadata)

    async def search_keyword(self, *, query: str, limit: int = 10) -> list[DataHit]:
        limit = max(1, int(limit))
        hits, metadata = await self._candidates(self._search_lexical(query, limit), LEXICAL)
        return await self._answered(hits, metadata)

    # -- LLM-facing wrappers -------------------------------------------------

    async def search_tool(self, *, query: str, limit: int = 10) -> str:
        return _serialize(await self.search(query=query, limit=limit))

    async def search_symbols_tool(self, *, query: str, limit: int = 10) -> str:
        return _serialize(await self.search_symbols(query=query, limit=limit))

    async def search_keyword_tool(self, *, query: str, limit: int = 10) -> str:
        return _serialize(await self.search_keyword(query=query, limit=limit))

    # -- internals -----------------------------------------------------------

    async def _candidates(
        self, raw: list[DataHit], source: str
    ) -> tuple[list[RetrievalHit], dict[str, dict[str, Any]]]:
        """Backend results as findings, with everything stale already gone.

        This is the currency boundary, and it sits here rather than after the
        ranking on purpose. A stale hit that reaches a fusion has already
        distorted it — it took a rank, it displaced something current, and the
        scores of everything around it moved. Dropping it afterwards would leave
        a ranking that had been computed with it in.

        Qdrant is where this actually bites: `write_document` upserts before it
        prunes, deliberately, so that a document is never briefly absent — which
        means chunks of two processing revisions are findable during a rewrite.
        OpenSearch replaces a document under one id and cannot hold two.
        """
        hits: list[RetrievalHit] = []
        metadata: dict[str, dict[str, Any]] = {}
        carrier = CHUNK if source == VECTOR else DOCUMENT

        pending: list[tuple[RetrievalHit, dict[str, Any]]] = []
        for hit in raw:
            payload = hit.metadata
            document_id = str(
                payload.get("document_id") or payload.get("id") or hit.id
            )
            carrier_id = (
                str(payload.get("chunk_id") or hit.id) if carrier == CHUNK else document_id
            )
            pending.append(
                (
                    RetrievalHit(
                        source=source,
                        carrier=carrier,
                        carrier_id=carrier_id,
                        content=hit.content,
                        raw_score=hit.score,
                        provenance=Provenance(
                            document_id=document_id,
                            chunk_id=carrier_id if carrier == CHUNK else "",
                            source_revision_id=str(payload.get("source_revision_id") or ""),
                            processing_revision_id=str(
                                payload.get("processing_revision_id") or ""
                            ),
                            position=_number(payload, "position"),
                            start_offset=_number(payload, "start_offset"),
                            end_offset=_number(payload, "end_offset"),
                            path=str(payload.get("path") or ""),
                            source_name=str(payload.get("source") or ""),
                            external_id=str(payload.get("external_id") or ""),
                        ),
                        fields=as_fields(payload),
                    ),
                    payload,
                )
            )

        published = await self._repository.current_processing_revisions(
            [found.provenance.document_id for found, _ in pending]
        )
        for found, payload in pending:
            current = published.get(found.provenance.document_id)
            if current is None or current != found.provenance.processing_revision_id:
                logger.debug(
                    "data.candidate.stale | document=%s found=%s published=%s",
                    found.provenance.document_id,
                    found.provenance.processing_revision_id,
                    current,
                )
                continue
            hits.append(found)
            metadata[found.carrier_id] = {
                k: v for k, v in payload.items() if k != "content"
            }
        return hits, metadata

    @staticmethod
    def _answer(
        carrier_id: str, hit: RetrievalHit, resolved: dict[str, ResolvedText]
    ) -> str:
        """The authoritative text, or the payload where the corpus has none.

        A revision written before the corpus stored its content has no
        authoritative text, and the payload is then all there is. It is a
        degraded answer and it is logged as one, rather than being silently
        indistinguishable from a resolved hit.
        """
        found = resolved.get(carrier_id)
        if found is not None:
            return found.text
        logger.warning(
            "data.resolve.unstored | document=%s revision=%s — answering from the "
            "index payload because the corpus has no stored content for this "
            "revision; reindex to make it authoritative",
            hit.provenance.document_id,
            hit.provenance.processing_revision_id,
        )
        return hit.content

    async def _answered(
        self, hits: list[RetrievalHit], metadata: dict[str, dict[str, Any]]
    ) -> list[DataHit]:
        """A single-capability result, resolved and in the model-facing shape."""
        resolved = await self.resolve(hits)
        return [
            DataHit(
                id=hit.carrier_id,
                score=hit.raw_score,
                content=self._answer(hit.carrier_id, hit, resolved),
                metadata=metadata.get(hit.carrier_id, {}),
                sources=(hit.source,),
            )
            for hit in hits
        ]

    async def _search_dense(self, query: str, limit: int) -> list[DataHit] | None:
        """None when no embedding capability is configured or reachable.

        A missing embedding provider degrades hybrid search to keyword-only
        rather than failing: partial results beat no results, and the caller can
        still see which backends contributed.
        """
        if self._embedding is None:
            logger.debug("data.search.dense_skipped | no embedding client configured")
            return None
        try:
            embedded = await self._embedding.embed([query])
            # `query_points`, not the removed `search`. qdrant-client dropped
            # `search` after 1.9, and because a failure here is caught and
            # logged rather than raised, the whole semantic half returned
            # nothing against a real Qdrant while every test passed against a
            # fake that still had the old method.
            response = self._vector.query_points(
                collection_name=self._collection,
                query=list(embedded.vectors[0]),
                limit=limit,
                score_threshold=self._score_threshold or None,
                with_payload=True,
            )
            hits = getattr(response, "points", response)
        except Exception:
            logger.exception("data.search.dense | collection=%s", self._collection)
            return None

        results: list[DataHit] = []
        for hit in hits:
            payload = getattr(hit, "payload", None) or {}
            results.append(
                DataHit(
                    id=str(getattr(hit, "id", "")),
                    score=float(getattr(hit, "score", 0.0)),
                    content=str(payload.get("content", "")),
                    metadata={k: v for k, v in payload.items() if k != "content"},
                    sources=("vector",),
                )
            )
        return results

    def _run_lexical(self, body: dict[str, Any], label: str) -> list[DataHit]:
        try:
            response = self._lexical.search(index=self._index, body=body)
        except Exception:
            logger.exception("data.search.%s | index=%s", label, self._index)
            return []
        hits = ((response or {}).get("hits") or {}).get("hits") or []
        results: list[DataHit] = []
        for hit in hits:
            source = hit.get("_source") or {}
            results.append(
                DataHit(
                    id=str(hit.get("_id") or source.get("id", "")),
                    score=float(hit.get("_score") or 0.0),
                    content=str(source.get("content", "")),
                    metadata={k: v for k, v in source.items() if k != "content"},
                    sources=("lexical",),
                )
            )
        return results

    def _search_lexical(self, query: str, limit: int) -> list[DataHit]:
        tokens = sorted(set(_TOKEN_RE.findall(query)))
        should: list[dict[str, Any]] = []
        for token in tokens:
            should.append({"term": {"symbols": {"value": token, "boost": 20}}})
            should.append({"term": {"fqns": {"value": token, "boost": 18}}})
        for token in tokens:
            if len(token) >= 3:
                should.append({"wildcard": {"symbols": {"value": f"*{token}*", "boost": 12}}})
        for token in tokens:
            should.append({"match": {"path": {"query": token, "boost": 8}}})
        should.append({"match": {"content": {"query": query, "boost": 5}}})

        return self._run_lexical(
            {
                "size": limit,
                "query": {"bool": {"should": should, "minimum_should_match": 1}},
                "_source": True,
            },
            "lexical",
        )

    def _search_symbols(self, query: str, limit: int) -> list[DataHit]:
        return self._run_lexical(
            {
                "size": limit,
                "query": {
                    "bool": {
                        "should": [
                            {"term": {"symbols": {"value": query, "boost": 25}}},
                            {"term": {"fqns": {"value": query, "boost": 20}}},
                            {"wildcard": {"symbols": {"value": f"*{query}*", "boost": 15}}},
                            {"wildcard": {"fqns": {"value": f"*{query}*", "boost": 12}}},
                            {"match": {"path": {"query": query, "boost": 5}}},
                        ],
                        "minimum_should_match": 1,
                    }
                },
                "_source": True,
            },
            "symbols",
        )
