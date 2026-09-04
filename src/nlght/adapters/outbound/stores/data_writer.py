# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Write-only index adapter for the data store.

A tool like any other — activated by a resource row, restricted by access
policy — but under a `kind` disjoint from `data_store`, so a retrieval
activation can never reach a write operation. It declares no signatures, so it
never becomes a callable contract in a request-scoped tool catalog.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from typing import Any, ClassVar

from nlght.adapters.outbound.persistence.ingestion_repository import (
    SqlAlchemyIngestionRepository,
)
from nlght.adapters.outbound.stores.connections import StoreConnections
from nlght.core.ingestion import ChunkProjection, stable_digest
from nlght.core.tools.tool import ToolBase, ToolOption, ToolSignature

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class IndexCounts:
    """What a target's indexes actually hold, as opposed to what is recorded.

    ``lexical_documents`` is directly comparable to the number of index-state
    rows for the target — one document is one OpenSearch document. The vector
    counts are points, not documents, so they are a signal rather than an
    equation: a collection that is empty while the state claims a populated
    corpus is the signature of a wiped index.
    """

    lexical_documents: int
    vector_points: int


try:  # pragma: no cover - import guard
    from qdrant_client import QdrantClient as _QdrantClient
    from qdrant_client.models import (
        Distance,
        FieldCondition,
        Filter,
        FilterSelector,
        MatchValue,
        PointStruct,
        VectorParams,
    )

    _HAS_QDRANT = True
except ImportError:  # pragma: no cover - import guard
    _QdrantClient = None
    PointStruct = None
    VectorParams = None
    Distance = None
    Filter = None
    FieldCondition = None
    MatchValue = None
    FilterSelector = None
    _HAS_QDRANT = False

_DOCUMENT_MAPPING = {
    "mappings": {
        "properties": {
            "id": {"type": "keyword"},
            "path": {"type": "keyword"},
            "content": {"type": "text"},
            "source_revision_id": {"type": "keyword"},
            "processing_revision_id": {"type": "keyword"},
            "source": {"type": "keyword"},
            "external_id": {"type": "keyword"},
            "symbols": {"type": "keyword"},
            "fqns": {"type": "keyword"},
            "semantics": {"type": "keyword"},
        }
    }
}

try:  # pragma: no cover - import guard
    from opensearchpy import OpenSearch as _OpenSearch

    _HAS_OPENSEARCH = True
except ImportError:  # pragma: no cover - import guard
    _OpenSearch = None
    _HAS_OPENSEARCH = False


PROJECTION_SCHEMA_VERSION = "2"
"""What shape this writer projects a document into.

Bumped whenever the payload gains, loses or re-means a field. Version 2 added
the chunk locator — position and offsets — without which a hit names no span of
its document. Bumping it is what makes an already-indexed corpus be written
again; nothing else in the recorded state describes the payload.
"""


def _point_id(name: str) -> str:
    """A stable Qdrant point ID derived from an arbitrary string.

    Qdrant only accepts an unsigned integer or a UUID as a point ID, but chunk
    and document identities here are content hashes. A UUIDv5 over the name is
    deterministic, so re-indexing the same chunk upserts the same point rather
    than duplicating it — the original hash is kept in the payload.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, name))


class DataIndexWriterTool(ToolBase):
    """Replaces one document's payload in the data index.

    Config: ``qdrant_url``, ``opensearch_url``, optional ``collection``,
    ``index``, ``qdrant_api_key``, ``opensearch_username``,
    ``opensearch_password``, ``timeout``.
    """

    KIND: ClassVar[str] = "data_index_writer"
    PROVIDER: ClassVar[str] = "qdrant+opensearch"

    def __init__(
        self,
        *,
        name: str,
        config: dict[str, Any],
        store_connections: StoreConnections | None = None,
        **kwargs: Any,  # noqa: ANN401 (forwarded into ToolBase.__init__ typed kw surface)
    ) -> None:
        super().__init__(name=name, config=config, **kwargs)
        if not _HAS_QDRANT or not _HAS_OPENSEARCH:
            raise ImportError(
                "DataIndexWriterTool needs both backends. Install them with: "
                "pip install 'nlght-ai[vector-store,lexical-store]'"
            )
        pg_url = str(config.get("pg_url", "")).strip()
        if not pg_url:
            raise ValueError(
                f"data index writer '{name}' requires 'pg_url' in its resource config, "
                "pointing at the knowledge database (run 'nlght-ai migrate-knowledge')."
            )
        # The runtime's pools, or a private set when constructed outside it.
        self._connections = store_connections or StoreConnections()
        # This activation owns every half of the write path: the index state that
        # decides whether a document needs reindexing belongs to the same
        # activation as the indexes it describes. Two writers pointing at
        # different indexes must not share one record of what is indexed — which
        # the shared pool preserves, because that record is keyed by `target`.
        self.repository = SqlAlchemyIngestionRepository(
            self._connections.engine(pg_url, pool_pre_ping=True)
        )
        self._collection = str(config.get("collection", "default"))
        self._index = str(config.get("index", self._collection))
        self._distance = str(config.get("distance", "cosine")).lower()
        self._timeout = int(config.get("timeout", 30))
        self._qdrant_url = config["qdrant_url"]
        self._qdrant_api_key = config.get("qdrant_api_key")
        self._opensearch_url = config["opensearch_url"]
        self._opensearch_username = config.get("opensearch_username")
        self._opensearch_password = config.get("opensearch_password")

    def projection_fingerprint(self, *, provider: str, model: str, dimension: int) -> str:
        """How this target projects a document, as one comparable value.

        Kept apart from `processing_revision_id`, which says what the processed
        *document* is. This says how that document is turned into index entries,
        and the two move for different reasons: re-chunking changes the document,
        swapping the embedding model does not.

        Before this existed, neither did the second half. `_is_current` compared
        the source hash, the enricher hash and the index generation — all three
        of which are unchanged when the embedding model, its dimension, the
        distance metric or the payload schema change. So changing an embedding
        model silently reindexed nothing: every document read as already current,
        the run reported success, and the collection kept vectors from a model
        the queries were no longer embedded with.
        """
        return stable_digest(
            PROJECTION_SCHEMA_VERSION,
            self._distance,
            provider,
            model,
            str(dimension),
        )

    @classmethod
    def options(cls) -> list[ToolOption]:
        return [
            ToolOption("pg_url", "string",
                       "Knowledge database URL; holds the index state that decides "
                       "which documents need reindexing",
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
            ToolOption("distance", "string", "Vector distance",
                       choices=["cosine", "dot", "euclid"], default="cosine"),
        ]

    @classmethod
    def signatures(cls) -> list[ToolSignature]:
        # Never offered to a model: indexing is workflow work, not a tool call.
        return []

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

    @property
    def target(self) -> str:
        """Identity of what this writer writes into, for index-state bookkeeping."""
        return f"{self.name}:{self._collection}"

    def ensure_ready(self, *, dimension: int) -> bool:
        """Create the collections and index this writer needs, once per process.

        Indexing into a missing Qdrant collection or an unmapped OpenSearch
        index fails or silently produces an unsearchable document, so the
        bootstrap runs before the first write rather than being a deployment
        prerequisite nobody documented.

        Returns whether anything had to be **created**. That is the signal that
        the index this writer is pointed at is not the one the recorded index
        state describes — an emptied Qdrant, a dropped OpenSearch index, a fresh
        environment on an old database — and the caller rotates the target's
        index generation on it, which makes every document count as unindexed
        again. Without it, a wiped index plus an intact index state produces a
        run that writes nothing and reports success.

        A failure to create is raised, never logged and passed over: continuing
        would write into an index whose mapping was never applied.
        """
        created = False
        # A dimension of zero means this run produced no vectors at all — a
        # pipeline with no embed step. Its OpenSearch index still needs its
        # mapping, so only the vector half is skipped.
        for collection in (self._collection,) if dimension else ():
            if not self._connections.needs_bootstrap("qdrant", self._qdrant_url, collection):
                continue
            if not self._vector.collection_exists(collection):
                logger.info("data_writer.collection.create | name=%s dim=%d", collection, dimension)
                try:
                    self._vector.create_collection(
                        collection_name=collection,
                        vectors_config=VectorParams(size=dimension, distance=self._distance_enum()),
                    )
                except Exception:
                    # Two workers may bootstrap the same empty deployment at
                    # once. Losing that race is not a failure — the collection
                    # is there. Anything else is.
                    if not self._vector.collection_exists(collection):
                        raise
                created = True
            self._connections.mark_bootstrapped("qdrant", self._qdrant_url, collection)

        if self._connections.needs_bootstrap("opensearch", self._opensearch_url, self._index):
            if not self._lexical.indices.exists(index=self._index):
                try:
                    self._lexical.indices.create(index=self._index, body=_DOCUMENT_MAPPING)
                except Exception:
                    if not self._lexical.indices.exists(index=self._index):
                        raise
                logger.info("data_writer.index.create | name=%s", self._index)
                created = True
            self._connections.mark_bootstrapped("opensearch", self._opensearch_url, self._index)
        return created

    def observed_counts(self) -> IndexCounts:
        """What the indexes actually hold right now, for reconciliation.

        Refreshes the OpenSearch index first: documents are written with
        ``refresh=False``, so an unrefreshed count would report a divergence
        that is only a delay. A collection or index that does not exist counts
        as empty rather than raising — "it is gone" is the finding, not an
        error.
        """
        try:
            self._lexical.indices.refresh(index=self._index)
            lexical_documents = int(self._lexical.count(index=self._index)["count"])
        except Exception:
            logger.warning("data_writer.count.lexical_absent | index=%s", self._index)
            lexical_documents = 0

        return IndexCounts(
            lexical_documents=lexical_documents,
            vector_points=self._count_points(self._collection),
        )

    def _count_points(self, collection: str) -> int:
        if not self._vector.collection_exists(collection):
            logger.warning("data_writer.count.collection_absent | name=%s", collection)
            return 0
        return int(self._vector.count(collection_name=collection, exact=True).count)

    def _distance_enum(self) -> Any:  # noqa: ANN401 (qdrant-client is untyped)
        mapping = {"cosine": Distance.COSINE, "dot": Distance.DOT, "euclid": Distance.EUCLID}
        if self._distance not in mapping:
            raise ValueError(f"unsupported distance '{self._distance}'")
        return mapping[self._distance]

    async def write_document(
        self,
        *,
        document_id: str,
        path: str,
        text: str,
        source_revision_id: str,
        processing_revision_id: str,
        metadata: dict[str, Any],
        chunks: list[ChunkProjection],
    ) -> None:
        """Replace one document: write the new revision, then prune the old one.

        Write-then-prune, not delete-then-write. Deleting first left a window in
        which the document had no chunks at all and a search missed it entirely.
        Writing first means the document is always present: for a moment it may
        carry chunks of both revisions, which shows a stale passage beside a
        fresh one — the far milder failure of the two.

        A chunk's point id is derived from its content hash, so a chunk that did
        not change keeps its id and the upsert overwrites it in place, carrying
        the new ``processing_revision_id`` with it. The prune then removes what
        the previous revision left behind: same document, older revision. That is
        a constant-size filter, unlike naming every surviving chunk id, and it is
        what keeps a document whose chunk count shrank from retaining orphans.

        Both fassungen go into the payload. ``processing_revision_id`` is what a
        reader checks against the published revision and what the prune filters
        on; ``source_revision_id`` is what a citation names, and it does not move
        when only the pipeline changed.
        """
        if chunks:
            self._vector.upsert(
                collection_name=self._collection,
                points=[
                    PointStruct(
                        id=_point_id(chunk.chunk_id),
                        vector=list(chunk.vector),
                        payload={
                            "chunk_id": chunk.chunk_id,
                            "document_id": document_id,
                            "source_revision_id": source_revision_id,
                            "processing_revision_id": processing_revision_id,
                            "path": path,
                            # The locator. `content` is kept beside it for
                            # ranking and for debugging a payload by eye, and is
                            # explicitly not what a reader answers with: the span
                            # is cut from the stored revision (ADR-0062).
                            "position": chunk.position,
                            "start_offset": chunk.start_offset,
                            "end_offset": chunk.end_offset,
                            "content": chunk.content,
                            "semantics": list(chunk.semantics),
                        },
                    )
                    for chunk in chunks
                ],
            )
        self._prune_superseded_vectors(document_id, processing_revision_id=processing_revision_id)
        # One document is one OpenSearch document, so this replaces it atomically
        # and needs no prune of its own.
        self._lexical.index(
            index=self._index,
            id=document_id,
            body={
                "id": document_id,
                "path": path,
                "content": text,
                "source_revision_id": source_revision_id,
                "processing_revision_id": processing_revision_id,
                **metadata,
            },
            refresh=False,
        )

    async def delete_document(self, *, document_id: str) -> None:
        """Remove a document from both halves of the index.

        Every call here raises on failure. The caller clears the document's
        index state immediately afterwards, so a failure reported as success
        would drop the document out of the database's view while leaving it in
        the index — where no later run would ever find it again, because it
        appears in no further snapshot.
        """
        self._delete_vectors(document_id)
        self._lexical.delete(index=self._index, id=document_id, ignore=[404], refresh=False)

    def _delete_vectors(self, document_id: str, *, collection: str | None = None) -> None:
        """Delete every point of a document. Raises if the store refuses."""
        self._vector.delete(
            collection_name=collection or self._collection,
            # The qdrant client rejects a raw dict here — it wants a typed
            # points selector.
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id),
                        )
                    ]
                )
            ),
        )

    def _prune_superseded_vectors(self, document_id: str, *, processing_revision_id: str) -> None:
        """Delete this document's points that belong to an earlier revision.

        Raises if the store refuses. A swallowed failure here would leave the
        previous revision's chunks in place, and the caller would then record
        the document as current — so its stale text would stay searchable and
        no later run would ever touch the document again.
        """
        self._vector.delete(
            collection_name=self._collection,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id),
                        )
                    ],
                    must_not=[
                        FieldCondition(
                            key="processing_revision_id",
                            match=MatchValue(value=processing_revision_id),
                        )
                    ],
                )
            ),
        )
