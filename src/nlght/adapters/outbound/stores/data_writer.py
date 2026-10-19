# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Write data-store projections without exposing an LLM-callable operation.

The adapter uses a separate resource kind from retrieval and declares no tool
signatures. Index state is scoped to its target so writers for different
collections do not share a false account of what is indexed.
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
    rows for the target. One document is one OpenSearch document. The vector
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
the chunk locator (position and offsets), without which a hit names no span of
its document. Bumping it is what makes an already-indexed corpus be written
again; nothing else in the recorded state describes the payload.
"""


def _point_id(name: str) -> str:
    """Derive a repeatable Qdrant UUID from an arbitrary content identity.

    Args:
        name (str): Original chunk or document identity.

    Returns:
        str: A UUIDv5 string suitable for Qdrant point IDs.
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
        # different indexes must not share one record of what is indexed, which
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
        """Identify the index projection independently of the processed document.

        Args:
            provider (str): Embedding provider name.
            model (str): Embedding model name.
            dimension (int): Embedding vector dimension.

        Returns:
            str: Digest of schema version, distance metric, provider, model, and dimension.
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
        """Identify this writer's index target for per-target state bookkeeping.

        Returns:
            str: Resource name and collection name joined as one target ID.
        """
        return f"{self.name}:{self._collection}"

    def ensure_ready(self, *, dimension: int) -> bool:
        """Create missing index structures before the first write in this process.

        Args:
            dimension (int): Embedding dimension. Zero skips vector collection setup.

        Returns:
            bool: True when an index or collection was created and the target generation
                must rotate.

        Side Effects:
            May create a Qdrant collection or OpenSearch index and records bootstrap
                state. Creation failures propagate.
        """
        created = False
        # A dimension of zero means this run produced no vectors at all, as in a
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
                    # once. Losing that race is not a failure because the collection
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
        """Read actual index counts for reconciliation with recorded state.

        Returns:
            IndexCounts: OpenSearch document count and Qdrant point count. Missing indexes count as zero.

        Side Effects:
            Refreshes and queries OpenSearch, queries Qdrant, and logs missing indexes.
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
        """Upsert a new document revision before pruning old vector points.

        Args:
            document_id (str): Stable identity of the document.
            path (str): Source path stored in index payloads.
            text (str): Whole document text for the lexical index.
            source_revision_id (str): Revision identified by citations.
            processing_revision_id (str): Published processing revision for currency checks.
            metadata (dict[str, Any]): Additional lexical document fields.
            chunks (list[ChunkProjection]): Embedded chunk projections for the vector index.

        Side Effects:
            Upserts Qdrant points, prunes older points for the document, then replaces
                its OpenSearch document.
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
        """Remove one document from both index backends.

        Args:
            document_id (str): Stable identity of the document to remove.

        Side Effects:
            Deletes Qdrant points and the OpenSearch document. Backend failures propagate.
        """
        self._delete_vectors(document_id)
        self._lexical.delete(index=self._index, id=document_id, ignore=[404], refresh=False)

    def _delete_vectors(self, document_id: str, *, collection: str | None = None) -> None:
        """Delete every vector point belonging to one document.

        Args:
            document_id (str): Document whose points must be removed.
            collection (str | None): Optional collection override. Defaults to this
                writer's collection.

        Side Effects:
            Deletes matching Qdrant points. A backend failure propagates.
        """
        self._vector.delete(
            collection_name=collection or self._collection,
            # The Qdrant client rejects a raw dict here and requires a typed
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
        """Remove a document's vector points from older processing revisions.

        Args:
            document_id (str): Document whose old points must be removed.
            processing_revision_id (str): Revision whose points must remain.

        Side Effects:
            Deletes stale Qdrant points. A backend failure propagates.
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
