# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import pytest

# The writer builds qdrant model objects at call time; without the extra there
# is nothing to assert against, so skip rather than fail in a pruned env.
pytest.importorskip("qdrant_client")

from qdrant_client.models import FilterSelector  # noqa: E402

from nlght.adapters.outbound.stores import data_writer as dw  # noqa: E402
from nlght.adapters.outbound.stores.connections import StoreConnections  # noqa: E402
from nlght.core.ingestion import ChunkProjection  # noqa: E402


class _FakeConnections(StoreConnections):
    """A registry that hands out the test's fakes instead of opening clients.

    The writer reaches its backends through the runtime's shared pools, so a
    test substitutes the pools rather than reaching past a lazy attribute.
    """

    def __init__(self, **by_kind: object) -> None:
        super().__init__()
        self._by_kind = by_kind

    def client(self, key, factory, *, close=None):  # type: ignore[no-untyped-def]
        kind = str(key[0])
        if kind in self._by_kind:
            return self._by_kind[kind]
        return super().client(key, factory, close=close)


class _FakeVector:
    def __init__(self, *, fail_delete: bool = False) -> None:
        self.calls: list[dict] = []
        self.upserts: list[dict] = []
        # The order the writer touched the store in — the difference between
        # replacing a document and briefly removing it.
        self.order: list[str] = []
        self._fail_delete = fail_delete

    def delete(self, **kwargs: object) -> None:
        if self._fail_delete:
            raise RuntimeError("qdrant refused the delete")
        self.order.append("delete")
        self.calls.append(kwargs)

    def upsert(self, **kwargs: object) -> None:
        self.order.append("upsert")
        self.upserts.append(kwargs)


class _FakeLexical:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def index(self, **kwargs: object) -> None:
        self.calls.append(kwargs)


def _make_writer(
    monkeypatch: pytest.MonkeyPatch,
    collection: str = "data-main",
    *,
    vector: _FakeVector | None = None,
    lexical: _FakeLexical | None = None,
) -> dw.DataIndexWriterTool:
    monkeypatch.setattr(dw, "_HAS_QDRANT", True)
    monkeypatch.setattr(dw, "_HAS_OPENSEARCH", True)
    return dw.DataIndexWriterTool(
        name="w",
        config={
            # Lazy engine — never connected in this test.
            "pg_url": "sqlite+aiosqlite:///:memory:",
            "qdrant_url": "http://qdrant",
            "opensearch_url": "http://opensearch",
            "collection": collection,
        },
        store_connections=_FakeConnections(
            qdrant=vector or _FakeVector(),
            opensearch=lexical or _FakeLexical(),
        ),
    )


def test_delete_vectors_passes_a_typed_filter_selector_not_a_dict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression: a raw dict points_selector made qdrant raise
    # "Unsupported points selector type: <class 'dict'>", so the delete-then-write
    # never cleared a document's old chunks on reindex.
    fake = _FakeVector()
    writer = _make_writer(monkeypatch, vector=fake)

    writer._delete_vectors("doc-123")

    assert len(fake.calls) == 1
    call = fake.calls[0]
    assert call["collection_name"] == "data-main"
    assert isinstance(call["points_selector"], FilterSelector)


def test_delete_vectors_targets_the_named_collection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeVector()
    writer = _make_writer(monkeypatch, vector=fake)

    writer._delete_vectors("doc-123", collection="data-main-identity")

    assert fake.calls[0]["collection_name"] == "data-main-identity"
    assert isinstance(fake.calls[0]["points_selector"], FilterSelector)


async def test_write_document_upserts_uuid_point_ids_keeping_the_hash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression: chunk ids are content hashes, which qdrant rejects as point
    # ids ("valid values are either an unsigned integer or a UUID"). They must be
    # mapped to a deterministic UUID, with the original hash kept in the payload.
    import uuid

    fake_vector = _FakeVector()
    writer = _make_writer(monkeypatch, vector=fake_vector)

    chunk_hash = "d4a52e6de11adbcb4e62387a8ee22153b2f6825880d731e080cfd505b4593395"
    await writer.write_document(
        document_id="doc-1",
        path="testdocs/notes/axiome.md",
        text="body",
        source_revision_id="src-1",
        processing_revision_id="revision-1",
        metadata={"source": "testdocs"},
        chunks=[
            ChunkProjection(
                chunk_id=chunk_hash, position=0, start_offset=0,
                end_offset=10, content="chunk text", vector=(0.1, 0.2),
            )
        ],
    )

    points = fake_vector.upserts[0]["points"]
    assert len(points) == 1
    # Must parse as a UUID, and be stable for the same chunk hash.
    assert uuid.UUID(str(points[0].id))
    assert str(points[0].id) == str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_hash))
    assert points[0].payload["chunk_id"] == chunk_hash
    assert points[0].payload["document_id"] == "doc-1"


# -- replacing a document ----------------------------------------------------


async def test_a_document_is_never_absent_while_it_is_replaced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression: delete-then-write left a window in which the document had no
    # chunks at all and a search missed it entirely. Write first, prune second.
    vector = _FakeVector()
    writer = _make_writer(monkeypatch, vector=vector)

    await writer.write_document(
        document_id="doc-1",
        path="a.md",
        text="body",
        source_revision_id="src-1",
        processing_revision_id="revision-2",
        metadata={},
        chunks=[
            ChunkProjection(
                chunk_id="chunk-hash", position=0, start_offset=0,
                end_offset=10, content="chunk text", vector=(0.1, 0.2),
            )
        ],
    )

    assert vector.order == ["upsert", "delete"]


async def test_the_prune_removes_only_the_previous_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A constant-size filter — same document, any revision but this one — so a
    # document whose chunk count shrank keeps no orphans and a document with
    # thousands of chunks needs no list of them.
    vector = _FakeVector()
    writer = _make_writer(monkeypatch, vector=vector)

    await writer.write_document(
        document_id="doc-1",
        path="a.md",
        text="body",
        source_revision_id="src-1",
        processing_revision_id="revision-2",
        metadata={},
        chunks=[
            ChunkProjection(
                chunk_id="chunk-hash", position=0, start_offset=0,
                end_offset=10, content="chunk text", vector=(0.1, 0.2),
            )
        ],
    )

    prune = vector.calls[0]["points_selector"].filter
    assert prune.must[0].match.value == "doc-1"
    assert prune.must_not[0].key == "processing_revision_id"
    assert prune.must_not[0].match.value == "revision-2"


async def test_every_chunk_carries_the_revision_it_belongs_to(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    vector = _FakeVector()
    writer = _make_writer(monkeypatch, vector=vector)

    await writer.write_document(
        document_id="doc-1",
        path="a.md",
        text="body",
        source_revision_id="src-1",
        processing_revision_id="revision-2",
        metadata={},
        chunks=[
            ChunkProjection(
                chunk_id="chunk-hash", position=0, start_offset=0,
                end_offset=10, content="chunk text", vector=(0.1, 0.2),
            )
        ],
    )

    assert vector.upserts[0]["points"][0].payload["processing_revision_id"] == "revision-2"


async def test_a_refused_prune_raises_instead_of_being_logged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # It used to be swallowed. The caller then recorded the document as indexed,
    # so the previous revision's chunks stayed searchable forever and no later
    # run ever touched the document again.
    writer = _make_writer(monkeypatch, vector=_FakeVector(fail_delete=True))

    with pytest.raises(RuntimeError, match="refused the delete"):
        await writer.write_document(
            document_id="doc-1",
            path="a.md",
            text="body",
            source_revision_id="src-1",
            processing_revision_id="revision-2",
            metadata={},
            chunks=[
            ChunkProjection(
                chunk_id="chunk-hash", position=0, start_offset=0,
                end_offset=10, content="chunk text", vector=(0.1, 0.2),
            )
        ],
        )


async def test_a_refused_tombstone_removal_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # The caller clears the index state right after this. Reporting success
    # would drop the document out of the database's view while leaving it in the
    # index, where no later run would ever find it again.
    writer = _make_writer(monkeypatch, vector=_FakeVector(fail_delete=True))

    with pytest.raises(RuntimeError, match="refused the delete"):
        await writer.delete_document(document_id="doc-1")


async def test_a_chunk_payload_names_the_span_it_came_from(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the locator a chunk hit names no part of its document.

    The payload keeps its own copy of the text for ranking and for reading a
    point by eye, but that copy is not what a reader answers with — the span is
    cut from the stored revision (ADR-0062). These three numbers are what make
    that cut possible at all, and they used to be computed and then dropped at
    the `DocumentChunk` → `EmbeddedChunk` boundary.
    """
    fake_vector = _FakeVector()
    writer = _make_writer(monkeypatch, vector=fake_vector)

    await writer.write_document(
        document_id="doc-1",
        path="a.md",
        text="body",
        source_revision_id="src-1",
        processing_revision_id="revision-1",
        metadata={},
        chunks=[
            ChunkProjection(
                chunk_id="chunk-hash", position=3, start_offset=1820,
                end_offset=1964, content="chunk text", vector=(0.1, 0.2),
                semantics=("calculatePrice",),
            )
        ],
    )

    payload = fake_vector.upserts[0]["points"][0].payload
    assert (payload["position"], payload["start_offset"], payload["end_offset"]) == (
        3, 1820, 1964,
    )
    # Both fassungen, so a hit can be checked for currency and cited.
    assert payload["source_revision_id"] == "src-1"
    assert payload["processing_revision_id"] == "revision-1"
    # The terms travel with their own chunk rather than being zipped back on by
    # position from a parallel list.
    assert payload["semantics"] == ["calculatePrice"]


def test_swapping_the_embedding_model_changes_the_projection_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The case that used to reindex nothing.

    Neither hash describes the projection, and the generation only rotates when
    a collection is created — so before this value existed, a model swap left
    every document reading as current and the collection holding vectors from a
    model the queries were no longer embedded with.
    """
    writer = _make_writer(monkeypatch, vector=_FakeVector())

    before = writer.projection_fingerprint(provider="local", model="minilm", dimension=384)
    swapped = writer.projection_fingerprint(provider="local", model="bge-large", dimension=384)
    resized = writer.projection_fingerprint(provider="local", model="minilm", dimension=768)
    same = writer.projection_fingerprint(provider="local", model="minilm", dimension=384)

    assert before != swapped
    assert before != resized
    assert before == same
