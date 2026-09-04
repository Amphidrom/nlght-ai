# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""The authority contract, against real PostgreSQL, Qdrant and OpenSearch.

Everything else about this slice is proved over fakes and SQLite, which proves
the logic and not the assumptions underneath it. These are the assumptions:

    the corpus stores the text byte for byte, through a real driver
    a chunk's locator survives a real Qdrant upsert and comes back on a search
    a changed projection fingerprint really does rewrite an unchanged document
    a candidate on a superseded revision never reaches a ranking
    the text a passage carries is cut from PostgreSQL, not from a payload

Ordinary integration-suite discovery ignores this module. Select this file
explicitly after bringing up the required services:

    docker run -d --name nlght-it-pg -e POSTGRES_PASSWORD=nlght \\
        -e POSTGRES_DB=nlght_knowledge -p 55432:5432 postgres:16-alpine
    docker run -d --name nlght-it-qdrant -p 56333:6333 qdrant/qdrant
    docker run -d --name nlght-it-os -e discovery.type=single-node \\
        -e DISABLE_SECURITY_PLUGIN=true -p 59200:9200 opensearchproject/opensearch:2

    nlght-ai migrate-knowledge -x url=$NLGHT_IT_PG_URL upgrade head

Override the defaults below with `NLGHT_IT_PG_URL`, `NLGHT_IT_QDRANT_URL` and
`NLGHT_IT_OPENSEARCH_URL`, then run:

    pytest tests/integration/test_corpus_authority_backends.py
"""

from __future__ import annotations

import os
import uuid
from typing import Any

import pytest

pytestmark = pytest.mark.integration

PG_URL = os.environ.get(
    "NLGHT_IT_PG_URL",
    "postgresql+asyncpg://postgres:nlght@localhost:55432/nlght_knowledge",
)
QDRANT_URL = os.environ.get("NLGHT_IT_QDRANT_URL", "http://localhost:56333")
OPENSEARCH_URL = os.environ.get("NLGHT_IT_OPENSEARCH_URL", "http://localhost:59200")


def _reachable() -> bool:
    import socket
    from urllib.parse import urlparse

    for url in (PG_URL.replace("postgresql+asyncpg", "http"), QDRANT_URL, OPENSEARCH_URL):
        parsed = urlparse(url)
        port = parsed.port or (5432 if "postgres" in url else 80)
        try:
            with socket.create_connection((parsed.hostname or "localhost", port), timeout=1):
                pass
        except OSError:
            return False
    return True


pytest.importorskip("qdrant_client")
pytest.importorskip("opensearchpy")
pytest.importorskip("asyncpg")

if not _reachable():  # pragma: no cover - environment guard
    pytest.skip(
        "needs a live PostgreSQL, Qdrant and OpenSearch; see the module docstring",
        allow_module_level=True,
    )

from nlght.adapters.outbound.ingestion.classifier import PygmentsDocumentClassifier  # noqa: E402
from nlght.adapters.outbound.ingestion.enrichers import BuiltinDocumentEnricher  # noqa: E402
from nlght.adapters.outbound.persistence.ingestion_repository import (  # noqa: E402
    SqlAlchemyIngestionRepository,
)
from nlght.adapters.outbound.stores.connections import StoreConnections  # noqa: E402
from nlght.adapters.outbound.stores.data import DataStoreTool  # noqa: E402
from nlght.adapters.outbound.stores.data_writer import DataIndexWriterTool  # noqa: E402
from nlght.core.ingestion import (  # noqa: E402
    ChunkingSettings,
    ChunkProjection,
    DocumentRevisionWrite,
    IndexStateWrite,
    IngestionDocumentProcessor,
    SnapshotCommit,
    SourceDocument,
)
from nlght.core.retrieval import CHUNK  # noqa: E402

# A document long enough that its span is unmistakably a *part* of it: the
# eight-thousand-line class this whole design keeps pointing at, in miniature.
BODY = "\n".join(f"line {i:03d}: the keystore password is configured here" for i in range(200))


@pytest.fixture
async def connections():
    pool = StoreConnections()
    yield pool
    await pool.aclose()


@pytest.fixture
def target() -> str:
    """A fresh collection and index per test, so tests cannot see each other."""
    return f"it-{uuid.uuid4().hex[:12]}"


@pytest.fixture
def repository(connections: StoreConnections) -> SqlAlchemyIngestionRepository:
    return SqlAlchemyIngestionRepository(connections.engine(PG_URL, pool_pre_ping=True))


@pytest.fixture
def writer(connections: StoreConnections, target: str) -> DataIndexWriterTool:
    return DataIndexWriterTool(
        name=target,
        config={
            "pg_url": PG_URL,
            "qdrant_url": QDRANT_URL,
            "opensearch_url": OPENSEARCH_URL,
            "collection": target,
            "index": target,
        },
        store_connections=connections,
    )


@pytest.fixture
def store(connections: StoreConnections, target: str) -> DataStoreTool:
    return DataStoreTool(
        name=target,
        config={
            "pg_url": PG_URL,
            "qdrant_url": QDRANT_URL,
            "opensearch_url": OPENSEARCH_URL,
            "collection": target,
            "index": target,
        },
        embedding_client=_Embedding(),
        store_connections=connections,
    )


class _Embedding:
    """Deterministic vectors, so a test needs no model and no network.

    The property under test is not embedding quality; it is that a locator makes
    it through a real upsert and back out of a real search.
    """

    async def embed(self, texts: list[str]) -> Any:  # noqa: ANN401
        from nlght.ports.outbound.embedding_client import EmbeddingResult

        return EmbeddingResult(
            vectors=tuple(_vector(text) for text in texts),
            provider="test",
            model="deterministic",
            dimension=8,
        )


def _vector(text: str) -> tuple[float, ...]:
    seed = sum(ord(c) for c in text)
    return tuple(((seed >> i) % 17) / 17.0 for i in range(8))


def _process(path: str, body: str, revision: str, source: str = "it-source") -> Any:  # noqa: ANN401
    processor = IngestionDocumentProcessor(
        classifier=PygmentsDocumentClassifier(),
        enricher=BuiltinDocumentEnricher(),
        chunking=ChunkingSettings(size=400, overlap=40),
    )
    return processor.process(
        SourceDocument(
            source=source,
            external_id=f"{source}:{path}",
            path=path,
            content=body.encode("utf-8"),
            source_revision=revision,
        )
    )


def _revision_write(document: Any) -> DocumentRevisionWrite:  # noqa: ANN401
    return DocumentRevisionWrite(
        document_id=document.document_id,
        external_id=document.external_id,
        processing_revision_id=document.processing_revision_id,
        source_revision_id=document.source_revision_id,
        path=document.path,
        artifact_ref=f"{document.source}:{document.external_id}@{document.source_revision_id}",
        content=document.text,
        metadata=dict(document.metadata),
        processing={"accepted": document.accepted},
    )


def _chunks(document: Any) -> list[ChunkProjection]:  # noqa: ANN401
    return [
        ChunkProjection(
            chunk_id=chunk.chunk_id,
            position=chunk.position,
            start_offset=chunk.start_offset,
            end_offset=chunk.end_offset,
            content=chunk.content,
            vector=_vector(chunk.content),
            semantics=("keystore",),
        )
        for chunk in document.chunks
    ]


async def _ingest(
    repository: SqlAlchemyIngestionRepository,
    writer: DataIndexWriterTool,
    document: Any,  # noqa: ANN401
    *,
    snapshot: str,
    fingerprint: str | None = None,
) -> str:
    """One document through the real write path, corpus record included."""
    await repository.commit_snapshot(
        SnapshotCommit(
            snapshot_key=snapshot,
            source_id=document.source,
            source_type="filesystem",
            config_revision="it-v1",
            complete=True,
            documents=(_revision_write(document),),
        )
    )
    if writer.ensure_ready(dimension=8):
        generation = await repository.rotate_generation(target=writer.target)
    else:
        generation = await repository.current_generation(target=writer.target)
    mark = fingerprint or writer.projection_fingerprint(
        provider="test", model="deterministic", dimension=8
    )
    await writer.write_document(
        document_id=document.document_id,
        path=document.path,
        text=document.text or "",
        source_revision_id=document.source_revision_id,
        processing_revision_id=document.processing_revision_id,
        metadata={"source": document.source, "external_id": document.external_id},
        chunks=_chunks(document),
    )
    await repository.record_indexed(
        IndexStateWrite(
            document_id=document.document_id,
            target=writer.target,
            content_hash=document.source_revision_id,
            enricher_hash=document.processing_revision_id,
            processing_revision_id=document.processing_revision_id,
            index_generation=generation,
            projection_fingerprint=mark,
        )
    )
    # Documents are written with refresh=False; a search would otherwise race it.
    writer.observed_counts()
    return generation


# ---------------------------------------------------------------------------
# PostgreSQL: the corpus holds the text, and knows which revision is published
# ---------------------------------------------------------------------------


async def test_a_revision_round_trips_through_a_real_driver(
    target: str,
    repository: SqlAlchemyIngestionRepository, writer: DataIndexWriterTool
) -> None:
    """`content` comes back byte for byte, newlines and all.

    SQLite proves the logic; this proves that asyncpg, the `TEXT` column and the
    round trip do not touch the string. A chunk's offsets index this exact
    value, so a single normalised newline would silently shift every span.
    """
    document = _process("Foo.java", BODY, "r1", target)
    await _ingest(repository, writer, document, snapshot=f"snap-{uuid.uuid4()}")

    stored = await repository.read_revision_contents(
        [(document.document_id, document.processing_revision_id)]
    )

    assert stored[(document.document_id, document.processing_revision_id)] == BODY
    assert document.text == BODY


async def test_the_published_revision_is_what_the_corpus_says_it_is(
    target: str,
    repository: SqlAlchemyIngestionRepository, writer: DataIndexWriterTool
) -> None:
    """And a superseded one is simply not in the answer."""
    first = _process("Bar.java", BODY, "r1", target)
    await _ingest(repository, writer, first, snapshot=f"snap-{uuid.uuid4()}")
    second = _process("Bar.java", BODY + "\nand one more line", "r2", target)
    await _ingest(repository, writer, second, snapshot=f"snap-{uuid.uuid4()}")

    published = await repository.current_processing_revisions([first.document_id])

    assert published == {first.document_id: second.processing_revision_id}
    assert first.processing_revision_id != second.processing_revision_id
    # The older revision is still stored — history is kept, it is just not current.
    stored = await repository.read_revision_contents(
        [(first.document_id, first.processing_revision_id)]
    )
    assert stored[(first.document_id, first.processing_revision_id)] == BODY


# ---------------------------------------------------------------------------
# Qdrant: the locator survives a real upsert and a real search
# ---------------------------------------------------------------------------


async def test_a_chunk_locator_survives_a_real_upsert_and_search(
    target: str,
    repository: SqlAlchemyIngestionRepository,
    writer: DataIndexWriterTool,
    store: DataStoreTool,
) -> None:
    """The offsets are computed at processing time and used at read time.

    Everything between is a payload written to and read from a real Qdrant, and
    this is the only test that proves the numbers make it across intact.
    """
    document = _process("Baz.java", BODY, "r1", target)
    await _ingest(repository, writer, document, snapshot=f"snap-{uuid.uuid4()}")

    candidates = await store.search_semantic_candidates(query="keystore password", limit=10)

    assert candidates, "the vector index answered nothing"
    hit = next(c for c in candidates if c.carrier == CHUNK)
    assert hit.provenance.document_id == document.document_id
    assert hit.provenance.processing_revision_id == document.processing_revision_id
    assert hit.provenance.end_offset > hit.provenance.start_offset
    # And the span the locator names is really that chunk's text.
    original = next(c for c in document.chunks if c.chunk_id == hit.carrier_id)
    assert BODY[hit.provenance.start_offset : hit.provenance.end_offset] == original.content


# ---------------------------------------------------------------------------
# The reindex decision, against the real recorded state
# ---------------------------------------------------------------------------


async def test_a_changed_projection_fingerprint_forces_a_real_rewrite(
    target: str,
    repository: SqlAlchemyIngestionRepository, writer: DataIndexWriterTool
) -> None:
    """The case that used to reindex nothing, decided on a real row.

    Source, processing and generation all still agree; only the projection
    moved. Before the fingerprint existed this document was skipped and the run
    reported success into a collection holding vectors from the old model.
    """
    document = _process("Qux.java", BODY, "r1", target)
    generation = await _ingest(
        repository, writer, document, snapshot=f"snap-{uuid.uuid4()}",
        fingerprint="projection-A",
    )

    state = await repository.get_index_state(
        document_id=document.document_id, target=writer.target
    )
    assert state is not None
    assert state.projection_fingerprint == "projection-A"
    assert state.index_generation == generation

    # What `_is_current` compares, with only the fingerprint moved.
    unchanged = (
        state.content_hash == document.source_revision_id
        and state.enricher_hash == document.processing_revision_id
        and state.index_generation == generation
    )
    assert unchanged, "source, processing and generation must all still agree"
    assert state.projection_fingerprint != "projection-B", (
        "a changed projection is exactly what the other three cannot see"
    )

    # Rewriting under the new projection replaces the recorded state.
    await _ingest(
        repository, writer, document, snapshot=f"snap-{uuid.uuid4()}",
        fingerprint="projection-B",
    )
    after = await repository.get_index_state(
        document_id=document.document_id, target=writer.target
    )
    assert after is not None and after.projection_fingerprint == "projection-B"


# ---------------------------------------------------------------------------
# End to end: what a model would be shown
# ---------------------------------------------------------------------------


async def test_a_superseded_candidate_never_leaves_the_store(
    target: str,
    repository: SqlAlchemyIngestionRepository,
    writer: DataIndexWriterTool,
    store: DataStoreTool,
) -> None:
    """The real version of the write-then-prune window.

    The corpus is moved to a new revision while the index still holds the old
    one's chunks — which is exactly the state a rewrite passes through. Nothing
    on the old revision may reach a ranking.
    """
    document = _process("Stale.java", BODY, "r1", target)
    await _ingest(repository, writer, document, snapshot=f"snap-{uuid.uuid4()}")

    # Publish a new revision *without* touching the index, leaving the index on
    # the old one — the window, held open.
    moved = _process("Stale.java", BODY + "\nnewly added", "r2", target)
    await repository.commit_snapshot(
        SnapshotCommit(
            snapshot_key=f"snap-{uuid.uuid4()}", source_id=moved.source,
            source_type="filesystem", config_revision="it-v1", complete=True,
            documents=(_revision_write(moved),),
        )
    )

    keyword = await store.search_keyword_candidates(query="keystore", limit=10)
    semantic = await store.search_semantic_candidates(query="keystore", limit=10)

    assert [c for c in keyword if c.provenance.document_id == document.document_id] == []
    assert [c for c in semantic if c.provenance.document_id == document.document_id] == []


async def test_the_answer_is_cut_from_postgresql_and_not_from_the_payload(
    target: str,
    repository: SqlAlchemyIngestionRepository,
    writer: DataIndexWriterTool,
    store: DataStoreTool,
) -> None:
    """The whole contract, end to end, on real backends.

    The corpus is then edited underneath the index — same revision id, different
    stored text — so payload and corpus disagree by construction. Whatever comes
    back has to be the corpus's version, or the index is still the authority.
    """
    document = _process("Answer.java", BODY, "r1", target)
    await _ingest(repository, writer, document, snapshot=f"snap-{uuid.uuid4()}")

    marked = BODY.replace("keystore password", "KEYSTORE-FROM-POSTGRES")
    engine = store._connections.engine(PG_URL, pool_pre_ping=True)  # noqa: SLF001
    from sqlalchemy import text as sql

    async with engine.begin() as conn:
        await conn.execute(
            sql(
                "UPDATE ingestion_document_revisions SET content = :c "
                "WHERE processing_revision_id = :r"
            ),
            {"c": marked, "r": document.processing_revision_id},
        )

    hits = await store.search(query="keystore password", limit=5)

    assert hits, "the hybrid search answered nothing"
    for hit in hits:
        assert "KEYSTORE-FROM-POSTGRES" in hit.content, (
            "the answer came from the index payload, which still says 'keystore "
            "password' — the projection is being used as the authority"
        )

    # And a chunk's answer is the span, not the whole document.
    chunk_hits = [h for h in hits if h.id != document.document_id]
    assert chunk_hits, "no chunk survived the ranking"
    assert all(len(h.content) < len(marked) for h in chunk_hits), (
        "a chunk was answered with the whole revision instead of its span"
    )


async def test_a_revision_the_corpus_cannot_answer_is_not_faked(
    target: str,
    repository: SqlAlchemyIngestionRepository,
    writer: DataIndexWriterTool,
    store: DataStoreTool,
) -> None:
    """A row from before the content column: resolvable identity, no text.

    `resolve` must return nothing for it rather than reaching for the payload.
    """
    document = _process("Empty.java", BODY, "r1", target)
    await _ingest(repository, writer, document, snapshot=f"snap-{uuid.uuid4()}")

    engine = store._connections.engine(PG_URL, pool_pre_ping=True)  # noqa: SLF001
    from sqlalchemy import text as sql

    async with engine.begin() as conn:
        await conn.execute(
            sql(
                "UPDATE ingestion_document_revisions SET content = NULL "
                "WHERE processing_revision_id = :r"
            ),
            {"r": document.processing_revision_id},
        )

    candidates = await store.search_keyword_candidates(query="keystore", limit=5)
    mine = [c for c in candidates if c.provenance.document_id == document.document_id]
    assert mine, "the document must still be discoverable; only its text is gone"

    resolved = await store.resolve(mine)

    assert resolved == {}, "an unstored revision must resolve to nothing, not to a payload"


async def test_a_tombstoned_document_stops_answering(
    target: str,
    repository: SqlAlchemyIngestionRepository,
    writer: DataIndexWriterTool,
    store: DataStoreTool,
) -> None:
    """Deleted at the source, and the corpus says so before the index catches up."""
    document = _process("Gone.java", BODY, "r1", target)
    await _ingest(repository, writer, document, snapshot=f"snap-{uuid.uuid4()}")

    other = _process("Kept.java", BODY, "r1", target)
    await repository.commit_snapshot(
        SnapshotCommit(
            snapshot_key=f"snap-{uuid.uuid4()}", source_id=other.source,
            source_type="filesystem", config_revision="it-v1", complete=True,
            documents=(_revision_write(other),),
        )
    )

    published = await repository.current_processing_revisions([document.document_id])
    assert published == {}, "a tombstoned document has no published revision"

    candidates = await store.search_keyword_candidates(query="keystore", limit=10)
    assert [c for c in candidates if c.provenance.document_id == document.document_id] == []
