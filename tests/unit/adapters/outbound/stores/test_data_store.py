# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Data store tool: hybrid fusion, degradation, and the text-query contract."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from nlght.adapters.outbound.stores import data as data_module
from nlght.adapters.outbound.stores.connections import StoreConnections
from nlght.adapters.outbound.stores.data import DataStoreTool
from nlght.ports.outbound.embedding_client import EmbeddingResult


class _FakeConnections(StoreConnections):
    """A registry that hands out the test's fakes instead of opening clients."""

    def __init__(self, **by_kind: object) -> None:
        super().__init__()
        self._by_kind = by_kind

    def client(self, key, factory, *, close=None):  # type: ignore[no-untyped-def]
        kind = str(key[0])
        if kind in self._by_kind:
            return self._by_kind[kind]
        return super().client(key, factory, close=close)


class _Corpus:
    """The authority, as the store sees it: what is published and what it says.

    Every document is published at the revision the payload names unless a test
    says otherwise — which is what makes a stale candidate a deliberate setup
    rather than the default.
    """

    def __init__(
        self,
        published: dict[str, str] | None = None,
        contents: dict[tuple[str, str], str] | None = None,
    ) -> None:
        self.published = published
        self.contents = contents or {}
        self.asked: list[str] = []

    async def current_processing_revisions(self, document_ids):  # type: ignore[no-untyped-def]
        self.asked.extend(document_ids)
        if self.published is not None:
            return {k: v for k, v in self.published.items() if k in set(document_ids)}
        return {identity: f"proc-{identity}" for identity in document_ids}

    async def read_revision_contents(self, keys):  # type: ignore[no-untyped-def]
        return {key: self.contents[key] for key in keys if key in self.contents}


class _Hit:
    def __init__(self, identity: str, score: float, content: str = "") -> None:
        self.id = identity
        self.score = score
        self.payload = {
            "content": content or identity,
            "path": f"{identity}.md",
            "document_id": identity,
            "chunk_id": f"chunk-{identity}",
            "processing_revision_id": f"proc-{identity}",
        }


class _FakeQdrant:
    """Mirrors the client's real surface, `query_points` and a response object.

    It used to expose `search`, which qdrant-client dropped after 1.9. The store
    called the method that no longer existed, the AttributeError was caught and
    logged rather than raised, and the semantic half of every hybrid search
    returned nothing against a real Qdrant — while every unit test passed,
    because the fake still had the old method. A fake that keeps a removed API
    alive does not test the adapter, it tests itself.
    """

    def __init__(self, hits: list[_Hit] | None = None, *, fail: bool = False) -> None:
        self.hits = hits or []
        self.fail = fail
        self.calls: list[dict[str, object]] = []

    def query_points(self, **kwargs: object) -> Any:  # noqa: ANN401 (QueryResponse)
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("qdrant unreachable")
        return SimpleNamespace(points=self.hits)


class _FakeOpenSearch:
    def __init__(self, ids: list[str] | None = None, *, fail: bool = False) -> None:
        self.ids = ids if ids is not None else []
        self.fail = fail
        self.bodies: list[dict[str, Any]] = []

    def search(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
        self.bodies.append(body)
        if self.fail:
            raise RuntimeError("opensearch unreachable")
        return {
            "hits": {
                "hits": [
                    {
                        "_id": identity,
                        "_score": 10.0 - position,
                        "_source": {
                            "content": identity,
                            "id": identity,
                            "processing_revision_id": f"proc-{identity}",
                        },
                    }
                    for position, identity in enumerate(self.ids)
                ]
            }
        }


class _FakeEmbedding:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.texts: list[list[str]] = []

    async def embed(self, texts) -> EmbeddingResult:
        self.texts.append(list(texts))
        if self.fail:
            raise RuntimeError("model not loaded")
        return EmbeddingResult(
            vectors=((0.1, 0.2, 0.3),), provider="fake", model="fake-model", dimension=3
        )


@pytest.fixture(autouse=True)
def _backends_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(data_module, "_HAS_QDRANT", True)
    monkeypatch.setattr(data_module, "_HAS_OPENSEARCH", True)


def _tool(
    vector: _FakeQdrant | None = None,
    lexical: _FakeOpenSearch | None = None,
    embedding: _FakeEmbedding | None = None,
    corpus: _Corpus | None = None,
) -> DataStoreTool:
    # The store reaches its backends through the runtime's shared pools, so the
    # fakes are handed in as those pools rather than assigned past a property.
    # The corpus is assigned past the constructor for the same reason the
    # knowledge store's is: the activation decides which database it reads, and
    # an injectable repository would hand that decision to whatever built it.
    tool = DataStoreTool(
        name="data",
        config={
            "pg_url": "sqlite+aiosqlite:///:memory:",
            "qdrant_url": "http://qdrant:6333",
            "opensearch_url": "http://opensearch:9200",
        },
        embedding_client=embedding,
        store_connections=_FakeConnections(
            qdrant=vector or _FakeQdrant(),
            opensearch=lexical or _FakeOpenSearch(),
        ),
    )
    tool._repository = corpus or _Corpus()  # type: ignore[assignment]
    return tool


def test_the_model_asks_in_text_never_with_a_vector() -> None:
    parameters = {
        signature.name: [p.name for p in signature.parameters]
        for signature in DataStoreTool.signatures()
    }
    assert parameters == {
        "data-search": ["query", "limit"],
        "data-search-symbols": ["query", "limit"],
        "data-search-keyword": ["query", "limit"],
    }


async def test_a_text_query_is_embedded_before_the_vector_search() -> None:
    embedding = _FakeEmbedding()
    vector = _FakeQdrant([_Hit("a", 0.9)])
    tool = _tool(vector, _FakeOpenSearch(), embedding)

    await tool.search(query="how does auth work", limit=5)

    assert embedding.texts == [["how does auth work"]]
    # `query`, the parameter `query_points` actually takes.
    assert vector.calls[0]["query"] == [0.1, 0.2, 0.3]
    assert vector.calls[0]["limit"] == 5


async def test_both_backends_answer_and_each_finding_names_its_own() -> None:
    """A document and a chunk of it are two findings, not one seen twice.

    The two backends answer at different granularities on purpose: OpenSearch
    holds the document, Qdrant holds chunks of it. Sharing a `document_id` makes
    them related, and relatedness is not identity (`core/retrieval/hit.py`) — so
    they are ranked together and neither is folded into the other.

    This is why the store cannot show cross-backend reinforcement, and the
    reason is the carrier rather than the old mismatched key spaces.
    """
    tool = _tool(
        _FakeQdrant([_Hit("shared", 0.9), _Hit("vector-only", 0.8)]),
        _FakeOpenSearch(["shared", "lexical-only"]),
        _FakeEmbedding(),
    )

    hits = await tool.search(query="auth", limit=10)

    by_id = {hit.id: hit for hit in hits}
    # The document and the chunk of the same document, both present, apart.
    assert by_id["shared"].sources == ("lexical",)
    assert by_id["chunk-shared"].sources == ("vector",)
    assert by_id["chunk-vector-only"].sources == ("vector",)
    assert by_id["lexical-only"].sources == ("lexical",)
    # A chunk is named by its own hash, not by the point id derived from it.
    assert "chunk-shared" in by_id


async def test_search_without_an_embedding_client_degrades_to_keyword_only() -> None:
    tool = _tool(_FakeQdrant([_Hit("a", 0.9)]), _FakeOpenSearch(["b"]), None)

    hits = await tool.search(query="auth")

    assert [hit.id for hit in hits] == ["b"]
    assert hits[0].sources == ("lexical",)


async def test_an_unreachable_vector_backend_still_returns_keyword_results() -> None:
    tool = _tool(_FakeQdrant(fail=True), _FakeOpenSearch(["b"]), _FakeEmbedding())

    hits = await tool.search(query="auth")

    assert [hit.id for hit in hits] == ["b"]


async def test_a_failing_embedding_model_does_not_fail_the_search() -> None:
    tool = _tool(_FakeQdrant([_Hit("a", 0.9)]), _FakeOpenSearch(["b"]), _FakeEmbedding(fail=True))

    hits = await tool.search(query="auth")

    assert [hit.id for hit in hits] == ["b"]


async def test_both_backends_down_yields_no_results_rather_than_an_error() -> None:
    tool = _tool(_FakeQdrant(fail=True), _FakeOpenSearch(fail=True), _FakeEmbedding())

    assert await tool.search(query="auth") == []


async def test_symbol_search_uses_the_lexical_backend_only() -> None:
    embedding = _FakeEmbedding()
    lexical = _FakeOpenSearch(["Widget"])
    tool = _tool(_FakeQdrant(), lexical, embedding)

    hits = await tool.search_symbols(query="Widget")

    assert [hit.id for hit in hits] == ["Widget"]
    assert embedding.texts == []
    assert {"term": {"symbols": {"value": "Widget", "boost": 25}}} in (
        lexical.bodies[0]["query"]["bool"]["should"]
    )


async def test_keyword_search_skips_embedding_entirely() -> None:
    embedding = _FakeEmbedding()
    tool = _tool(_FakeQdrant(), _FakeOpenSearch(["exact"]), embedding)

    hits = await tool.search_keyword(query="ERR_CONN_RESET")

    assert [hit.id for hit in hits] == ["exact"]
    assert embedding.texts == []


async def test_fusion_is_rank_based_not_score_based() -> None:
    # BM25 scores dwarf cosine similarities; a sum would let the lexical backend
    # decide every ranking on its own. The fusion itself now lives in
    # `core/retrieval` and is pinned there; this holds the store to using it.
    tool = _tool(_FakeQdrant([_Hit("v", 0.91)]), _FakeOpenSearch(["l"]), _FakeEmbedding())

    fused = await tool.search(query="tls", limit=10)

    assert {hit.id for hit in fused} == {"chunk-v", "l"}
    assert fused[0].score == pytest.approx(fused[1].score)


async def test_fusion_respects_the_limit() -> None:
    tool = _tool(lexical=_FakeOpenSearch([str(i) for i in range(20)]))

    assert len(await tool.search(query="tls", limit=5)) == 5


async def test_the_llm_wrapper_keeps_its_shape() -> None:
    """The model-facing contract is unchanged: id, score, content, sources, metadata."""
    tool = _tool(_FakeQdrant([_Hit("a", 0.9)]), _FakeOpenSearch(["a"]), _FakeEmbedding())

    payload = json.loads(await tool.search_tool(query="auth", limit=2))

    assert {entry["id"] for entry in payload} == {"a", "chunk-a"}
    assert all(set(entry) >= {"id", "score", "content", "sources"} for entry in payload)
    assert sorted(entry["sources"][0] for entry in payload) == ["lexical", "vector"]


# ---------------------------------------------------------------------------
# the currency boundary and the authority (ADR-0062)
# ---------------------------------------------------------------------------


async def test_a_stale_candidate_never_reaches_the_ranking() -> None:
    """The reason the check sits in the store and not after the fusion.

    `write_document` upserts before it prunes — deliberately, so a document is
    never briefly absent — so Qdrant holds chunks of two processing revisions
    during a rewrite. A stale one that reached the ranking would already have
    distorted it: it took a rank, displaced something current, and moved the
    scores around it. Removing it afterwards leaves a ranking computed with it
    in.
    """
    corpus = _Corpus(published={"fresh": "proc-fresh"})  # "stale" is not published
    tool = _tool(
        _FakeQdrant([_Hit("stale", 0.99), _Hit("fresh", 0.5)]),
        _FakeOpenSearch(["stale", "fresh"]),
        _FakeEmbedding(),
        corpus=corpus,
    )

    hits = await tool.search(query="auth", limit=10)

    assert {hit.id for hit in hits} == {"fresh", "chunk-fresh"}
    # Not merely absent from the answer — absent from what was ranked.
    candidates = await tool.search_semantic_candidates(query="auth", limit=10)
    assert [hit.provenance.document_id for hit in candidates] == ["fresh"]


async def test_a_tombstoned_document_has_no_published_revision() -> None:
    """A deleted document answers nothing, by the same route as a stale one."""
    tool = _tool(
        lexical=_FakeOpenSearch(["gone"]),
        corpus=_Corpus(published={}),
    )

    assert await tool.search_keyword(query="anything") == []


async def test_a_chunk_is_answered_with_the_span_cut_from_the_stored_revision() -> None:
    """The whole point: the text comes from PostgreSQL, not from the payload.

    The payload deliberately holds a different string here. If it ever appears
    in the answer, the store is serving a projection as though it were the
    corpus.
    """
    document = "0123456789ABCDEFGHIJ"
    corpus = _Corpus(contents={("d", "proc-d"): document})
    hit = _Hit("d", 0.9, content="WHAT THE PAYLOAD HAPPENS TO SAY")
    hit.payload["start_offset"] = 4
    hit.payload["end_offset"] = 10
    tool = _tool(_FakeQdrant([hit]), _FakeOpenSearch([]), _FakeEmbedding(), corpus=corpus)

    hits = await tool.search(query="auth", limit=10)

    assert [h.content for h in hits] == ["456789"]


async def test_a_document_is_answered_whole_because_it_names_no_span() -> None:
    """No invented range, no index snippet — the finding named the document."""
    corpus = _Corpus(contents={("d", "proc-d"): "the whole revision, all of it"})
    tool = _tool(lexical=_FakeOpenSearch(["d"]), corpus=corpus)

    hits = await tool.search_keyword(query="auth")

    assert [h.content for h in hits] == ["the whole revision, all of it"]


async def test_currency_is_asked_once_per_capability_not_once_per_hit() -> None:
    """Batched, because a search asks it of every candidate.

    Content is the expensive read and happens after the ranking; this one is
    identity only and happens before it, so it has to be one query.
    """
    corpus = _Corpus()
    tool = _tool(lexical=_FakeOpenSearch(["a", "b", "c"]), corpus=corpus)

    await tool.search_keyword_candidates(query="auth", limit=10)

    assert corpus.asked == ["a", "b", "c"]
