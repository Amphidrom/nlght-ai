# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Knowledge store tool: search strategy, quarantine, and degradation."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from nlght.adapters.outbound.stores import knowledge as knowledge_module
from nlght.adapters.outbound.stores.connections import StoreConnections
from nlght.adapters.outbound.stores.knowledge import KnowledgeStoreTool
from nlght.core.knowledge import KnowledgeObject


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


class _FakeOpenSearch:
    def __init__(self, hits: list[str] | None = None, *, fail: bool = False) -> None:
        self.hits = hits if hits is not None else []
        self.fail = fail
        self.requests: list[tuple[str, dict[str, Any]]] = []

    def search(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
        self.requests.append((index, body))
        if self.fail:
            raise RuntimeError("opensearch unreachable")
        return {"hits": {"hits": [{"_source": {"id": identity}} for identity in self.hits]}}


class _FakeRepository:
    def __init__(self, visible: set[str] | None = None) -> None:
        self.visible = visible if visible is not None else set()
        self.calls: list[tuple[tuple[str, ...], bool]] = []

    async def resolve(self, identities, *, include_quarantined: bool = False):
        self.calls.append((tuple(identities), include_quarantined))
        now = datetime(2026, 1, 1, tzinfo=UTC)
        return [
            KnowledgeObject(
                identity=identity,
                kind="fact",
                type="dependency",
                confidence=0.9,
                payload={"subject": "Spring", "predicate": "requires", "object": "Java 17"},
                review_required=False,
                review_reason=None,
                product=None,
                version=None,
                metadata={},
                created_at=now,
                updated_at=now,
            )
            for identity in identities
            if identity in self.visible
        ]

    async def upsert(self, write): ...
    async def pending_review(self, *, limit: int = 50): return []
    async def record_review(self, **kwargs): ...


def _tool(
    client: _FakeOpenSearch,
    repository: _FakeRepository,
    **config: str,
) -> KnowledgeStoreTool:
    # The store builds its repository from its own pg_url and accepts no
    # injection, so the fake is substituted after construction. The OpenSearch
    # client comes from the runtime's shared pools, which the test supplies.
    # A lazy SQLite URL keeps construction offline.
    tool = KnowledgeStoreTool(
        name="knowledge",
        config={
            "pg_url": "sqlite+aiosqlite:///:memory:",
            "os_url": "http://opensearch:9200",
            **config,
        },
        store_connections=_FakeConnections(opensearch=client),
    )
    tool._repository = repository  # type: ignore[assignment]
    return tool


@pytest.fixture(autouse=True)
def _opensearch_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(knowledge_module, "_HAS_OPENSEARCH", True)


def test_construction_without_a_knowledge_database_is_refused() -> None:
    # The knowledge graph is a separate database; without its URL the store
    # must refuse rather than silently reading platform tables.
    with pytest.raises(ValueError, match="requires 'pg_url'"):
        KnowledgeStoreTool(name="knowledge", config={"os_url": "http://x"})


def test_write_operations_are_not_exposed_to_a_model() -> None:
    names = {signature.name for signature in KnowledgeStoreTool.signatures()}
    assert names == {
        "knowledge-query-facts",
        "knowledge-query-rules",
        "knowledge-query-patterns",
        "knowledge-query-decisions",
    }


async def test_each_kind_queries_its_own_index() -> None:
    client = _FakeOpenSearch()
    tool = _tool(client, _FakeRepository())

    await tool.query_facts(query="java")
    await tool.query_rules(query="java")
    await tool.query_patterns(query="java")
    await tool.query_decisions(query="java")

    assert [index for index, _ in client.requests] == ["facts", "rules", "patterns", "decisions"]


async def test_quarantined_results_are_dropped_even_when_the_index_returns_them() -> None:
    # The search index may lag behind a review decision; PostgreSQL is authority.
    client = _FakeOpenSearch(["visible", "quarantined"])
    repository = _FakeRepository(visible={"visible"})
    tool = _tool(client, repository)

    results = await tool.query_facts(query="java")

    assert [item.identity for item in results] == ["visible"]
    assert repository.calls == [(("visible", "quarantined"), False)]


async def test_a_filter_only_query_needs_no_search_text() -> None:
    client = _FakeOpenSearch()
    tool = _tool(client, _FakeRepository())

    await tool.query_facts(type="dependency", min_confidence=0.5, limit=3)

    _, body = client.requests[0]
    assert body["size"] == 3
    assert body["query"]["bool"]["filter"] == [
        {"term": {"type": "dependency"}},
        {"range": {"confidence": {"gte": 0.5}}},
    ]


async def test_a_text_query_asks_the_wording_the_terms_and_the_structure() -> None:
    """Three strategies, and the wording is one of them now.

        keywords        explicit term lookups
        structure       phrase queries against subject, rule_text and the rest
        observed_text   the sentence the claim was read from, as language

    It used to end in a `text_all` bag assembled from every field value, which
    carried no phrase and no grammar and for several kinds only repeated the
    fields beside it.
    """
    client = _FakeOpenSearch()
    tool = _tool(client, _FakeRepository())

    await tool.query_facts(query="spring boot")

    _, body = client.requests[0]
    should = body["query"]["bool"]["should"]
    assert {"term": {"keywords": {"value": "spring", "boost": 12}}} in should
    assert {"match_phrase": {"subject": {"query": "spring boot", "boost": 10}}} in should
    assert {"match_phrase": {"observed_text": {"query": "spring boot", "boost": 11}}} in should
    assert {"match": {"observed_text": {"query": "spring boot", "boost": 6}}} in should
    assert not any("text_all" in str(clause) for clause in should)


async def test_a_question_in_the_sources_own_words_outranks_a_bare_term_match() -> None:
    """Why the wording is boosted above a single-field term match.

    A question is usually a sentence, and the sentence a claim was read from is
    the closest thing the index holds to an answer — so a phrase hit on it is
    worth more than one loose term matching a structured field, and less than a
    phrase hit on the field that names the business question.
    """
    client = _FakeOpenSearch()
    tool = _tool(client, _FakeRepository())

    await tool.query_rules(query="grace period 30 seconds")

    _, body = client.requests[0]
    phrase = next(
        clause["match_phrase"]["observed_text"]["boost"]
        for clause in body["query"]["bool"]["should"]
        if "match_phrase" in clause and "observed_text" in clause["match_phrase"]
    )
    loose = next(
        clause["match"]["observed_text"]["boost"]
        for clause in body["query"]["bool"]["should"]
        if "match" in clause and "observed_text" in clause["match"]
    )
    assert phrase > loose


async def test_filters_still_apply_alongside_a_text_query() -> None:
    client = _FakeOpenSearch()
    tool = _tool(client, _FakeRepository())

    await tool.query_facts(query="spring", type="dependency")

    _, body = client.requests[0]
    assert body["query"]["bool"]["filter"] == [{"term": {"type": "dependency"}}]


async def test_an_unreachable_search_backend_degrades_to_no_results() -> None:
    repository = _FakeRepository(visible={"a"})
    tool = _tool(_FakeOpenSearch(["a"], fail=True), repository)

    assert await tool.query_facts(query="java") == []
    assert repository.calls == []


async def test_an_index_prefix_scopes_every_index() -> None:
    client = _FakeOpenSearch()
    tool = _tool(client, _FakeRepository(), index_prefix="tenant-a-")

    await tool.query_rules(query="x")

    assert client.requests[0][0] == "tenant-a-rules"


async def test_the_llm_wrapper_returns_serialised_assertions() -> None:
    tool = _tool(_FakeOpenSearch(["visible"]), _FakeRepository(visible={"visible"}))

    payload = json.loads(await tool.query_facts_tool(query="java"))

    assert payload == [
        {
            "identity": "visible",
            "kind": "fact",
            "type": "dependency",
            "confidence": 0.9,
            "product": None,
            "version": None,
            "subject": "Spring",
            "predicate": "requires",
            "object": "Java 17",
        }
    ]


def test_the_resource_config_carries_the_knowledge_connection() -> None:
    names = {option.name for option in KnowledgeStoreTool.options()}

    # The connection belongs to the activation, so two stores can point at
    # different knowledge databases.
    assert "pg_url" in names
    assert {option.name for option in KnowledgeStoreTool.options() if option.required} == {
        "pg_url", "os_url"
    }
