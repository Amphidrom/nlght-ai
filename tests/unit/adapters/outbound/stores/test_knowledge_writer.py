# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Knowledge index writer: a refused write is not a completed one."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from nlght.adapters.outbound.stores import knowledge_writer as writer_module
from nlght.adapters.outbound.stores.connections import StoreConnections
from nlght.adapters.outbound.stores.knowledge_writer import KnowledgeIndexWriterTool
from nlght.core.knowledge import KnowledgeObject


class _FakeConnections(StoreConnections):
    """A registry that hands out the test's fake instead of opening a client."""

    def __init__(self, client: object) -> None:
        super().__init__()
        self._client = client

    def client(self, key, factory, *, close=None):  # type: ignore[no-untyped-def]
        if str(key[0]) == "opensearch":
            return self._client
        return super().client(key, factory, close=close)


class _FakeOpenSearch:
    def __init__(self, *, fail_index: bool = False, fail_delete: bool = False) -> None:
        self.indexed: list[dict[str, Any]] = []
        self.deleted: list[dict[str, Any]] = []
        self._fail_index = fail_index
        self._fail_delete = fail_delete

    def index(self, **kwargs: Any) -> None:  # noqa: ANN401 (opensearch-py is untyped)
        if self._fail_index:
            raise RuntimeError("opensearch refused the write")
        self.indexed.append(kwargs)

    def delete(self, **kwargs: Any) -> None:  # noqa: ANN401 (opensearch-py is untyped)
        if self._fail_delete:
            raise RuntimeError("opensearch refused the delete")
        self.deleted.append(kwargs)


def _assertion() -> KnowledgeObject:
    now = datetime(2026, 8, 27, tzinfo=UTC)
    return KnowledgeObject(
        identity="fact-1",
        kind="fact",
        type="capability",
        confidence=0.9,
        payload={"subject": "nlght", "predicate": "supports", "object": "fan-out"},
        review_required=False,
        review_reason=None,
        product=None,
        version=None,
        metadata={"keywords": ["fan-out"]},
        created_at=now,
        updated_at=now,
    )


def _writer(client: _FakeOpenSearch, monkeypatch: pytest.MonkeyPatch) -> KnowledgeIndexWriterTool:
    monkeypatch.setattr(writer_module, "_HAS_OPENSEARCH", True)
    return KnowledgeIndexWriterTool(
        name="knowledge-index",
        config={
            # Lazy engine, never connected in this test.
            "pg_url": "sqlite+aiosqlite:///:memory:",
            "os_url": "http://opensearch:9200",
        },
        store_connections=_FakeConnections(client),
    )


async def test_an_assertion_is_written_under_its_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _FakeOpenSearch()

    await _writer(client, monkeypatch).index(_assertion())

    assert client.indexed[0]["id"] == "fact-1"
    assert client.indexed[0]["body"]["subject"] == "nlght"


async def test_a_refused_write_raises_instead_of_being_logged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # It used to catch and log, leaving the caller unable to tell a written
    # assertion from an unwritten one. The caller is what decides whether the
    # assertion counts as persisted, and once approval is what triggers
    # indexing, a swallowed failure is an assertion PostgreSQL calls retrievable
    # and search cannot find.
    writer = _writer(_FakeOpenSearch(fail_index=True), monkeypatch)

    with pytest.raises(RuntimeError, match="refused the write"):
        await writer.index(_assertion())


async def test_a_refused_removal_raises_too(monkeypatch: pytest.MonkeyPatch) -> None:
    # The mirror: a removal reported as done while the document stays indexed
    # leaves rejected or merged-away content findable, and nothing tries again.
    writer = _writer(_FakeOpenSearch(fail_delete=True), monkeypatch)

    with pytest.raises(RuntimeError, match="refused the delete"):
        await writer.remove(identity="fact-1", kind="fact")


async def test_removing_what_is_already_gone_is_not_a_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The wanted state is "absent", and it is absent. The 404 stays ignored.
    client = _FakeOpenSearch()

    await _writer(client, monkeypatch).remove(identity="fact-1", kind="fact")

    assert client.deleted[0]["ignore"] == [404]


class _FakeIndices:
    """An index-management fake with controllable existing indexes and create failures.

    It can simulate another writer creating an index during a race."""

    def __init__(
        self,
        present: set[str] | None = None,
        *,
        fail_create: set[str] | None = None,
        created_by_someone_else: set[str] | None = None,
        stale: set[str] | None = None,
    ) -> None:
        self.present = present if present is not None else set()
        #: Indexes whose mapping predates the wording field, as a deployment
        #: that has not been rebuilt since would have.
        self.stale = stale or set()
        self.created: list[str] = []
        self.create_attempts: list[str] = []
        self._fail_create = fail_create or set()
        self._raced = created_by_someone_else or set()

    def exists(self, *, index: str) -> bool:
        return index in self.present

    def get_mapping(self, *, index: str) -> dict[str, Any]:
        """The mapping a deployment actually has, which may predate a field.

        Args:
            index (str): Index name used by the test.

        Returns:
            dict[str, Any]: Index mapping returned by the test backend.
        """
        if index in self.stale:
            return {index: {"mappings": {"properties": {"subject": {"type": "text"}}}}}
        return {
            index: {
                "mappings": {
                    "properties": {"subject": {"type": "text"}, "observed_text": {"type": "text"}}
                }
            }
        }

    def create(self, *, index: str, body: dict[str, Any]) -> None:
        self.create_attempts.append(index)
        if index in self._raced:
            # Another writer got there between our `exists` and this call.
            self.present.add(index)
            raise RuntimeError("resource_already_exists_exception")
        if index in self._fail_create:
            raise RuntimeError("opensearch refused the create")
        self.present.add(index)
        self.created.append(index)


class _FakeCluster(_FakeOpenSearch):
    def __init__(self, indices: _FakeIndices) -> None:
        super().__init__()
        self.indices = indices


#: One index per kind, named by `index_for`. No prefix is configured here, so
#: these are the bare names the writer builds.
_KINDS = {"facts", "rules", "patterns", "decisions"}


def test_an_index_that_exists_is_left_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    indices = _FakeIndices(present=set(_KINDS))

    _writer(_FakeCluster(indices), monkeypatch).ensure_ready()

    assert indices.create_attempts == []


def test_an_index_that_predates_the_wording_is_reported_as_degraded(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
) -> None:
    """Report an index as degraded when its mapping lacks observed wording."""
    indices = _FakeIndices(present=set(_KINDS), stale=set(_KINDS))

    with caplog.at_level("WARNING"):
        writer = _writer(_FakeCluster(indices), monkeypatch)
        writer.ensure_ready()

    assert writer.schema_state() == dict.fromkeys(
        ("fact", "rule", "pattern", "decision"), "stale"
    )
    assert any("schema=old" in record.message for record in caplog.records)
    assert any("rebuild_required" in record.message for record in caplog.records)


def test_a_rebuilt_index_reports_itself_current(monkeypatch: pytest.MonkeyPatch) -> None:
    indices = _FakeIndices(present=set(_KINDS))

    writer = _writer(_FakeCluster(indices), monkeypatch)

    assert set(writer.schema_state().values()) == {"current"}


def test_a_stale_index_is_not_failed_and_not_migrated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Degraded is allowed on purpose: the corpus is still searchable through its
    # structured fields and keywords. What is not allowed is being silent, and
    # rewriting somebody's mapping underneath them is a migration, not a check.
    indices = _FakeIndices(present=set(_KINDS), stale=set(_KINDS))

    _writer(_FakeCluster(indices), monkeypatch).ensure_ready()

    assert indices.create_attempts == []


def test_a_missing_index_is_created_with_its_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    indices = _FakeIndices()

    _writer(_FakeCluster(indices), monkeypatch).ensure_ready()

    assert set(indices.created) == _KINDS


def test_only_the_missing_indexes_are_created(monkeypatch: pytest.MonkeyPatch) -> None:
    """A deployment half-way through a rollout, or one index dropped by hand.

    The check is per index rather than per deployment, so an existing corpus is
    not disturbed and a single dropped index is repaired on the next run.
    """
    indices = _FakeIndices(present={"facts", "rules"})

    _writer(_FakeCluster(indices), monkeypatch).ensure_ready()

    assert set(indices.created) == {"patterns", "decisions"}


def test_a_failed_create_is_raised_rather_than_logged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Raise an index-creation failure before reporting persistence success."""
    indices = _FakeIndices(fail_create={"rules"})

    with pytest.raises(RuntimeError, match="refused the create"):
        _writer(_FakeCluster(indices), monkeypatch).ensure_ready()


def test_a_failed_create_is_retried_rather_than_remembered_as_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retry index creation after a failed bootstrap attempt."""
    indices = _FakeIndices(fail_create={"rules"})
    connections = _FakeConnections(_FakeCluster(indices))
    monkeypatch.setattr(writer_module, "_HAS_OPENSEARCH", True)
    config = {"pg_url": "sqlite+aiosqlite:///:memory:", "os_url": "http://opensearch:9200"}
    writer = KnowledgeIndexWriterTool(
        name="knowledge-index", config=config, store_connections=connections
    )

    with pytest.raises(RuntimeError):
        writer.ensure_ready()

    # The outage clears. The same process tries again through the same memo.
    indices._fail_create = set()
    writer.ensure_ready()

    assert set(indices.present) == _KINDS
    assert sorted(indices.created) == sorted(_KINDS)


def test_losing_the_create_race_is_not_a_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Accept another worker creating the index during bootstrap.

    The writer checks current index state after the create race."""
    indices = _FakeIndices(created_by_someone_else=set(_KINDS))

    _writer(_FakeCluster(indices), monkeypatch).ensure_ready()

    assert indices.created == []          # nobody here created one
    assert set(indices.present) == _KINDS  # and they are all there


def _claim(observed: str = "", **fields: str) -> KnowledgeObject:
    now = datetime(2026, 8, 29, tzinfo=UTC)
    return KnowledgeObject(
        identity="fact-1", kind="fact", type="capability", confidence=0.9,
        payload=dict(fields), review_required=False, review_reason=None,
        product=None, version=None, metadata={}, created_at=now, updated_at=now,
        observed_text=observed,
    )


async def test_the_indexed_text_is_what_the_source_said(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Index observed source wording instead of concatenated structure fields."""
    client = _FakeOpenSearch()
    sentence = "When graceful shutdown is enabled, the application waits for active requests."

    await _writer(client, monkeypatch).index(
        _claim(sentence, subject="application", predicate="waits_for",
               object="active_requests", condition="graceful_shutdown_enabled")
    )

    body = client.indexed[0]["body"]
    assert body["observed_text"] == sentence
    assert "text_all" not in body, "the assembled bag is gone, not merely unused"


async def test_the_structured_fields_are_indexed_beside_the_wording(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Neither replaces the other.

        observed_text   language, for BM25 and for quoting
        structure       precise field and phrase queries, and filtering

    Dropping the bag removed a *duplicate* of the fields, never the fields.
    """
    client = _FakeOpenSearch()

    await _writer(client, monkeypatch).index(
        _claim("The application waits for active requests.",
               subject="application", predicate="waits_for", object="active_requests")
    )

    body = client.indexed[0]["body"]
    assert body["subject"] == "application"
    assert body["predicate"] == "waits_for"
    assert body["object"] == "active_requests"
    assert body["observed_text"]


async def test_a_claim_with_no_recorded_wording_is_indexed_without_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Written before revisions existed, or by a path that keeps none. Its
    # structure is still searchable, and nothing assembles a sentence to fill
    # the gap, which is the one thing ADR-0048 forbids.
    client = _FakeOpenSearch()

    await _writer(client, monkeypatch).index(
        _claim("", subject="application", predicate="waits_for", object="active_requests")
    )

    body = client.indexed[0]["body"]
    assert body["observed_text"] == ""
    assert body["subject"] == "application"
