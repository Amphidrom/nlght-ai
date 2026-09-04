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
            # Lazy engine — never connected in this test.
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
    # assertion counts as persisted — and once approval is what triggers
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


# ---------------------------------------------------------------------------
# Readiness: "persisted" may not be true while "findable" is impossible
# ---------------------------------------------------------------------------

class _FakeIndices:
    """The index-management half of an OpenSearch client, with a controllable store.

    `present` is the deployment's actual state, so a test can start with some
    indexes already there and a create can be made to fail while another writer
    puts the index in place — which is the race the real client cannot avoid.
    """

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
        """The mapping a deployment actually has, which may predate a field."""
        if index in self.stale:
            return {index: {"mappings": {"properties": {"subject": {"type": "text"}}}}}
        return {
            index: {
                "mappings": {
                    "properties": {"subject": {"type": "text"}, "observed_text": {"type": "text"}}
                }
            }
        }

    def create(self, *, index: str, body: dict[str, Any]) -> None:  # noqa: ARG002
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
    """A search that quietly got worse is the hardest regression to find.

    An index created before the wording was indexed still accepts every document
    and answers every query — it simply cannot rank on `observed_text`. Nothing
    fails, so nothing says so, and months later somebody debugs relevance while
    the only problem is a rebuild nobody ran.
    """
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
    # structured fields and keywords. What is not allowed is being silent — and
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
    """The whole point of the slice.

    It used to catch and log, so a run committed to PostgreSQL, indexed into a
    mapping that was never applied, and reported success — "persisted" true
    while "findable" was structurally impossible.

    There is no best effort here: every kind's index is required, because a kind
    whose mapping is missing is a kind no query can find.
    """
    indices = _FakeIndices(fail_create={"rules"})

    with pytest.raises(RuntimeError, match="refused the create"):
        _writer(_FakeCluster(indices), monkeypatch).ensure_ready()


def test_a_failed_create_is_retried_rather_than_remembered_as_done(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The second defect, and the one that made the first unrecoverable.

    `mark_bootstrapped` ran on the failure path too, so the next call in the
    process skipped the index it had just failed to create — the error was not
    ignored but cached as success. A transient failure therefore became
    permanent for the lifetime of the worker.

    Retrying must also leave nothing half-made: the second attempt creates
    exactly the index the first could not.
    """
    indices = _FakeIndices(fail_create={"rules"})
    connections = _FakeConnections(_FakeCluster(indices))
    monkeypatch.setattr(writer_module, "_HAS_OPENSEARCH", True)
    config = {"pg_url": "sqlite+aiosqlite:///:memory:", "os_url": "http://opensearch:9200"}
    writer = KnowledgeIndexWriterTool(
        name="knowledge-index", config=config, store_connections=connections
    )

    with pytest.raises(RuntimeError):
        writer.ensure_ready()

    # The outage clears; the same process tries again through the same memo.
    indices._fail_create = set()  # noqa: SLF001
    writer.ensure_ready()

    assert set(indices.present) == _KINDS
    assert sorted(indices.created) == sorted(_KINDS)


def test_losing_the_create_race_is_not_a_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two workers bootstrapping one empty deployment.

    Both see the index missing and both try; the second is told it already
    exists. That is the outcome it wanted, so it is success — and recognised by
    asking again rather than by matching an exception type, because
    `opensearchpy` is untyped and its error classes move between versions.

    At most one create takes effect, and neither writer raises.
    """
    indices = _FakeIndices(created_by_someone_else=set(_KINDS))

    _writer(_FakeCluster(indices), monkeypatch).ensure_ready()

    assert indices.created == []          # nobody here created one
    assert set(indices.present) == _KINDS  # and they are all there


# ---------------------------------------------------------------------------
# text_all is a bag, and a stable one
# ---------------------------------------------------------------------------

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
    """The wording is indexed, not a bag assembled from the field values.

    `text_all` used to hold every value of the payload joined together —

        graceful_shutdown_enabled active_requests waits_for application

    which is not language. It carries no phrase, no grammar and no synonym, it
    cannot be quoted back to a reader, and for a pattern it was literally
    `description + pattern_name`, both already indexed beside it. What a lexical
    index should rank on is the sentence the claim was read from.
    """
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
    # the gap — which is the one thing ADR-0048 forbids.
    client = _FakeOpenSearch()

    await _writer(client, monkeypatch).index(
        _claim("", subject="application", predicate="waits_for", object="active_requests")
    )

    body = client.indexed[0]["body"]
    assert body["observed_text"] == ""
    assert body["subject"] == "application"
