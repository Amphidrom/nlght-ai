# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Search may only see the knowledge the store is allowed to hand out.

Part of the search contract, not of context building. If a quarantined or
retired claim reaches a `RetrievalHit`, slice 1 is already wrong — and a later
step filtering it out would be compensating for a leak instead of fixing it.

    approved and live       → retrievable
    awaiting or refused     → not retrievable
    merged into another     → not retrievable
    retired by its source   → see below

Against the **real** repository, not another stub. The index is stubbed, because
an index is an external service and it deliberately knows nothing about
visibility — it answers with identities, and the repository decides which of them
a reader may see (ADR-0032). Stubbing the repository would test the mock.
"""

from __future__ import annotations

import pathlib
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pytest
import pytest_asyncio

from nlght.adapters.outbound.stores.connections import StoreConnections
from nlght.adapters.outbound.stores.knowledge import KnowledgeStoreTool
from nlght.adapters.outbound.workflow.steps.retrieval import RetrievalSearchStep
from nlght.core.entry.context import RequestContext
from nlght.core.knowledge import (
    DERIVED_ANCHOR,
    FactPayload,
    KnowledgeWrite,
    LineageWrite,
    ObservedSection,
    Proposition,
    entity_key,
)
from nlght.core.protocol.protocol import ProtocolKind
from nlght.core.retrieval import ASSERTION, KNOWLEDGE
from nlght.core.trigger.trigger import Trigger, TriggerKind
from nlght.core.workflow.step import WorkflowStepContext

pytestmark = pytest.mark.integration

DOCUMENT = "doc-visibility"
SLOT = "unit:visibility#claims"


class _Emitter:
    async def emit(self, signal: object) -> None: ...


def _ctx() -> WorkflowStepContext:
    context = RequestContext(
        correlation_id="run-1", request_id="rid", received_at=datetime(2026, 1, 1, tzinfo=UTC),
        path="/", method="POST", headers={}, query_params={}, client_host=None,
    )
    ctx = WorkflowStepContext(
        correlation_id="run-1",
        trigger=Trigger(
            kind=TriggerKind.INBOUND_EVENT, protocol=ProtocolKind.GENERIC_JSON,
            operation="ask", payload={}, context=context,
        ),
        model="", messages=[], stream=False, emitter=_Emitter(),
    )
    ctx.metadata["retrieval.question"] = "which java version is required"
    return ctx


@dataclass
class _Index:
    """The search backend, which knows nothing about who may see what.

    It answers with identities and a ranking. That is the whole of its job, and
    it is why the visibility rule cannot live here: an index is eventually
    consistent with the graph, so a claim quarantined a second ago is still in
    it.
    """

    identities: list[str] = field(default_factory=list)
    asked: list[str] = field(default_factory=list)

    def search(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:  # noqa: ARG002
        self.asked.append(index)
        # The shape the store reads: identities live in `_source`, because the
        # index is asked for exactly that field and nothing else.
        return {"hits": {"hits": [{"_source": {"id": item}} for item in self.identities]}}

    def close(self) -> None: ...


@pytest_asyncio.fixture
async def corpus():
    """A knowledge database with one visible claim and three invisible ones."""
    directory = pathlib.Path(tempfile.mkdtemp())
    url = f"sqlite+aiosqlite:///{(directory / 'knowledge.db').as_posix()}"
    connections = StoreConnections()
    engine = connections.engine(url)

    from sqlalchemy import event

    from nlght.adapters.outbound.persistence.knowledge_models import KnowledgeBase

    @event.listens_for(engine.sync_engine, "connect")
    def _enforce_foreign_keys(dbapi_connection, _record):  # noqa: ANN001, ANN202
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    async with engine.begin() as conn:
        await conn.run_sync(KnowledgeBase.metadata.create_all)

    from nlght.adapters.outbound.persistence.knowledge_repository import (
        SqlAlchemyKnowledgeRepository,
    )

    repository = SqlAlchemyKnowledgeRepository(engine)
    yield url, connections, repository
    await engine.dispose()


def _claim(
    text: str, obj: str, *, run: str = "run-1", review: bool = False,
    fingerprint: str | None = None,
):
    """One sighting, as the pipeline would produce it.

    The fingerprint addresses the *proposition*, so it moves only when the
    structured claim does — which is why a rewording that leaves the structure
    alone keeps one graph node, and a claim whose structure moves leaves a second
    one behind. The tests that care about the difference say which they mean.
    """
    lineage = LineageWrite(
        entity=entity_key("fact", predicate="requires", subject="spring", object=obj),
        kind="fact", text=text, fingerprint=fingerprint or f"fp:{obj}",
        extraction_version="v1",
        document_id=DOCUMENT, document_revision=f"{run}-rev", document_path="visibility.adoc",
        slot_id=SLOT, run_id=run,
        proposition=Proposition({"subject": "spring", "predicate": "requires", "object": obj}),
    )
    graph = KnowledgeWrite(
        identity=lineage.fingerprint,
        payload=FactPayload(subject="spring", predicate="requires", object=obj),
        type="dependency", confidence=0.9, source_id="corpus", run_id=run,
        review_required=review,
        review_reason="awaiting a reviewer" if review else None,
    )
    return graph, lineage


async def _record(repository, claims, *, run: str, texts: list[str]):  # noqa: ANN001
    return await repository.record_document(
        document_id=DOCUMENT,
        sections=[ObservedSection.as_read(
            anchor=SLOT, anchor_strength=DERIVED_ANCHOR,
            content="|".join(sorted(texts)),
            unit_ordinal=SLOT, assertions=tuple(claims),
        )],
        run_id=run,
    )


def _store(url: str, connections: StoreConnections, index: _Index) -> KnowledgeStoreTool:
    """The real activation, reading the real database through its own pg_url."""
    tool = KnowledgeStoreTool(
        name="knowledge",
        config={"pg_url": url, "os_url": "http://opensearch:9200"},
        store_connections=connections,
    )
    # Only the external search service is replaced. The repository behind the
    # tool is the real one, built by the tool from its own configuration.
    tool._connections.client = lambda *args, **kwargs: index  # type: ignore[assignment]  # noqa: SLF001
    return tool


def _step(store: KnowledgeStoreTool) -> RetrievalSearchStep:
    step = RetrievalSearchStep(config={"knowledge_store": "knowledge", "kinds": ["fact"]})

    async def _activate(source: str, caller: Any = None) -> Any:  # noqa: ANN401
        return store if source == KNOWLEDGE else None

    step._activate = _activate  # type: ignore[method-assign]  # noqa: SLF001
    return step


async def _search(corpus, index_holds: list[str]):  # noqa: ANN001
    url, connections, _ = corpus
    index = _Index(identities=index_holds)
    result = await _step(_store(url, connections, index)).run(_ctx())
    return result.ctx.metadata["retrieval.hits"]


async def test_an_approved_claim_is_retrievable(corpus) -> None:
    _, _, repository = corpus
    await _record(repository, [_claim("Spring Boot requires Java 17.", "java 17")],
                  run="run-1", texts=["Spring Boot requires Java 17."])

    hits = await _search(corpus, ["fp:java 17"])

    assert [item.best.carrier_id for item in hits] == ["fp:java 17"]
    assert hits[0].best.carrier == ASSERTION


async def test_a_claim_awaiting_review_is_not_retrievable(corpus) -> None:
    """The index still holds it — the repository is what keeps it out.

    An index is eventually consistent with the graph, so a claim quarantined a
    second ago is still findable in it. That is exactly why the rule lives in
    `resolve()` and why this test hands the index the identity anyway.
    """
    _, _, repository = corpus
    await _record(
        repository,
        [_claim("Spring Boot requires Java 17.", "java 17"),
         _claim("Spring Boot requires Maven.", "maven", review=True)],
        run="run-1",
        texts=["Spring Boot requires Java 17.", "Spring Boot requires Maven."],
    )

    hits = await _search(corpus, ["fp:java 17", "fp:maven"])

    assert [item.best.carrier_id for item in hits] == ["fp:java 17"]


async def test_a_claim_merged_into_another_is_not_retrievable(corpus) -> None:
    # Merged away is the second way to stop being canonical, and retrieval must
    # see neither (ADR-0032).
    from sqlalchemy.ext.asyncio import AsyncSession

    url, connections, repository = corpus
    await _record(
        repository,
        [_claim("Spring Boot requires Java 17.", "java 17"),
         _claim("Spring Boot requires Maven.", "maven")],
        run="run-1",
        texts=["Spring Boot requires Java 17.", "Spring Boot requires Maven."],
    )
    from nlght.adapters.outbound.persistence.knowledge_models import Knowledge

    async with AsyncSession(connections.engine(url)) as session, session.begin():
        merged = await session.get(Knowledge, "fp:maven")
        merged.merged_into = "fp:java 17"

    hits = await _search(corpus, ["fp:java 17", "fp:maven"])

    assert [item.best.carrier_id for item in hits] == ["fp:java 17"]


async def test_only_visible_claims_reach_the_ranking(corpus) -> None:
    """The whole point, stated as the fusion sees it.

    Three identities come back from the index and one claim reaches the ranking.
    Nothing downstream has to know that the other two ever existed.
    """
    _, _, repository = corpus
    await _record(
        repository,
        [_claim("Spring Boot requires Java 17.", "java 17"),
         _claim("Spring Boot requires Maven.", "maven", review=True),
         _claim("Spring Boot requires Gradle.", "gradle", review=True)],
        run="run-1",
        texts=["a", "b", "c"],
    )

    hits = await _search(corpus, ["fp:java 17", "fp:maven", "fp:gradle"])

    assert len(hits) == 1
    assert all(item.best.source == KNOWLEDGE for item in hits)


async def test_a_corpus_of_only_invisible_claims_returns_nothing(corpus) -> None:
    # An empty knowledge result, not an error and not a leak.
    _, _, repository = corpus
    await _record(repository, [_claim("Spring Boot requires Maven.", "maven", review=True)],
                  run="run-1", texts=["Spring Boot requires Maven."])

    assert await _search(corpus, ["fp:maven"]) == ()


async def test_a_fingerprint_addresses_a_recorded_form_and_never_the_claim(corpus) -> None:
    """The guard on what `KnowledgeRevision.fingerprint == Knowledge.identity` means.

    It is the technical address of the graph record a revision wrote, and it must
    never drift back into meaning *the identity of the claim*. Those are the two
    things this model has kept apart from the beginning: a claim reworded by the
    source is one assertion at a new state, and its fingerprint moves while its
    `assertion_id` does not.

        revision 1  fingerprint P1  "must not exceed 30 seconds"
        revision 2  fingerprint P2  "must not exceed 60 seconds"

        P1 != P2, and the assertion is the same assertion throughout.
    """
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import AsyncSession

    from nlght.adapters.outbound.persistence.knowledge_models import (
        KnowledgeRevision,
        KnowledgeVariant,
    )

    url, connections, repository = corpus
    first = await _record(
        repository,
        [_claim("Spring Boot requires Java 17.", "java 17", fingerprint="fp:state-1")],
        run="run-1", texts=["Spring Boot requires Java 17."],
    )
    # The source says it differently and the model reads it differently, so the
    # structured claim moves: same business question, new recorded form.
    second = await _record(
        repository,
        [_claim("Spring Boot requires Java 17 or newer.", "java 17",
                run="run-2", fingerprint="fp:state-2")],
        run="run-2", texts=["Spring Boot requires Java 17 or newer."],
    )

    assert second.recorded[0].assertion_id == first.recorded[0].assertion_id
    assert second.recorded[0].revision == 2

    async with AsyncSession(connections.engine(url)) as session:
        fingerprints = (
            await session.execute(
                select(KnowledgeRevision.fingerprint)
                .join(KnowledgeVariant, KnowledgeVariant.variant_id == KnowledgeRevision.variant_id)
                .where(KnowledgeVariant.assertion_id == first.recorded[0].assertion_id)
                .order_by(KnowledgeRevision.revision)
            )
        ).scalars().all()

    assert len(set(fingerprints)) == 2, "one claim, two recorded forms"


async def test_a_superseded_state_is_not_retrievable(corpus) -> None:
    """A claim that moved on does not answer with what it used to say.

    A graph node is addressed by the fingerprint of the revision that wrote it,
    so a reworded claim leaves a *second* node behind carrying the old wording.
    Both were canonical, and a search could return

        "Spring Boot requires Java 17."
        "Spring Boot requires Java 17 or newer."

    as two live answers to one question. A state the corpus has moved past is not
    what the corpus says (ADR-0032), so only the current one is retrievable —
    while a review surface, asking for everything, still sees both.
    """
    _, _, repository = corpus
    await _record(
        repository,
        [_claim("Spring Boot requires Java 17.", "java 17", fingerprint="fp:state-1")],
        run="run-1", texts=["Spring Boot requires Java 17."],
    )
    await _record(
        repository,
        [_claim("Spring Boot requires Java 17 or newer.", "java 17",
                run="run-2", fingerprint="fp:state-2")],
        run="run-2", texts=["Spring Boot requires Java 17 or newer."],
    )

    # The index still holds both — it is not the place this is decided.
    hits = await _search(corpus, ["fp:state-1", "fp:state-2"])

    assert [item.best.content for item in hits] == ["Spring Boot requires Java 17 or newer."]


async def test_a_retired_claim_is_not_retrievable(corpus) -> None:
    """Retired by its source, and therefore gone from retrieval.

    The source stopped saying it, so a reader must not be handed it. This is the
    case the graph cannot answer on its own — retirement is recorded on the
    assertion, and `resolve()` filters on the graph's `review_required` and
    `merged_into` — so it is worth its own test rather than being assumed to
    follow from the other two.
    """
    _, _, repository = corpus
    await _record(
        repository,
        [_claim("Spring Boot requires Java 17.", "java 17"),
         _claim("Spring Boot requires Maven.", "maven")],
        run="run-1",
        texts=["Spring Boot requires Java 17.", "Spring Boot requires Maven."],
    )
    # The source drops the Maven sentence: the section changes, so the claim it
    # held is retired rather than merely unmentioned.
    await _record(repository, [_claim("Spring Boot requires Java 17.", "java 17", run="run-2")],
                  run="run-2", texts=["Spring Boot requires Java 17."])

    hits = await _search(corpus, ["fp:java 17", "fp:maven"])

    assert [item.best.carrier_id for item in hits] == ["fp:java 17"]
