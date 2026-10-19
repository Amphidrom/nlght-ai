# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Verify current document support for knowledge claims.

Evidence records past observations. Current support identifies documents that
still carry a claim after a complete reread (ADR-0044).
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from nlght.adapters.outbound.persistence.knowledge_models import (
    KnowledgeAssertion,
    KnowledgeDocumentSupport,
    KnowledgeEvidence,
)
from nlght.adapters.outbound.persistence.knowledge_repository import (
    SqlAlchemyKnowledgeRepository,
)
from nlght.core.knowledge import (
    DERIVED_ANCHOR,
    KnowledgeWrite,
    LineageWrite,
    ObservedSection,
    Proposition,
    RulePayload,
    entity_key,
)

pytestmark = pytest.mark.integration

RULE = "Expenses above CHF 500 require approval."
DOC_A = "doc-a"
DOC_B = "doc-b"


@pytest_asyncio.fixture
async def repository():
    from sqlalchemy import event
    from sqlalchemy.ext.asyncio import create_async_engine

    from nlght.adapters.outbound.persistence.knowledge_models import KnowledgeBase

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")

    @event.listens_for(engine.sync_engine, "connect")
    def _enforce_foreign_keys(dbapi_connection, _record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    async with engine.begin() as conn:
        await conn.run_sync(KnowledgeBase.metadata.create_all)
    repository = SqlAlchemyKnowledgeRepository(engine)
    repository._test_engine = engine
    yield repository
    await engine.dispose()


def _lineage(
    *,
    text: str = RULE,
    fingerprint: str = "fp-500",
    rule_property: str = "approval_threshold",
    document_id: str = DOC_A,
    document_path: str = "policies/expenses.adoc",
    run_id: str = "run-1",
    document_revision: str = "rev-1",
) -> LineageWrite:
    return LineageWrite(
        entity=entity_key("rule", subject="expense", rule_property=rule_property),
        kind="rule",
        scope=(),
        text=text,
        proposition=Proposition(
            {"rule_text": text, "subject": "expense", "rule_property": rule_property}
        ),
        fingerprint=fingerprint,
        extraction_version="ver-1",
        document_id=document_id,
        document_revision=document_revision,
        document_path=document_path,
        run_id=run_id,
    )


def _graph(identity: str = "fp-500", run_id: str = "run-1") -> KnowledgeWrite:
    return KnowledgeWrite(
        identity=identity,
        type="policy",
        payload=RulePayload(
            rule_text=RULE, subject="expense", rule_property="approval_threshold"
        ),
        confidence=0.9,
        source_id="corpus",
        run_id=run_id,
    )


async def _observe(
    repository,
    *lineages: LineageWrite,
    document_id: str = DOC_A,
    run_id: str = "run-1",
    anchor: str = "#threshold",
    content: str | None = None,
):
    """Record a complete document observation through the production path.

    Only a complete read may update current document support.

    Args:
        repository (SqlAlchemyKnowledgeRepository): Repository under test.
        *lineages (LineageWrite): Claims observed in this document.
        document_id (str): Stable document identity.
        run_id (str): Identity of the observation run.
        anchor (str): Section anchor used for the observation.
        content (str | None): Source text override for this observation.

    Returns:
        DocumentOutcome: Reconciliation result for the document.

    Side Effects:
        Persists the complete observation and updates current support.
    """
    sections = (
        [
            ObservedSection.as_read(
                anchor=anchor,
                anchor_strength=DERIVED_ANCHOR,
                content=content if content is not None else "|".join(
                    line.text for line in lineages
                ),
                unit_ordinal=anchor,
                assertions=tuple(
                    (_graph(line.fingerprint, run_id=run_id), line) for line in lineages
                ),
            )
        ]
        if lineages
        else []
    )
    return await repository.record_document(
        document_id=document_id, sections=sections, run_id=run_id
    )


async def _rows(repository, model):
    async with AsyncSession(repository._test_engine) as session:
        return list((await session.execute(select(model))).scalars().all())


# 1. A carries the state

async def test_a_document_that_carries_a_state_is_its_current_support(
    repository,
) -> None:
    await _observe(repository, _lineage())

    claim = (await repository.resolve(["fp-500"]))[0]

    assert claim.assertion_id and claim.assertion_id != claim.identity
    assert claim.revision_id
    assert [entry.document_id for entry in claim.support] == [DOC_A]
    assert claim.support[0].observed_document_revision == "rev-1"
    assert claim.support[0].document_path == "policies/expenses.adoc"
    assert claim.support[0].slot_id, "the section it stands in, as provenance"


async def test_re_observing_the_same_section_is_one_support_not_two(
    repository,
) -> None:
    # Support is replaced from what a run saw, not accumulated. A source read on
    # five runs is one citation, not five.
    await _observe(repository, _lineage())
    await _observe(repository, _lineage(run_id="run-2"), run_id="run-2")

    claim = (await repository.resolve(["fp-500"]))[0]

    assert len(claim.support) == 1
    assert len(await _rows(repository, KnowledgeEvidence)) == 2, "both sightings kept"


# 2. Two documents, and neither owns the claim

async def test_two_documents_both_stay_support_and_neither_becomes_the_owner(
    repository,
) -> None:
    await _observe(repository, _lineage())
    await _observe(
        repository,
        _lineage(document_id=DOC_B, document_path="handbook/spending.adoc",
                 run_id="run-2"),
        document_id=DOC_B, run_id="run-2", anchor="#spending",
    )

    claim = (await repository.resolve(["fp-500"]))[0]

    assert {entry.document_id for entry in claim.support} == {DOC_A, DOC_B}
    assert {entry.document_path for entry in claim.support} == {
        "policies/expenses.adoc", "handbook/spending.adoc",
    }


async def test_a_claim_carries_no_flat_document_of_its_own(repository) -> None:
    # `Provenance.document_id` is where a document or a chunk says which file it
    # is. For a claim it stays empty: filling it would mean choosing one of the
    # sources as the origin, and every citation afterwards would name it.
    from nlght.adapters.outbound.workflow.steps.retrieval.search import (
        RetrievalSearchStep,
    )

    await _observe(repository, _lineage())
    claim = (await repository.resolve(["fp-500"]))[0]

    hits = await RetrievalSearchStep._knowledge(
        RetrievalSearchStep(config={}),
        _StubStore([claim]), "expenses", _StubPlan(),
        lambda *_args, **_kwargs: None,
    )

    assert hits[0].provenance.document_id == "", "no document owns a claim"
    assert hits[0].provenance.assertion_id == claim.assertion_id
    assert hits[0].provenance.knowledge_revision_id == claim.revision_id
    assert [entry.document_id for entry in hits[0].provenance.support] == [DOC_A]


class _StubStore:
    def __init__(self, claims: list) -> None:
        self._claims = claims

    async def query_rules(self, **_kwargs: object) -> list:
        return self._claims


class _StubPlan:
    kinds = ("rule",)

    def fetch_for(self, _source: str) -> int:
        return 10


# 3. A document drops the claim. Another keeps it alive

async def test_a_document_that_dropped_the_claim_stops_being_current_support(
    repository,
) -> None:
    """Drop a document's current support when it no longer carries the claim.

    Preserve the assertion because another document still supports it. Keep the
    former sighting as historical evidence.
    """
    await _observe(repository, _lineage())
    await _observe(
        repository,
        _lineage(document_id=DOC_B, document_path="handbook/spending.adoc",
                 run_id="run-2"),
        document_id=DOC_B, run_id="run-2", anchor="#spending",
    )

    # Doc A read in full, and it no longer says it.
    outcome = await _observe(repository, document_id=DOC_A, run_id="run-3")

    assert outcome.retired == (), "doc B still carries it, so the claim lives"
    live = [row for row in await _rows(repository, KnowledgeAssertion)
            if row.retired_at is None]
    assert len(live) == 1

    claim = (await repository.resolve(["fp-500"]))[0]
    assert [entry.document_id for entry in claim.support] == [DOC_B]

    evidence = {row.document_id for row in await _rows(repository, KnowledgeEvidence)}
    assert evidence == {DOC_A, DOC_B}, "history is not rewritten"


# 4. A partial run says nothing about what a document carries

async def test_a_partial_run_never_removes_support(repository) -> None:
    """A run that did not see all of a document cannot say what it carries.

    `record_document` requires a complete observation. Partial persistence
    writes evidence but leaves current support unchanged.
    """
    await _observe(repository, _lineage())
    before = (await repository.resolve(["fp-500"]))[0].support

    # The partial path: a document written but never compared.
    await repository.persist(_graph(run_id="run-2"), _lineage(run_id="run-2"))

    claim = (await repository.resolve(["fp-500"]))[0]
    assert claim.support == before, "a partial reading removed nothing"


async def test_an_unchanged_section_keeps_its_support_when_the_model_stays_silent(
    repository,
) -> None:
    """Keep support when the stable-slot guard retains a claim.

    The stable-slot guard exists because a model asked twice about byte-identical
    text answered differently (ADR-0044). Dropping the support row for a claim
    the same run refused to retire would remove a valid citation because of
    model nondeterminism.
    """
    second = _lineage(
        text="Expenses above CHF 5000 require board approval.",
        fingerprint="fp-5000",
        # A different business question, so this is a second claim in the same
        # section and not a second state of the first.
        rule_property="board_approval_threshold",
    )
    section = f"{RULE}|{second.text}"
    await _observe(repository, _lineage(), second, content=section)
    before = (await repository.resolve(["fp-5000"]))[0].support
    assert before, "it was carried to begin with"

    # The same section, byte-identical, and the model returns only one of its
    # two claims this time.
    await _observe(
        repository, _lineage(run_id="run-2"), run_id="run-2", content=section,
    )

    claim = (await repository.resolve(["fp-5000"]))[0]
    assert claim.support == before, "the guard refused to retire it; support agrees"


# 5. A state the corpus has moved past keeps its own support

async def test_a_citation_of_the_canonical_state_uses_only_that_state_s_support(
    repository,
) -> None:
    moved = "Expenses above CHF 550 require approval."
    await _observe(repository, _lineage())
    await _observe(
        repository,
        _lineage(text=moved, fingerprint="fp-550", document_id=DOC_B,
                 document_path="handbook/spending.adoc", run_id="run-2"),
        document_id=DOC_B, run_id="run-2", anchor="#spending",
    )

    current = await repository.resolve(["fp-550"])
    superseded = await repository.resolve(["fp-500"])

    assert [entry.document_id for entry in current[0].support] == [DOC_B]
    assert superseded == [], "the older state is not canonical at all"


# 6. The quarantine view shows more, and means the same

async def test_including_quarantined_does_not_change_what_support_means(
    repository,
) -> None:
    """Keep support semantics identical in review and retrieval views.

    `include_quarantined` widens *which nodes* come back. It must not quietly
    widen what `support` is, or a reviewer and a reader would be looking at two
    different notions of "carried by" under one name.
    """
    await _observe(repository, _lineage())
    await _observe(
        repository,
        _lineage(document_id=DOC_B, document_path="handbook/spending.adoc",
                 run_id="run-2"),
        document_id=DOC_B, run_id="run-2", anchor="#spending",
    )
    await _observe(repository, document_id=DOC_A, run_id="run-3")

    plain = (await repository.resolve(["fp-500"]))[0]
    widened = (await repository.resolve(["fp-500"], include_quarantined=True))[0]

    assert widened.support == plain.support == (plain.support[0],)
    assert plain.support[0].document_id == DOC_B


# The wording, and the passage

async def test_the_wording_comes_from_the_revision_and_not_from_the_support(
    repository,
) -> None:
    # ADR-0048, unchanged by any of this: the sentence is read from the state it
    # was recorded on. Support says *where* it is carried and never *what* it says.
    await _observe(repository, _lineage())

    claim = (await repository.resolve(["fp-500"]))[0]

    assert claim.observed_text == RULE
    assert not hasattr(claim.support[0], "text")


async def test_a_passage_carries_the_support_unchanged(repository) -> None:
    from nlght.core.context import ContextBudget, build_context
    from nlght.core.retrieval import (
        ASSERTION,
        KNOWLEDGE,
        Provenance,
        RetrievalHit,
        fuse,
        rank_within_sources,
    )

    await _observe(repository, _lineage())
    await _observe(
        repository,
        _lineage(document_id=DOC_B, document_path="handbook/spending.adoc",
                 run_id="run-2"),
        document_id=DOC_B, run_id="run-2", anchor="#spending",
    )
    claim = (await repository.resolve(["fp-500"]))[0]

    hit = RetrievalHit(
        source=KNOWLEDGE, carrier=ASSERTION, carrier_id=claim.identity,
        content=claim.observed_text,
        provenance=Provenance(
            assertion_id=claim.assertion_id,
            knowledge_revision_id=claim.revision_id,
            support=claim.support,
        ),
    )
    selection = build_context(
        fuse(rank_within_sources([hit])), ContextBudget(max_chars=500)
    )

    assert selection.passages[0].provenance.support == claim.support
    assert len(selection.passages[0].provenance.support) == 2


async def test_support_rows_are_replaced_and_not_accumulated(repository) -> None:
    # The table is a projection of a current state, so it must not grow with
    # every run. A row per run would be the evidence table again, under a name
    # that promises the opposite.
    for run in ("run-1", "run-2", "run-3"):
        await _observe(repository, _lineage(run_id=run), run_id=run)

    assert len(await _rows(repository, KnowledgeDocumentSupport)) == 1


async def test_the_report_counts_claims_nothing_currently_carries(repository) -> None:
    """The operator number for migration 0019.

    Claims without a complete current observation resolve without a citable
    source. The report counts them until a complete reread establishes support.
    """
    # The partial path: evidence and a graph node, and nothing saying a document
    # still carries it.
    await repository.persist(_graph(), _lineage())

    report = await repository.lineage_report()
    assert report.claims_without_current_support == 1
    assert "claims_without_current_support" in report.as_summary()

    # A complete observation settles it.
    await _observe(repository, _lineage(run_id="run-2"), run_id="run-2")

    assert (await repository.lineage_report()).claims_without_current_support == 0


async def test_an_empty_section_is_a_shape_production_does_not_produce(
    repository,
) -> None:
    """Pin the effect of an empty section supplied outside the production path.

    `knowledge.persist` sends an empty section list when there are no candidates.
    A supplied empty section has no extraction version, so the stable-slot guard
    cannot retain its claims. Recording it retires the claims and their support.
    """
    await _observe(repository, _lineage())
    assert (await repository.resolve(["fp-500"]))[0].support

    outcome = await repository.record_document(
        document_id=DOC_A,
        sections=[
            ObservedSection.as_read(
                anchor="#threshold", anchor_strength=DERIVED_ANCHOR,
                content=RULE, unit_ordinal="#threshold", assertions=(),
            )
        ],
        run_id="run-2",
    )

    assert len(outcome.retired) == 1, "a section that says nothing held nothing"
    assert await repository.resolve(["fp-500"]) == []
    assert await _rows(repository, KnowledgeDocumentSupport) == []
    # And the sighting is still there: history is never rewritten.
    assert len(await _rows(repository, KnowledgeEvidence)) == 1
