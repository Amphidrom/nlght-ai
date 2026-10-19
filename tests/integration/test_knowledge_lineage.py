# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Verify assertion, variant, revision, and evidence lineage.

The tests use persisted rows to check continuity, approval, provenance, and
graph reachability across complete observations.
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from nlght.adapters.outbound.persistence.knowledge_models import (
    Knowledge,
    KnowledgeAssertion,
    KnowledgeEvidence,
    KnowledgeRevision,
    KnowledgeVariant,
)
from nlght.adapters.outbound.persistence.knowledge_repository import (
    SqlAlchemyKnowledgeRepository,
)
from nlght.core.knowledge import LineageWrite, Proposition, entity_key

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def sqlite_engine():
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
    yield engine
    await engine.dispose()


def _write(
    *,
    text: str = "Expenses above CHF 500 require approval.",
    fingerprint: str = "fp-500",
    scope: tuple[str, ...] = ("employee",),
    document_revision: str = "rev-1",
    run_id: str = "run-1",
    normative_change: bool = False,
) -> LineageWrite:
    return LineageWrite(
        entity=entity_key("rule", subject="expense", rule_property="approval_threshold"),
        kind="rule",
        scope=scope,
        text=text,
        proposition=Proposition(
            {"rule_text": text, "subject": "expense",
             "rule_property": "approval_threshold"}
        ),
        fingerprint=fingerprint,
        extraction_version="ver-1",
        document_id="doc-1",
        document_revision=document_revision,
        slot_id="slot-1",
        run_id=run_id,
        normative_change=normative_change,
    )


async def _rows(engine, model):
    async with AsyncSession(engine) as session:
        return list((await session.execute(select(model))).scalars().all())


async def test_a_new_claim_creates_the_whole_chain(sqlite_engine) -> None:
    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)

    recorded = await repository.record_lineage(_write())

    assert recorded.revision == 1
    assert recorded.review_state == "approved"
    assert len(await _rows(sqlite_engine, KnowledgeAssertion)) == 1
    assert len(await _rows(sqlite_engine, KnowledgeVariant)) == 1
    assert len(await _rows(sqlite_engine, KnowledgeRevision)) == 1
    assert len(await _rows(sqlite_engine, KnowledgeEvidence)) == 1


async def test_the_assertion_id_is_a_surrogate_not_the_entity_key(sqlite_engine) -> None:
    # The property everything rests on. A key derived from content would move
    # when the content moves, taking the approval with it.
    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)
    write = _write()

    recorded = await repository.record_lineage(write)

    assert recorded.assertion_id != write.entity.key
    assert recorded.assertion_id.startswith("a_")


async def test_an_unchanged_claim_does_not_grow_a_second_revision(sqlite_engine) -> None:
    """The acceptance metric, at the table that would show a violation.

    A second run over an unchanged source must leave the corpus alone. If it
    wrote a revision anyway, the history would fill with states nothing
    distinguishes and the diff would have nothing to compare against.
    """
    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)
    first = await repository.record_lineage(_write())

    second = await repository.record_lineage(_write(run_id="run-2"))

    assert second.assertion_id == first.assertion_id
    assert second.revision == 1
    assert len(await _rows(sqlite_engine, KnowledgeRevision)) == 1


async def test_a_second_run_records_that_it_saw_the_claim_again(sqlite_engine) -> None:
    # No new revision, but a new sighting: the evidence is how a later run knows
    # document D still asserts this, which is what licenses *not* retracting it.
    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)
    await repository.record_lineage(_write(run_id="run-1"))

    await repository.record_lineage(_write(run_id="run-2", document_revision="rev-2"))

    evidence = await _rows(sqlite_engine, KnowledgeEvidence)
    assert len(evidence) == 2
    assert {row.document_revision for row in evidence} == {"rev-1", "rev-2"}


async def test_a_retried_run_does_not_record_the_same_sighting_twice(sqlite_engine) -> None:
    # Same run, same document: one sighting seen twice because a worker crashed.
    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)
    await repository.record_lineage(_write(run_id="run-1"))

    await repository.record_lineage(_write(run_id="run-1"))

    assert len(await _rows(sqlite_engine, KnowledgeEvidence)) == 1


async def test_a_reworded_rule_continues_the_same_assertion(sqlite_engine) -> None:
    """Continue a rule assertion when its wording changes.

    A rule's subject and property identify it. A wording revision preserves the
    assertion and its approval when the normative value is unchanged.
    """
    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)
    first = await repository.record_lineage(_write())

    second = await repository.record_lineage(
        _write(text="Expenses over CHF 500 need signing off.", fingerprint="fp-500b")
    )

    assert second.assertion_id == first.assertion_id
    assert second.revision == 2
    assert second.supersedes == 1
    assert len(await _rows(sqlite_engine, KnowledgeAssertion)) == 1


async def test_a_rewording_keeps_the_approval(sqlite_engine) -> None:
    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)
    await repository.record_lineage(_write())

    second = await repository.record_lineage(
        _write(text="Expenses over CHF 500 need signing off.", fingerprint="fp-500b")
    )

    assert second.review_state == "approved"


async def test_a_normative_change_keeps_the_assertion_and_drops_the_approval(
    sqlite_engine,
) -> None:
    """Keep rule identity but clear approval after a normative value change.

    Changing CHF 500 to CHF 550 alters the approved threshold.
    """
    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)
    first = await repository.record_lineage(_write())

    second = await repository.record_lineage(
        _write(
            text="Expenses above CHF 550 require approval.",
            fingerprint="fp-550",
            normative_change=True,
        )
    )

    assert second.assertion_id == first.assertion_id
    assert second.review_state == "needs_review"


async def test_two_scopes_holding_at_once_are_two_variants_of_one_assertion(
    sqlite_engine,
) -> None:
    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)
    employees = await repository.record_lineage(_write(scope=("employee",)))

    executives = await repository.record_lineage(
        _write(scope=("executive",), text="Expenses above CHF 5000 require approval.",
               fingerprint="fp-5000")
    )

    assert executives.assertion_id == employees.assertion_id
    assert executives.variant_id != employees.variant_id
    assert len(await _rows(sqlite_engine, KnowledgeAssertion)) == 1
    assert len(await _rows(sqlite_engine, KnowledgeVariant)) == 2


async def test_each_variant_keeps_its_own_revisions(sqlite_engine) -> None:
    # Two cases, each with its own history. A shared revision counter would make
    # one case's edit look like a state of the other.
    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)
    await repository.record_lineage(_write(scope=("employee",)))
    await repository.record_lineage(
        _write(scope=("executive",), text="CHF 5000.", fingerprint="fp-5000")
    )

    changed = await repository.record_lineage(
        _write(scope=("employee",), text="CHF 550.", fingerprint="fp-550")
    )

    assert changed.revision == 2


async def test_an_ambiguous_scope_opens_a_variant_and_says_so(sqlite_engine) -> None:
    # Two possible widenings: nothing is claimed, and the row records that the
    # variant was arrived at ambiguously so somebody can look at it.
    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)
    await repository.record_lineage(_write(scope=("employee",)))
    await repository.record_lineage(
        _write(scope=("contractor",), text="CHF 500 for contractors.", fingerprint="fp-c")
    )

    both = await repository.record_lineage(
        _write(scope=("employee", "contractor"), text="CHF 500 for both.", fingerprint="fp-b")
    )

    variants = {row.variant_id: row for row in await _rows(sqlite_engine, KnowledgeVariant)}
    assert len(variants) == 3
    assert variants[both.variant_id].resolution == "ambiguous"


#
#     entity, variant, revision, and evidence are committed together
#
# A crash between any two of them must leave nothing behind. Half a lineage is
# worse than none: an assertion with no revision claims a business question
# nothing says anything about, and a revision with no evidence claims a state
# nothing was ever seen to assert, and the diff would then read that absence as
# "the document stopped saying this" and retract it.


class _Crash(RuntimeError):
    """Stands in for a worker dying mid-transaction."""


def _crash_on_insert(engine, table: str):
    """Fail the first INSERT into one table, as a crash at that step would.

    Args:
        engine (Any): Database engine used by the test.
        table (str): Table whose insert is intercepted.
    """
    from contextlib import contextmanager

    from sqlalchemy import event

    @contextmanager
    def _installed():
        def _listener(conn, cursor, statement, parameters, context, executemany):
            if statement.lstrip().upper().startswith("INSERT INTO") and table in statement:
                raise _Crash(f"crashed before writing {table}")

        event.listen(engine.sync_engine, "before_cursor_execute", _listener)
        try:
            yield
        finally:
            event.remove(engine.sync_engine, "before_cursor_execute", _listener)

    return _installed()


async def _counts(engine) -> dict[str, int]:
    return {
        "assertions": len(await _rows(engine, KnowledgeAssertion)),
        "variants": len(await _rows(engine, KnowledgeVariant)),
        "revisions": len(await _rows(engine, KnowledgeRevision)),
        "evidence": len(await _rows(engine, KnowledgeEvidence)),
    }


@pytest.mark.parametrize(
    "table",
    ["knowledge_assertions", "knowledge_variants", "knowledge_revisions", "knowledge_evidence"],
)
async def test_a_crash_at_any_step_leaves_nothing_behind(sqlite_engine, table: str) -> None:
    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)

    with _crash_on_insert(sqlite_engine, table), pytest.raises(_Crash):
        await repository.record_lineage(_write())

    assert await _counts(sqlite_engine) == {
        "assertions": 0, "variants": 0, "revisions": 0, "evidence": 0,
    }


@pytest.mark.parametrize(
    "table",
    ["knowledge_assertions", "knowledge_variants", "knowledge_revisions", "knowledge_evidence"],
)
async def test_the_retry_after_a_crash_produces_one_clean_lineage(
    sqlite_engine, table: str
) -> None:
    """The run is retried under the same identity and finishes the job.

    Exactly one of everything: not a second assertion for the quantity, not a
    second revision for a state that was never committed, and not a duplicate
    sighting.
    """
    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)
    with _crash_on_insert(sqlite_engine, table), pytest.raises(_Crash):
        await repository.record_lineage(_write())

    recorded = await repository.record_lineage(_write())

    assert recorded.revision == 1
    assert await _counts(sqlite_engine) == {
        "assertions": 1, "variants": 1, "revisions": 1, "evidence": 1,
    }


async def test_a_retry_of_a_run_that_succeeded_changes_nothing(sqlite_engine) -> None:
    # The other crash: the lineage committed and the worker died before the
    # execution was marked done. The retry sees the same claim under the same
    # run and must add nothing at all.
    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)
    first = await repository.record_lineage(_write())
    before = await _counts(sqlite_engine)

    second = await repository.record_lineage(_write())

    assert (second.assertion_id, second.variant_id, second.revision) == (
        first.assertion_id, first.variant_id, first.revision
    )
    assert await _counts(sqlite_engine) == before


async def test_the_same_key_under_another_scheme_is_refused(sqlite_engine) -> None:
    """Reject a key produced by a different identity scheme.

    Accepting it would merge incompatible meanings. Ignoring it would duplicate
    the same business question instead of requiring an explicit migration.
    """
    from nlght.core.knowledge import EntityKey, EntityKeyVersionMismatch

    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)
    write = _write()
    await repository.record_lineage(write)

    with pytest.raises(EntityKeyVersionMismatch, match="1"):
        await repository.record_lineage(
            LineageWrite(
                entity=EntityKey(version="2", key=write.entity.key),
                kind=write.kind, text=write.text, fingerprint=write.fingerprint,
                proposition=write.proposition,
                extraction_version=write.extraction_version,
                document_id=write.document_id, document_revision=write.document_revision,
                run_id="run-2", scope=write.scope,
            )
        )

    assert len(await _rows(sqlite_engine, KnowledgeAssertion)) == 1


async def test_a_different_key_under_another_scheme_is_simply_new(sqlite_engine) -> None:
    # The version alone is not the problem. A genuinely different quantity
    # under a newer scheme is a new assertion like any other.
    from nlght.core.knowledge import EntityKey

    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)
    await repository.record_lineage(_write())

    await repository.record_lineage(
        LineageWrite(
            entity=EntityKey(version="2", key="a-different-quantity"),
            kind="rule", text="Travel must be booked centrally.", fingerprint="fp-travel",
            proposition=Proposition({"rule_text": "Travel must be booked centrally."}),
            extraction_version="ver-1", document_id="doc-1", document_revision="rev-1",
            run_id="run-2",
        )
    )

    assert len(await _rows(sqlite_engine, KnowledgeAssertion)) == 2


#
# While identity is being migrated, both the graph and the lineage are written.
# Two transactions would let one land without the other, and the two records of
# one assertion would then disagree when telling which
# of them is right is hardest. So they commit together or not at all.


def _knowledge_write(identity: str = "fp-500", run_id: str = "run-1"):
    """Build the graph node addressed by the revision fingerprint.

    Production persistence derives both identities from the same content hash.
    Keeping them aligned makes the node reachable from its lineage.

    Args:
        identity (str): Claim identity used for the write.
        run_id (str): Ingestion run identifier.
    """
    from nlght.core.knowledge import KnowledgeWrite, RulePayload

    return KnowledgeWrite(
        identity=identity, type="security",
        payload=RulePayload(
            rule_text="Expenses above CHF 500 require approval.",
            subject="expense", rule_property="approval_threshold",
        ),
        confidence=0.9, source_id="source-1", run_id=run_id,
    )


async def _graph_rows(engine):
    from nlght.adapters.outbound.persistence.knowledge_models import Knowledge

    return await _rows(engine, Knowledge)


async def test_the_graph_and_the_lineage_are_written_together(sqlite_engine) -> None:
    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)

    node, recorded = await repository.persist(_knowledge_write(), _write())

    assert node.identity == "fp-500"
    assert recorded is not None
    assert len(await _graph_rows(sqlite_engine)) == 1
    assert await _counts(sqlite_engine) == {
        "assertions": 1, "variants": 1, "revisions": 1, "evidence": 1,
    }


async def test_a_failure_in_the_lineage_rolls_back_the_graph(sqlite_engine) -> None:
    """The direction that would leave an assertion with no history.

    Without the shared transaction the graph would hold a claim the lineage has
    never heard of, so the diff would find no evidence for it and read that as
    the document having stopped asserting it.
    """
    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)

    with _crash_on_insert(sqlite_engine, "knowledge_revisions"), pytest.raises(_Crash):
        await repository.persist(_knowledge_write(), _write())

    assert await _graph_rows(sqlite_engine) == []
    assert await _counts(sqlite_engine) == {
        "assertions": 0, "variants": 0, "revisions": 0, "evidence": 0,
    }


async def test_a_failure_in_the_graph_rolls_back_the_lineage(sqlite_engine) -> None:
    """Roll back lineage when its graph write fails.

    A recorded assertion without a graph node would appear to exist in lineage
    while remaining invisible to retrieval.
    """
    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)

    with _crash_on_insert(sqlite_engine, "knowledge_rules"), pytest.raises(_Crash):
        await repository.persist(_knowledge_write(), _write())

    assert await _graph_rows(sqlite_engine) == []
    assert await _counts(sqlite_engine) == {
        "assertions": 0, "variants": 0, "revisions": 0, "evidence": 0,
    }


async def test_the_retry_after_either_failure_writes_both_once(sqlite_engine) -> None:
    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)
    with _crash_on_insert(sqlite_engine, "knowledge_evidence"), pytest.raises(_Crash):
        await repository.persist(_knowledge_write(), _write())

    node, recorded = await repository.persist(_knowledge_write(), _write())

    assert node.confidence == pytest.approx(0.9)
    assert recorded is not None and recorded.revision == 1
    assert len(await _graph_rows(sqlite_engine)) == 1
    assert await _counts(sqlite_engine) == {
        "assertions": 1, "variants": 1, "revisions": 1, "evidence": 1,
    }


async def test_an_assertion_without_an_entity_key_writes_only_the_graph(
    sqlite_engine,
) -> None:
    """Avoid inventing lineage identity for a claim without an entity key.

    Wording cannot provide a stable business key across later model runs.
    """
    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)

    node, recorded = await repository.persist(_knowledge_write(), None)

    assert node.identity == "fp-500"
    assert recorded is None
    assert len(await _graph_rows(sqlite_engine)) == 1
    assert await _counts(sqlite_engine) == {
        "assertions": 0, "variants": 0, "revisions": 0, "evidence": 0,
    }


async def test_the_report_of_an_empty_corpus_says_so(sqlite_engine) -> None:
    report = await SqlAlchemyKnowledgeRepository(sqlite_engine).lineage_report()

    assert report.assertions == 0
    assert report.state_digest == ""


async def test_a_second_run_over_an_unchanged_claim_leaves_the_report_standing(
    sqlite_engine,
) -> None:
    """Keep revision counts stable across repeated observations of one claim.

    Evidence gains a new sighting, while the assertion and revision remain
    unchanged.
    """
    from nlght.core.knowledge import compare

    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)
    await repository.record_lineage(_write(run_id="run-1"))
    baseline = await repository.lineage_report()

    await repository.record_lineage(_write(run_id="run-2"))
    after = await repository.lineage_report()

    result = compare(baseline, after)
    assert result.unchanged, result.violations
    assert after.evidence == baseline.evidence + 1
    assert after.state_digest == baseline.state_digest


async def test_a_reworded_claim_shows_up_as_drift_in_the_report(sqlite_engine) -> None:
    # What a real rerun would look like if the model rephrased: one variant
    # gains a revision, and the comparison names it rather than leaving it in
    # a count somebody has to notice.
    from nlght.core.knowledge import compare

    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)
    await repository.record_lineage(_write(run_id="run-1"))
    baseline = await repository.lineage_report()

    await repository.record_lineage(
        _write(run_id="run-2", text="Expenses over CHF 500 need signing off.",
               fingerprint="fp-500b")
    )
    after = await repository.lineage_report()

    result = compare(baseline, after)
    assert not result.unchanged
    assert after.variants_beyond_first_revision == 1
    # The document's revision did not move, so nobody edited it and this is
    # drift. Name the document and slot where it happened instead of leaving it
    # in a count somebody has to notice.
    assert result.mode == "unchanged-source"
    assert [move.kind for move in result.movements] == ["revision added"]
    assert result.movements[0].expected is False
    assert "document_id=doc-1" in result.violations[0]


async def test_the_report_counts_graph_rows_without_lineage(sqlite_engine) -> None:
    # A rule persisted without the identifying fields reaches the graph and not
    # the lineage. Counting the gap per kind is what says whether the prompt or
    # the schema is at fault.
    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)
    await repository.persist(_knowledge_write(), None)

    report = await repository.lineage_report()

    assert report.graph_assertions == 1
    assert report.graph_without_lineage == 1
    assert report.graph_without_lineage_by_kind == {"rule": 1}
    # Not the other empty pointer: this node has no lineage at all, which is a
    # permanent condition and not an unfinished migration.
    assert report.graph_with_unresolved_revision == 0


async def test_the_report_counts_unresolved_migrations_apart(sqlite_engine) -> None:
    """Count graph nodes whose revision identity migration remains unresolved.

    They are withheld from retrieval and must remain distinguishable from nodes
    that never had business-key lineage.
    """
    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)
    stored, _ = await repository.persist(_knowledge_write(), _write())

    async with AsyncSession(sqlite_engine) as session, session.begin():
        node = await session.get(Knowledge, stored.identity)
        node.revision_id = None
        node.revision_unresolved = True

    report = await repository.lineage_report()

    assert report.graph_with_unresolved_revision == 1
    assert report.graph_without_lineage == 0, "lineage exists; only the relation is unproven"
    assert "graph_with_unresolved_revision" in report.as_summary()


async def test_a_revision_does_not_look_like_a_candidate_without_lineage(
    sqlite_engine,
) -> None:
    """Count unreachable graph nodes by identity rather than row totals.

    One assertion may legitimately own a graph node for each revision.
    """
    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)
    await repository.persist(_knowledge_write(), _write())
    await repository.persist(
        _knowledge_write(identity="fp-550", run_id="run-2"),
        _write(text="Expenses above CHF 550 require approval.", fingerprint="fp-550",
               run_id="run-2"),
    )

    report = await repository.lineage_report()

    # Two graph rows and one assertion at two revisions. Nothing is unreachable.
    assert report.graph_assertions == 2
    assert report.assertions == 1
    assert report.graph_without_lineage == 0


async def test_a_redescribed_claim_leaves_no_orphan_in_the_graph(sqlite_engine) -> None:
    """Address a redescribed claim's graph node through its resolved revision.

    Hashing the candidate's changed description would write an orphan node that
    lineage cannot reach or retract.
    """
    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)
    await repository.persist(_knowledge_write(), _write())

    # The same rule, same body, described with a different subject.
    stored, recorded = await repository.persist(
        _knowledge_write(identity="fp-500-redescribed", run_id="run-2"),
        _write(fingerprint="fp-500-redescribed", run_id="run-2"),
    )

    report = await repository.lineage_report()
    assert recorded is not None
    # No new state: the body did not move.
    assert recorded.revision == 1
    # And the node kept the address of the state that is actually recorded.
    assert stored.identity == "fp-500"
    assert report.graph_assertions == 1
    assert report.graph_without_lineage == 0


async def test_a_graph_row_no_revision_points_at_is_a_gap(sqlite_engine) -> None:
    # The real thing being counted: a claim stored in the graph that the lineage
    # cannot reach, so the diff will never see it.
    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)
    await repository.persist(_knowledge_write(), None)

    report = await repository.lineage_report()

    assert report.graph_without_lineage == 1
    assert report.graph_without_lineage_by_kind == {"rule": 1}


async def test_a_graph_row_with_its_lineage_is_not_counted_as_a_gap(sqlite_engine) -> None:
    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)
    await repository.persist(_knowledge_write(), _write())

    report = await repository.lineage_report()

    assert report.graph_without_lineage == 0
    assert report.entity_key_versions == {"1": 1}


#
#     A is replaced by B.
#     A splits into B and C.
#     A and B merge into C.
#
# An edge rather than a column, because a `superseded_by` field can say the
# first of those and neither of the others, and a corpus of policies does both
# routinely.


from nlght.adapters.outbound.persistence.knowledge_models import (  # noqa: E402
    KnowledgeAssertionLineage,
)
from nlght.core.knowledge import EntityKey  # noqa: E402


def _other(key: str, text: str, run_id: str = "run-2") -> LineageWrite:
    return LineageWrite(
        entity=EntityKey(version="1", key=key),
        kind="fact", text=text, fingerprint=f"fp-{key}",
        proposition=Proposition({"subject": key, "predicate": "asserts", "object": text}),
        extraction_version="ver-1", document_id="doc-1",
        document_revision="rev-2", run_id=run_id,
    )


async def test_one_assertion_can_take_another_s_place(sqlite_engine) -> None:
    """Record succession between two distinct assertions explicitly.

    A platform cannot infer that two different CEO propositions replace each
    other without domain knowledge.
    """
    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)
    alice = await repository.record_lineage(_write(text="CEO is Alice.", fingerprint="fp-a"))
    bob = await repository.record_lineage(_other("ceo-bob", "CEO is Bob."))

    written = await repository.record_supersession(
        predecessors=[alice.assertion_id], successors=[bob.assertion_id],
        reason="the page names a new chief executive", recorded_by="reviewer-1",
    )

    assert written == 1
    assert await repository.supersessions_of(alice.assertion_id) == (bob.assertion_id,)
    rows = {row.assertion_id: row for row in await _rows(sqlite_engine, KnowledgeAssertion)}
    assert rows[alice.assertion_id].retired_at is not None
    # The predecessor stays: deleting it would take its evidence and its review
    # history with it, and a citation naming it would find nothing.
    assert rows[bob.assertion_id].retired_at is None


async def test_one_assertion_can_be_split_into_two(sqlite_engine) -> None:
    # A rule pulled apart into two. A `superseded_by` column could not say it.
    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)
    original = await repository.record_lineage(_write(fingerprint="fp-o"))
    first = await repository.record_lineage(_other("split-a", "First half."))
    second = await repository.record_lineage(_other("split-b", "Second half."))

    written = await repository.record_supersession(
        predecessors=[original.assertion_id],
        successors=[first.assertion_id, second.assertion_id],
    )

    assert written == 2
    assert set(await repository.supersessions_of(original.assertion_id)) == {
        first.assertion_id, second.assertion_id
    }


async def test_two_assertions_can_be_folded_into_one(sqlite_engine) -> None:
    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)
    first = await repository.record_lineage(_write(fingerprint="fp-1"))
    second = await repository.record_lineage(_other("merge-b", "The other one."))
    merged = await repository.record_lineage(_other("merge-c", "Both, together."))

    written = await repository.record_supersession(
        predecessors=[first.assertion_id, second.assertion_id],
        successors=[merged.assertion_id],
    )

    assert written == 2
    rows = {row.assertion_id: row for row in await _rows(sqlite_engine, KnowledgeAssertion)}
    assert rows[first.assertion_id].retired_at is not None
    assert rows[second.assertion_id].retired_at is not None
    assert rows[merged.assertion_id].retired_at is None


async def test_the_edge_carries_who_recorded_it_and_why(sqlite_engine) -> None:
    # A succession nobody can account for later is one nobody can undo either.
    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)
    alice = await repository.record_lineage(_write(fingerprint="fp-a"))
    bob = await repository.record_lineage(_other("ceo-bob", "CEO is Bob."))

    await repository.record_supersession(
        predecessors=[alice.assertion_id], successors=[bob.assertion_id],
        reason="the page names a new chief executive", recorded_by="reviewer-1",
    )

    edge = (await _rows(sqlite_engine, KnowledgeAssertionLineage))[0]
    assert edge.relation == "supersedes"
    assert edge.reason == "the page names a new chief executive"
    assert edge.recorded_by == "reviewer-1"


async def test_recording_the_same_supersession_again_writes_nothing(sqlite_engine) -> None:
    # A retry after a crash is not a second event.
    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)
    alice = await repository.record_lineage(_write(fingerprint="fp-a"))
    bob = await repository.record_lineage(_other("ceo-bob", "CEO is Bob."))
    await repository.record_supersession(
        predecessors=[alice.assertion_id], successors=[bob.assertion_id]
    )

    written = await repository.record_supersession(
        predecessors=[alice.assertion_id], successors=[bob.assertion_id]
    )

    assert written == 0
    assert len(await _rows(sqlite_engine, KnowledgeAssertionLineage)) == 1


async def test_an_assertion_cannot_supersede_itself(sqlite_engine) -> None:
    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)
    recorded = await repository.record_lineage(_write())

    with pytest.raises(ValueError, match="itself"):
        await repository.record_supersession(
            predecessors=[recorded.assertion_id], successors=[recorded.assertion_id]
        )


async def test_superseding_something_that_is_not_there_is_refused(sqlite_engine) -> None:
    # An edge to a missing assertion would read as "replaced" while nothing
    # replaced it.
    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)
    recorded = await repository.record_lineage(_write())

    with pytest.raises(ValueError, match="no assertion"):
        await repository.record_supersession(
            predecessors=[recorded.assertion_id], successors=["a_nothing"]
        )


async def test_a_supersession_needs_both_sides(sqlite_engine) -> None:
    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)
    recorded = await repository.record_lineage(_write())

    with pytest.raises(ValueError, match="each side"):
        await repository.record_supersession(
            predecessors=[recorded.assertion_id], successors=[]
        )


#
# Three scenarios, and they end differently on purpose. The distinction is what
# an assertion *is* for each kind:
#
#     fact   identity is the whole proposition
#     rule   identity is subject + property, the wording is the body
#
# So a rewording writes a revision for a rule and none for a fact, and a fact
# whose proposition moved is simply a different assertion.


def _fact_lineage(
    key: str, text: str, fingerprint: str, *, document_revision: str, run_id: str
) -> LineageWrite:
    from nlght.core.knowledge import EntityKey

    return LineageWrite(
        entity=EntityKey(version="1", key=key),
        kind="fact", text=text, fingerprint=fingerprint,
        proposition=Proposition({"subject": key, "predicate": "asserts", "object": text}),
        extraction_version="ver-1", document_id="doc-1",
        document_revision=document_revision, run_id=run_id, slot_id="slot-1",
    )


async def test_a_reworded_source_writes_a_revision_and_keeps_the_proposition(
    sqlite_engine,
) -> None:
    """Record new observed wording without changing the fact proposition.

    The same assertion and fingerprint gain a revision for the new source text.
    Approval carries because the structured claim is unchanged.
    """
    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)
    first = await repository.record_lineage(
        _fact_lineage(
            "requires-spring-maven", "Spring benötigt Maven.", "fp-spring-maven",
            document_revision="rev-1", run_id="run-1",
        )
    )

    second = await repository.record_lineage(
        _fact_lineage(
            "requires-spring-maven", "Maven wird von Spring benötigt.", "fp-spring-maven",
            document_revision="rev-2", run_id="run-2",
        )
    )

    assert second.assertion_id == first.assertion_id
    assert second.revision == 2
    assert second.review_state == "approved", "a rewording does not spend an approval"

    revisions = sorted(
        await _rows(sqlite_engine, KnowledgeRevision), key=lambda row: row.revision
    )
    assert [row.text for row in revisions] == [
        "Spring benötigt Maven.",
        "Maven wird von Spring benötigt.",
    ]
    # Both forms of one claim, so "what did the source say when revision 1 was
    # written" stays answerable, which is what the history is for.
    assert len({row.fingerprint for row in revisions}) == 1

    evidence = await _rows(sqlite_engine, KnowledgeEvidence)
    assert {row.document_revision for row in evidence} == {"rev-1", "rev-2"}


async def test_a_materially_changed_fact_is_a_different_assertion(sqlite_engine) -> None:
    """Create a new assertion for a materially different fact.

    The old assertion retains its evidence and review history. No succession is
    inferred from wording alone.
    """
    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)
    seventeen = await repository.record_lineage(
        _fact_lineage(
            "requires-spring-java-17", "Spring requires Java 17.", "fp-java-17",
            document_revision="rev-1", run_id="run-1",
        )
    )

    twentyone = await repository.record_lineage(
        _fact_lineage(
            "requires-spring-java-21", "Spring requires Java 21.", "fp-java-21",
            document_revision="rev-2", run_id="run-2",
        )
    )

    assert twentyone.assertion_id != seventeen.assertion_id
    assert len(await _rows(sqlite_engine, KnowledgeAssertion)) == 2
    assert await repository.supersessions_of(seventeen.assertion_id) == ()
    rows = {row.assertion_id: row for row in await _rows(sqlite_engine, KnowledgeAssertion)}
    assert rows[seventeen.assertion_id].retired_at is None


async def test_a_reworded_rule_does_write_a_revision(sqlite_engine) -> None:
    """Write a revision when a rule is reworded without changing its key.

    The same subject and property preserve the assertion and its approval.
    """
    repository = SqlAlchemyKnowledgeRepository(sqlite_engine)
    first = await repository.record_lineage(_write(fingerprint="fp-500"))

    second = await repository.record_lineage(
        _write(
            text="Expenses over CHF 500 need signing off.",
            fingerprint="fp-500-reworded",
            document_revision="rev-2",
            run_id="run-2",
        )
    )

    assert second.assertion_id == first.assertion_id
    assert second.revision == 2
    assert second.supersedes == 1
    assert second.review_state == "approved"
