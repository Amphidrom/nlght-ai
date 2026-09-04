# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Keeping a claim whole, and keeping it apart from what was decided about it.

`knowledge_propositions` holds what a *fact* says — content-addressed, so one
claim seen twice is one row. What was judged about a pair of assertions lives in
`knowledge_equivalence_assessments` and is tested beside it, because the two pull
in opposite directions: content addressing wants one row per claim, while an
audit must be able to hold two different answers to one question.

The judgement half used to be keyed on propositions and is not any more — a fact
is the only kind that has one, so three kinds could never be audited. It compares
observations now (`test_knowledge_equivalence_audit.py`).
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from nlght.adapters.outbound.persistence.knowledge_models import (
    KnowledgeAssertion,
    KnowledgeEquivalenceAssessment,
    KnowledgeProposition,
)
from nlght.adapters.outbound.persistence.knowledge_repository import (
    SqlAlchemyKnowledgeRepository,
)
from nlght.core.knowledge import Judge, LineageWrite, Proposition, entity_key
from nlght.core.knowledge.equivalence import SAME, AssertionObservation

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def repository():
    from sqlalchemy import event
    from sqlalchemy.ext.asyncio import create_async_engine

    from nlght.adapters.outbound.persistence.knowledge_models import KnowledgeBase

    engine = create_async_engine("sqlite+aiosqlite:///:memory:")

    @event.listens_for(engine.sync_engine, "connect")
    def _enforce_foreign_keys(dbapi_connection, _record):  # noqa: ANN001, ANN202
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    async with engine.begin() as conn:
        await conn.run_sync(KnowledgeBase.metadata.create_all)
    repository = SqlAlchemyKnowledgeRepository(engine)
    repository._test_engine = engine  # noqa: SLF001
    yield repository
    await engine.dispose()


def _p(**fields: object) -> Proposition:
    return Proposition(fields)


async def _rows(repository, model):  # noqa: ANN001, ANN201
    async with AsyncSession(repository._test_engine) as session:  # noqa: SLF001
        return list((await session.execute(select(model))).scalars().all())


_TRANSFER = _p(predicate="transfers", sender="Alice", amount="CHF 500", recipient="Bob")

#: Held equal across the sightings below on purpose. These are tests of where a
#: proposition is *stored*, not of what continues an assertion — letting the key
#: move would turn every one of them into a test of the matcher.
_ENTITY = entity_key("fact", predicate="transfers", subject="alice")


def _lineage(
    proposition: Proposition,
    *,
    text: str,
    kind: str = "fact",
    entity: object = _ENTITY,
    slot_id: str | None = None,
    continues: str | None = None,
) -> LineageWrite:
    return LineageWrite(
        entity=entity,
        kind=kind,
        text=text,
        fingerprint=proposition.fingerprint,
        extraction_version="ver-1",
        document_id="doc-transfers",
        document_revision="rev-1",
        run_id="run-1",
        slot_id=slot_id,
        continues=continues,
        proposition=proposition,
    )


# ---------------------------------------------------------------------------
# The claim
# ---------------------------------------------------------------------------

async def test_a_four_role_claim_is_stored_whole(repository) -> None:
    """The claim this whole slice began with.

    "Alice transfers CHF 500 to Bob" had a well-formed entity key and nowhere to
    live, so a model that read the sentence correctly had its answer discarded as
    malformed.
    """
    await repository.record_proposition(_TRANSFER)

    [row] = await _rows(repository, KnowledgeProposition)
    assert row.fields == {
        "predicate": "transfers",
        "sender": "Alice",
        "amount": "CHF 500",
        "recipient": "Bob",
    }


async def test_nested_structure_and_list_order_survive_the_database(repository) -> None:
    await repository.record_proposition(
        _p(predicate="limits", detail={"unit": "CHF", "bounds": ["min", "max"]})
    )

    [row] = await _rows(repository, KnowledgeProposition)
    assert row.fields["detail"]["bounds"] == ["min", "max"]


async def test_the_same_claim_seen_twice_is_one_row(repository) -> None:
    # Content addressing. Not identity — that is the next test.
    first = await repository.record_proposition(_TRANSFER)
    second = await repository.record_proposition(_TRANSFER)

    assert first == second
    assert len(await _rows(repository, KnowledgeProposition)) == 1


async def test_one_claim_said_two_ways_is_two_rows(repository) -> None:
    """The property that keeps this table from becoming identity again.

    These may well be one claim. Whether they are is a *judgement*, recorded in
    the other table — and a store that merged them here would have decided it by
    hashing, which is the content-derived identity four ADRs went into removing.
    """
    await repository.record_proposition(_p(predicate="transfers", sender="Alice"))
    await repository.record_proposition(_p(predicate="transfers", actor="Alice"))

    assert len(await _rows(repository, KnowledgeProposition)) == 2


async def test_a_fingerprint_carries_the_scheme_that_made_it(repository) -> None:
    # Normalisation will change, and an old value must not look like it lives in
    # the new namespace — the same reason `entity_key_version` is its own column.
    await repository.record_proposition(_TRANSFER)

    [row] = await _rows(repository, KnowledgeProposition)
    assert row.fingerprint_version
    assert row.fingerprint != str(row.proposition_id)


# ---------------------------------------------------------------------------
# Who owns a proposition
# ---------------------------------------------------------------------------

async def test_a_revision_names_the_structure_it_recorded(repository) -> None:
    """The link that makes a stored claim readable, and where it hangs.

    On the revision, because a revision is one *state* of a claim. Reaching it
    from the graph node instead would have needed a `proposition_id` on the node,
    and a node has one — so the next test could not hold.
    """
    from nlght.adapters.outbound.persistence.knowledge_models import KnowledgeRevision

    await repository.record_lineage(_lineage(_TRANSFER, text="Alice transfers CHF 500 to Bob"))

    [proposition] = await _rows(repository, KnowledgeProposition)
    [revision] = await _rows(repository, KnowledgeRevision)
    assert revision.proposition_id == proposition.proposition_id


async def test_one_assertion_keeps_the_structure_of_every_revision(repository) -> None:
    """The cardinality, as the thing that breaks if it is wrong.

        Assertion A
          Revision 1 → Proposition P1
          Revision 2 → Proposition P2

    One assertion, two states, two structures — and both still readable. A single
    `proposition_id` on the assertion or on the graph node would pass every other
    test here and lose exactly this: the earlier form of every claim that was
    ever reworded, overwritten by the later one.

    The rows are written directly rather than through `record_lineage`, and that
    is the point of the test rather than a shortcut around it. Driving it through
    the resolver would have made it depend on *why* P1 and P2 landed on one
    assertion — today's matching semantics — and this asks only whether the store
    can hold the shape at all. Why two structures may continue one assertion is
    an equivalence judgement, and it gets its own test when it is wired.
    """
    from nlght.adapters.outbound.persistence.knowledge_models import (
        KnowledgeRevision,
        KnowledgeVariant,
    )

    first = await repository.record_proposition(_TRANSFER)
    second = await repository.record_proposition(
        _p(predicate="transfers", sender="Alice", amount="CHF 500", beneficiary="Bob")
    )
    assert first != second

    async with AsyncSession(repository._test_engine) as session, session.begin():  # noqa: SLF001
        session.add(
            KnowledgeAssertion(assertion_id="a-1", kind="fact", entity_key="k", entity_key_version="1")
        )
        await session.flush()
        session.add(KnowledgeVariant(variant_id="v-1", assertion_id="a-1", scope=[], resolution="new"))
        await session.flush()
        for number, proposition_id in enumerate((first, second), start=1):
            session.add(
                KnowledgeRevision(
                    variant_id="v-1",
                    revision=number,
                    text=f"state {number}",
                    fingerprint=f"fp-{number}",
                    extraction_version="ver-1",
                    review_state="approved",
                    proposition_id=proposition_id,
                )
            )

    revisions = sorted(await _rows(repository, KnowledgeRevision), key=lambda row: row.revision)
    fields = {
        row.proposition_id: row.fields for row in await _rows(repository, KnowledgeProposition)
    }

    assert len(await _rows(repository, KnowledgeAssertion)) == 1
    assert [sorted(fields[row.proposition_id]) for row in revisions] == [
        ["amount", "predicate", "recipient", "sender"],
        ["amount", "beneficiary", "predicate", "sender"],
    ]


async def test_every_kind_carries_one(repository) -> None:
    """A proposition is a property of a knowledge assertion, not of facts.

    It was fact-only, and the asymmetry produced a special path everywhere it
    touched: a rule had no structured form, so equivalence could not judge one,
    and a revision could name a fact's structure and nothing else's.

    Generalising it invents nothing, which is why it is safe. `fields` never had
    privileged roles — no subject, no predicate — so a rule's fields are a
    rule's, and the platform reads none of them as meaning anything.
    """
    from nlght.adapters.outbound.persistence.knowledge_models import KnowledgeRevision

    rule = _p(rule_text="Expenses above CHF 500 require approval.",
              subject="expense", rule_property="approval_threshold")
    await repository.record_lineage(
        _lineage(rule, text="Expenses above CHF 500 require approval.", kind="rule")
    )

    [revision] = await _rows(repository, KnowledgeRevision)
    assert revision.proposition_id is not None
    [stored] = await _rows(repository, KnowledgeProposition)
    assert stored.fields["rule_text"] == "Expenses above CHF 500 require approval."


# ---------------------------------------------------------------------------
# A verdict may not outlive the premise it was reached under
# ---------------------------------------------------------------------------

_OTHER = _p(predicate="transfers", actor="Alice", amount="CHF 500", beneficiary="Bob")
_OTHER_ENTITY = entity_key("fact", predicate="transfers", actor="alice")


async def _standing_in(repository, slot_id: str) -> str:  # noqa: ANN001
    """One assertion already recorded, in the slot named."""
    placed = await repository.record_lineage(
        _lineage(_TRANSFER, text="Alice transfers CHF 500 to Bob", slot_id=slot_id)
    )
    return placed.assertion_id


async def test_a_verdict_applies_only_where_its_candidate_still_stands(repository) -> None:
    """The window between judging and writing, closed.

    The judgement is formed outside the transaction, against the slot this
    section is *expected* to resolve to; the authoritative placement happens
    inside it and can land elsewhere — the registry moved between the two, or
    recovery answered differently.

        pre-resolution     → slot A, and the judge is asked about slot A's claims
        verdict            → yes
        authoritative      → slot B
        applied?           → no

    Otherwise a verdict reached about one slot continues an assertion standing in
    another, which is precisely what scoping the question to a slot was for. The
    assessment stays in the audit either way: the judge really was asked and
    really did answer, and **judged yes is not applied yes**.
    """
    standing = await _standing_in(repository, "slot-a")
    judge = Judge(classifier="proposition-equivalence", model="m", version="1")
    # The judgement is about two *observations*, which is the unit every kind
    # has — the proposition beside it is only a fact's representation.
    left = await repository.record_observation(
        AssertionObservation.of(_TRANSFER, observed_text="Alice transfers CHF 500 to Bob"),
        run_id="run-1", document_id="doc-transfers", slot_id="slot-a",
    )
    right = await repository.record_observation(
        AssertionObservation.of(_OTHER, observed_text="Alice sends CHF 500 to the beneficiary Bob"),
        run_id="run-1", document_id="doc-transfers", slot_id="slot-b",
    )
    await repository.record_equivalence(left=left, right=right, verdict=SAME, judge=judge)

    placed = await repository.record_lineage(
        _lineage(
            _OTHER,
            text="Alice sends CHF 500 to the beneficiary Bob",
            entity=_OTHER_ENTITY,
            slot_id="slot-b",
            continues=standing,
        )
    )

    assert placed.assertion_id != standing
    assert len(await _rows(repository, KnowledgeAssertion)) == 2
    # Asked and answered, and the record of that does not depend on it being used.
    [assessment] = await _rows(repository, KnowledgeEquivalenceAssessment)
    assert assessment.verdict == SAME


async def test_a_verdict_does_apply_where_its_candidate_still_stands(repository) -> None:
    """The control, without which the guard above could pass by refusing everything.

    Same claim, same judgement, and the placement lands where the question was
    asked. The assertion continues.
    """
    standing = await _standing_in(repository, "slot-a")

    placed = await repository.record_lineage(
        _lineage(
            _OTHER,
            text="Alice sends CHF 500 to the beneficiary Bob",
            entity=_OTHER_ENTITY,
            slot_id="slot-a",
            continues=standing,
        )
    )

    assert placed.assertion_id == standing
    assert len(await _rows(repository, KnowledgeAssertion)) == 1


# ---------------------------------------------------------------------------
# Every stored claim is reachable
# ---------------------------------------------------------------------------

async def test_a_claim_with_no_revision_is_refused_rather_than_stored_silently(
    repository,
) -> None:
    """The invariant, as the thing that makes it impossible to violate.

    A fact keeps its structure on the revision a sighting produces. Written
    without lineage it produced a graph row with no payload, no proposition and
    no revision — a node asserting nothing, which is a quieter failure than
    losing the claim rather than a milder one.

    Measured before it was closed:

        knowledge               rows=1     ← a node that says nothing
        knowledge_facts         rows=0
        knowledge_revisions     rows=0
        knowledge_propositions  rows=0     ← the claim is nowhere

    By the time a write reaches the repository this is a programming error and
    not a data condition: extraction rejects a fact that cannot name what it
    claims, and the persist step counts and skips one whose evidence names no
    document. So it raises rather than storing something unreachable.
    """
    from nlght.core.knowledge import KnowledgeWrite

    with pytest.raises(ValueError, match="no revision to keep it on"):
        await repository.persist(
            KnowledgeWrite(
                identity="fp-1",
                payload=None,
                proposition=_TRANSFER,
                type="transfer",
                confidence=0.9,
                source_id="corpus",
                run_id="run-1",
            ),
            None,
        )

    assert await _rows(repository, KnowledgeAssertion) == []


async def test_a_kind_without_a_proposition_still_stores_without_lineage(
    repository,
) -> None:
    """And the guard stops exactly there.

    `rule`, `pattern` and `decision` *are* their payloads, so one written without
    lineage is still completely readable — it has no structure that needs a
    revision to live on. A corpus written before entity keys existed is full of
    them, and refusing those would be taking knowledge back to fix a problem
    they do not have.
    """
    from nlght.adapters.outbound.persistence.knowledge_models import KnowledgeRule
    from nlght.core.knowledge import KnowledgeWrite, RulePayload

    node, recorded = await repository.persist(
        KnowledgeWrite(
            identity="fp-rule",
            payload=RulePayload(rule_text="Expenses above CHF 500 require approval."),
            type="policy",
            confidence=0.9,
            source_id="corpus",
            run_id="run-1",
        ),
        None,
    )

    assert recorded is None
    assert node.payload["rule_text"] == "Expenses above CHF 500 require approval."
    assert len(await _rows(repository, KnowledgeRule)) == 1


# ---------------------------------------------------------------------------
# Retrieval reads one thing, not four
# ---------------------------------------------------------------------------

async def test_retrieval_answers_from_the_proposition_for_every_kind(repository) -> None:
    """Otherwise the model is unified and the reading is not.

    Before, retrieval reached the proposition for a fact that had no legacy row
    and the kind-specific table for everything else — so what a caller got back
    depended on which kind it asked about and whether that kind's projection had
    survived. Four tables, four answers, one of them the whole claim and three
    of them a shape.

    A revision names its proposition for every kind now, so that is what
    retrieval reads. The kind-specific tables remain for consumers that want one
    of those shapes; they are not where the claim lives.
    """
    from nlght.core.knowledge import entity_key

    cases = {
        "fact": ("Spring Boot requires Java 17.",
                 {"subject": "Spring Boot", "predicate": "requires", "object": "Java 17"},
                 entity_key("fact", predicate="requires", subject="spring boot",
                            object="java 17")),
        "rule": ("Applications must enable graceful shutdown.",
                 {"rule_text": "Applications must enable graceful shutdown."},
                 entity_key("rule", subject="application", rule_property="shutdown")),
        "pattern": ("Health statuses are exported as metrics.",
                    {"pattern_name": "health_as_metrics", "description": "export"},
                    entity_key("pattern", pattern_name="health_as_metrics")),
        "decision": ("We will use PostgreSQL for persistence.",
                     {"decision": "Use PostgreSQL", "effect": "persistence"},
                     entity_key("decision", subject="persistence",
                                decision_type="datastore")),
    }

    from nlght.core.knowledge import KnowledgeWrite
    from nlght.core.knowledge.legacy import legacy_payload

    identities = []
    for kind, (wording, fields, key) in cases.items():
        proposition = _p(**fields)
        stored, placed = await repository.persist(
            KnowledgeWrite(
                identity=f"fp-{kind}",
                # The projection, derived — absent for a claim it cannot hold.
                payload=legacy_payload(kind, proposition),
                _kind=kind,
                proposition=proposition,
                type="t", confidence=0.9, source_id="corpus", run_id="run-1",
            ),
            LineageWrite(
                entity=key, kind=kind, text=wording, fingerprint=f"fp-{kind}",
                extraction_version="v1", document_id="d1", document_revision="r1",
                run_id="run-1", slot_id="s1", proposition=proposition,
            ),
        )
        identities.append((kind, stored.identity, fields))

    for kind, identity, fields in identities:
        [stored] = await repository.resolve([identity], include_quarantined=True)
        assert stored.payload == fields, f"{kind} did not come back whole"


# ---------------------------------------------------------------------------
# What happens to the revisions that predate the symmetry
# ---------------------------------------------------------------------------

async def test_a_new_revision_without_a_proposition_is_refused(repository) -> None:
    """Every revision created from now on has one, for every kind.

    A revision is a state of a claim, and a state with no structured claim in it
    is a row saying only that something happened. The rule is enforced where the
    revision is made rather than trusted at four call sites.
    """
    from nlght.core.knowledge import entity_key

    with pytest.raises(ValueError, match="must say what the claim is"):
        await repository.record_lineage(
            LineageWrite(
                entity=entity_key("rule", subject="a", rule_property="b"),
                kind="rule", text="Something the source says.", fingerprint="fp-x",
                extraction_version="v1", document_id="d1", document_revision="r1",
                run_id="run-1", slot_id="s1",
            )
        )


async def test_a_historical_revision_stays_readable_through_its_payload(
    repository,
) -> None:
    """The 19 rows a live corpus already has, and what is owed to them.

    They were written before every kind had a proposition, so their
    `proposition_id` is `NULL` and always will be. Retrieval falls back to the
    kind's payload for those, which still holds everything they ever said.

    What must **never** happen is a backfill from that payload. Not because the
    payload lost something — a stored projection *is* lossless — that is
    what `Represented | NotRepresentable` guarantees, and a payload that could
    not hold the whole claim was never written. What is not guaranteed is the
    inverse: reading a payload back does not reproduce the proposition a past run
    actually produced. The same three columns can be reached from more than one
    structure, and the run that wrote them is gone.

    So a backfilled proposition is a plausible guess wearing the clothes of an
    observation. A missing proposition is a gap; a fabricated one is a lie in an
    audit.
    """
    from nlght.adapters.outbound.persistence.knowledge_models import (
        Knowledge,
        KnowledgeRevision,
        KnowledgeRule,
    )
    from nlght.core.knowledge import KnowledgeWrite, RulePayload, entity_key

    await repository.persist(
        KnowledgeWrite(
            identity="fp-old",
            payload=RulePayload(rule_text="Expenses above CHF 500 require approval."),
            _kind="rule", type="policy", confidence=0.9,
            source_id="corpus", run_id="run-0",
        ),
        LineageWrite(
            entity=entity_key("rule", subject="expense", rule_property="threshold"),
            kind="rule", text="Expenses above CHF 500 require approval.",
            fingerprint="fp-old", extraction_version="v0",
            document_id="d1", document_revision="r1", run_id="run-0", slot_id="s1",
            proposition=_p(rule_text="Expenses above CHF 500 require approval."),
        ),
    )

    # Simulate the historical state: the revision loses its proposition, exactly
    # as the rows written before this change have it.
    async with AsyncSession(repository._test_engine) as session, session.begin():  # noqa: SLF001
        for revision in (await session.execute(select(KnowledgeRevision))).scalars():
            revision.proposition_id = None

    [stored] = await repository.resolve(["fp-old"], include_quarantined=True)

    assert stored.payload["rule_text"] == "Expenses above CHF 500 require approval."
    assert len(await _rows(repository, KnowledgeRule)) == 1
    assert len(await _rows(repository, Knowledge)) == 1


def test_nothing_derives_a_proposition_from_a_payload() -> None:
    """The backfill, guarded where it would be written.

    `legacy.py` projects a proposition *into* a payload. Nothing goes the other
    way, and nothing may: a payload is a lossy shape, so a proposition built from
    one is a structure nobody extracted.
    """
    import inspect  # noqa: PLC0415

    from nlght.adapters.outbound.persistence import knowledge_repository  # noqa: PLC0415
    from nlght.core.knowledge import legacy  # noqa: PLC0415

    for module in (legacy, knowledge_repository):
        source = inspect.getsource(module)
        for payload in ("FactPayload(", "RulePayload(", "PatternPayload(", "DecisionPayload("):
            for line in source.splitlines():
                if payload in line and "Proposition(" in line:
                    raise AssertionError(
                        f"{module.__name__} builds a proposition beside a payload: {line}"
                    )
