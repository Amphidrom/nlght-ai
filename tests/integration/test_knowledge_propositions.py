# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Verify stored propositions and their relationship to assertion revisions.

Content addressing deduplicates identical structures. Equivalence decisions
remain separate audit records because hashing cannot establish claim identity.
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


def _p(**fields: object) -> Proposition:
    return Proposition(fields)


async def _rows(repository, model):
    async with AsyncSession(repository._test_engine) as session:
        return list((await session.execute(select(model))).scalars().all())


_TRANSFER = _p(predicate="transfers", sender="Alice", amount="CHF 500", recipient="Bob")

#: Held equal across the sightings below on purpose. These are tests of where a
#: proposition is *stored*, not of what continues an assertion. Letting the key
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
    # This checks content addressing. The next test checks identity.
    first = await repository.record_proposition(_TRANSFER)
    second = await repository.record_proposition(_TRANSFER)

    assert first == second
    assert len(await _rows(repository, KnowledgeProposition)) == 1


async def test_one_claim_said_two_ways_is_two_rows(repository) -> None:
    """Keep differently worded propositions separate until equivalence is judged.

    The proposition store must not merge claims based on a content hash.
    """
    await repository.record_proposition(_p(predicate="transfers", sender="Alice"))
    await repository.record_proposition(_p(predicate="transfers", actor="Alice"))

    assert len(await _rows(repository, KnowledgeProposition)) == 2


async def test_a_fingerprint_carries_the_scheme_that_made_it(repository) -> None:
    # Normalisation will change, and an old value must not look like it lives in
    # the new namespace for the same reason `entity_key_version` has its own column.
    await repository.record_proposition(_TRANSFER)

    [row] = await _rows(repository, KnowledgeProposition)
    assert row.fingerprint_version
    assert row.fingerprint != str(row.proposition_id)


async def test_a_revision_names_the_structure_it_recorded(repository) -> None:
    """Store each observed structure on the revision that recorded it.

    A graph node cannot represent every historical structure of an assertion.
    """
    from nlght.adapters.outbound.persistence.knowledge_models import KnowledgeRevision

    await repository.record_lineage(_lineage(_TRANSFER, text="Alice transfers CHF 500 to Bob"))

    [proposition] = await _rows(repository, KnowledgeProposition)
    [revision] = await _rows(repository, KnowledgeRevision)
    assert revision.proposition_id == proposition.proposition_id


async def test_one_assertion_keeps_the_structure_of_every_revision(repository) -> None:
    """Keep distinct proposition structures readable across revisions.

    The test writes rows directly to verify storage cardinality independent of
    resolver matching decisions.
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

    async with AsyncSession(repository._test_engine) as session, session.begin():
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
    """Store a proposition for each knowledge kind.

    Rule, pattern, and decision fields retain their own structure without the
    platform assigning special semantic roles to those fields.
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


_OTHER = _p(predicate="transfers", actor="Alice", amount="CHF 500", beneficiary="Bob")
_OTHER_ENTITY = entity_key("fact", predicate="transfers", actor="alice")


async def _standing_in(repository, slot_id: str) -> str:
    """One assertion already recorded, in the slot named.

    Args:
        repository (Any): Repository used by the test.
        slot_id (str): Identifier of the claim slot.

    Returns:
        str: Identifier of the assertion currently occupying the slot.
    """
    placed = await repository.record_lineage(
        _lineage(_TRANSFER, text="Alice transfers CHF 500 to Bob", slot_id=slot_id)
    )
    return placed.assertion_id


async def test_a_verdict_applies_only_where_its_candidate_still_stands(repository) -> None:
    """Apply an equivalence verdict only to the slot it assessed.

    The authoritative slot can change between judgment and persistence. Keep
    the judgment in the audit even when its verdict cannot be applied.
    """
    standing = await _standing_in(repository, "slot-a")
    judge = Judge(classifier="proposition-equivalence", model="m", version="1")
    # The judgement is about two *observations*, which is the unit every kind
    # has. The proposition beside it is a representation of that fact.
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


async def test_a_claim_with_no_revision_is_refused_rather_than_stored_silently(
    repository,
) -> None:
    """Reject a fact whose structure has no revision to attach to.

    Without lineage, the graph node would contain no readable proposition or
    payload. Treat this as a programming error.
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
    """Allow non-fact legacy claims without lineage or propositions.

    Rule, pattern, and decision payloads remain readable without a revision.
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


async def test_retrieval_answers_from_the_proposition_for_every_kind(repository) -> None:
    """Read current claims from their revision propositions across all kinds.

    Kind-specific tables remain projections for consumers that need their
    particular shapes.
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
                # The derived projection is absent for a claim it cannot hold.
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
    """Read historical revisions without propositions from their stored payloads.

    A backfill would guess at an observation the original extraction never
    recorded. Retrieval preserves the old data without fabricating provenance.
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
    async with AsyncSession(repository._test_engine) as session, session.begin():
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
    import inspect

    from nlght.adapters.outbound.persistence import knowledge_repository
    from nlght.core.knowledge import legacy

    for module in (legacy, knowledge_repository):
        source = inspect.getsource(module)
        for payload in ("FactPayload(", "RulePayload(", "PatternPayload(", "DecisionPayload("):
            for line in source.splitlines():
                if payload in line and "Proposition(" in line:
                    raise AssertionError(
                        f"{module.__name__} builds a proposition beside a payload: {line}"
                    )
