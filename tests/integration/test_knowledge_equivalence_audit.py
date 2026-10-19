# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Persist observations and equivalence assessments before continuation.

An assessment names both observations and its judge. Only after that audit
record exists may an affirmative decision continue an assertion.
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from nlght.core.knowledge import Judge, Proposition
from nlght.core.knowledge.equivalence import (
    AMBIGUOUS,
    DIFFERENT,
    SAME,
    UNJUDGED,
    AssertionObservation,
)

pytestmark = pytest.mark.integration

RUN = "run-1"
DOCUMENT = "doc-1"
SLOT = "slot-1"


@pytest_asyncio.fixture
async def repository():
    from sqlalchemy import event
    from sqlalchemy.ext.asyncio import create_async_engine

    from nlght.adapters.outbound.persistence.knowledge_models import KnowledgeBase
    from nlght.adapters.outbound.persistence.knowledge_repository import (
        SqlAlchemyKnowledgeRepository,
    )

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


async def _rows(repository, model):
    async with AsyncSession(repository._test_engine) as session:
        return list((await session.execute(select(model))).scalars().all())


def _rule(text: str, **fields: str) -> AssertionObservation:
    return AssertionObservation(kind="rule", observed_text=text, representation=fields)


def _fact(text: str, **fields: str) -> AssertionObservation:
    return AssertionObservation(kind="fact", observed_text=text, representation=fields)


_OLD = _rule(
    "you have to set up the environment when building the application",
    subject="aot_conditions", rule_property="environment_setup",
)
_NEW = _rule(
    "the environment must be set up at build time",
    subject="ahead_of_time", rule_property="build_environment",
)


async def test_an_observation_of_any_kind_is_durable(repository) -> None:
    """Every kind, because three of them have no proposition to be stored as.

    This is what makes the audit possible at all: the pair being judged has to
    exist before the judgement can point at it, and `knowledge_propositions`
    can only hold a fact.
    """
    from nlght.adapters.outbound.persistence.knowledge_models import (
        KnowledgeAssertionObservation,
    )

    for observation in (
        _OLD,
        _fact("Alice transfers CHF 500 to Bob.",
              subject="Alice", predicate="transfers", amount="CHF 500", recipient="Bob"),
        AssertionObservation(kind="decision", observed_text="We will use PostgreSQL.",
                             representation={"decision": "PostgreSQL", "effect": "persistence"}),
        AssertionObservation(kind="pattern", observed_text="Export health as metrics.",
                             representation={"pattern_name": "health_as_metrics",
                                             "description": "export statuses"}),
    ):
        await repository.record_observation(
            observation, run_id=RUN, document_id=DOCUMENT, slot_id=SLOT
        )

    stored = await _rows(repository, KnowledgeAssertionObservation)

    assert {row.kind for row in stored} == {"rule", "fact", "decision", "pattern"}
    assert all(row.observed_text for row in stored)
    assert all(row.representation for row in stored)


async def test_the_same_observation_seen_twice_is_two_observations(repository) -> None:
    """Record separate observations for separate readings of one claim."""
    from nlght.adapters.outbound.persistence.knowledge_models import (
        KnowledgeAssertionObservation,
    )

    first = await repository.record_observation(
        _OLD, run_id="run-1", document_id=DOCUMENT, slot_id=SLOT
    )
    second = await repository.record_observation(
        _OLD, run_id="run-2", document_id=DOCUMENT, slot_id=SLOT
    )

    assert first != second
    assert len(await _rows(repository, KnowledgeAssertionObservation)) == 2


@pytest.mark.parametrize("verdict", [SAME, DIFFERENT, AMBIGUOUS, UNJUDGED])
async def test_every_verdict_is_recorded(repository, verdict: str) -> None:
    """Including the ones that merge nothing.

    "These were judged different" is a finding, and a trail holding only the
    joins would show a judge that never disagreed and never hesitated.
    """
    from nlght.adapters.outbound.persistence.knowledge_models import (
        KnowledgeEquivalenceAssessment,
    )

    left = await repository.record_observation(
        _OLD, run_id=RUN, document_id=DOCUMENT, slot_id=SLOT
    )
    right = await repository.record_observation(
        _NEW, run_id=RUN, document_id=DOCUMENT, slot_id=SLOT
    )

    await repository.record_equivalence(
        left=left, right=right, verdict=verdict,
        judge=Judge(classifier="proposition-equivalence", model="m", version="1"),
        run_id=RUN,
    )

    [row] = await _rows(repository, KnowledgeEquivalenceAssessment)
    assert row.verdict == verdict
    assert {row.left_observation_id, row.right_observation_id} == {left, right}


async def test_one_pair_may_be_judged_more_than_once(repository) -> None:
    # Append-only, and no unique key on the pair: the same judge asked twice can
    # answer differently, and that instability is what an auditor most wants to
    # see. Reusing an answer instead of asking again is a cache, and a cache is
    # a different table.
    from nlght.adapters.outbound.persistence.knowledge_models import (
        KnowledgeEquivalenceAssessment,
    )

    left = await repository.record_observation(
        _OLD, run_id=RUN, document_id=DOCUMENT, slot_id=SLOT
    )
    right = await repository.record_observation(
        _NEW, run_id=RUN, document_id=DOCUMENT, slot_id=SLOT
    )

    await repository.record_equivalence(left=left, right=right, verdict=SAME)
    await repository.record_equivalence(left=left, right=right, verdict=AMBIGUOUS)

    rows = await _rows(repository, KnowledgeEquivalenceAssessment)
    assert sorted(row.verdict for row in rows) == [AMBIGUOUS, SAME]


async def test_an_unjudged_pair_names_no_model(repository) -> None:
    # Nobody was asked is not a decision a model made, and recording one would
    # put false evidence in a trail people read.
    from nlght.adapters.outbound.persistence.knowledge_models import (
        KnowledgeEquivalenceAssessment,
    )

    left = await repository.record_observation(
        _OLD, run_id=RUN, document_id=DOCUMENT, slot_id=SLOT
    )
    right = await repository.record_observation(
        _NEW, run_id=RUN, document_id=DOCUMENT, slot_id=SLOT
    )

    await repository.record_equivalence(left=left, right=right, verdict=UNJUDGED)

    [row] = await _rows(repository, KnowledgeEquivalenceAssessment)
    assert row.classifier is None and row.model is None


async def test_the_assessment_outlives_a_merge_that_never_happens(repository) -> None:
    """The proof that it is a reason and not a description.

    An assessment written after the merge could only ever describe what
    happened; one written before it is the record of why. The difference is
    visible exactly here: the judgement stands even though nothing was joined,
    so it cannot have been derived from the outcome.
    """
    from nlght.adapters.outbound.persistence.knowledge_models import (
        KnowledgeAssertion,
        KnowledgeEquivalenceAssessment,
    )

    left = await repository.record_observation(
        _OLD, run_id=RUN, document_id=DOCUMENT, slot_id=SLOT
    )
    right = await repository.record_observation(
        _NEW, run_id=RUN, document_id=DOCUMENT, slot_id=SLOT
    )
    await repository.record_equivalence(left=left, right=right, verdict=AMBIGUOUS)

    # Nothing continued anything: no assertion was written at all.
    assert await _rows(repository, KnowledgeAssertion) == []
    assert len(await _rows(repository, KnowledgeEquivalenceAssessment)) == 1


async def test_the_kind_boundary_is_not_recorded_as_a_model_judgement(repository) -> None:
    """Keep deterministic cross-kind rejection out of model-judgment records."""
    from nlght.adapters.outbound.persistence.knowledge_models import (
        KnowledgeEquivalenceAssessment,
    )
    from nlght.core.knowledge.equivalence import settled

    rule = _rule("Secrets must never be logged.",
                 subject="secret", rule_property="logging_prohibition")
    fact = _fact("Secrets must never be logged.",
                 subject="secret", predicate="must_not", object="be_logged")

    assert settled(rule, fact) == DIFFERENT, "decided without asking anything"

    rows = await _rows(repository, KnowledgeEquivalenceAssessment)
    assert [row for row in rows if row.classifier == "proposition-equivalence"] == []


async def test_a_reworded_rule_can_be_judged_and_continued(repository) -> None:
    """Judge reworded rules and continue the stored assertion when equivalent.

    The rule has a proposition. The model judgment resolves descriptions whose
    entity-key fields differ despite expressing the same claim."""
    from nlght.adapters.outbound.persistence.knowledge_models import (
        KnowledgeAssertionObservation,
        KnowledgeEquivalenceAssessment,
    )

    left = await repository.record_observation(
        _OLD, run_id="run-1", document_id=DOCUMENT, slot_id=SLOT
    )
    right = await repository.record_observation(
        _NEW, run_id="run-2", document_id=DOCUMENT, slot_id=SLOT
    )
    await repository.record_equivalence(
        left=left, right=right, verdict=SAME,
        judge=Judge(classifier="proposition-equivalence", model="qwen3.8:27B", version="1"),
        run_id="run-2",
    )

    observations = await _rows(repository, KnowledgeAssertionObservation)
    assert {row.kind for row in observations} == {"rule"}
    # The wording is the source's, and the representation says which assertion
    # inside it was judged. Neither is reconstructed from the other.
    assert {row.observed_text for row in observations} == {
        _OLD.observed_text, _NEW.observed_text
    }

    [assessment] = await _rows(repository, KnowledgeEquivalenceAssessment)
    assert assessment.verdict == SAME
    assert assessment.model == "qwen3.8:27B"


async def test_a_fact_keeps_its_proposition_table_as_well(repository) -> None:
    # The observation is the comparison unit. It does not replace the structured
    # fact. `knowledge_propositions` goes on holding what a fact *is*,
    # content-addressed, while an observation records that somebody looked.
    from nlght.adapters.outbound.persistence.knowledge_models import (
        KnowledgeAssertionObservation,
        KnowledgeProposition,
    )

    proposition = Proposition(
        {"subject": "Alice", "predicate": "transfers", "amount": "CHF 500", "recipient": "Bob"}
    )
    await repository.record_proposition(proposition)
    await repository.record_observation(
        AssertionObservation.of(proposition, observed_text="Alice transfers CHF 500 to Bob."),
        run_id=RUN, document_id=DOCUMENT, slot_id=SLOT,
    )

    assert len(await _rows(repository, KnowledgeProposition)) == 1
    assert len(await _rows(repository, KnowledgeAssertionObservation)) == 1


async def test_comparing_against_many_candidates_records_one_observation(
    repository,
) -> None:
    """The invariant the whole model rests on.

        observation  a sighting happened
        assessment   a judgement between two sightings
        revision     the state that resulted

    One incoming sighting compared against three candidates is **one** new
    observation and three assessments. If the candidate side were synthesised at
    comparison time, a sighting would become a sighting *per comparison*: the
    audit would be complete and historically false, and the row would carry the
    run that did the comparing rather than the run that did the seeing.

    So the candidate side must point at the historical observation the candidate
    actually came from.
    """
    from nlght.adapters.outbound.persistence.knowledge_models import (
        KnowledgeAssertionObservation,
        KnowledgeEquivalenceAssessment,
    )

    # Three claims already standing, each sighted once, in an earlier run.
    standing = [
        await repository.record_observation(
            _rule(f"Rule number {n} as the source states it.",
                  subject=f"subject_{n}", rule_property="limit"),
            run_id="run-1", document_id=DOCUMENT, slot_id=SLOT,
        )
        for n in range(3)
    ]
    assert len(await _rows(repository, KnowledgeAssertionObservation)) == 3

    # One new sighting, judged against all three.
    incoming = await repository.record_observation(
        _NEW, run_id="run-2", document_id=DOCUMENT, slot_id=SLOT
    )
    for candidate in standing:
        await repository.record_equivalence(
            left=incoming, right=candidate, verdict=DIFFERENT,
            judge=Judge(classifier="proposition-equivalence", model="m", version="1"),
            run_id="run-2",
        )

    observations = await _rows(repository, KnowledgeAssertionObservation)
    assessments = await _rows(repository, KnowledgeEquivalenceAssessment)

    # Four sightings happened, so there are four rows, not four plus three
    # copies made while comparing.
    assert len(observations) == 4
    assert len(assessments) == 3
    # Every assessment points at rows that already existed.
    referenced = {row.left_observation_id for row in assessments} | {
        row.right_observation_id for row in assessments
    }
    assert referenced <= {row.observation_id for row in observations}
    # And each candidate keeps the run that saw it, not the run that compared it.
    seen_in = {row.observation_id: row.run_id for row in observations}
    assert {seen_in[candidate] for candidate in standing} == {"run-1"}
    assert seen_in[incoming] == "run-2"


async def test_assessing_a_sighting_does_not_re_sight_its_candidates(repository) -> None:
    """Record one incoming observation while comparing historical candidates."""
    from sqlalchemy.ext.asyncio import AsyncSession as _Session

    from nlght.adapters.outbound.persistence.knowledge_models import (
        KnowledgeAssertionObservation,
    )
    from nlght.core.knowledge import LineageWrite, entity_key
    from nlght.core.knowledge.proposition import Candidate

    # Three rules already in the slot, each sighted once by an earlier run.
    for number in range(3):
        await repository.record_lineage(
            LineageWrite(
                entity=entity_key("rule", subject=f"subject_{number}", rule_property="limit"),
                kind="rule",
                text=f"Rule number {number} as the source states it.",
                fingerprint=f"fp-{number}",
                extraction_version="ver-1",
                document_id=DOCUMENT,
                document_revision="rev-1",
                run_id="run-1",
                slot_id=SLOT,
                proposition=Proposition(
                    {"rule_text": f"Rule number {number} as the source states it.",
                     "subject": f"subject_{number}", "rule_property": "limit"}
                ),
            )
        )
    before = len(await _rows(repository, KnowledgeAssertionObservation))

    incoming = LineageWrite(
        entity=entity_key("rule", subject="something_else", rule_property="limit"),
        kind="rule",
        text="A rule the ladder cannot place.",
        fingerprint="fp-new",
        extraction_version="ver-1",
        document_id=DOCUMENT,
        document_revision="rev-2",
        run_id="run-2",
        slot_id=SLOT,
        proposition=Proposition(
            {"rule_text": "A rule the ladder cannot place.",
             "subject": "something_else", "rule_property": "limit"}
        ),
    )
    candidates = [
        Candidate(
            assertion_id=f"a_{number}",
            entity_key=f"k{number}",
            entity_key_version="1",
            text=f"Rule number {number} as the source states it.",
            document_id=DOCUMENT,
            slot_ids=frozenset({SLOT}),
            kind="rule",
        )
        for number in range(3)
    ]

    async with _Session(repository._test_engine) as session:
        await repository._assess(session, incoming, candidates, None, SLOT)

    after = await _rows(repository, KnowledgeAssertionObservation)

    assert len(after) - before == 1, (
        "one sighting was assessed, so one observation may be recorded. "
        "the candidates were seen in an earlier run and already have theirs"
    )
    assert {row.run_id for row in after} == {"run-1", "run-2"}
