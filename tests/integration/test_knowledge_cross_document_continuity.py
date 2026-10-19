# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Link equivalent claims found in different documents.

Candidate searches remain bounded by kind and other narrowing conditions.
A matching judgement joins claim continuity without losing each sighting.
"""

from __future__ import annotations

import pytest
import pytest_asyncio

from nlght.adapters.outbound.persistence.knowledge_models import (
    KnowledgeAssertion,
    KnowledgeEquivalenceAssessment,
)
from nlght.adapters.outbound.persistence.knowledge_repository import (
    SqlAlchemyKnowledgeRepository,
)
from nlght.core.knowledge import (
    KnowledgeWrite,
    LineageWrite,
    Proposition,
    entity_key,
)
from nlght.core.knowledge.equivalence import DIFFERENT, SAME, AssertionObservation, Judge
from nlght.core.knowledge.knowledge import FactPayload
from nlght.core.knowledge.slot import DERIVED_ANCHOR, ObservedSection

pytestmark = pytest.mark.integration

DOC_A = "doc-management-a"
DOC_B = "doc-management-b"
SENTENCE = "The management server uses the application's port."


class _Judge:
    """A judge that answers the same way every time, and says who it is."""

    def __init__(self, verdict: str) -> None:
        self._verdict = verdict
        self.asked: list[tuple[str, str]] = []

    @property
    def judge(self) -> Judge:
        return Judge(classifier="test", model="stub", version="1")

    async def classify(self, a: AssertionObservation, b: AssertionObservation) -> str:
        self.asked.append((a.observed_text, b.observed_text))
        return self._verdict


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


def _pair(fields: dict[str, str], *, document: str, run: str, text: str = SENTENCE):
    """The graph half and the lineage half of one sighting.

    Args:
        fields (dict[str, str]): Fields used to build the claim.
        document (str): Document or document identifier for the test.
        run (str): Run identifier for the sighting.
        text (str): Observed source text.
    """
    lineage = LineageWrite(
        entity=entity_key("fact", **fields),
        kind="fact",
        text=text,
        fingerprint=f"fp-{document}-{fields['predicate']}",
        extraction_version="v1",
        document_id=document,
        document_revision=f"{document}-rev-1",
        document_path=f"testdocs/{document}.adoc",
        run_id=run,
        proposition=Proposition(dict(fields)),
    )
    graph = KnowledgeWrite(
        identity=lineage.fingerprint,
        payload=FactPayload(
            subject=fields["subject"], predicate=fields["predicate"], object=fields["object"]
        ),
        type="config", confidence=0.9, source_id="corpus", run_id=run,
    )
    return graph, lineage


async def _record(repository, fields, *, document, run, judge=None, text=SENTENCE):
    section = ObservedSection.as_read(
        anchor=f"{document}#port",
        anchor_strength=DERIVED_ANCHOR,
        content=text,
        unit_ordinal=f"{document}#port",
        assertions=(_pair(fields, document=document, run=run, text=text),),
    )
    return await repository.record_document(
        document_id=document, sections=[section], run_id=run, assess=judge
    )


async def _rows(repository, model):
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import AsyncSession

    async with AsyncSession(repository._test_engine) as session:
        return list((await session.execute(select(model))).scalars().all())


USES = {"subject": "management_server", "predicate": "uses", "object": "application_port"}
USES_PORT = {
    "subject": "management_server", "predicate": "uses_port", "object": "application_port"
}


async def test_a_second_document_reinforces_rather_than_duplicates(repository) -> None:
    """Join equivalent claims from two documents after the judge confirms them."""
    judge = _Judge(SAME)

    first = await _record(repository, USES, document=DOC_A, run="run-1", judge=judge)
    second = await _record(repository, USES_PORT, document=DOC_B, run="run-2", judge=judge)

    assert judge.asked, "the pair never reached the judge"
    assert second.recorded[0].assertion_id == first.recorded[0].assertion_id
    live = [row for row in await _rows(repository, KnowledgeAssertion) if row.retired_at is None]
    assert len(live) == 1, "one claim in two documents is one assertion"


async def test_only_a_yes_joins_them(repository) -> None:
    # Retrieval brings a pair together. It does not decide. A judge that says
    # they are different leaves two assertions, exactly as before.
    judge = _Judge(DIFFERENT)

    first = await _record(repository, USES, document=DOC_A, run="run-1", judge=judge)
    second = await _record(repository, USES_PORT, document=DOC_B, run="run-2", judge=judge)

    assert judge.asked
    assert second.recorded[0].assertion_id != first.recorded[0].assertion_id


async def test_the_judgement_is_recorded_whatever_it_said(repository) -> None:
    judge = _Judge(DIFFERENT)

    await _record(repository, USES, document=DOC_A, run="run-1", judge=judge)
    await _record(repository, USES_PORT, document=DOC_B, run="run-2", judge=judge)

    [assessment] = await _rows(repository, KnowledgeEquivalenceAssessment)
    assert assessment.verdict == DIFFERENT
    assert assessment.classifier == "test"


async def test_both_propositions_survive_the_merge(repository) -> None:
    """Preserve both observed propositions after equivalent claims are joined."""
    from nlght.adapters.outbound.persistence.knowledge_models import KnowledgeProposition

    judge = _Judge(SAME)
    await _record(repository, USES, document=DOC_A, run="run-1", judge=judge)
    await _record(repository, USES_PORT, document=DOC_B, run="run-2", judge=judge)

    predicates = {
        str(row.fields.get("predicate")) for row in await _rows(repository, KnowledgeProposition)
    }

    assert predicates == {"uses", "uses_port"}


async def test_a_claim_sharing_one_term_is_never_offered(repository) -> None:
    """Require more than one shared term before offering a candidate to the judge."""
    judge = _Judge(SAME)
    await _record(repository, USES, document=DOC_A, run="run-1", judge=judge)

    unrelated = {
        "subject": "management_server", "predicate": "logs_to", "object": "stdout",
    }
    await _record(
        repository, unrelated, document=DOC_B, run="run-2", judge=judge,
        text="The management server logs to stdout.",
    )

    assert judge.asked == [], "one shared term is a coincidence, not a candidate"
    live = [row for row in await _rows(repository, KnowledgeAssertion) if row.retired_at is None]
    assert len(live) == 2


async def test_a_claim_of_another_kind_is_never_offered(repository) -> None:
    # `kind` is part of the identity namespace, so the answer is already `no` and
    # the judge is never asked a question nothing could change.
    judge = _Judge(SAME)
    await _record(repository, USES, document=DOC_A, run="run-1", judge=judge)

    rule = LineageWrite(
        entity=entity_key("rule", subject="management_server", rule_property="application_port"),
        kind="rule", text="The management server must use the application's port.",
        fingerprint="fp-rule", extraction_version="v1",
        document_id=DOC_B, document_revision="b-rev-1", document_path="testdocs/b.adoc",
        run_id="run-2",
        proposition=Proposition({
            "subject": "management_server", "rule_property": "application_port",
            "rule_text": "The management server must use the application's port.",
        }),
    )
    graph = KnowledgeWrite(
        identity="fp-rule",
        payload=None, proposition=rule.proposition, _kind="rule",
        type="config", confidence=0.9, source_id="corpus", run_id="run-2",
    )
    await repository.record_document(
        document_id=DOC_B,
        sections=[ObservedSection.as_read(
            anchor="b#port", anchor_strength=DERIVED_ANCHOR,
            content=rule.text,
            unit_ordinal="b#port", assertions=((graph, rule),),
        )],
        run_id="run-2", assess=judge,
    )

    assert judge.asked == []


async def test_a_third_sighting_of_the_joined_claim_still_finds_it(repository) -> None:
    # The merge has to hold: once two documents carry one assertion, a later run
    # over either of them continues that assertion rather than opening a third.
    judge = _Judge(SAME)
    first = await _record(repository, USES, document=DOC_A, run="run-1", judge=judge)
    await _record(repository, USES_PORT, document=DOC_B, run="run-2", judge=judge)

    again = await _record(repository, USES_PORT, document=DOC_B, run="run-3", judge=judge)

    assert again.recorded[0].assertion_id == first.recorded[0].assertion_id
    live = [row for row in await _rows(repository, KnowledgeAssertion) if row.retired_at is None]
    assert len(live) == 1


async def test_the_reach_is_capped(repository) -> None:
    """Not global, and the code says how far.

    A cap the corpus cannot exceed is what makes "bounded" true of the resolver
    rather than of this particular test data.
    """
    assert SqlAlchemyKnowledgeRepository.PLAUSIBLE_MINIMUM_TERMS == 2
    assert SqlAlchemyKnowledgeRepository.PLAUSIBLE_LIMIT == 8

    judge = _Judge(DIFFERENT)
    await _record(repository, USES, document=DOC_A, run="run-1", judge=judge)
    for index in range(12):
        await _record(
            repository,
            {"subject": "management_server", "predicate": f"relates_{index}",
             "object": "application_port"},
            document=f"doc-{index}", run=f"run-{index}", judge=judge,
            text=f"The management server relates {index} to the application's port.",
        )

    # The last sighting saw at most the cap, however many claims were plausible.
    assert len(judge.asked) <= SqlAlchemyKnowledgeRepository.PLAUSIBLE_LIMIT * 13
    last_round = [pair for pair in judge.asked if "relates 11" in pair[0]]
    assert len(last_round) <= SqlAlchemyKnowledgeRepository.PLAUSIBLE_LIMIT


async def test_a_claim_with_too_little_structure_reaches_nothing(repository) -> None:
    # A proposition with one field could match anything that mentions its one
    # value, so it is not offered a search at all.
    judge = _Judge(SAME)
    await _record(repository, USES, document=DOC_A, run="run-1", judge=judge)

    # With `predicate` excluded, only the subject offers overlap. Searching on
    # that alone would reach every claim about the management server.
    thin_fields = {"predicate": "exists", "subject": "management_server"}
    thin = LineageWrite(
        entity=entity_key("fact", **thin_fields),
        kind="fact", text="The management server exists.", fingerprint="fp-thin",
        extraction_version="v1", document_id=DOC_B, document_revision="b-1",
        document_path="testdocs/b.adoc", run_id="run-2",
        proposition=Proposition(dict(thin_fields)),
    )
    graph = KnowledgeWrite(
        identity="fp-thin",
        payload=None, proposition=thin.proposition, _kind="fact",
        type="config", confidence=0.9, source_id="corpus", run_id="run-2",
    )
    await repository.record_document(
        document_id=DOC_B,
        sections=[ObservedSection.as_read(
            anchor="b#exists", anchor_strength=DERIVED_ANCHOR,
            content=thin.text,
            unit_ordinal="b#exists", assertions=((graph, thin),),
        )],
        run_id="run-2", assess=judge,
    )

    assert judge.asked == []
