# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Nothing is forgotten before it becomes an element.

The reduction was made blind to what an element is made of. Deciding what may
*reach* the reduction was not: a score threshold and a per-storage-type count
removed candidates first, before retention existed and before anything could weigh
what losing them would cost.

The defect is reproduced below rather than described. It is the reason this slice
exists, and it is the test that must never go green for the wrong reason.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from nlght.core.hive_mind.builder import ContextSnapshot, MentalModelBuilder
from nlght.core.hive_mind.elements import result_elements, turn_elements
from nlght.core.hive_mind.models import (
    Level,
    MentalElement,
    Presentation,
    RelevanceInputs,
    Representation,
    Retention,
    SessionResult,
    TurnSummary,
)
from nlght.core.hive_mind.relevance import RelevanceEngine, ScoringStrategy, reduce_to_budget


def _built(snapshot: ContextSnapshot, **kwargs):  # noqa: ANN003, ANN202
    return MentalModelBuilder(relevance_engine=RelevanceEngine(**kwargs)).build(
        turn_id="t", context=snapshot,
        signal_entities=["keystore"], intents=["configure"],
    )


# The defect, exactly as it was found
# ---------------------------------------------------------------------------

def test_a_durable_user_fact_is_not_lost_to_a_count() -> None:
    """Thirteen results stored, and the important one was the eleventh.

    `max_results = 10` removed it before the adapter could mark it IMPORTANT and
    before the reduction could decide whether it was worth its room. Twelve task
    results — merely USEFUL — survived in its place, for no reason except that
    they were newer and there were more of them.
    """
    now = datetime.now(UTC)
    tasks = [
        SessionResult(content=f"task result {n}", entities=["keystore"], created_at=now)
        for n in range(12)
    ]
    durable = SessionResult(
        content="the user prefers metric units", entities=["units"],
        tags=["user_fact"], created_at=now - timedelta(hours=1),
    )

    model = _built(ContextSnapshot(known_results=[*tasks, durable]))
    elements = result_elements(model)

    assert len(model.known_results) == 13, "nothing was discarded"
    kept = {element.at(Level.FULL).text for element in elements}
    assert any("metric units" in text for text in kept)
    important = [e for e in elements if e.retention is Retention.IMPORTANT]
    assert len(important) == 1, "and it reached the engine as IMPORTANT"


def test_a_low_scoring_element_still_reaches_the_engine() -> None:
    """Score 0.14 under LINEAR, and it used to be below the 0.15 threshold.

    "This does not match the moment" is not "this may be forgotten". Only
    retention and what else wants the room can answer the second, and the
    threshold answered it first.
    """
    now = datetime.now(UTC)
    unrelated = SessionResult(
        content="the user prefers metric units", entities=["units"],
        tags=["user_fact"], created_at=now - timedelta(hours=1),
    )

    model = _built(
        ContextSnapshot(known_results=[unrelated]),
        strategy=ScoringStrategy.LINEAR,
    )

    assert model.known_results[0].score.total < 0.15
    assert len(result_elements(model)) == 1


def test_every_turn_reaches_the_engine() -> None:
    # The third storage type had a third rule, and it was the only one that never
    # consulted relevance: turns were cut by position.
    turns = [
        TurnSummary(turn_nr=n, user_input=f"turn {n}", intent="ask", topic="t")
        for n in range(20)
    ]

    model = _built(ContextSnapshot(recent_turns=turns))

    assert len(model.recent_turns) == 20
    # The adapter still shows the last three, which is a *presentation* decision
    # about how much conversation a model reads — not a discard, and reversible
    # without anything having been thrown away.
    assert len(turn_elements(model)) == 3


def test_all_three_sorts_arrive_in_one_pool() -> None:
    now = datetime.now(UTC)
    model = _built(ContextSnapshot(
        known_results=[SessionResult(content="a result", created_at=now)],
        recent_turns=[TurnSummary(turn_nr=1, user_input="a turn", intent="ask", topic="t")],
    ))

    pool = [*result_elements(model), *turn_elements(model)]

    assert len(pool) == 2
    # One reduction over both, deciding between them rather than allocating to
    # each.
    view = reduce_to_budget(pool, budget=1_000)
    assert len(view.kept) == 2


# The selection is type-blind too, now
# ---------------------------------------------------------------------------

def _element(kind: str, element_id: str) -> MentalElement:
    return MentalElement(
        element_id=element_id, kind=kind,
        representations=(
            Representation(level=Level.FULL, text="  - identical", cost=6),
            Representation(level=Level.OMIT, text="", cost=0),
        ),
        presentation=Presentation("A heading:", order=10),
        retention=Retention.USEFUL,
        signals=RelevanceInputs(entities=("alpha",), identity=element_id),
    )


def test_identical_signals_are_ranked_identically_whatever_the_kind() -> None:
    """Selection is now as blind as reduction was made.

    Two elements alike in everything the engine may look at, differing only in
    what sort of thing they are, must be scored and ordered the same.
    """
    engine = RelevanceEngine()
    now = datetime.now(UTC)

    as_memory = engine.rank(
        [_element("result", "a"), _element("atom", "b")], ["alpha"], "x", now
    )
    as_passage = engine.rank(
        [_element("passage", "a"), _element("turn", "b")], ["alpha"], "x", now
    )

    assert [e.element_id for e in as_memory] == [e.element_id for e in as_passage]
    assert {e.relevance for e in as_memory} == {e.relevance for e in as_passage}


def test_ranking_returns_everything_it_was_given() -> None:
    engine = RelevanceEngine()
    elements = [_element("result", f"e{n}") for n in range(50)]

    assert len(engine.rank(elements, [], "x", datetime.now(UTC))) == 50
