# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""SessionResult → MentalElement.

The adapter carries the one judgement the renderer used to make inline: a result
tagged `user_fact` is a fact about the person and belongs under a different
heading, held harder, than a task outcome. That judgement is legitimate here and
nowhere downstream.
"""

from __future__ import annotations

from datetime import UTC, datetime

from nlght.core.hive_mind.elements import (
    SECTION_TASK_RESULTS,
    SECTION_USER_FACTS,
    result_elements,
)
from nlght.core.hive_mind.models import (
    MentalModel,
    RelevanceScore,
    Retention,
    ScoredResult,
    SessionResult,
)

from .conftest import assert_element_contract, assert_no_invented_short_form


def _model(*results: ScoredResult) -> MentalModel:
    return MentalModel(
        turn_id="t", built_at=datetime.now(UTC), known_results=list(results)
    )


def _result(content: str, *, tags: list[str] | None = None, score: float = 0.7):
    return ScoredResult(
        result=SessionResult(content=content, tags=tags or []),
        score=RelevanceScore(total=score),
    )


def test_a_user_fact_and_a_task_result_are_told_apart_by_the_adapter() -> None:
    elements = result_elements(_model(
        _result("the user prefers metric units", tags=["user_fact"]),
        _result("the build finished"),
    ))

    by_section = {e.presentation.section: e for e in elements}
    assert set(by_section) == {
        SECTION_USER_FACTS.section, SECTION_TASK_RESULTS.section,
    }
    assert "metric units" in by_section[SECTION_USER_FACTS.section].at(
        list(by_section[SECTION_USER_FACTS.section].levels)[0]
    ).text


def test_a_fact_about_the_person_is_held_harder_than_a_task_outcome() -> None:
    """Retention is about what losing it costs, and the two differ.

    A fact about the user outlives the task that produced it; a task result is
    about a moment that has passed. That is a statement about the information,
    which is why the adapter may make it — and why the reduction, which cannot,
    is simply told.
    """
    fact, outcome = result_elements(_model(
        _result("prefers metric units", tags=["user_fact"]),
        _result("the build finished"),
    ))

    assert fact.retention is Retention.IMPORTANT
    assert outcome.retention is Retention.USEFUL


def test_the_relevance_is_the_engine_s_score_and_not_a_new_one() -> None:
    element = result_elements(_model(_result("x", score=0.42)))[0]

    assert element.relevance == 0.42


def test_the_payload_is_carried_rather_than_converted() -> None:
    scored = _result("the build finished")

    element = result_elements(_model(scored))[0]

    assert element.payload is scored.result


def test_the_element_contract_holds() -> None:
    for element in result_elements(_model(
        _result("a fact", tags=["user_fact"]), _result("an outcome"),
    )):
        assert_element_contract(element)
        assert_no_invented_short_form(element)
