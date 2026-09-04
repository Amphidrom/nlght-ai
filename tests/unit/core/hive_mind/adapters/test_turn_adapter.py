# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""TurnSummary → MentalElement.

Note which "summary" this is. `MentalModel.summary` is a line about the *current*
turn and is framing — it describes the moment rather than the world, and is not
reducible. `TurnSummary` objects in `recent_turns` are past conversation, which is
knowledge, and they go through this adapter.
"""

from __future__ import annotations

from datetime import UTC, datetime

from nlght.core.hive_mind.elements import SECTION_CONVERSATION, turn_elements
from nlght.core.hive_mind.models import MentalModel, Retention, TurnSummary

from .conftest import assert_element_contract, assert_no_invented_short_form


def _model(*turns: TurnSummary) -> MentalModel:
    return MentalModel(turn_id="t", built_at=datetime.now(UTC), recent_turns=list(turns))


def _turn(nr: int, user: str = "", result: str = "") -> TurnSummary:
    return TurnSummary(turn_nr=nr, user_input=user, intent="ask", topic="t",
                       result_summary=result)


def test_only_the_last_three_turns_are_offered() -> None:
    # The slice is the prompt's long-standing behaviour and is kept deliberately:
    # how much conversation a model sees is a prompt decision, not something to
    # change while removing a coupling.
    elements = turn_elements(_model(*(_turn(n, f"turn {n}") for n in range(1, 6))))

    assert [e.element_id for e in elements] == ["turn:3", "turn:4", "turn:5"]


def test_a_turn_reads_as_one_line_built_from_its_fields() -> None:
    element = turn_elements(_model(_turn(7, "what is the limit?", "100 characters")))[0]

    text = element.at(element.levels[0]).text
    assert "[7]" in text
    assert "user: what is the limit?" in text
    assert "→ 100 characters" in text


def test_conversation_is_held_harder_than_a_passing_result() -> None:
    """Losing the middle of a conversation is worse than saying less about it.

    A question is asked *within* a conversation, so a gap in it changes what the
    remaining turns appear to mean.
    """
    element = turn_elements(_model(_turn(1, "hello")))[0]

    assert element.retention is Retention.IMPORTANT
    assert element.presentation.section == SECTION_CONVERSATION.section


def test_the_element_contract_holds() -> None:
    for element in turn_elements(_model(_turn(1, "a"), _turn(2, "b"))):
        assert_element_contract(element)
        assert_no_invented_short_form(element)
