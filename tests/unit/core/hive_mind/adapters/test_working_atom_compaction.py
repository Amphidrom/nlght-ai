# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""The first honest compact representation in the platform.

Until now every adapter offered `FULL` and `OMIT`, so the reduction could forget
but never shorten — half of what the architecture was built for was unproven. A
working atom can now be said briefly, and briefly means *the subject without the
substance*, built from fields the atom has:

    FULL     [SPEC] candidate:ssl-keystore-path | the keystore lives at …
    COMPACT  [SPEC] [fact] candidate:ssl-keystore-path

Not a truncation. A truncation ends mid-claim and reads as a quotation of
something the source never finished saying.
"""

from __future__ import annotations

from datetime import UTC, datetime

from nlght.core.hive_mind.elements import context_atom_elements, planning_atom_elements
from nlght.core.hive_mind.models import (
    AtomType,
    Level,
    MentalModel,
    RelevanceScore,
    Retention,
    ScoredAtom,
    WorkingAtom,
)
from nlght.core.hive_mind.relevance import reduce_to_budget

from .conftest import assert_element_contract

LONG = "the keystore lives at /etc/ssl/private and is read at start-up by the server"


def _atom(*, key: str = "ssl-keystore-path", kind: str = "fact",
          content: str = LONG, atom_type: AtomType = AtomType.RESULT) -> ScoredAtom:
    return ScoredAtom(
        atom=WorkingAtom(atom_type=atom_type, content=content, task_id="t",
                         key=key, kind=kind),
        score=RelevanceScore(total=0.8),
    )


def _model(*atoms: ScoredAtom) -> MentalModel:
    return MentalModel(turn_id="t", built_at=datetime.now(UTC), active_atoms=list(atoms))


def _element(**kwargs):  # noqa: ANN003, ANN202
    return context_atom_elements(_model(_atom(**kwargs)))[0]


# What the atom offers
# ---------------------------------------------------------------------------

def test_an_atom_with_a_key_can_be_said_briefly() -> None:
    element = _element()

    assert element.levels == (Level.FULL, Level.COMPACT, Level.OMIT)
    brief = element.at(Level.COMPACT)
    assert "candidate:ssl-keystore-path" in brief.text
    assert "[fact]" in brief.text
    assert "/etc/ssl/private" not in brief.text, "the substance is what was given up"


def test_the_brief_form_is_derived_and_never_a_quotation() -> None:
    element = _element()

    assert element.at(Level.COMPACT).derived is True
    assert element.at(Level.FULL).derived is False


def test_an_atom_without_a_key_offers_no_brief_form() -> None:
    # Nothing to build one from, so nothing is offered — rather than parsing a
    # shorthand back out of the content, which is where it used to hide.
    element = _element(key="")

    assert element.levels == (Level.FULL, Level.OMIT)


def test_a_brief_form_that_would_not_be_cheaper_is_not_offered() -> None:
    """Uniformity is not worth a representation that saves nothing.

    `MentalElement` refuses an element whose shorter form costs more, so an
    adapter that produced one to keep the shapes tidy would fail outright. This
    is that guarantee, honoured on the way in.
    """
    element = _element(content="x", key="a-very-long-candidate-key-indeed")

    assert element.levels == (Level.FULL, Level.OMIT)


def test_planning_atoms_compact_the_same_way() -> None:
    element = planning_atom_elements(_model(_atom(atom_type=AtomType.SPEC)))[0]

    assert Level.COMPACT in element.levels


def test_the_element_contract_still_holds_with_three_levels() -> None:
    assert_element_contract(_element())


# What the reduction now does with it
# ---------------------------------------------------------------------------

def test_the_budget_walks_full_then_compact_then_gone() -> None:
    element = _element()
    full = element.at(Level.FULL).cost
    brief = element.at(Level.COMPACT).cost

    assert reduce_to_budget([element], budget=full).kept[0][1].level is Level.FULL
    assert reduce_to_budget([element], budget=brief).kept[0][1].level is Level.COMPACT
    assert reduce_to_budget([element], budget=brief - 1).forgotten == (element,)


def test_an_important_atom_is_shortened_while_useful_ones_are_dropped() -> None:
    """The behaviour the whole architecture was built for, finally demonstrable.

    Under pressure the important thing is said briefly rather than lost, and what
    pays for it is the merely useful being forgotten. Neither could be shown
    before, because no adapter offered a middle level.
    """
    from dataclasses import replace

    important = replace(_element(), element_id="important", retention=Retention.IMPORTANT)
    useful = [
        replace(_element(key=f"k{n}"), element_id=f"useful:{n}", retention=Retention.USEFUL)
        for n in range(3)
    ]

    view = reduce_to_budget([important, *useful], budget=important.at(Level.COMPACT).cost)

    kept = {element.element_id: representation.level for element, representation in view.kept}
    assert kept == {"important": Level.COMPACT}
    assert {element.element_id for element in view.forgotten} == {
        "useful:0", "useful:1", "useful:2",
    }


def test_the_provenance_of_a_shortened_atom_is_its_own_atom_still() -> None:
    scored = _atom()
    element = context_atom_elements(_model(scored))[0]

    view = reduce_to_budget([element], budget=element.at(Level.COMPACT).cost)

    assert view.kept[0][0].payload is scored.atom
