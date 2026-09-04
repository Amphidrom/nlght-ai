# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""WorkingAtom → MentalElement.

Two judgements live here: which atoms read as context rather than as planning,
and how an atom's embedded `[kind:…]` prefix is normalised. Both used to be in
the renderer.
"""

from __future__ import annotations

from datetime import UTC, datetime

from nlght.core.hive_mind.elements import (
    SECTION_PLANNING,
    SECTION_TASK_CONTEXT,
    context_atom_elements,
    planning_atom_elements,
)
from nlght.core.hive_mind.models import (
    AtomType,
    MentalModel,
    RelevanceScore,
    ScoredAtom,
    WorkingAtom,
)

from .conftest import assert_element_contract, assert_no_invented_short_form


def _atom(content: str, atom_type: AtomType = AtomType.RESULT,
          tags: list[str] | None = None, score: float = 0.8) -> ScoredAtom:
    return ScoredAtom(
        atom=WorkingAtom(atom_type=atom_type, content=content, task_id="t1",
                         tags=tags or []),
        score=RelevanceScore(total=score),
    )


def _model(*atoms: ScoredAtom) -> MentalModel:
    return MentalModel(turn_id="t", built_at=datetime.now(UTC), active_atoms=list(atoms))


def test_readable_atoms_become_context_and_the_rest_become_planning() -> None:
    model = _model(
        _atom("a result", AtomType.RESULT),
        _atom("a plan", AtomType.PLAN),
    )

    context = context_atom_elements(model)
    planning = planning_atom_elements(model)

    assert [e.presentation.section for e in context] == [SECTION_TASK_CONTEXT.section]
    assert [e.presentation.section for e in planning] == [SECTION_PLANNING.section]


def test_an_embedded_kind_prefix_is_shown_once_and_not_twice() -> None:
    # The content may already carry the prefix from whoever wrote it. Stripping
    # and re-adding is presentation embedded in stored data — not fixed here,
    # but not lost either.
    element = context_atom_elements(_model(
        _atom("[kind:decision] | we chose Postgres", tags=["kind:decision"])
    ))[0]

    text = element.at(element.levels[0]).text
    assert text.count("[kind:decision]") == 1
    assert "we chose Postgres" in text


def test_tags_filter_which_atoms_are_offered_at_all() -> None:
    model = _model(
        _atom("wanted", tags=["kind:decision"]),
        _atom("unwanted", tags=["kind:noise"]),
    )

    offered = context_atom_elements(model, tags=["kind:decision"])

    assert len(offered) == 1
    assert "wanted" in offered[0].at(offered[0].levels[0]).text


def test_the_reference_resolver_runs_before_any_cost_is_computed() -> None:
    """The placement is a correctness matter, not a tidiness one.

    A resolver may return different text — and a different number of entries.
    Running it after an element is priced would make every cost a lie and the
    budget guarantee with it, so the cost must be of what the resolver produced.
    """
    from nlght.core.hive_mind.models import estimate_tokens

    resolved = context_atom_elements(
        _model(_atom("short")),
        resolve_references=lambda entries: [e + " and a great deal more text" for e in entries],
    )

    element = resolved[0]
    text = element.at(element.levels[0]).text
    assert "a great deal more text" in text
    assert element.at(element.levels[0]).cost == estimate_tokens(text)


def test_the_element_contract_holds() -> None:
    model = _model(_atom("a result", AtomType.RESULT), _atom("a plan", AtomType.PLAN))
    for element in [*context_atom_elements(model), *planning_atom_elements(model)]:
        assert_element_contract(element)
        assert_no_invented_short_form(element)
