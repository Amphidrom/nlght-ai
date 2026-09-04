# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""The kind contract, held in place.

What this file can and cannot prove is worth stating, because the difference is
the whole reason the live corpus exists:

    CI          the contract is present, complete, and reaches the model
    live corpus the model actually classifies the gate matrix correctly

A string test cannot show that a model reads a boundary the way it was meant. It
shows that the boundary did not quietly disappear — which is a real failure mode,
since the contract is assembled from data and a deleted entry leaves every
remaining string looking correct.

So the expectations below are spelled out **independently** of the module they
check. A test that iterated `DEFINITIONS` and asserted each entry appears in the
prompt would pass after a definition was deleted: the loop would simply be
shorter. Naming the four here is what makes a deletion red.
"""

from __future__ import annotations

import pytest

from nlght.core.knowledge import DECISION, FACT, PATTERN, RULE
from nlght.core.knowledge.classification import (
    AMBIGUITY_GATES,
    BOUNDARIES,
    DEFINITIONS,
    GATE_MATRIX,
    contract,
)

KINDS = {FACT, RULE, DECISION, PATTERN}


def test_every_kind_is_defined_and_no_others_are() -> None:
    """Four definitions, named here so removing one fails."""
    assert {item.kind for item in DEFINITIONS} == KINDS
    assert len(DEFINITIONS) == len(KINDS), "a kind is defined twice"
    assert all(item.definition.strip() for item in DEFINITIONS)


def test_the_gate_matrix_has_exactly_one_clear_sentence_per_kind() -> None:
    # Read against each other, so each is the answer to "why not the neighbour".
    assert {item.kind for item in GATE_MATRIX} == KINDS
    assert len(GATE_MATRIX) == len(KINDS)


@pytest.mark.parametrize(
    "boundary",
    [
        "a sentence is not a rule merely because it sounds technical",
        "a conditional is not a rule",
        "a rule is not a decision merely because it recommends an action",
        "a decision is not a pattern merely because the choice would be reusable",
        "a pattern is not a fact merely because it describes what is often done",
        "useful architecture is not a pattern",
    ],
)
def test_each_boundary_stands(boundary: str) -> None:
    """The lines that are actually crossed.

    Every misclassification the corpus produced came from one of these
    resemblances rather than from a claim nobody could place, so each is named
    here rather than counted.
    """
    assert any(boundary in line for line in BOUNDARIES), boundary


@pytest.mark.parametrize(
    ("sentence", "kind"),
    [
        ("PostgreSQL can be used for persistence.", FACT),
        ("You should use PostgreSQL.", RULE),
        ("We use PostgreSQL.", FACT),
        ("Using a separate management port is a common approach.", PATTERN),
        ("Metrics are exported to Prometheus when the registry is on the classpath.", FACT),
        ("Schema scripts run before data scripts.", FACT),
        ("A layered jar separates dependencies from application classes.", FACT),
    ],
)
def test_each_ambiguous_case_keeps_its_answer(sentence: str, kind: str) -> None:
    # Named with the answer, so a case silently flipping is a failure rather
    # than a new convention.
    matched = [item for item in AMBIGUITY_GATES if item.sentence == sentence]

    assert matched, f"the contract no longer covers {sentence!r}"
    assert matched[0].kind == kind
    assert matched[0].because.strip(), "an answer without a reason teaches nothing"


def test_every_contrast_names_a_real_kind() -> None:
    for item in (*GATE_MATRIX, *AMBIGUITY_GATES):
        assert item.kind in KINDS, item


def test_the_rendered_contract_carries_every_part() -> None:
    """Assembled, not written out — so the prompt cannot drift from the data."""
    rendered = contract()

    for item in DEFINITIONS:
        assert item.kind in rendered
        assert item.definition in rendered
    for line in BOUNDARIES:
        assert line in rendered
    for item in (*GATE_MATRIX, *AMBIGUITY_GATES):
        assert item.sentence in rendered
        assert item.because in rendered


def test_the_contract_states_that_one_sentence_may_say_several_things() -> None:
    """Atomicity belongs to the contract, not beside it.

    A kind is a property of an assertion and never of a sentence, so a sentence
    asserting two things yields two claims with two kinds — and they share their
    wording, which is correct rather than a duplicate.
    """
    rendered = contract().lower()

    assert "several atomic assertions" in rendered
    assert "classified separately" in rendered
    assert "same observed_text" in rendered


def test_exactly_one_kind_is_demanded() -> None:
    assert "exactly one kind per assertion" in contract().lower()
