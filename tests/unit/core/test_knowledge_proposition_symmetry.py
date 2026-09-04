# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""A proposition is a property of every knowledge assertion, or of none.

It was a property of facts, and the asymmetry produced a special path everywhere
it touched: equivalence could not judge a `rule` because it took a
`FactProposition`; a revision could name a fact's structure and nothing else's;
every new surface had to ask which kind it was holding, and a fourth kind would
have needed a fifth answer.

Generalising it invents nothing, which is the reason it is safe. `fields` never
had privileged roles — no `subject`, no `predicate`, no ontology — so a rule's
fields are a rule's and the platform reads none of them as meaning anything. What
changed is that all four kinds now have the same three levels:

    Proposition           the structured interpretation
    observed_text         the wording, exactly as the source has it
    legacy payload        one consumer's shape, derived, lossless or absent

The last two must never be confused with the first. A proposition may not replace
the wording, and the wording may not be rebuilt from a proposition.
"""

from __future__ import annotations

import pytest

from nlght.core.knowledge import DECISION, FACT, PATTERN, RULE, ExtractedItem
from nlght.core.knowledge.knowledge_proposition import Proposition
from nlght.core.knowledge.legacy import project
from nlght.core.knowledge.projection import Represented

KINDS = (FACT, RULE, PATTERN, DECISION)

#: One well-formed candidate per kind, each with the sentence it was read from.
_CASES: dict[str, tuple[str, dict[str, str]]] = {
    FACT: (
        "Spring Boot requires Java 17.",
        {"subject": "Spring Boot", "predicate": "requires", "object": "Java 17"},
    ),
    RULE: (
        "Applications must enable graceful shutdown.",
        {"rule_text": "Applications must enable graceful shutdown.",
         "subject": "application", "rule_property": "shutdown_requirement"},
    ),
    PATTERN: (
        "Health statuses are exported as metrics so a dashboard can chart them.",
        {"pattern_name": "health_as_metrics", "description": "export statuses as metrics"},
    ),
    DECISION: (
        "We will use PostgreSQL for persistence.",
        {"decision": "Use PostgreSQL", "effect": "persistence is handled by PostgreSQL",
         "subject": "persistence", "decision_type": "datastore_choice"},
    ),
}


def _item(kind: str) -> ExtractedItem:
    wording, content = _CASES[kind]
    return ExtractedItem(
        kind=kind, type="t", content=dict(content), confidence=0.9, observed_text=wording
    )


@pytest.mark.parametrize("kind", KINDS)
def test_every_kind_produces_a_proposition(kind: str) -> None:
    proposition = _item(kind).proposition

    assert isinstance(proposition, Proposition)
    assert dict(proposition.fields) == _CASES[kind][1]


@pytest.mark.parametrize("kind", KINDS)
def test_the_proposition_keeps_everything_the_extraction_produced(kind: str) -> None:
    # Preserved, not interpreted. Nothing is dropped for being unfamiliar and
    # nothing is renamed into a role the platform prefers.
    item = _item(kind)

    assert set(item.proposition.fields) == set(item.content)


@pytest.mark.parametrize("kind", KINDS)
def test_a_fingerprint_addresses_the_structure_for_every_kind(kind: str) -> None:
    """Content address, and never an identity — the same rule for all four.

    Two claims with one fingerprint are byte-identical after normalisation. Two
    with different ones may still be the same claim, which is the question
    equivalence exists to ask.
    """
    wording, content = _CASES[kind]
    same = Proposition(dict(content))
    reordered = Proposition(dict(reversed(list(content.items()))))
    moved = Proposition({**content, "note": "something else"})

    assert same.fingerprint == reordered.fingerprint
    assert same.fingerprint != moved.fingerprint


@pytest.mark.parametrize("kind", KINDS)
def test_the_legacy_payload_is_derived_and_lossless(kind: str) -> None:
    # A projection of the proposition, for every kind — not a second thing
    # written beside it that could disagree.
    item = _item(kind)
    derived = project(kind, item.proposition)

    assert isinstance(derived, Represented)
    assert item.payload() is not None


@pytest.mark.parametrize("kind", KINDS)
def test_a_claim_the_legacy_row_cannot_hold_gets_none_rather_than_a_shortened_one(
    kind: str,
) -> None:
    """The n-ary case, now for every kind rather than for facts alone.

    A row that drops a field reads as a claim and is not the one that was made.
    So the projection refuses, the proposition keeps everything, and the claim is
    stored whole.
    """
    wording, content = _CASES[kind]
    richer = Proposition({**content, "qualifier": "in production only"})

    assert project(kind, richer).__class__.__name__ == "NotRepresentable"
    assert "qualifier" in dict(richer.fields)


@pytest.mark.parametrize("kind", KINDS)
def test_the_wording_and_the_proposition_move_independently(kind: str) -> None:
    """Both matter, for different reasons, and neither substitutes for the other.

        observed_text   what the source says, quotable
        proposition     the structured interpretation of it

    They may coincide — a rule's `rule_text` often *is* the sentence — and
    coinciding is not being one thing. What matters is that neither is derived
    from the other, so moving one leaves the other exactly where it was.
    """
    wording, content = _CASES[kind]
    item = _item(kind)

    assert item.observed_text
    assert item.verified_against(f"Intro.\n\n{item.observed_text}\n\nMore.")

    # The sentence moves; the structured claim does not.
    reworded = ExtractedItem(
        kind=kind, type="t", content=dict(content), confidence=0.9,
        observed_text=f"Put another way: {wording}",
    )
    assert reworded.proposition == item.proposition
    assert reworded.observed_text != item.observed_text

    # The structured claim moves; the sentence does not.
    reinterpreted = ExtractedItem(
        kind=kind, type="t", content={**content, "qualifier": "in production"},
        confidence=0.9, observed_text=wording,
    )
    assert reinterpreted.proposition != item.proposition
    assert reinterpreted.observed_text == item.observed_text


def test_no_production_code_still_speaks_of_a_fact_proposition() -> None:
    """The asymmetry is gone from the API, not merely unused.

    A name left behind is a special path waiting to be reintroduced by somebody
    who reads it as a hint that facts are different.
    """
    import pathlib  # noqa: PLC0415

    root = pathlib.Path(__file__).resolve().parents[3] / "src"
    offenders = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*.py")
        if "FactProposition" in path.read_text(encoding="utf-8")
    ]

    assert offenders == []


# ---------------------------------------------------------------------------
# The consequences, stated where they would be undone
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", KINDS)
def test_no_kind_gets_a_shortened_row(kind: str) -> None:
    """`NotRepresentable` is not a fact's privilege.

    A rule whose proposition carries something `knowledge_rules` has no column
    for must produce **no** row, exactly as a four-role fact does. Inventing a
    shortened one there would be the same defect the fact path was fixed for,
    and quieter — a rule row reads plausibly whatever is missing from it.
    """
    _, content = _CASES[kind]
    richer = Proposition({**content, "qualifier": "in production only"})

    item = ExtractedItem(
        kind=kind, type="t", content=dict(richer.fields), confidence=0.9,
        observed_text=_CASES[kind][0],
    )

    assert item.payload() is None
    # And nothing was lost: the claim is whole in the proposition.
    assert "qualifier" in dict(item.proposition.fields)


@pytest.mark.parametrize("kind", KINDS)
def test_the_wording_stays_out_of_the_content_address(kind: str) -> None:
    """`observed_text` belongs to the sighting, never to the proposition.

    Folding it into the fingerprint would make one claim into two whenever a
    source was reworded, which is the distinction the whole model rests on:

        same proposition + different wording
            → one claim, two observed forms, two revisions

    A content address that moved with the wording could not express that.
    """
    wording, content = _CASES[kind]
    proposition = Proposition(dict(content))

    assert "observed_text" not in dict(proposition.fields)
    # Two sightings of one claim, worded differently, address the same content.
    first = ExtractedItem(kind=kind, type="t", content=dict(content),
                          confidence=0.9, observed_text=wording)
    second = ExtractedItem(kind=kind, type="t", content=dict(content),
                           confidence=0.9, observed_text=f"Put another way: {wording}")

    assert first.proposition.fingerprint == second.proposition.fingerprint


def test_the_proposition_type_names_no_kind() -> None:
    # The guard against the asymmetry returning as a field. A `Proposition` that
    # knew its kind would invite kind-specific branches inside it, and the kind
    # is already reachable through the revision and the assertion.
    import dataclasses  # noqa: PLC0415

    names = {f.name for f in dataclasses.fields(Proposition)}

    assert names == {"fields"}
