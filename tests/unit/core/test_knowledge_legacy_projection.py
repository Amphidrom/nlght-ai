# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Derive legacy payloads without losing proposition fields.

A representable proposition yields a lossless legacy view. Other shapes stay
complete in their canonical representation.
"""

from __future__ import annotations

from nlght.core.knowledge import NotRepresentable, Proposition, Represented
from nlght.core.knowledge.legacy import legacy_fact_payload


def _p(**fields: object) -> Proposition:
    return Proposition(fields)


def test_the_shape_the_corpus_is_full_of_still_makes_its_old_row() -> None:
    """Backwards compatible, and that is the point of deriving rather than replacing."""
    result = legacy_fact_payload(_p(subject="spring", predicate="requires", object="maven"))

    assert isinstance(result, Represented)
    assert (result.value.subject, result.value.predicate, result.value.object) == (
        "spring",
        "requires",
        "maven",
    )


def test_a_claim_with_more_parts_makes_no_row_rather_than_a_shortened_one() -> None:
    """"Alice transfers CHF 500 to Bob".

    Cut down to fit, it becomes "Alice transfers CHF 500" or "Alice transfers to
    Bob": well-formed, reading as true, and not what the source said. The
    proposition is stored whole; the legacy view simply has nothing to show.
    """
    result = legacy_fact_payload(
        _p(subject="Alice", predicate="transfers", amount="CHF 500", recipient="Bob")
    )

    assert isinstance(result, NotRepresentable)
    assert "amount" in result.reason and "recipient" in result.reason


def test_a_refusal_names_what_the_claim_carries() -> None:
    result = legacy_fact_payload(
        _p(subject="Alice", predicate="transfers", amount="CHF 500", recipient="Bob")
    )

    assert isinstance(result, NotRepresentable)
    assert result.available == ("amount", "predicate", "recipient", "subject")


def test_a_claim_whose_parts_are_named_differently_makes_no_row() -> None:
    """Avoid guessing legacy field roles from unfamiliar proposition names."""
    result = legacy_fact_payload(_p(actor="Alice", action="transfers", target="Bob"))

    assert isinstance(result, NotRepresentable)


def test_a_missing_part_makes_no_row_rather_than_an_empty_column() -> None:
    result = legacy_fact_payload(_p(subject="spring", predicate="requires"))

    assert isinstance(result, NotRepresentable)
    assert "no object" in result.reason


def test_a_derived_row_says_exactly_what_the_proposition_says() -> None:
    """Require a derived legacy row to preserve the complete proposition."""
    proposition = _p(subject="spring", predicate="requires", object="maven")

    result = legacy_fact_payload(proposition)

    assert isinstance(result, Represented)
    derived = {
        "subject": result.value.subject,
        "predicate": result.value.predicate,
        "object": result.value.object,
    }
    assert derived == dict(proposition.fields)


def test_deriving_twice_gives_the_same_row() -> None:
    # A derivation depends only on the proposition. Clock, run, and state
    # cannot affect it, or repeated derivations could disagree silently.
    proposition = _p(subject="spring", predicate="requires", object="maven")

    assert legacy_fact_payload(proposition) == legacy_fact_payload(proposition)


def test_only_the_legacy_module_names_the_legacy_shape() -> None:
    """The guard on where this knowledge is allowed to live.

    `subject`/`predicate`/`object` is the old payload's shape. Naming it here,
    where the old payload is built, is honest; naming it in `projection.py` made
    it look like a privileged form of a proposition, and that had to be taken
    back out once already.
    """
    import inspect

    import nlght.core.knowledge.legacy as legacy
    import nlght.core.knowledge.projection as projection

    shape_words = {"subject", "predicate", "object", "triple"}

    def names_a_shape(module: object) -> bool:
        body = inspect.getsource(module)
        return any(f'"{word}"' in body for word in shape_words)

    assert names_a_shape(legacy), "the legacy payload's shape is named where it is built"
    assert not names_a_shape(projection), "and nowhere else in core"
