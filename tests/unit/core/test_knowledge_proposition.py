# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Holding what was extracted, whole, without deciding what it means.

The one question this answers:

    can we keep, completely, what the extraction produced?

Not what a proposition is made of, not which field names a claim of a kind
should have, not which of them carry identity, not whether two differently shaped
claims say the same thing. Each of those is a later question, and answering one
here by accident is how the last three abstractions in this area went wrong:

    the entity key was the model's word choice           (ADR-0043)
    `predicate + roles` would have been its label choice
    `as_triple` made the legacy payload the privileged form

So the rule the tests below enforce:

    **Representation may preserve meaning. It may not invent any.**

Unstable field names are not a defect here. A model that says `recipient` on one
pass and `target` on the next has described one claim two ways, and noticing that
is equivalence — a separate question, on two complete structures, which is only
askable once both can be held completely.

`predicate` gets no privilege. It is a field the current extraction happens to
produce, not a distinguished part of a proposition.
"""

from __future__ import annotations

from nlght.core.knowledge import Proposition


def test_the_shape_the_corpus_is_full_of_survives_a_round_trip() -> None:
    fields = {"subject": "spring", "predicate": "requires", "object": "java 17"}

    restored = Proposition.from_dict(Proposition(fields).as_dict())

    assert restored.fields == fields


def test_a_claim_with_roles_the_payload_cannot_hold_survives() -> None:
    """The claim this slice exists for.

    "Alice transfers CHF 500 to Bob" has four parts, a well-formed entity key and
    nowhere to be stored — a model that read the sentence correctly had its
    answer discarded as malformed.
    """
    fields = {
        "subject": "Alice",
        "predicate": "transfers",
        "amount": "CHF 500",
        "recipient": "Bob",
    }

    restored = Proposition.from_dict(Proposition(fields).as_dict())

    assert restored.fields == fields


def test_field_names_nothing_recognises_survive() -> None:
    """No allow-list, and no notion of a name being wrong.

    Whether `zeta` is a sensible role is not a representation question. Deciding
    it here would be inventing meaning, and dropping the field would be losing
    what the extraction actually said.
    """
    fields = {"a": "1", "zeta": "2", "role_47": "3"}

    restored = Proposition.from_dict(Proposition(fields).as_dict())

    assert restored.fields == fields


def test_structure_and_order_inside_a_value_survive_exactly() -> None:
    """If the extraction expressed order, it is not lost.

    Which is *not* a claim that a relation's arguments are ordered. It says only
    that where order was expressed, it comes back.
    """
    fields = {
        "predicate": "requires",
        "sequence": ["first", "second", "third"],
        "detail": {"unit": "CHF", "amount": "500", "bounds": ["min", "max"]},
    }

    restored = Proposition.from_dict(Proposition(fields).as_dict())

    assert list(restored.fields["sequence"]) == ["first", "second", "third"]
    assert list(restored.fields["detail"]["bounds"]) == ["min", "max"]
    assert restored.fields["detail"]["unit"] == "CHF"


def test_the_order_of_field_names_carries_no_meaning() -> None:
    """A mapping is a mapping.

    Two extractions that named the same parts in a different order described the
    same structure, and treating that as a difference would make every claim
    depend on how a model happened to emit its JSON.
    """
    first = Proposition({"subject": "Alice", "predicate": "transfers", "recipient": "Bob"})
    second = Proposition({"recipient": "Bob", "predicate": "transfers", "subject": "Alice"})

    assert first == second


def test_the_order_inside_a_list_does_carry_meaning() -> None:
    # The other side of the same line. An explicit list is an expression of
    # order, and losing it would be losing something the extraction said.
    assert Proposition({"steps": ["a", "b"]}) != Proposition({"steps": ["b", "a"]})


def test_predicate_is_not_privileged() -> None:
    """It is a field the current extraction happens to produce.

    Treating it as a distinguished part would be the field form deciding the
    model — the same move as `predicate + roles`, arriving one layer down.
    """
    with_predicate = Proposition({"predicate": "transfers", "subject": "Alice"})
    without = Proposition({"action": "transfers", "actor": "Alice"})

    assert set(with_predicate.fields) == {"predicate", "subject"}
    assert set(without.fields) == {"action", "actor"}
    assert not hasattr(with_predicate, "predicate")


def test_two_namings_of_one_claim_are_not_merged_here() -> None:
    """Equivalence is a later question, on two complete structures.

    `recipient` and `target` may well express one role. Deciding that during
    representation would be inventing meaning, and it is exactly the defect that
    keeps coming back in new clothes.
    """
    named_one_way = Proposition({"predicate": "transfers", "recipient": "Bob"})
    named_another = Proposition({"predicate": "transfers", "target": "Bob"})

    assert named_one_way != named_another


def test_an_empty_proposition_is_refused() -> None:
    # A claim that says nothing is not a claim. This is the one judgement
    # representation is allowed: that there is something to hold.
    import pytest  # noqa: PLC0415

    with pytest.raises(ValueError, match="at least one field"):
        Proposition({})
