# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Specify equivalence decisions between complete propositions.

Only an affirmative judgement permits a merge. Rejection, ambiguity, and the
absence of a judgement remain distinct outcomes.
"""

from __future__ import annotations

from nlght.core.knowledge import Proposition
from nlght.core.knowledge.equivalence import (
    AMBIGUOUS,
    DIFFERENT,
    SAME,
    UNJUDGED,
    AssertionObservation,
    Judge,
    PropositionEquivalence,
    equivalent,
    may_merge,
    settled,
)


def _p(**fields: object) -> AssertionObservation:
    """Build an observation from proposition fields.

    Args:
        **fields (object): Named proposition fields.

    Returns:
        AssertionObservation: Observation containing the proposition.
    """
    return AssertionObservation.of(Proposition(fields))


def test_the_same_fields_are_the_same_proposition() -> None:
    assert equivalent(
        _p(subject="spring", predicate="requires", object="maven"),
        _p(subject="spring", predicate="requires", object="maven"),
    ) == SAME


def test_field_order_does_not_make_two_propositions() -> None:
    # A claim must not depend on how a model emitted its JSON.
    assert equivalent(
        _p(subject="spring", predicate="requires", object="maven"),
        _p(object="maven", predicate="requires", subject="spring"),
    ) == SAME


def test_a_moved_quantity_is_a_different_proposition() -> None:
    """Java 17 against Java 21, and no judgement is needed or wanted.

    Whether the second supersedes the first is a lineage question. Whether they
    are one claim is not: they are two.
    """
    assert equivalent(
        _p(subject="spring", predicate="requires", object="java 17"),
        _p(subject="spring", predicate="requires", object="java 21"),
    ) == DIFFERENT


def test_a_quantity_buried_in_the_structure_still_counts() -> None:
    # A comparison that only looked at the top level would call these one claim.
    assert equivalent(
        _p(predicate="limits", detail={"unit": "CHF", "amount": "500"}),
        _p(predicate="limits", detail={"unit": "CHF", "amount": "550"}),
    ) == DIFFERENT


def test_a_judgement_cannot_talk_a_moved_quantity_into_being_one_claim() -> None:
    # The decidable answers are not opinions.
    assert equivalent(
        _p(subject="spring", predicate="requires", object="java 17"),
        _p(subject="spring", predicate="requires", object="java 21"),
        verdict=SAME,
    ) == DIFFERENT


def test_active_and_passive_are_one_claim_when_the_judgement_says_so() -> None:
    """"Spring requires Maven" and "Maven is required by Spring"."""
    assert equivalent(
        _p(subject="spring", predicate="requires", object="maven"),
        _p(subject="maven", predicate="required_by", object="spring"),
        verdict=SAME,
    ) == SAME


def test_renamed_roles_are_one_claim_when_the_judgement_says_so() -> None:
    # `key/value/origin` against `data_key/data_value/property_source` is a
    # difference in extraction labels, not in the claim, and *that* is what
    # ADR-0043 removed from identity, arriving here where it can be decided
    # rather than assumed.
    assert equivalent(
        _p(predicate="provides access to", subject="SanitizableData",
           key="key", value="value", origin="PropertySource"),
        _p(predicate="provides access to", subject="SanitizableData",
           data_key="key", data_value="value", property_source="PropertySource"),
        verdict=SAME,
    ) == SAME


def test_swapped_roles_are_two_claims_when_the_judgement_says_so() -> None:
    """Keep swapped roles distinct when the judge rejects equivalence."""
    assert equivalent(
        _p(predicate="transfers", sender="Alice", recipient="Bob"),
        _p(predicate="transfers", sender="Bob", recipient="Alice"),
        verdict=DIFFERENT,
    ) == DIFFERENT


def test_a_swap_and_a_rename_are_the_same_shape() -> None:
    """The measurement behind the decision above, asserted so it is not re-argued.

    If a rule over the structure could tell these apart, the swap would be
    decidable and no judgement would be needed for it.
    """
    from itertools import permutations

    def renaming_exists(left: dict[str, str], right: dict[str, str]) -> bool:
        names = list(left)
        return any(
            all(left[a] == right[b] for a, b in zip(names, order, strict=True))
            for order in permutations(right)
        )

    swap = ({"sender": "Alice", "recipient": "Bob"}, {"sender": "Bob", "recipient": "Alice"})
    rename = ({"sender": "Alice", "recipient": "Bob"}, {"actor": "Alice", "target": "Bob"})

    assert renaming_exists(*swap)
    assert renaming_exists(*rename)


def test_doubt_never_merges() -> None:
    """Keep ambiguous equivalence decisions from merging claims."""
    assert equivalent(
        _p(predicate="transfers", sender="Alice", recipient="Bob"),
        _p(predicate="transfers", actor="Alice", target="Bob"),
        verdict=AMBIGUOUS,
    ) == AMBIGUOUS


def test_nobody_asked_is_its_own_answer() -> None:
    """Not doubt, not permission, and above all not `no`.

    Storing it as `no` would let the record claim later that something decided
    these were different claims, when the truth is that no classifier was
    configured. A trail that cannot tell those apart is worse than none: it reads
    as evidence.
    """
    result = equivalent(
        _p(predicate="transfers", sender="Alice", recipient="Bob"),
        _p(predicate="transfers", actor="Alice", target="Bob"),
    )

    assert result == UNJUDGED
    assert not may_merge(result)


def test_all_three_ways_of_not_being_yes_leave_the_claims_apart() -> None:
    # Four states, one operational consequence. Distinct because they mean
    # different things to a reader, collapsed because they mean the same thing to
    # a writer.
    assert may_merge(SAME)
    assert not may_merge(DIFFERENT)
    assert not may_merge(AMBIGUOUS)
    assert not may_merge(UNJUDGED)


def test_a_decision_is_about_a_pair_and_says_who_made_it() -> None:
    """Record an equivalence decision for a pair with its judge."""
    left = Proposition({"predicate": "transfers", "sender": "Alice", "recipient": "Bob"})
    right = Proposition({"predicate": "transfers", "actor": "Alice", "target": "Bob"})

    decision = PropositionEquivalence.between(
        left, right, SAME,
        judge=Judge(classifier="proposition-equivalence", model="qwen3.8:27B", version="1"),
    )

    assert decision.left != decision.right
    assert decision.may_merge
    assert str(decision.judge) == "proposition-equivalence/qwen3.8:27B/1"


def test_a_pair_is_stored_one_way_round() -> None:
    """Store each unordered comparison pair once."""
    left = Proposition({"predicate": "transfers", "sender": "Alice", "recipient": "Bob"})
    right = Proposition({"predicate": "transfers", "actor": "Alice", "target": "Bob"})

    forward = PropositionEquivalence.between(left, right, SAME)
    backward = PropositionEquivalence.between(right, left, SAME)

    assert (forward.left, forward.right) == (backward.left, backward.right)


def test_an_unjudged_decision_names_no_judge() -> None:
    # `None` rather than a placeholder, because a placeholder is something a
    # reader can mistake for a judge.
    decision = PropositionEquivalence(left="a", right="b", verdict=UNJUDGED)

    assert decision.judge is None
    assert not decision.may_merge


def test_a_fingerprint_refers_to_a_structure_and_is_not_an_identity() -> None:
    """Treat a fingerprint as a structural key, not claim identity."""
    # The fingerprint belongs to the proposition, which is a *fact's*
    # representation, not to the claim wrapper the judge is handed.
    same_words = Proposition(
        {"subject": "spring", "predicate": "requires", "object": "maven"}
    )
    reordered = Proposition(
        {"object": "maven", "predicate": "requires", "subject": "spring"}
    )
    one_claim_named_twice = Proposition(
        {"subject": "maven", "predicate": "required_by", "object": "spring"}
    )

    assert same_words.fingerprint == reordered.fingerprint
    assert same_words.fingerprint != one_claim_named_twice.fingerprint
    assert equivalent(
        AssertionObservation.of(same_words), AssertionObservation.of(one_claim_named_twice), verdict=SAME
    ) == SAME


def test_nothing_canonical_is_produced() -> None:
    """Keep equivalence judgement separate from canonicalisation."""
    import nlght.core.knowledge.equivalence as equivalence

    exported = {name for name in dir(equivalence) if not name.startswith("_")}

    assert not exported & {"canonical", "canonicalise", "merge", "normal_form", "unify"}


def _observed(kind: str, text: str, **representation: object):
    return AssertionObservation(kind=kind, observed_text=text, representation=representation)


def test_the_same_assertion_said_differently_may_continue() -> None:
    """Continue an assertion across different wordings when judged equivalent."""
    old = _observed("rule", "you have to set up the environment when building the application",
                    subject="aot_conditions", rule_property="environment_setup")
    new = _observed("rule", "the environment must be set up at build time",
                    subject="ahead_of_time", rule_property="build_environment")

    assert settled(old, new) is None, "nothing structural can answer this"
    assert may_merge(equivalent(old, new, verdict=SAME))


def test_a_different_assertion_does_not_continue() -> None:
    # Same shape, same kind, different claim about the world.
    old = _observed("rule", "The grace period must not exceed 30 seconds.",
                    subject="shutdown", rule_property="grace_period", rule_text="30 seconds")
    new = _observed("rule", "The grace period must not exceed 60 seconds.",
                    subject="shutdown", rule_property="grace_period", rule_text="60 seconds")

    # Decided without a model: a moved quantity is a different claim.
    assert settled(old, new) == DIFFERENT
    assert not may_merge(equivalent(old, new))


def test_one_sentence_carrying_two_assertions_is_not_one_assertion() -> None:
    """Keep two assertions from one sentence distinct."""
    sentence = (
        "When shutdown is enabled, the application waits for requests "
        "and rejects new requests."
    )
    waits = _observed("fact", sentence,
                      subject="application", predicate="waits_for", object="active_requests")
    rejects = _observed("fact", sentence,
                        subject="application", predicate="rejects", object="new_requests")

    assert waits.observed_text == rejects.observed_text
    # Not settled as the same, and not merged without a judge saying so.
    assert settled(waits, rejects) is None
    assert not may_merge(equivalent(waits, rejects))
    # And a judge that said no is respected, sentence or no sentence.
    assert not may_merge(equivalent(waits, rejects, verdict=DIFFERENT))


def test_two_kinds_are_settled_without_a_model() -> None:
    """`kind` is part of the identity namespace, so the answer is already known.

    Decided rather than refused: `no` is the true answer, and settling it here
    means the model is never asked a question nothing could change.
    """
    rule = _observed("rule", "Secrets must never be logged.",
                     subject="secret", rule_property="logging_prohibition")
    fact = _observed("fact", "Secrets must never be logged.",
                     subject="secret", predicate="must_not", object="be_logged")

    assert settled(rule, fact) == DIFFERENT
    assert not may_merge(equivalent(rule, fact))
    # The judge is never consulted, so even an affirmative verdict cannot join them.
    assert not may_merge(equivalent(rule, fact, verdict=SAME))


def test_doubt_and_silence_never_continue_an_assertion() -> None:
    old = _observed("decision", "We will use PostgreSQL for persistence.",
                    subject="persistence", decision="PostgreSQL", effect="used for persistence")
    new = _observed("decision", "Persistence is handled by PostgreSQL.",
                    subject="storage", decision="PostgreSQL", effect="handles persistence")

    for verdict in (DIFFERENT, AMBIGUOUS, UNJUDGED, None):
        assert not may_merge(equivalent(old, new, verdict=verdict))
    assert may_merge(equivalent(old, new, verdict=SAME))
