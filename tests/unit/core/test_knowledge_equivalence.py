# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Whether two complete propositions say the same thing.

A judgement, and only that. Nothing here produces a canonical form, stores
anything, or hands back a merged proposition — a "canonical" anything would be a
data model chosen by the side that was only supposed to compare, which is the
move this slice keeps having to undo.

    yes         one proposition, said differently
    no          a judge looked and said they are different
    ambiguous   a judge looked and could not tell
    unjudged    nobody was asked

The last three all mean *do not merge*, and they stay four states rather than
two. Recording "nobody was asked" as `no` would let the trail claim later that
something decided these were different claims, when the truth is that no
classifier was configured — an audit trail that cannot tell those apart is worse
than none, because it reads as evidence.

The conservative direction is deliberately *not* the one `semantic_change` takes.
There, doubt sends a revision to review, because a review nobody needed costs a
minute. Here, doubt refuses to merge, because two claims wrongly joined lose one
of them and nothing afterwards shows which. The safe direction follows the harm.
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
    """A fact's claim, which is what the judge and `settled` compare.

    Facts keep `Proposition` as their representation; the `Claim` around it
    carries the kind — a judgement never crosses one — and the sentence it was
    read from, which is context and never the thing being compared.
    """
    return AssertionObservation.of(Proposition(fields))


# ---------------------------------------------------------------------------
# Decidable without knowing what the words mean
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Only meaning can decide
# ---------------------------------------------------------------------------

def test_active_and_passive_are_one_claim_when_the_judgement_says_so() -> None:
    """"Spring requires Maven" and "Maven is required by Spring"."""
    assert equivalent(
        _p(subject="spring", predicate="requires", object="maven"),
        _p(subject="maven", predicate="required_by", object="spring"),
        verdict=SAME,
    ) == SAME


def test_renamed_roles_are_one_claim_when_the_judgement_says_so() -> None:
    # `key/value/origin` against `data_key/data_value/property_source` is a
    # difference in extraction labels, not in the claim — and *that* is what
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
    """And this is why the swap is delegated rather than settled.

    A swap and a rename are indistinguishable over the structure — both admit a
    bijection of field names carrying one to the other — and they must end
    opposite. Only knowing that `sender` and `recipient` are different roles,
    while `sender` and `actor` are the same one, separates them.
    """
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
    from itertools import permutations  # noqa: PLC0415

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


# ---------------------------------------------------------------------------
# What doubt does
# ---------------------------------------------------------------------------

def test_doubt_never_merges() -> None:
    """`ambiguous` is an answer, and it is not a quiet `yes`.

    Two claims wrongly joined lose one of them, and nothing afterwards shows
    which — which is why the safe direction here is the opposite of the one
    `semantic_change` takes with the same word.
    """
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
    """It is a statement about A *and* B, so it is not a field on either.

    Hanging it on A would make a statement about a relationship look like a
    property of a thing, and the next reader would use it as one. And a verdict
    without its judge cannot be revisited when the judge changes — which it will,
    because the undecidable band is answered by a model whose version moves.
    """
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
    """"X is equivalent to Y" is the same decision as "Y is equivalent to X".

    Storing both would let the corpus hold two answers to one question. That says
    nothing about whether `transfers(A, B)` equals `transfers(B, A)` — the claims
    are directed and the *equivalence relation between claims* is symmetric, and
    confusing the two is the mistake this slice keeps stepping around.
    """
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
    """Which is the distinction the whole module rests on.

    Same fingerprint means byte-identical after normalisation. *Different*
    fingerprints say nothing about whether the claims are one — that is the
    question being asked, and a fingerprint that answered it would be the
    canonical form this refuses to build.
    """
    # The fingerprint belongs to the proposition, which is a *fact's*
    # representation — not to the claim wrapper the judge is handed.
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
    """The guard on the whole module.

    A canonical form would be a data model chosen by the side that was only
    supposed to compare — and where a proposition is stored is a separate,
    undecided question.
    """
    import nlght.core.knowledge.equivalence as equivalence  # noqa: PLC0415

    exported = {name for name in dir(equivalence) if not name.startswith("_")}

    assert not exported & {"canonical", "canonicalise", "merge", "normal_form", "unify"}


# ---------------------------------------------------------------------------
# The contract: one question, for every kind
# ---------------------------------------------------------------------------

def _observed(kind: str, text: str, **representation: object):  # noqa: ANN201
    return AssertionObservation(kind=kind, observed_text=text, representation=representation)


def test_the_same_assertion_said_differently_may_continue() -> None:
    """The case a live run found missing.

        "…you have to set up the environment when building…"
        "…the environment must be set up at build time"

    Same rule, two wordings, and the model chose different fields for each — so
    the entity key moved, every rung of the matching ladder failed, and the
    corpus opened a second assertion and retired the first. Only a judgement
    over the two observations can join them.
    """
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
    """`observed_text` is authoritative for the wording and is not an identity.

        "When shutdown is enabled, the application waits for requests
         and rejects new requests."

    Two assertions, one sentence. A judgement made on the sentence alone would
    merge them, so the sentence is never what is compared — the representation
    says which assertion inside it is being asked about.
    """
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
    # Even a judge saying yes cannot join them — it is never consulted.
    assert not may_merge(equivalent(rule, fact, verdict=SAME))


def test_doubt_and_silence_never_continue_an_assertion() -> None:
    old = _observed("decision", "We will use PostgreSQL for persistence.",
                    subject="persistence", decision="PostgreSQL", effect="used for persistence")
    new = _observed("decision", "Persistence is handled by PostgreSQL.",
                    subject="storage", decision="PostgreSQL", effect="handles persistence")

    for verdict in (DIFFERENT, AMBIGUOUS, UNJUDGED, None):
        assert not may_merge(equivalent(old, new, verdict=verdict))
    assert may_merge(equivalent(old, new, verdict=SAME))
