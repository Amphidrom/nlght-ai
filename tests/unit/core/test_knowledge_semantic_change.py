# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""The seven cases that decide whether an approval survives an edit.

The gate on this part of the design. Each case is one edit to a rule's body and
one expected answer, and the four that must come back `normative` and the three
that must not are not variations of one thing — they produce opposite outcomes
for a reviewer:

    normative  a new revision, needing review, the old approval not carried
    rewrite    the same claim in other words, the approval standing
    unchanged  nothing worth recording at all

Getting them the wrong way round is expensive in both directions. Read a changed
threshold as a rewording and a person's approval covers a value they never saw;
read a rewording as a change and the queue fills with sentences nobody edited.

Three of the seven cannot be decided by looking at the strings, and that is a
fact about language rather than a gap in the code: `must` becoming `must not`
changes the rule and `must` becoming `shall` does not, and no rule over
characters separates those without knowing what the words mean. They are
delegated as one decision with three answers.

**What the delegated cases prove, and what they do not.** With a stub standing in
for the classifier they prove the decision is asked for, honoured, and
conservative when the answer is doubt. They do not prove any particular model
gets `must not` right — that is a property of the model, and only a live run can
speak to it.
"""

from __future__ import annotations

from nlght.core.knowledge.semantics import (
    AMBIGUOUS,
    NORMATIVE,
    REWRITE,
    UNCHANGED,
    semantic_change,
)


def _asked(answer: str) -> str:
    """What the classifier said, standing in for the model.

    The judgement itself is the model's. What these assert is that it is asked
    for, that its answer decides, and that doubt falls the conservative way —
    not that any particular model gets `must not` right. That is a property of
    the model, and `tests/local_e2e/test_semantic_change_live.py` puts the same
    seven cases to a real one.
    """
    return answer


# ---------------------------------------------------------------------------
# Materially different — the approval does not carry
# ---------------------------------------------------------------------------

def test_must_becoming_must_not_is_normative() -> None:
    assert semantic_change(
        "Requests must be signed.", "Requests must not be signed.",
        verdict=_asked(NORMATIVE),
    ) == NORMATIVE


def test_may_becoming_must_is_normative() -> None:
    # A permission turning into an obligation. Everything downstream of the rule
    # changes, and one word carries it.
    assert semantic_change(
        "Clients may retry a failed request.", "Clients must retry a failed request.",
        verdict=_asked(NORMATIVE),
    ) == NORMATIVE


def test_enabled_becoming_disabled_is_normative() -> None:
    assert semantic_change(
        "Caching is enabled by default.", "Caching is disabled by default.",
        verdict=_asked(NORMATIVE),
    ) == NORMATIVE


def test_a_moved_quantity_is_normative() -> None:
    # The one materially different case that is decidable without meaning: the
    # sentence is identical and the number is not.
    assert semantic_change(
        "Expenses above CHF 500 require approval.",
        "Expenses above CHF 550 require approval.",
    ) == NORMATIVE


# ---------------------------------------------------------------------------
# The same claim — the approval stands
# ---------------------------------------------------------------------------

def test_must_becoming_shall_is_only_a_rewrite() -> None:
    """The case that makes a word list impossible.

    `must` appears in this pair and in the first one, and the two must come out
    opposite. Any rule keyed on the word is wrong for one of them.
    """
    assert semantic_change(
        "Requests must be signed.", "Requests shall be signed.",
        verdict=_asked(REWRITE),
    ) == REWRITE


def test_a_paraphrase_is_only_a_rewrite() -> None:
    assert semantic_change(
        "Expenses above CHF 500 require approval.",
        "Anything over CHF 500 must be signed off.",
    ) == REWRITE


def test_punctuation_and_spacing_change_nothing() -> None:
    assert semantic_change(
        "Lines are limited to 100 characters",
        "Lines  are   limited to 100 characters.",
    ) == UNCHANGED


# ---------------------------------------------------------------------------
# What a classifier's answer is allowed to do
# ---------------------------------------------------------------------------

def test_doubt_resolves_to_normative_not_to_rewrite() -> None:
    """A review nobody needed costs a minute; an approval nobody gave costs trust.

    So the conservative direction is the reviewable one. This is the whole reason
    `ambiguous` exists as a distinct answer rather than being folded into either
    outcome at the classifier.
    """
    assert semantic_change(
        "Requests must be signed.", "Requests should be signed.", verdict=_asked(AMBIGUOUS)
    ) == NORMATIVE


def test_a_classifier_cannot_override_a_moved_quantity() -> None:
    # The decidable answers are not opinions. A model calling a changed threshold
    # a rewording does not make it one.
    assert semantic_change(
        "Expenses above CHF 500 require approval.",
        "Expenses above CHF 550 require approval.",
        verdict=_asked(REWRITE),
    ) == NORMATIVE


def test_no_classifier_leaves_a_rewording_alone() -> None:
    """Absent is not the same as unsure.

    Treating every unclassified rewording as normative would fill the review
    queue with sentences nobody edited, which is the outcome this design exists
    to prevent. An unconfigured classifier leaves the deterministic answer
    standing; doubt a classifier *expresses* is what resolves conservatively.
    """
    assert semantic_change(
        "Expenses above CHF 500 require approval.",
        "Anything over CHF 500 must be signed off.",
    ) == REWRITE
