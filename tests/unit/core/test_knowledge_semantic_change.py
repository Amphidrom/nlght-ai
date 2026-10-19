# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Classify rule edits that preserve or change normative meaning.

A normative change starts a reviewable revision. A wording change can retain
the prior approval.
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
    """Return a scripted semantic-change verdict for the classifier.

    Args:
        answer (str): Model verdict to supply.

    Returns:
        str: Verdict passed to semantic_change().
        """
    return answer


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
