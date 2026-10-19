# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Specify deterministic entity linking and slot matching.

An existing entity key resolves before similarity is consulted. Unresolved
candidates use a global assignment and three score bands. Continuity of an
assertion and validity of its approval are separate decisions.
"""

from __future__ import annotations

import pytest

from nlght.core.knowledge.lineage import assign as _assign
from nlght.core.knowledge.lineage import band as _band
from nlght.core.knowledge.lineage import link as _link


def test_a_resolving_entity_key_never_reaches_the_similarity_matcher() -> None:
    """Resolve an existing entity key without consulting similarity."""
    known = {"rule|expense|approval_threshold": "a42"}

    assert _link("rule|expense|approval_threshold", known) == "a42"


def test_an_unknown_entity_key_is_a_new_assertion_not_a_near_match() -> None:
    # Without a hit the answer is "none", and minting a new assertion is the
    # caller's job. Quietly attaching it to the closest thing is how a matcher
    # starts defining identity.
    known = {"rule|expense|approval_threshold": "a42"}

    assert _link("rule|expense|reimbursement_deadline", known) is None


def test_the_assignment_is_global_rather_than_greedy() -> None:
    """Choose the globally best assignment over a greedy first match."""
    scores = {("A", "X"): 0.91, ("A", "Y"): 0.89, ("B", "X"): 0.90, ("B", "Y"): 0.40}

    assert _assign(scores) == {"A": "Y", "B": "X"}


def test_an_unmatched_old_assertion_is_not_forced_onto_a_poor_candidate() -> None:
    # Two old assertions, one new one that only resembles the first. The second
    # is gone. Forcing every old assertion into an assignment would report
    # that retraction as a rewording.
    scores = {("A", "X"): 0.97, ("B", "X"): 0.12}

    assert _assign(scores) == {"A": "X"}


def test_a_new_assertion_with_no_predecessor_stays_unmatched() -> None:
    scores = {("A", "X"): 0.96, ("A", "Y"): 0.10}

    assert _assign(scores) == {"A": "X"}


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (1.00, "same"),
        (0.96, "same"),
        (0.95, "same"),
        (0.94, "ambiguous"),
        (0.85, "ambiguous"),
        (0.80, "ambiguous"),
        (0.79, "new"),
        (0.10, "new"),
    ],
)
def test_the_score_falls_into_one_of_three_bands(score: float, expected: str) -> None:
    """Classify matching scores into same, ambiguous, and new bands."""
    assert _band(score) == expected


def test_the_ambiguous_band_is_not_by_itself_a_review_case() -> None:
    # Sending every uncertain match to a reviewer would refill the queue the
    # design exists to keep empty. Ambiguity is a request for a better matcher
    # first, and a person only when those have also failed.
    assert _band(0.88) == "ambiguous"
    assert _band(0.88) != "review"


def test_a_rewording_continues_the_same_assertion_at_a_new_revision() -> None:
    """Preserve assertion identity across a wording-only revision."""
    from nlght.core.knowledge.lineage import continue_assertion

    continued = continue_assertion(
        assertion_id="a_7f93",
        revision=17,
        new_text="Ausgaben von mehr als CHF 500 müssen genehmigt werden.",
    )

    assert continued.assertion_id == "a_7f93"
    assert continued.revision == 18


def test_a_material_change_retires_the_assertion_and_opens_a_new_one() -> None:
    # The other side: not everything continues. A materially different claim is
    # a withdrawal and a new assertion, recorded as an event rather than
    # inferred later from a distance score.
    from nlght.core.knowledge.lineage import supersede_assertion

    retired, created = supersede_assertion(assertion_id="a17", new_text="Something else entirely.")

    assert retired.assertion_id == "a17"
    assert retired.retired_at is not None
    assert created.assertion_id != "a17"


def test_a_changed_threshold_keeps_the_assertion_and_invalidates_the_approval() -> None:
    """Keep rule identity while requiring review of a changed threshold."""
    from nlght.core.knowledge.lineage import continue_assertion

    continued = continue_assertion(
        assertion_id="a42",
        revision=7,
        new_text="Spesen über CHF 550 benötigen eine Genehmigung.",
        normative_change=True,
    )

    assert continued.assertion_id == "a42"
    assert continued.review_state == "needs_review"


def test_a_pure_rewording_keeps_the_approval() -> None:
    from nlght.core.knowledge.lineage import continue_assertion

    continued = continue_assertion(
        assertion_id="a42",
        revision=7,
        new_text="Ausgaben über CHF 500 sind genehmigungspflichtig.",
        normative_change=False,
    )

    assert continued.review_state == "approved"
