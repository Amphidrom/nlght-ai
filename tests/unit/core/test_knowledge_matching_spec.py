# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Executable specification for entity linking and slot matching.

**The matcher does not decide identity.** That is a schema decision — see
`test_knowledge_entity_key_spec.py` — and the first thing a newly extracted
assertion gets is a deterministic lookup on its entity key. A hit is the
assertion; no score is consulted. What is left for the matcher is narrower and
honest: when the extraction did not canonicalise, *which existing entity does
this probably correspond to*, and, in a slot, which old assertion continues as
which new one.

That boundary is the point. A score may propose a link; it may never define
what the database considers identity, or a faulty matcher rewrites history.

Three properties are specified here:

  1. Deterministic first. Similarity is only reached when the key does not
     resolve.
  2. The assignment is **global**, not greedy. With one to five assertions per
     slot the cost is irrelevant and greedy demonstrably picks worse.
  3. The score has **three bands**. The middle band is ambiguous linking and
     escalates to further matchers before it reaches a person — and a person
     then rules on a rare entity-linking ambiguity, not on every text edit.

And one thing that is not the same question: whether the assertion continues,
and whether a person's approval of it still stands.
"""

from __future__ import annotations

import pytest

from nlght.core.knowledge.lineage import assign as _assign
from nlght.core.knowledge.lineage import band as _band
from nlght.core.knowledge.lineage import link as _link

# ---------------------------------------------------------------------------
# 1. Deterministic before similarity
# ---------------------------------------------------------------------------

def test_a_resolving_entity_key_never_reaches_the_similarity_matcher() -> None:
    """The normal case, and it must not be a judgement call.

    Run two extracts `(rule, expense, approval_threshold)` with a new amount.
    That key already exists, so the assertion is that entity — decided by a
    lookup, not by a distance. Embeddings enter only where the schema failed to
    canonicalise.
    """
    known = {"rule|expense|approval_threshold": "a42"}

    assert _link("rule|expense|approval_threshold", known) == "a42"


def test_an_unknown_entity_key_is_a_new_assertion_not_a_near_match() -> None:
    # Without a hit the answer is "none", and minting a new assertion is the
    # caller's job. Quietly attaching it to the closest thing is how a matcher
    # starts defining identity.
    known = {"rule|expense|approval_threshold": "a42"}

    assert _link("rule|expense|reimbursement_deadline", known) is None


# ---------------------------------------------------------------------------
# 2. Global assignment, where similarity is genuinely needed
# ---------------------------------------------------------------------------

def test_the_assignment_is_global_rather_than_greedy() -> None:
    """The worked counter-example, which greedy gets wrong by construction.

              X     Y
        A   .91   .89
        B   .90   .40

    Greedy takes the largest score first — A→X at .91 — and has then only B→Y
    at .40 left, total 1.31. The optimal assignment is A→Y and B→X, total 1.79.
    The consequence is not academic: B is matched to something it barely
    resembles, so a rule that simply moved is recorded as one assertion
    retracted and another created, and its approval is gone.
    """
    scores = {("A", "X"): 0.91, ("A", "Y"): 0.89, ("B", "X"): 0.90, ("B", "Y"): 0.40}

    assert _assign(scores) == {"A": "Y", "B": "X"}


def test_an_unmatched_old_assertion_is_not_forced_onto_a_poor_candidate() -> None:
    # Two old assertions, one new one that only resembles the first. The second
    # is gone, and saying so is the point — an assignment that pairs everything
    # would report a retraction as a rewording.
    scores = {("A", "X"): 0.97, ("B", "X"): 0.12}

    assert _assign(scores) == {"A": "X"}


def test_a_new_assertion_with_no_predecessor_stays_unmatched() -> None:
    scores = {("A", "X"): 0.96, ("A", "Y"): 0.10}

    assert _assign(scores) == {"A": "X"}


# ---------------------------------------------------------------------------
# 3. Three bands
# ---------------------------------------------------------------------------

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
    """A binary threshold has to be wrong in one direction at the boundary.

    Set it high and reworded assertions are retracted and re-reviewed; set it
    low and a materially changed rule is carried through on an old approval.
    The middle band refuses to guess: it escalates to a structured proposition
    comparison, then an equivalence classifier, and only then to a person.
    """
    assert _band(score) == expected


def test_the_ambiguous_band_is_not_by_itself_a_review_case() -> None:
    # Sending every uncertain match to a reviewer would refill the queue the
    # design exists to keep empty. Ambiguity is a request for a better matcher
    # first, and a person only when those have also failed.
    assert _band(0.88) == "ambiguous"
    assert _band(0.88) != "review"


# ---------------------------------------------------------------------------
# 4. A continuation keeps the assertion, not the wording
# ---------------------------------------------------------------------------

def test_a_rewording_continues_the_same_assertion_at_a_new_revision() -> None:
    """The property the surrogate id exists for.

        old: "Spesen über CHF 500 benötigen eine Genehmigung."
        new: "Ausgaben von mehr als CHF 500 müssen genehmigt werden."

    Same rule, different words. The assertion_id must not move — if it were
    derived from content it would have to, and the approval would go with it.
    """
    from nlght.core.knowledge.lineage import continue_assertion  # noqa: PLC0415

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
    from nlght.core.knowledge.lineage import supersede_assertion  # noqa: PLC0415

    retired, created = supersede_assertion(assertion_id="a17", new_text="Something else entirely.")

    assert retired.assertion_id == "a17"
    assert retired.retired_at is not None
    assert created.assertion_id != "a17"


# ---------------------------------------------------------------------------
# Identity and approval are two questions
# ---------------------------------------------------------------------------

def test_a_changed_threshold_keeps_the_assertion_and_invalidates_the_approval() -> None:
    """The case that separates the two questions.

        revision 7: expenses > CHF 500 require approval   APPROVED
        revision 8: expenses > CHF 550 require approval   NEEDS_REVIEW

    It is the same business rule — the approval threshold for expenses — so the
    assertion continues. But the normative value changed, so the approval a
    person gave against 500 cannot stand for 550. Treating "same assertion" and
    "approval still valid" as one decision gets one of the two wrong: either the
    rule is torn up and re-reviewed for a rewording, or a changed limit is
    served under an old approval.
    """
    from nlght.core.knowledge.lineage import continue_assertion  # noqa: PLC0415

    continued = continue_assertion(
        assertion_id="a42",
        revision=7,
        new_text="Spesen über CHF 550 benötigen eine Genehmigung.",
        normative_change=True,
    )

    assert continued.assertion_id == "a42"
    assert continued.review_state == "needs_review"


def test_a_pure_rewording_keeps_the_approval() -> None:
    from nlght.core.knowledge.lineage import continue_assertion  # noqa: PLC0415

    continued = continue_assertion(
        assertion_id="a42",
        revision=7,
        new_text="Ausgaben über CHF 500 sind genehmigungspflichtig.",
        normative_change=False,
    )

    assert continued.review_state == "approved"
