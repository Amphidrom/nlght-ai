# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Whether a claim's body moved, and whether an approval survives it.

One question with three answers, asked of every changed body:

    unchanged   only differences no proposition depends on
    rewrite     the same proposition in other words — the assertion, its variant
                and its approval all stand
    normative   the same business quantity in a materially different state — a
                new revision, needing review, and the approval given to the old
                state is not carried

The three are not shades of one scale. `rewrite` and `normative` produce opposite
outcomes for a reviewer, and getting them the wrong way round is expensive in
both directions: read a changed threshold as a rewording and a person's approval
covers a value they never saw; read a rewording as a change and the review queue
fills with sentences nobody edited.

**Two of the three answers are decidable here and one is not.** Whitespace and
punctuation are decidable, and so is a moved quantity. Whether "must" becoming
"must not" changed the rule is a fact about a language, not about a string — and
any deterministic rule for it *is* a word list, one that would need `not`,
`never`, `unless`, `disabled`, `may`, `shall`, and their equivalents in every
language a customer writes documentation in. That list is never finished and
nobody can maintain it.

So the undecidable band is delegated, as exactly one business decision with three
answers, and doubt resolves conservatively: a classifier that cannot tell yields
`normative`, because a review nobody needed costs a minute and an approval nobody
gave costs trust.

`fingerprint` decides none of this. It is a content hash, and it says whether two
bodies are byte-identical — which is neither identity nor normative continuity.
"""

from __future__ import annotations

from nlght.core.knowledge.proposition import material_content, normalise

UNCHANGED = "unchanged"
REWRITE = "rewrite"
NORMATIVE = "normative"

#: What a classifier says when it cannot tell. Not a fourth outcome — it resolves
#: to `normative`, and exists so "unsure" and "the same claim" cannot be confused
#: at the point where the difference is decided.
AMBIGUOUS = "ambiguous"


def settled(old: str, new: str) -> str | None:
    """The answer where one can be had without understanding the words.

    Returns `None` for the band only meaning can decide, which is precisely the
    band worth spending a model call on: same quantities, different words.
    """
    if normalise(old) == normalise(new):
        return UNCHANGED
    if material_content(old) != material_content(new):
        # A threshold, a version, a port or a count that moved. The sentence
        # around it may be identical, and it is still a different claim about
        # the world.
        return NORMATIVE
    return None


def semantic_change(old: str, new: str, *, verdict: str | None = None) -> str:
    """How this body changed, given what a classifier made of it.

    `verdict` is the classifier's answer for the undecidable band, or `None` when
    none was asked for. `None` is not the same as doubt: an unconfigured
    classifier leaves the deterministic answer standing, because treating every
    unclassified rewording as normative would fill the review queue with
    sentences nobody edited — the outcome this whole design exists to prevent.
    Doubt the classifier *expresses* is different, and resolves to `normative`.
    """
    decided = settled(old, new)
    if decided is not None:
        return decided
    if verdict == REWRITE:
        return REWRITE
    if verdict in (NORMATIVE, AMBIGUOUS):
        return NORMATIVE
    return REWRITE
