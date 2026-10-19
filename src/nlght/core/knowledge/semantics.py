# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Classify how revised wording affects approval continuity."""

from __future__ import annotations

from nlght.core.knowledge.proposition import material_content, normalise

UNCHANGED = "unchanged"
REWRITE = "rewrite"
NORMATIVE = "normative"

#: Keep classifier uncertainty distinct from a rewrite before resolving it to
#: a normative change that requires review.
AMBIGUOUS = "ambiguous"


def settled(old: str, new: str) -> str | None:
    """Classify wording changes that are decidable without a model.

    Args:
        old (str): Previous claim wording.
        new (str): Current claim wording.

    Returns:
        str | None: Unchanged or normative verdict. None when wording alone cannot decide.
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
    """Combine deterministic wording evidence with a classifier verdict.

    Args:
        old (str): Previous claim wording.
        new (str): Current claim wording.
        verdict (str | None): Optional classifier verdict for an unresolved change.

    Returns:
        str: Unchanged, rewrite, or normative classification.
    """
    decided = settled(old, new)
    if decided is not None:
        return decided
    if verdict == REWRITE:
        return REWRITE
    if verdict in (NORMATIVE, AMBIGUOUS):
        return NORMATIVE
    return REWRITE
