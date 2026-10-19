# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Resolve assertion continuity by exact signals, never resemblance."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from nlght.core.knowledge.lineage import AMBIGUOUS_FROM

#: Preserve the matching rung so an exact continuation can be distinguished
#: from an unresolved resemblance in lineage reports.
SLOT_TEXT = "slot_text"
SLOT_KEY = "slot_key"
DOCUMENT_TEXT = "document_text"
KEY = "key"
AMBIGUOUS = "ambiguous"
NEW = "new"


def normalise(text: str) -> str:
    """Normalize wording without discarding material words.

    Args:
        text (str): Source text.

    Returns:
        str: Lowercase text with folded whitespace and trimmed edge punctuation.
    """
    return " ".join(str(text or "").split()).strip(" .;,:").lower()


def similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, normalise(left), normalise(right)).ratio()


def material_content(text: str) -> frozenset[str]:
    """Isolate quantity-bearing tokens from claim wording.

    Args:
        text (str): Source text.

    Returns:
        frozenset[str]: Lowercase tokens containing digits, with edge punctuation removed.
    """
    return frozenset(
        token.strip(".,;:()[]{}\"'").lower()
        for token in str(text or "").split()
        if any(character.isdigit() for character in token)
    )


@dataclass(slots=True, frozen=True)
class Candidate:
    """An assertion already in the corpus, as the matcher sees it."""

    assertion_id: str
    entity_key: str
    #: Compare scheme and key together because schemes can reuse one string for
    #: different claims.
    entity_key_version: str
    text: str
    document_id: str
    #: Source wording can describe claims of different kinds. Keep each kind in
    #: its own identity namespace even when the wording is identical.
    kind: str = ""
    #: A claim can be observed in several slots, so any prior slot can support
    #: a continuation.
    slot_ids: frozenset[str] = field(default_factory=frozenset)


@dataclass(slots=True, frozen=True)
class PropositionMatch:
    """Which assertion this proposition is, and how that was decided."""

    assertion_id: str | None
    resolution: str

    @property
    def is_continuation(self) -> bool:
        return self.assertion_id is not None


def _unique(candidates: Sequence[Candidate]) -> Candidate | None:
    """Accept a match only when exactly one candidate remains.

    Args:
        candidates (Sequence[Candidate]): Candidate sequence.

    Returns:
        Candidate | None: The sole candidate or None.
    """
    return candidates[0] if len(candidates) == 1 else None


def match_proposition(
    *,
    entity_key: str,
    entity_key_version: str,
    text: str,
    slot_id: str | None,
    document_id: str,
    candidates: Iterable[Candidate],
    kind: str,
) -> PropositionMatch:
    """Match a same-kind claim by exact slot, document, text, or key.

    Args:
        entity_key (str): Stable identity of the claim subject.
        entity_key_version (str): Scheme version for the identity key.
        text (str): Current claim wording.
        slot_id (str | None): Resolved section slot, if available.
        document_id (str): Source document identity.
        candidates (Iterable[Candidate]): Existing claims eligible for continuation.
        kind (str): Knowledge kind to match.

    Returns:
        PropositionMatch: Continuation, ambiguous match, or new-claim resolution.
    """
    # A shared source sentence can yield several kinds. Reject unknown kinds
    # rather than letting one kind's assertion continue another's lineage.
    known = [c for c in candidates if c.kind == kind]
    wording = normalise(text)

    in_slot = [
        candidate
        for candidate in known
        if slot_id is not None and slot_id in candidate.slot_ids
    ]

    same_wording = [c for c in in_slot if normalise(c.text) == wording]
    if (found := _unique(same_wording)) is not None:
        return PropositionMatch(found.assertion_id, SLOT_TEXT)

    scheme = (entity_key_version, entity_key)
    same_key_here = [c for c in in_slot if (c.entity_key_version, c.entity_key) == scheme]
    if (found := _unique(same_key_here)) is not None:
        return PropositionMatch(found.assertion_id, SLOT_KEY)

    # A renamed or reordered section can move a claim to a new slot.
    in_document = [c for c in known if c.document_id == document_id]
    moved = [c for c in in_document if normalise(c.text) == wording]
    if (found := _unique(moved)) is not None:
        return PropositionMatch(found.assertion_id, DOCUMENT_TEXT)

    # Outside the document, require the key so shared boilerplate does not
    # merge unrelated claims.
    same_key = [c for c in known if (c.entity_key_version, c.entity_key) == scheme]
    if (found := _unique(same_key)) is not None:
        return PropositionMatch(found.assertion_id, KEY)
    if same_key:
        return PropositionMatch(None, AMBIGUOUS)

    # Similarity can signal doubt inside one slot, but cannot establish
    # identity. Across documents it can pair unrelated shared subjects.
    plausible = [
        candidate
        for candidate in in_slot
        if similarity(candidate.text, text) >= AMBIGUOUS_FROM
    ]
    if len(plausible) > 1:
        # Picking one plausible claim would silently inherit the wrong lineage.
        return PropositionMatch(None, AMBIGUOUS)
    return PropositionMatch(None, NEW)
