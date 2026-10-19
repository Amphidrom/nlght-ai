# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Represent assertion continuity, supersession, and score matching."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime

from nlght.core.knowledge.entity import EntityKey, new_assertion_id
from nlght.core.knowledge.knowledge import KnowledgeObject
from nlght.core.knowledge.knowledge_proposition import Proposition

APPROVED = "approved"
NEEDS_REVIEW = "needs_review"

SUPERSEDES = "supersedes"
"""The one lineage relation there is, named so a second can be added later.

An edge carries its relation rather than the table meaning only one thing:
"replaced by" is what is needed now, and "split into" or "merged with" may want
to be told apart from it without another table.
"""

SAME = "same"
AMBIGUOUS = "ambiguous"
NEW = "new"

SAME_FROM = 0.95
"""At or above this, two assertions are the same one."""

AMBIGUOUS_FROM = 0.80
"""Keep uncertain matches separate until structure or equivalence can decide them."""


def band(score: float) -> str:
    """Classify a match score without forcing an uncertain match.

    Args:
        score (float): Similarity score.

    Returns:
        str: Same, ambiguous, or new band.
    """
    if score >= SAME_FROM:
        return SAME
    if score >= AMBIGUOUS_FROM:
        return AMBIGUOUS
    return NEW


def link(entity_key: str, known: Mapping[str, str]) -> str | None:
    """Resolve an entity key through known assertion identities.

    Args:
        entity_key (str): Stable entity key to resolve.
        known (Mapping[str, str]): Entity keys mapped to assertion IDs.

    Returns:
        str | None: Existing assertion ID or None.
    """
    return known.get(entity_key)


def assign(scores: Mapping[tuple[str, str], float]) -> dict[str, str]:
    """Choose a maximum-total-score one-to-one assignment.

    Args:
        scores (Mapping[tuple[str, str], float]): Scores for candidate old and new assertion pairs.

    Returns:
        dict[str, str]: Accepted old-to-new assertion pairing.
    """
    candidates = {
        pair: score for pair, score in scores.items() if score >= AMBIGUOUS_FROM
    }
    lefts = sorted({left for left, _ in candidates})
    rights = sorted({right for _, right in candidates})

    best: tuple[float, dict[str, str]] = (0.0, {})

    def search(index: int, taken: frozenset[str], total: float, chosen: dict[str, str]) -> None:
        nonlocal best
        if index == len(lefts):
            if total > best[0]:
                best = (total, dict(chosen))
            return
        left = lefts[index]
        # Leaving one unpaired is a real option: it is what "this assertion is
        # gone" looks like.
        search(index + 1, taken, total, chosen)
        for right in rights:
            if right in taken or (left, right) not in candidates:
                continue
            chosen[left] = right
            search(index + 1, taken | {right}, total + candidates[(left, right)], chosen)
            del chosen[left]

    search(0, frozenset(), 0.0, {})
    return best[1]


@dataclass(slots=True, frozen=True)
class LineageWrite:
    """Carry one observed claim and the context needed to place it.

    The repository resolves assertion, variant, and revision continuity. The
    caller supplies `normative_change` after comparing substance. `proposition`
    holds the complete extracted structure for every built-in kind, and the
    resulting revision records that structure.
    """

    entity: EntityKey
    kind: str
    text: str
    fingerprint: str
    extraction_version: str
    document_id: str
    document_revision: str
    run_id: str
    scope: tuple[str, ...] = ()
    slot_id: str | None = None
    #: Keep the source path for provenance without making renames change identity.
    document_path: str = ""
    source_span: dict[str, object] | None = None
    normative_change: bool = False
    proposition: Proposition | None = None
    #: Record one event per sighting so comparisons do not falsely assign an
    #: earlier candidate to the current run.
    observation_id: uuid.UUID | None = None
    #: The assertion an equivalence judgement placed this sighting on, when the
    #: matching ladder found none. Decided by the caller because deciding it
    #: needs a model, and a model round trip may not happen inside the
    #: transaction that writes the claim.
    #:
    #: Weaker than the ladder by construction: it is consulted only where exact
    #: agreement found nothing, so a judgement can never override what the source
    #: plainly said.
    continues: str | None = None
    #: Keep the candidate search scope because slot and corpus verdicts have
    #: different revalidation requirements.
    #:
    #:     "slot"    stale when authoritative placement lands elsewhere
    #:     "corpus"  independent of slot membership
    #:
    continues_scope: str = ""

    def __post_init__(self) -> None:
        missing = [
            name
            for name in (
                "kind", "text", "fingerprint", "extraction_version",
                "document_id", "document_revision", "run_id",
            )
            if not str(getattr(self, name)).strip()
        ]
        if missing:
            raise ValueError(f"lineage write is missing {', '.join(missing)}")



@dataclass(slots=True, frozen=True)
class RecordedLineage:
    """Where a sighting landed: which assertion, which case, which state."""

    assertion_id: str
    variant_id: str
    revision: int
    review_state: str
    supersedes: int | None
    #: Keep the fingerprint of the resolved state. Two revisions can share a
    #: content hash, so the hash does not identify the revision.
    fingerprint: str = ""
    #: Point the graph node at an exact revision to resolve current state.
    revision_id: str = ""
    #: Record the matching rung so exact continuation and ambiguity remain
    #: distinguishable in reports.
    resolution: str = "key"


@dataclass(slots=True, frozen=True)
class SlotSupport:
    """Report whether one slot carries a claim without deciding retirement.

    Document-wide support controls retirement because a claim can move between
    slots while the document still asserts it.
    """

    assertion_id: str
    document_id: str
    slot_id: str
    active: bool


@dataclass(slots=True, frozen=True)
class DocumentOutcome:
    """Collect the writes and retirements after one complete document read.

    Retraction compares document-wide support after all claims arrive. Slot
    movement alone cannot retire a claim that the document still asserts.
    """

    #: The graph objects written, so the caller can index what may be read
    #: without asking for them again.
    stored: tuple[KnowledgeObject, ...] = ()
    recorded: tuple[RecordedLineage, ...] = ()
    retired: tuple[RetiredAssertion, ...] = ()
    #: Slots whose section the document no longer has. A slot's life and a
    #: claim's life are separate: a section can disappear while everything it
    #: said is still asserted somewhere else.
    retired_slots: tuple[str, ...] = ()
    #: Where each claim is and is no longer supported, for reading rather than
    #: for deciding.
    supports: tuple[SlotSupport, ...] = ()


@dataclass(slots=True, frozen=True)
class AssertionRevision:
    """Record one state and its review status under a stable assertion ID."""

    assertion_id: str
    revision: int
    text: str
    review_state: str = APPROVED
    supersedes: int | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(slots=True, frozen=True)
class RetiredAssertion:
    """A claim that is no longer carried.

    `superseded_by` used to be required, which quietly said that retirement
    always means replacement. It does not: a source that stops asserting
    something replaces it with nothing, and that is the commoner case. Something
    took its place only when something did.
    """

    assertion_id: str
    retired_at: datetime
    #: Withdrawal needs the kind because indexes are partitioned by kind.
    kind: str = ""
    #: The graph identity of its last state, so the search index can be told.
    #: Retiring the lineage without withdrawing the document would leave the
    #: claim unfindable and retrievable at the same time.
    fingerprint: str = ""
    superseded_by: str | None = None


def continue_assertion(
    *,
    assertion_id: str,
    revision: int,
    new_text: str,
    normative_change: bool = False,
) -> AssertionRevision:
    """Create the next revision of one assertion.

    Args:
        assertion_id (str): Assertion to revise.
        revision (int): Current revision number.
        new_text (str): Wording of the next revision.
        normative_change (bool): Whether the change requires review.

    Returns:
        AssertionRevision: New assertion revision with matching review state.

    Raises:
        ValueError: New text is blank or the current revision is below one.
    """
    if not new_text.strip():
        raise ValueError("an assertion revision must have text")
    if revision < 1:
        raise ValueError("revisions start at 1")
    return AssertionRevision(
        assertion_id=assertion_id,
        revision=revision + 1,
        text=new_text,
        review_state=NEEDS_REVIEW if normative_change else APPROVED,
        supersedes=revision,
    )


def supersede_assertion(
    *, assertion_id: str, new_text: str
) -> tuple[RetiredAssertion, AssertionRevision]:
    """Retire one assertion and open a distinct claim.

    Args:
        assertion_id (str): Assertion being retired.
        new_text (str): Wording of the distinct new claim.

    Returns:
        tuple[RetiredAssertion, AssertionRevision]: Retirement record and first revision
            of a new assertion.

    Raises:
        ValueError: The new claim text is blank.

    Side Effects:
        Reads the clock and generates a new surrogate ID.
    """
    if not new_text.strip():
        raise ValueError("an assertion revision must have text")
    created = AssertionRevision(
        assertion_id=new_assertion_id(), revision=1, text=new_text,
        review_state=NEEDS_REVIEW,
    )
    retired = RetiredAssertion(
        assertion_id=assertion_id,
        retired_at=datetime.now(UTC),
        superseded_by=created.assertion_id,
    )
    return retired, created
