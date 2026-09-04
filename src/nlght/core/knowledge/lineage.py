# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""An assertion over its lifetime: continuation, supersession, and matching.

An assertion is a surrogate id with a sequence of revisions under it. A rewording
is the same assertion at a new revision; a materially different claim is a
withdrawal and a new one, recorded as an event rather than inferred later from a
distance score.

Two questions are kept apart here, because they are not the same decision:

    is this still the same assertion?
    does the human approval of it still hold?

An expense limit moving from CHF 500 to 550 is the same rule — the same business
question — so the assertion continues. But a person approved five hundred, not
five fifty, so the approval does not. Treating the two as one decision gets one
of them wrong: either the rule is torn up and re-reviewed for a rewording, or a
changed limit is served under an old approval.

The matcher in this module never decides identity. An extracted entity key
resolves by lookup; similarity is reached only where extraction failed to
canonicalise, and then only to propose which existing entity a new assertion
probably belongs to.
"""

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
"""Between this and `SAME_FROM`, continuity is uncertain.

Not a review case by itself: sending every uncertain match to a person would
refill the queue this design exists to keep empty. Ambiguity asks for a better
matcher first — a structured comparison, then an equivalence classifier — and a
person only when those have also failed.
"""


def band(score: float) -> str:
    """Which of the three bands a match score falls into.

    A binary threshold has to be wrong in one direction at the boundary: set it
    high and reworded assertions are retracted and re-reviewed, set it low and a
    materially changed rule is carried through on an old approval. The middle
    band refuses to guess.
    """
    if score >= SAME_FROM:
        return SAME
    if score >= AMBIGUOUS_FROM:
        return AMBIGUOUS
    return NEW


def link(entity_key: str, known: Mapping[str, str]) -> str | None:
    """Which existing assertion this entity key is, if any.

    Deterministic and first. A key that resolves is the answer and no score is
    consulted; a key that does not resolve is a new assertion, and the caller
    mints one. Quietly attaching it to the nearest thing is how a matcher starts
    defining identity.
    """
    return known.get(entity_key)


def assign(scores: Mapping[tuple[str, str], float]) -> dict[str, str]:
    """Pair old assertions with new ones so the total score is highest.

    Global rather than greedy. Greedy takes the largest score first and is then
    forced into whatever is left — given A→X .91, A→Y .89, B→X .90, B→Y .40 it
    picks A→X and B→Y for 1.31, where A→Y and B→X score 1.79. The cost of
    getting that wrong is not academic: B is paired with something it barely
    resembles, so an assertion that merely moved is recorded as one retraction
    and one creation, and its approval is gone.

    A slot holds one to five assertions, so the exhaustive search this does is
    free. Pairs scoring below the ambiguous floor are not considered at all —
    an assignment that paired everything would report a retraction as a
    rewording.
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
    """One sighting of one claim, with everything needed to place it.

    The caller says what it saw and where; the repository decides whether that
    is a new assertion, a new case of an existing one, a new state of a case, or
    a claim it has already recorded.

    ``normative_change`` is the second question asked separately: whether the
    substance moved, and therefore whether an approval given against the earlier
    state can still stand. It is the caller's judgement because only the caller
    compared the two values.

    ``fact_proposition`` is the complete extracted structure a *fact* carried,
    and it lands on the revision this sighting produces. An assertion has many
    revisions and each records the form the claim took at the time — two
    propositions with different fingerprints can continue one assertion, when an
    equivalence judgement says they are one claim — so one proposition per
    assertion would flatten that history and lose the earlier form of anything
    reworded.

    Named for its kind on purpose, and rejected for any other. `rule`, `pattern`
    and `decision` have no propositions: they are documents of a fixed shape
    rather than n-ary claims, and a field called `proposition` on the general
    lineage contract would read as a universal proposition model that has not
    been decided. When one is, this becomes it — deliberately, not by a name
    that quietly implied it first.
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
    #: What the source calls the document — provenance for a reader, never part
    #: of identity. A rename must not open an assertion or write a revision, and
    #: nothing resolves a slot by it.
    document_path: str = ""
    source_span: dict[str, object] | None = None
    normative_change: bool = False
    proposition: Proposition | None = None
    #: The sighting this write was recorded as, so the revision it produces can
    #: name it. An observation is an event and is written once per sighting —
    #: never once per comparison, which would stamp a candidate seen last week
    #: with the run that merely looked at it today.
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
    #: Where that candidate was found, because it is also the premise the verdict
    #: was formed under — and the two premises are revalidated differently.
    #:
    #:     "slot"    judged as a claim standing in the slot this sighting was
    #:               expected to land in; a verdict is stale once the authoritative
    #:               placement lands somewhere else
    #:     "corpus"  judged as a live claim elsewhere that shared enough structure
    #:               to be worth asking about; slot membership was never part of it
    #:
    #: Carried rather than inferred. Inferring it made every cross-document
    #: verdict look stale, and treating them all as corpus-scoped would have
    #: quietly undone the slot revalidation.
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
    #: The fingerprint of the state this sighting landed on — which is *not*
    #: necessarily the one it arrived with. A sighting judged to be the same
    #: state keeps the fingerprint already recorded, and the graph node is
    #: addressed by this so the two cannot describe different things.
    #:
    #: A content address, and nothing else. It says what a state hashes to; it
    #: does not say which state a graph node stands for, and two revisions may
    #: carry the same one — see `revision_id`.
    fingerprint: str = ""
    #: The state itself, as an id rather than as a hash. The graph node points
    #: at it, which is what makes "is this node current" one hop instead of a
    #: question about every revision that happens to hash the same.
    revision_id: str = ""
    #: How the continuation was decided — which rung of the ladder answered, or
    #: that none did. Worth carrying: a corpus held together by exact agreement
    #: and one held together by resemblance deserve very different trust, and
    #: `ambiguous` names the cases a better signal should be tried on first.
    resolution: str = "key"


@dataclass(slots=True, frozen=True)
class SlotSupport:
    """Whether one slot still carries one claim.

    **Diagnostic, never a decision.** A slot losing a claim is not the claim
    going: it may have moved to another section, or lost its heading and fallen
    into the unscoped part of the document, and retiring it for that would be the
    ordering failure this design already paid for once. What retirement asks is
    whether the *document* still carries it anywhere.

    What this answers is the question a reviewer has when the number moves:
    *where* did it stop being said.
    """

    assertion_id: str
    document_id: str
    slot_id: str
    active: bool


@dataclass(slots=True, frozen=True)
class DocumentOutcome:
    """What one document's writes did, once the whole document was seen.

    Retraction is a set operation and cannot be assembled from single writes: a
    claim is gone because the source no longer contains it, which is only
    knowable once everything the source *does* contain has arrived.

    The container is the **document**, not the slot, and that distinction was
    paid for. A slot id is a section's position, so inserting a paragraph
    renumbers every section below it — and slot-scoped retraction then retired
    each unchanged claim from the slot it used to be in and re-created it in the
    one it moved to, losing its approval and its history to an edit that never
    touched it. The slot stays the container that *matches* a claim; the
    document is the one that decides a claim is gone.
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
    """One state of one assertion.

    ``assertion_id`` is the surrogate and does not move; ``revision`` counts the
    states. ``review_state`` answers the second question — whether the approval
    given against an earlier state still stands.
    """

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
    #: Which index holds it — the store is partitioned by kind, so withdrawing
    #: needs both halves of the address.
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
    """The same assertion, saying it differently — or saying something else.

    ``normative_change`` is the second question asked separately. False is a
    rewording and the approval stands; true means the substance moved (CHF 500
    to CHF 550) and the approval a person gave against the old value cannot
    carry, even though the rule is the same rule.
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
    """A different claim: retire the old assertion and open a new one.

    Not everything continues. A materially different claim is a withdrawal and a
    creation, and recording it as an event is what keeps it from having to be
    inferred later from a distance score.
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
