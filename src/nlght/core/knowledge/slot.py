# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Where an assertion stands: the container the incremental diff runs in.

Two runs compare one slot's assertions against the same slot's, so a slot that
loses its identity retracts and recreates everything inside it and takes the
approvals with them. That makes it the third surrogate in the model, for the same
reason as the other two — the path proposes the match, the id is minted once and
never derived.

    document_id   the document a slot belongs to; matching never crosses it
    slot_id       persisted, minted once
    anchors       every path this slot has been found by, and what each was worth
    content       a recovery aid, and only that

so the effective identity is `document_id + slot_id`, and the anchors and the
content are aids for finding a slot again *within* one document.

**An anchor is worth what its source made it worth.** An id the author wrote —
`[[expenses]]`, `id="expenses"` — survives a rename, a reorder, an insertion
above it and a rewrite. A heading path survives none of those; it finds the
section again while nobody renames a heading, and stops the moment somebody does.
Both resolve; they do not resolve with the same confidence, and the resolution
says which happened so a later reader is not left to assume.

The strength travels **on each anchor** rather than once per slot, and one
transition is the reason:

    authored → authored    ordinary
    derived  → authored    an upgrade; very likely the same section
    derived  → derived     the fallback rungs are doing the work
    authored → derived     a degradation — do not continue blindly

When an authored id is removed and only a heading path is left, a slot that
claimed one strength for its whole life would carry a lineage across a change
nobody could see.

**The current anchor is the primary signal.** A resolving anchor is the answer
and nothing else is consulted. Content is a recovery aid, reached only where no
anchor resolves, and it claims a slot only when it claims exactly one. Several
candidates mean nothing is claimed: that errs towards a document holding one slot
too many, which shows up as an assertion arriving twice, over two sections
quietly becoming one, which nobody sees.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from nlght.core.ingestion.document import stable_digest
from nlght.core.knowledge.pipeline import AUTHORED_ANCHOR, DERIVED_ANCHOR
from nlght.core.knowledge.proposition import normalise

#: How a slot was arrived at. The rung *and* what the matched anchor was worth,
#: because "found it exactly" means something different when the thing found was
#: an authored id than when it was a heading somebody may rename tomorrow.
AUTHORED_EXACT = "authored_exact"
AUTHORED_ALIAS = "authored_alias"
DERIVED_EXACT = "derived_exact"
DERIVED_ALIAS = "derived_alias"
CONTENT_RECOVERY = "content_recovery"
AMBIGUOUS = "ambiguous"
NEW = "new"

_CONTINUATIONS = frozenset(
    {AUTHORED_EXACT, AUTHORED_ALIAS, DERIVED_EXACT, DERIVED_ALIAS, CONTENT_RECOVERY}
)

RETIRED = "retired"


@dataclass(slots=True, frozen=True)
class SlotAnchor:
    """One path a slot has been found by, and how much it was worth.

    Temporal rather than a current value beside a list of old ones: the same path
    can legitimately come back — a heading renamed and renamed back — and a
    mapping from anchor to strength would collapse that history into one entry.
    """

    anchor: str
    strength: str
    valid_from: datetime | None = None
    valid_to: datetime | None = None

    @property
    def is_current(self) -> bool:
        return self.valid_to is None


@dataclass(slots=True, frozen=True)
class SlotMatch:
    """Which slot a section is, and how that was decided."""

    slot_id: str
    resolution: str

    @property
    def is_continuation(self) -> bool:
        return self.resolution in _CONTINUATIONS

    @property
    def is_authored(self) -> bool:
        """Whether the match rested on something the source itself wrote."""
        return self.resolution in (AUTHORED_EXACT, AUTHORED_ALIAS)


def _anchors(value: Mapping[str, Any]) -> Sequence[SlotAnchor]:
    return tuple(value.get("anchors") or ())


def _live(document_id: str, known: Mapping[str, Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    """This document's slots that are still in play.

    Retired ones are excluded rather than filtered later: a section that comes
    back must not silently reattach to a slot retired months ago. Whether the old
    lineage continues is a decision worth taking deliberately, and taking it
    implicitly through a resolver is what the surrogate ids exist to prevent.
    """
    return {
        slot_id: value
        for slot_id, value in known.items()
        if value.get("document_id") == document_id and value.get("status") != RETIRED
    }


def _resolution(strength: str, *, exact: bool) -> str:
    if strength == AUTHORED_ANCHOR:
        return AUTHORED_EXACT if exact else AUTHORED_ALIAS
    return DERIVED_EXACT if exact else DERIVED_ALIAS


def resolve_slot(
    document_id: str,
    anchor: str,
    content: str,
    known: Mapping[str, Mapping[str, Any]],
) -> SlotMatch:
    """Which slot of this document this section is — an existing one, or a new one.

    A ladder, and every rung requires a *unique* answer:

        the current anchor, exactly
        → the current anchor, among several sharing it, disambiguated by content
        → a path this slot used to have
        → the content, when it matches exactly one slot
        → otherwise a new slot

    Nothing here weighs a score. Two candidates at any rung mean the section is
    new, because choosing between them is the heuristic that would decide
    identity.
    """
    live = _live(document_id, known)

    def _matching(slot_id: str, *, current: bool) -> SlotAnchor | None:
        for candidate in _anchors(live[slot_id]):
            if candidate.anchor == anchor and candidate.is_current is current:
                return candidate
        return None

    by_anchor = {
        slot_id: found
        for slot_id in live
        if (found := _matching(slot_id, current=True)) is not None
    }
    if len(by_anchor) == 1:
        slot_id, found = next(iter(by_anchor.items()))
        return SlotMatch(slot_id, _resolution(found.strength, exact=True))
    if by_anchor:
        # A document with two sections under one heading path. The anchor cannot
        # decide, so the content says which is which — the second and last place
        # it is allowed to.
        same_content = [
            slot_id for slot_id in by_anchor if live[slot_id].get("content") == content
        ]
        if len(same_content) == 1:
            found = by_anchor[same_content[0]]
            return SlotMatch(same_content[0], _resolution(found.strength, exact=True))
        return SlotMatch(_mint(), AMBIGUOUS)

    by_alias = {
        slot_id: found
        for slot_id in live
        if (found := _matching(slot_id, current=False)) is not None
    }
    if len(by_alias) == 1:
        # Resolved without looking at the content: once a rename is recorded, a
        # source that reverts it is not a rediscovery. That is what the history
        # is for — a later run does not work the move out again every time.
        slot_id, found = next(iter(by_alias.items()))
        return SlotMatch(slot_id, _resolution(found.strength, exact=False))
    if by_alias:
        return SlotMatch(_mint(), AMBIGUOUS)

    # No anchor resolved. Only now is content consulted, and only when it points
    # at one slot: boilerplate repeated across a document would otherwise pull
    # every copy of it onto whichever slot happened to be first.
    #
    # `live` excludes retired slots, so this cannot quietly reactivate one. A
    # section whose text matches something retired months ago is a decision for a
    # person, not a resemblance for a resolver.
    by_content = [slot_id for slot_id, value in live.items() if value.get("content") == content]
    if len(by_content) == 1:
        return SlotMatch(by_content[0], CONTENT_RECOVERY)
    if by_content:
        return SlotMatch(_mint(), AMBIGUOUS)

    return SlotMatch(_mint(), NEW)


def retire_slot(
    slot_id: str, known: Mapping[str, Mapping[str, Any]], *, at: datetime | None = None
) -> dict[str, Any]:
    """Mark a slot retired, keeping what a later run would recognise it by.

    Dropping the row instead would make a section that comes back
    indistinguishable from one that was never there, and the decision whether to
    continue the old lineage could no longer be taken at all.

    The current anchor is closed rather than removed: it becomes the path this
    slot *had*, which is the one a returning section is most likely to arrive
    under.
    """
    value = known.get(slot_id)
    if value is None:
        raise KeyError(f"no slot '{slot_id}' to retire")
    closed_at = at or datetime.now(tz=None)
    anchors = tuple(
        anchor if not anchor.is_current else SlotAnchor(
            anchor=anchor.anchor,
            strength=anchor.strength,
            valid_from=anchor.valid_from,
            valid_to=closed_at,
        )
        for anchor in _anchors(value)
    )
    return {
        "slot_id": slot_id,
        "document_id": value.get("document_id"),
        "status": RETIRED,
        "anchors": anchors,
        "content": value.get("content"),
        "retired_at": closed_at,
    }


def section_fingerprint(content: str) -> str:
    """What a section says, as the registry compares it.

    A hash, and of the *normalised* text: a section reflowed or given a full stop
    has not changed what it says, and a fingerprint that moved on those would
    make every reformatting look like a rename.

    One function, because the side that writes it and the side that compares it
    disagreeing is a bug nothing would report — every recovery would simply stop
    working.
    """
    return stable_digest(normalise(content))


def parser_input_fingerprint(content: str) -> str:
    """What the model was actually shown, as a hash — and nothing normalised away.

    The other question about a section, and deliberately not the one above.
    `section_fingerprint` asks *is this probably the same section*, which is why
    it may fold a reflow and a full stop; this one asks *was the model asked
    exactly this*, and an answer that folds anything is answering a different
    question.

    Using the normalised hash for both froze a slot's claims across a real edit:
    a sentence given a full stop is a changed question, the guard read it as an
    unchanged section, and the corrected wording never became a revision.
    """
    return stable_digest(content)


@dataclass(slots=True, frozen=True)
class ObservedSection:
    """One section of a document, as this run read it.

    The business boundary, deliberately: what a section *is* to the registry is
    where it can be found and what it says, and nothing else. The ordinal the
    parser produced is carried under a name that says what it is and takes part
    in no decision — it renumbers when a paragraph is inserted above, which is
    exactly why it must never reach identity again.

    Two fingerprints of one text, because two different questions are asked of
    it (see `parser_input_fingerprint`). Build with `as_read` rather than by
    hand: taking them from one string is what stops them describing different
    text, which nothing downstream could detect.
    """

    anchor: str
    anchor_strength: str
    content_fingerprint: str
    #: Position, for tracing a claim back to the batch it came from. Never an
    #: input to slot resolution and never the identity of anything.
    unit_ordinal: str = ""
    assertions: tuple[Any, ...] = ()
    #: The exact input this section was read from. Empty where a caller could
    #: not say — a section with no parser text — and an empty one never lets the
    #: stable-slot guard fire, because "both unknown" is not "both the same".
    parser_input_fingerprint: str = ""

    @classmethod
    def as_read(
        cls,
        *,
        anchor: str,
        anchor_strength: str,
        content: str,
        unit_ordinal: str = "",
        assertions: tuple[Any, ...] = (),
    ) -> ObservedSection:
        """One section from the one text it was read from."""
        return cls(
            anchor=anchor,
            anchor_strength=anchor_strength,
            content_fingerprint=section_fingerprint(content),
            unit_ordinal=unit_ordinal,
            assertions=assertions,
            parser_input_fingerprint=parser_input_fingerprint(content),
        )

    @property
    def anchorable(self) -> bool:
        """Whether a persisted slot may be minted for this section at all.

        A section nothing can find again would get a slot that is new on every
        run, retiring and recreating everything it holds each time — worse than
        no slot, because it claims a stability it does not have.
        """
        return self.anchor_strength in (AUTHORED_ANCHOR, DERIVED_ANCHOR) and bool(self.anchor)


def slot_value(
    document_id: str,
    anchors: Sequence[SlotAnchor] = (),
    content: str = "",
    status: str = "live",
) -> dict[str, Any]:
    """One slot as the resolver reads it."""
    return {
        "document_id": document_id,
        "anchors": tuple(anchors),
        "content": content,
        "status": status,
    }


def current_anchor(value: Mapping[str, Any]) -> SlotAnchor | None:
    """The path this slot is found by now, if it still has one."""
    return next((anchor for anchor in _anchors(value) if anchor.is_current), None)


def degraded(value: Mapping[str, Any], arriving: str) -> bool:
    """Whether this slot's anchor stopped being something the source wrote.

    The transition a single strength per slot could not express. Continuing is
    still right — it is the same section — but doing it without noticing is how a
    lineage gets carried across a change nobody could see.
    """
    anchor = current_anchor(value)
    return anchor is not None and anchor.strength == AUTHORED_ANCHOR and arriving == DERIVED_ANCHOR


def _mint() -> str:
    return f"slot_{uuid.uuid4().hex[:16]}"
