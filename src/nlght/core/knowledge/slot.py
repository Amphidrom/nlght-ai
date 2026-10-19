# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Resolve document sections to persistent slot identities."""

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
    can legitimately return when a heading is renamed and later restored. A
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
        """Identify a match grounded in a source-authored anchor.

        Returns:
            bool: Whether the resolution is authored.
        """
        return self.resolution in (AUTHORED_EXACT, AUTHORED_ALIAS)


def _anchors(value: Mapping[str, Any]) -> Sequence[SlotAnchor]:
    return tuple(value.get("anchors") or ())


def _live(document_id: str, known: Mapping[str, Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    """Filter live slots belonging to one document.

    Args:
        document_id (str): Document whose slots are being considered.
        known (Mapping[str, Mapping[str, Any]]): Known slot values keyed by slot ID.

    Returns:
        dict[str, Mapping[str, Any]]: Live slot values belonging to the document.
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
    """Resolve one section through exact anchor and content signals.

    Args:
        document_id (str): Document containing the arriving section.
        anchor (str): Authored section anchor.
        content (str): Exact section content.
        known (Mapping[str, Mapping[str, Any]]): Known slot values available for matching.

    Returns:
        SlotMatch: Match to an existing slot or a newly minted slot.

    Side Effects:
        May generate a random slot ID for an unmatched section.
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
        # decide, so the content says which is which. This is the second and last place
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
        # is for. A later run does not work the move out again every time.
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
    """Close a slot while retaining its anchor history.

    Args:
        slot_id (str): Slot to close.
        known (Mapping[str, Mapping[str, Any]]): Known slot values keyed by slot ID.
        at (datetime | None): Retirement time. Defaults to the current time.

    Returns:
        dict[str, Any]: Copy of the slot value with its anchor history closed.

    Side Effects:
        Reads the clock when at is omitted.
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
    """Hash normalized section content for slot comparison.

    Args:
        content (str): Section text.

    Returns:
        str: Section fingerprint.
    """
    return stable_digest(normalise(content))


def parser_input_fingerprint(content: str) -> str:
    """Hash exact parser input without normalization.

    Args:
        content (str): Parser input text.

    Returns:
        str: Parser-input fingerprint.
    """
    return stable_digest(content)


@dataclass(slots=True, frozen=True)
class ObservedSection:
    """One section of a document, as this run read it.

    The business boundary, deliberately: what a section *is* to the registry is
    where it can be found and what it says, and nothing else. The ordinal the
    parser produced is carried under a name that says what it is and takes part
    in no decision. It renumbers when a paragraph is inserted above, which is
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
    #: not say. A section with no parser text is empty and never lets the
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
        """Record the exact section content seen by a parser.

        Args:
            anchor (str): Section anchor as parsed.
            anchor_strength (str): Strength of the parsed anchor.
            content (str): Exact section text.
            unit_ordinal (str): Parser position of the section.
            assertions (tuple[Any, ...]): Claims observed in the section.

        Returns:
            ObservedSection: Observed section with content and parser-input fingerprints.
        """
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
        """Permit a persistent slot only for a usable anchor.

        Returns:
            bool: Whether the section can be anchored.
        """
        return self.anchor_strength in (AUTHORED_ANCHOR, DERIVED_ANCHOR) and bool(self.anchor)


def slot_value(
    document_id: str,
    anchors: Sequence[SlotAnchor] = (),
    content: str = "",
    status: str = "live",
) -> dict[str, Any]:
    """Build a slot value for registry storage.

    Args:
        document_id (str): Document that owns the slot.
        anchors (Sequence[SlotAnchor]): Recorded anchor history.
        content (str): Current section content.
        status (str): Slot lifecycle status.

    Returns:
        dict[str, Any]: Registry value with immutable anchor history.
    """
    return {
        "document_id": document_id,
        "anchors": tuple(anchors),
        "content": content,
        "status": status,
    }


def current_anchor(value: Mapping[str, Any]) -> SlotAnchor | None:
    """Find the currently valid anchor of a slot.

    Args:
        value (Mapping[str, Any]): Slot value.

    Returns:
        SlotAnchor | None: Current SlotAnchor or None.
    """
    return next((anchor for anchor in _anchors(value) if anchor.is_current), None)


def degraded(value: Mapping[str, Any], arriving: str) -> bool:
    """Detect loss of an authored section anchor.

    Args:
        value (Mapping[str, Any]): Existing slot value.
        arriving (str): Anchor of the arriving section.

    Returns:
        bool: True when the authored anchor loses specificity.
    """
    anchor = current_anchor(value)
    return anchor is not None and anchor.strength == AUTHORED_ANCHOR and arriving == DERIVED_ANCHOR


def _mint() -> str:
    return f"slot_{uuid.uuid4().hex[:16]}"
