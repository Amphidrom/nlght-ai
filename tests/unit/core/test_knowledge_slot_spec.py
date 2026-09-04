# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Executable specification for slot identity — where an assertion stands.

A slot is the container that carries the incremental diff. Two runs compare the
assertions of one slot against the assertions of the same slot, so if a slot
loses its identity every assertion inside it is retracted and recreated, and
every approval goes with them. That makes this the third surrogate in the model,
and for the same reason as the other two:

    document_id   the document a slot belongs to; matching never crosses it
    slot_id       persisted UUID, minted once. Never derived.
    anchors       every path this slot has been found by, and what each was worth
    content       a recovery aid, and only that

so the effective identity is `document_id + slot_id`, and the anchors and the
content are aids for finding a slot again *within the same document*.

An anchor carries its own strength rather than the slot carrying one for its
whole life, because `authored → derived` is a degradation a single value cannot
express: an authored id removed leaves a heading path, and a slot still claiming
to be authored would carry a lineage across a change nobody could see. The
history is temporal rather than "current plus a list of old ones", so a path that
comes back — a heading renamed and renamed back — is two entries and not one.

**The heading path is the primary signal.** A resolving anchor is the answer and
nothing else is consulted. Content similarity is a *recovery aid* — reached only
where the anchor no longer resolves, to work out which vanished slot the new
section is. It never overrules an anchor and never merges two sections that both
still have one.

What this replaces: `split_into_units` builds `unit_id` from
`f"{source_id}:{document_id}:{len(units)}"`, an ordinal, so inserting a paragraph
renumbers everything after it and the whole document reads as rewritten.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from nlght.core.knowledge import (
    AUTHORED_ANCHOR,
    DERIVED_ANCHOR,
    resolve_slot,
    retire_slot,
)
from nlght.core.knowledge.slot import (
    AMBIGUOUS,
    AUTHORED_ALIAS,
    AUTHORED_EXACT,
    CONTENT_RECOVERY,
    DERIVED_ALIAS,
    DERIVED_EXACT,
    NEW,
    SlotAnchor,
    degraded,
)


def _resolve_slot(
    document_id: str, anchor: str, content: str, known: dict[str, dict[str, Any]]
) -> str:
    """The id alone, for the cases that only care which slot it is."""
    return resolve_slot(document_id, anchor, content, known).slot_id


def _retire_slot(slot_id: str, known: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return retire_slot(slot_id, known)


def _slot(
    anchor: str,
    content: str,
    *aliases: str,
    document: str = "doc-1",
    strength: str = DERIVED_ANCHOR,
) -> dict[str, Any]:
    """A slot with its anchor history, current last.

    The document is part of a slot, not context around it: matching never crosses
    one, so a resolver handed the whole store has to be able to tell.

    `strength` defaults to derived because a heading path is what most formats
    can offer, and defaulting to the stronger reading would make every test look
    better anchored than the corpus it stands for.
    """
    return {
        "document_id": document,
        "anchors": (
            *(
                SlotAnchor(anchor=alias, strength=strength, valid_to=_CLOSED)
                for alias in aliases
            ),
            SlotAnchor(anchor=anchor, strength=strength),
        ),
        "content": content,
    }


#: Any past instant; the resolver only asks whether an anchor is still current.
_CLOSED = datetime(2026, 1, 1, tzinfo=UTC)


_TRAVEL = "Employees must book travel through the corporate agency."
_EXPENSE = "Expenses above CHF 500 require approval by a line manager."


# ---------------------------------------------------------------------------
# 1. Renamed heading
# ---------------------------------------------------------------------------

def test_renaming_a_heading_keeps_the_slot_and_records_the_old_path() -> None:
    """The case the alias history exists for.

    Somebody retitles "## Travel approval" to "## Travel authorisation" and
    changes nothing underneath. The primary signal fails — that path is new —
    so content recovers the slot, and the new path is added to the slot rather
    than replacing its identity.
    """
    known = {"slot-91": _slot("/travel/approval", _TRAVEL)}

    resolved = _resolve_slot("doc-1", "/travel/authorisation", _TRAVEL, known)

    assert resolved == "slot-91"


def test_a_path_this_slot_had_before_resolves_without_looking_at_content() -> None:
    # Once a rename is recorded, a source that reverts it is not a discovery.
    # The alias resolves directly, which is the whole point of keeping history:
    # a later run does not rediscover the move every time.
    known = {"slot-91": _slot("/travel/authorisation", _TRAVEL, "/travel/approval")}

    assert _resolve_slot("doc-1", "/travel/approval", "wholly different text", known) == "slot-91"


# ---------------------------------------------------------------------------
# 2. Moved section
# ---------------------------------------------------------------------------

def test_a_section_moved_under_another_parent_keeps_its_slot() -> None:
    # /expenses/travel becomes /policy/travel-approval. Same section, different
    # place in the document. Its assertions must not be retracted for it.
    known = {"slot-91": _slot("/expenses/travel", _TRAVEL)}

    assert _resolve_slot("doc-1", "/policy/travel-approval", _TRAVEL, known) == "slot-91"


# ---------------------------------------------------------------------------
# 3. Inserted paragraph — the ordinal problem
# ---------------------------------------------------------------------------

def test_inserting_a_section_leaves_every_other_slot_alone() -> None:
    """The defect today's `unit_id` has by construction.

    A new introduction is written at the top of the document. With an ordinal,
    every section below it renumbers and the whole document is retracted and
    recreated. With paths, the sections that did not move do not move.
    """
    known = {
        "slot-1": _slot("/travel", _TRAVEL),
        "slot-2": _slot("/expenses", _EXPENSE),
    }

    assert _resolve_slot("doc-1", "/travel", _TRAVEL, known) == "slot-1"
    assert _resolve_slot("doc-1", "/expenses", _EXPENSE, known) == "slot-2"


def test_the_inserted_section_itself_is_a_new_slot() -> None:
    known = {
        "slot-1": _slot("/travel", _TRAVEL),
        "slot-2": _slot("/expenses", _EXPENSE),
    }

    resolved = _resolve_slot("doc-1", "/introduction", "This handbook describes...", known)

    assert resolved not in known


# ---------------------------------------------------------------------------
# 4. Deleted section
# ---------------------------------------------------------------------------

def test_a_deleted_section_does_not_drag_its_neighbours_with_it() -> None:
    # /travel is gone from the document. The run resolves what remains, and
    # nothing about the deletion may move /expenses — the assertions of a
    # section nobody edited must survive the removal of the one above it.
    known = {
        "slot-1": _slot("/travel", _TRAVEL),
        "slot-2": _slot("/expenses", _EXPENSE),
    }

    assert _resolve_slot("doc-1", "/expenses", _EXPENSE, known) == "slot-2"


def test_a_deleted_section_is_not_recovered_onto_a_surviving_one() -> None:
    """The trap in using content as a recovery aid.

    A section disappears and its neighbour happens to read similarly. The
    neighbour's own anchor still resolves, so it is not a candidate for
    recovery, and the deleted slot stays deleted rather than being reported as
    a move.
    """
    known = {
        "slot-1": _slot("/travel/approval", _TRAVEL),
        "slot-2": _slot("/travel/booking", _TRAVEL + " Book early."),
    }

    assert _resolve_slot("doc-1", "/travel/booking", _TRAVEL + " Book early.", known) == "slot-2"


# ---------------------------------------------------------------------------
# 5. Two sections with the same heading in one document
# ---------------------------------------------------------------------------

def test_two_sections_with_one_heading_stay_two_slots() -> None:
    """Where the primary signal cannot decide on its own.

    A handbook with "### Approval" under both Travel and Expenses gives two
    sections one path. The anchor is ambiguous, so content decides which is
    which — the second legitimate use of the recovery aid — and they stay two
    slots. Collapsing them would merge two unrelated sets of assertions into one
    diff, retracting each on every run because the other one replaced it.
    """
    known = {
        "slot-1": _slot("/handbook/approval", _TRAVEL),
        "slot-2": _slot("/handbook/approval", _EXPENSE),
    }

    assert _resolve_slot("doc-1", "/handbook/approval", _TRAVEL, known) == "slot-1"
    assert _resolve_slot("doc-1", "/handbook/approval", _EXPENSE, known) == "slot-2"


# ---------------------------------------------------------------------------
# 6. The negative case — similarity must not merge two living sections
# ---------------------------------------------------------------------------

def test_two_similar_sections_under_different_headings_stay_apart() -> None:
    """The failure mode the recovery aid could introduce.

    Two sections that read almost identically — a policy repeated for two
    departments, boilerplate copied between chapters — each keep their own
    anchor. Neither anchor is missing, so similarity is never consulted, and the
    two never collapse onto one slot.

    If they did, one section's assertions would replace the other's on every
    run: each would be retracted as "no longer asserted here" and recreated,
    the approvals lost, in a document nobody had edited.
    """
    known = {
        "slot-1": _slot("/sales/expenses", _EXPENSE),
        "slot-2": _slot("/engineering/expenses", _EXPENSE),
    }

    first = _resolve_slot("doc-1", "/sales/expenses", _EXPENSE, known)
    second = _resolve_slot("doc-1", "/engineering/expenses", _EXPENSE, known)

    assert first == "slot-1"
    assert second == "slot-2"
    assert first != second


def test_similarity_is_only_consulted_for_anchors_that_no_longer_resolve() -> None:
    # Stated on its own because it is the rule that makes the case above hold,
    # and a resolver could pass that test by accident while breaking this one.
    # A slot whose anchor is still present is not available for rehanging.
    known = {
        "slot-1": _slot("/sales/expenses", _EXPENSE),
        "slot-2": _slot("/engineering/expenses", _EXPENSE),
    }

    # A third, genuinely new section reading like both of them. It may not be
    # recovered onto either, because neither is missing.
    resolved = _resolve_slot("doc-1", "/marketing/expenses", _EXPENSE, known)

    assert resolved not in known


# ---------------------------------------------------------------------------
# The slot id is a surrogate
# ---------------------------------------------------------------------------

def test_a_slot_id_is_not_derived_from_its_anchor() -> None:
    """The property a later "just hash the path" refactor would break.

    The same path resolves to an existing slot in one document and to a new slot
    in another, and the same section keeps its id through a rename that changes
    its path. A derivation can do neither. As with the assertion and the
    variant, the path proposes the match and the id is minted once.
    """
    known = {"slot-91": _slot("/travel/approval", _TRAVEL)}

    kept = _resolve_slot("doc-1", "/travel/authorisation", _TRAVEL, known)
    elsewhere = _resolve_slot("doc-2", "/travel/approval", _TRAVEL, {})

    assert kept == "slot-91"
    assert elsewhere != "slot-91"


# ---------------------------------------------------------------------------
# Slot matching is scoped to one document
# ---------------------------------------------------------------------------
#
# Which makes the effective identity of a slot:
#
#     document_id + persisted slot_id
#
# and leaves the heading path and content similarity as what they are: aids for
# finding a slot again *within the same document*.

def test_two_pages_with_the_same_heading_and_the_same_text_are_two_slots() -> None:
    """Two documents, identical in both signals, and still not one slot.

        Page A: "Approval" / "Expenses above CHF 500 require approval."
        Page B: "Approval" / "Expenses above CHF 500 require approval."

    Both signals agree and both are irrelevant, because they are signals for
    finding a slot again inside one document. A shared slot would mean the two
    pages hold one diff: whatever A says, B is asked to have said too, so each
    run retracts one page's assertions in favour of the other's and hands back
    an approval for a document nobody edited.

    Boilerplate repeated across pages is the normal case for this, not an
    exotic one — a policy paragraph copied into every team's handbook.
    """
    boilerplate = "Expenses above CHF 500 require approval."
    known: dict[str, dict[str, Any]] = {}

    page_a = _resolve_slot("page-a", "/approval", boilerplate, known)
    known[page_a] = _slot("/approval", boilerplate, document="page-a")
    page_b = _resolve_slot("page-b", "/approval", boilerplate, known)

    assert page_a != page_b


def test_an_existing_slot_is_never_matched_from_another_document() -> None:
    # The same rule from the other side: a slot that exists in one document is
    # not a candidate for a section of a different one, however well it matches.
    known = {"slot-91": _slot("/travel/approval", _TRAVEL)}

    assert _resolve_slot("doc-2", "/travel/approval", _TRAVEL, known) != "slot-91"


def test_a_vanished_slot_is_not_recovered_onto_another_document() -> None:
    """Recovery is scoped too, not only exact resolution.

    Deleting a section from one document must not rehang its slot onto a
    similar section elsewhere. Without the scope the deletion would be recorded
    as a move and the assertions would silently change document — so a citation
    would point at a page that never made the claim.
    """
    known = {"slot-91": _slot("/travel/approval", _TRAVEL)}

    resolved = _resolve_slot("doc-2", "/travel/authorisation", _TRAVEL, known)

    assert resolved != "slot-91"


# ---------------------------------------------------------------------------
# Retirement — a deleted slot is retired, not forgotten
# ---------------------------------------------------------------------------
#
#     slot_id
#     document_id
#     status = retired
#     last_anchor
#     aliases[]
#
# Dropping the row instead would make a section that comes back indistinguishable
# from one that was never there, and the decision whether to continue the old
# lineage could no longer be taken at all.


def test_a_slot_whose_section_is_gone_is_retired_rather_than_deleted() -> None:
    known = {"slot-91": _slot("/travel/approval", _TRAVEL)}

    retired = _retire_slot("slot-91", known)

    assert retired["status"] == "retired"
    # Closed, not dropped: the path it had is what a returning section arrives
    # under, and nothing else could tell that from a section never seen before.
    assert [anchor.anchor for anchor in retired["anchors"]] == ["/travel/approval"]
    assert not any(anchor.is_current for anchor in retired["anchors"])


def test_a_retired_slot_keeps_the_paths_it_had() -> None:
    # Every path it was found by is what a later run would recognise it by, so
    # they outlive the section itself — and each keeps the strength it had, so a
    # decision to continue the lineage can be taken knowing what it rests on.
    known = {"slot-91": _slot("/travel/authorisation", _TRAVEL, "/travel/approval")}

    retired = _retire_slot("slot-91", known)

    assert [anchor.anchor for anchor in retired["anchors"]] == [
        "/travel/approval",
        "/travel/authorisation",
    ]
    assert all(anchor.strength == DERIVED_ANCHOR for anchor in retired["anchors"])


def test_a_retired_slot_is_not_resolved_into_by_accident() -> None:
    """Retirement is a record, not a candidate list.

    A section that comes back must not silently reattach to a slot that was
    retired months ago — whether the old lineage continues is a decision worth
    taking deliberately, and taking it implicitly through a similarity score is
    exactly the kind of thing the surrogate ids exist to prevent. Resolution
    therefore mints a new slot, and the retired one stays available to be
    consulted rather than being matched into.
    """
    known = {
        "slot-91": {**_slot("/travel/approval", _TRAVEL), "status": "retired"},
    }

    resolved = _resolve_slot("doc-1", "/travel/approval", _TRAVEL, known)

    assert resolved != "slot-91"


# ---------------------------------------------------------------------------
# The ladder, and where it refuses to decide
# ---------------------------------------------------------------------------
#
#     the anchor, exactly
#     → the anchor, among several sharing it, disambiguated by content
#     → a path this slot used to have
#     → the content, when it points at exactly one slot
#     → otherwise a new slot
#
# Every rung needs a unique answer. Two candidates anywhere mean the section is
# new, because choosing between them is the heuristic that would decide identity.


def test_the_resolution_says_which_rung_answered() -> None:
    # Carried out rather than swallowed: a run can log how many of its sections
    # were recovered by content rather than found by their path, which is what
    # a source drifting away from its anchors looks like before it hurts.
    known = {"slot-1": _slot("/travel", _TRAVEL)}

    assert resolve_slot("doc-1", "/travel", _TRAVEL, known).resolution == DERIVED_EXACT
    assert resolve_slot("doc-1", "/moved", _TRAVEL, known).resolution == CONTENT_RECOVERY
    assert resolve_slot("doc-1", "/new", "Unrelated prose.", known).resolution == NEW


def test_boilerplate_repeated_in_one_document_recovers_nothing() -> None:
    """The failure the uniqueness rule prevents.

    The same paragraph appears under three headings. Two of them are known; a
    third arrives under a path that does not resolve. Recovering it onto
    whichever matched first would pull one section's assertions onto another,
    and each run would then retract the loser's and recreate them.
    """
    known = {
        "slot-1": _slot("/sales/policy", _EXPENSE),
        "slot-2": _slot("/engineering/policy", _EXPENSE),
    }

    match = resolve_slot("doc-1", "/marketing/policy", _EXPENSE, known)

    assert match.slot_id not in known
    assert match.resolution == AMBIGUOUS
    assert not match.is_continuation


def test_two_slots_holding_one_alias_recover_nothing() -> None:
    # Two sections have both been renamed away from the same old path. Which of
    # them a section arriving under it belongs to is not answerable, and
    # answering it anyway would attach a history to the wrong section.
    known = {
        "slot-1": _slot("/a/new", _TRAVEL, "/old/path"),
        "slot-2": _slot("/b/new", _EXPENSE, "/old/path"),
    }

    match = resolve_slot("doc-1", "/old/path", _TRAVEL, known)

    assert match.slot_id not in known
    assert match.resolution == AMBIGUOUS


def test_three_sections_under_one_heading_need_distinct_content() -> None:
    # The anchor cannot decide and neither can the content when two of them read
    # alike. Nothing is claimed rather than one of the three being picked.
    known = {
        "slot-1": _slot("/handbook/approval", _TRAVEL),
        "slot-2": _slot("/handbook/approval", _EXPENSE),
        "slot-3": _slot("/handbook/approval", _EXPENSE),
    }

    assert resolve_slot("doc-1", "/handbook/approval", _TRAVEL, known).slot_id == "slot-1"
    assert (
        resolve_slot("doc-1", "/handbook/approval", _EXPENSE, known).resolution == AMBIGUOUS
    )


# ---------------------------------------------------------------------------
# Retirement keeps what a returning section would arrive under
# ---------------------------------------------------------------------------

def test_retiring_a_slot_closes_its_current_path_rather_than_dropping_it() -> None:
    """Its current path becomes a past one, because that is what a return uses.

    A section that comes back most likely arrives under the path it had when it
    left. Losing that would make the history the wrong shape for the one question
    it exists to answer.
    """
    known = {"slot-91": _slot("/travel/authorisation", _TRAVEL, "/travel/approval")}

    retired = _retire_slot("slot-91", known)

    closed = [anchor for anchor in retired["anchors"] if anchor.anchor == "/travel/authorisation"]
    assert len(closed) == 1
    assert closed[0].is_current is False


def test_retiring_a_slot_keeps_the_document_it_belonged_to() -> None:
    # Without it a retired slot could not be scoped back to its document, and
    # the boundary that holds everywhere else would have a hole in the archive.
    known = {"slot-91": _slot("/travel", _TRAVEL, document="page-a")}

    assert retire_slot("slot-91", known)["document_id"] == "page-a"


def test_retiring_a_slot_that_is_not_there_is_an_error() -> None:
    # Silently returning a retirement record for a slot nobody has would write
    # an archive entry for a section that never existed.
    with pytest.raises(KeyError, match="slot-91"):
        retire_slot("slot-91", {})


# ---------------------------------------------------------------------------
# What the match rested on
# ---------------------------------------------------------------------------

def test_an_authored_anchor_says_so_when_it_resolves() -> None:
    """The distinction one word could not carry.

    "Found it exactly" means something different when the thing found was an id
    the author wrote than when it was a heading somebody may rename tomorrow. A
    later reader of `resolution` should not have to assume which happened.
    """
    known = {
        "slot-91": _slot("expenses-policy", _EXPENSE, strength=AUTHORED_ANCHOR),
    }

    match = resolve_slot("doc-1", "expenses-policy", _EXPENSE, known)

    assert match.resolution == AUTHORED_EXACT
    assert match.is_authored
    assert match.is_continuation


def test_a_derived_anchor_says_so_too() -> None:
    known = {"slot-91": _slot("/expenses", _EXPENSE)}

    match = resolve_slot("doc-1", "/expenses", _EXPENSE, known)

    assert match.resolution == DERIVED_EXACT
    assert match.is_authored is False


def test_an_authored_alias_is_distinguishable_from_a_derived_one() -> None:
    # A rename recorded against an authored id and one recorded against a heading
    # path are both continuations and are not worth the same. Reporting both as
    # "alias" would flatten exactly the difference the strength was added for.
    authored = {
        "slot-91": _slot("expenses-v2", _EXPENSE, "expenses", strength=AUTHORED_ANCHOR),
    }
    derived = {"slot-92": _slot("/expenses/v2", _EXPENSE, "/expenses", document="doc-2")}

    assert resolve_slot("doc-1", "expenses", _EXPENSE, authored).resolution == AUTHORED_ALIAS
    assert resolve_slot("doc-2", "/expenses", _EXPENSE, derived).resolution == DERIVED_ALIAS


def test_a_path_that_comes_back_is_two_entries_and_not_one() -> None:
    """Why the history is temporal rather than a mapping.

    A heading renamed and renamed back gives the same path twice, at different
    times. A mapping from anchor to strength would collapse those into one entry
    and lose the fact that the slot ever moved.
    """
    known = {
        "slot-91": {
            "document_id": "doc-1",
            "anchors": (
                SlotAnchor("/expenses", DERIVED_ANCHOR, valid_to=_CLOSED),
                SlotAnchor("/spending", DERIVED_ANCHOR, valid_to=_CLOSED),
                SlotAnchor("/expenses", DERIVED_ANCHOR),
            ),
            "content": _EXPENSE,
        }
    }

    match = resolve_slot("doc-1", "/expenses", _EXPENSE, known)

    assert match.resolution == DERIVED_EXACT, "the current entry answers, not the closed one"
    assert match.slot_id == "slot-91"


def test_a_degradation_is_visible_without_changing_the_answer() -> None:
    """`authored → derived`, which a single strength per slot could not express.

    The authored id was removed from the source, so the section arrives with a
    heading path. It is still the same section and the slot still continues —
    what must not happen is continuing while believing the anchor is as strong as
    it was.
    """
    known = {"slot-91": _slot("expenses-policy", _EXPENSE, strength=AUTHORED_ANCHOR)}

    assert degraded(known["slot-91"], DERIVED_ANCHOR) is True
    assert degraded(known["slot-91"], AUTHORED_ANCHOR) is False


def test_an_upgrade_is_not_a_degradation() -> None:
    # derived → authored: somebody added an id to a heading. Very likely the same
    # section, and nothing to warn about.
    known = {"slot-91": _slot("/expenses", _EXPENSE)}

    assert degraded(known["slot-91"], AUTHORED_ANCHOR) is False


def test_content_recovery_still_cannot_reactivate_a_retired_slot() -> None:
    """The guardrail this rewrite must not have dropped.

    Content is the last rung and the most eager one, and a retired slot is
    precisely where eagerness costs something: a section whose text matches
    something retired months ago is a decision for a person, not a resemblance
    for a resolver. `_live` excludes retired slots before content is ever
    consulted, and this is what keeps that true.
    """
    known = {
        "slot-91": {
            **_slot("/travel/approval", _TRAVEL, strength=AUTHORED_ANCHOR),
            "status": "retired",
        }
    }

    # A different anchor, so nothing but the content could match.
    match = resolve_slot("doc-1", "/travel/booking", _TRAVEL, known)

    assert match.slot_id != "slot-91"
    assert match.resolution == NEW
