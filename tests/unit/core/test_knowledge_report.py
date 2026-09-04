# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Holding a rerun against the run before it.

The acceptance metric is a comparison, not a number, and for a long time it was
the wrong comparison. It could say *that* something moved and never *where*, so a
run over a corpus somebody had deliberately edited reported "a variant gained a
revision over an unchanged source" — the design working, printed as a failure.
Read that often enough and the line stops being read at all.

So movement is now attributed. Each claim carries the document and slot it stands
in, each document the revision it was last seen at, and a document whose revision
moved is one somebody edited. Nothing has to be told which those were — two
pictures say it — and that is what makes the two gates separable:

    unchanged source   nothing may move but the evidence
    changed source     movement inside edited documents is the point of the run;
                       movement anywhere else is still drift
"""

from __future__ import annotations

from nlght.core.knowledge import AssertionState, LineageReport, compare
from nlght.core.knowledge.report import CHANGED_SOURCE, UNCHANGED_SOURCE

DOC_A = "doc-actuator"
DOC_B = "doc-aot"


def _state(
    *,
    variant: str = "var-1",
    revision: int = 1,
    review: str = "approved",
    document: str = DOC_A,
    slot: str = "slot-1",
    retired: bool = False,
    kind: str = "rule",
) -> AssertionState:
    return AssertionState(
        assertion_id=f"a_{variant}",
        entity_key=f"key-{variant}",
        variant_id=variant,
        revision=revision,
        review_state=review,
        document_id=document,
        slot_id=slot,
        retired=retired,
        kind=kind,
    )


def _report(
    *states: AssertionState,
    documents: dict[str, str] | None = None,
    **overrides: object,
) -> LineageReport:
    values: dict[str, object] = {
        "assertions": len(states), "variants": len(states), "revisions": len(states),
        "evidence": len(states), "graph_assertions": len(states),
        "graph_without_lineage": 0, "entity_key_versions": {"1": len(states)},
        "evidence_without_document": 0,
        "states": [state.as_dict() for state in states],
        "documents": documents if documents is not None else {DOC_A: "rev-1", DOC_B: "rev-1"},
    }
    values.update(overrides)
    return LineageReport.from_dict(values)


# ---------------------------------------------------------------------------
# Nothing was edited
# ---------------------------------------------------------------------------

def test_a_rerun_that_changed_nothing_is_unchanged() -> None:
    result = compare(_report(_state()), _report(_state()))

    assert result.unchanged
    assert result.mode == UNCHANGED_SOURCE
    assert result.movements == ()


def test_more_evidence_is_the_run_happening_not_the_corpus_moving() -> None:
    """The one difference that is expected, and the reason the two are split.

    A second run is a second sighting, and recording that document D still
    asserts something is exactly what licenses not retracting it later. A
    comparison that called this a violation would cry wolf on every run.
    """
    result = compare(_report(_state()), _report(_state(), evidence=24))

    assert result.unchanged
    assert result.violations == ()
    assert any("evidence" in item for item in result.expected)


def test_a_revision_over_an_unedited_document_is_drift() -> None:
    """The failure the whole design exists to remove.

    No document revision moved, so nobody edited anything — and a claim gained a
    second state, which means the model was asked again and answered
    differently. Under incremental update that is a retraction and a re-review
    of something nobody touched.
    """
    result = compare(_report(_state()), _report(_state(revision=2)))

    assert not result.unchanged
    assert result.mode == UNCHANGED_SOURCE
    assert [move.kind for move in result.movements] == ["revision added"]
    assert result.movements[0].expected is False


def test_a_new_assertion_in_an_unedited_document_is_drift() -> None:
    result = compare(_report(_state()), _report(_state(), _state(variant="var-2")))

    assert not result.unchanged
    assert [move.kind for move in result.movements] == ["assertion added"]


# ---------------------------------------------------------------------------
# Something was edited
# ---------------------------------------------------------------------------

def test_a_revision_inside_an_edited_document_is_expected() -> None:
    """The line that used to read as a failure and is the result being sought.

    The document's revision moved, so somebody edited it, and a rule that was
    reworded gaining a second state is precisely what the run is for.
    """
    result = compare(
        _report(_state(document=DOC_B)),
        _report(_state(document=DOC_B, revision=2), documents={DOC_A: "rev-1", DOC_B: "rev-2"}),
    )

    assert result.unchanged
    assert result.mode == CHANGED_SOURCE
    assert result.changed_documents == (DOC_B,)
    assert result.movements[0].expected is True
    assert result.movements[0].source_revision == ("rev-1", "rev-2")


def test_an_edit_in_one_document_does_not_excuse_movement_in_another() -> None:
    """The guard that makes the changed-source gate worth anything.

    A run that edited one page must not be a licence for every other page to
    drift. Attribution is per document, never per run.
    """
    edited = _state(document=DOC_B, revision=2)
    untouched = _state(variant="var-9", document=DOC_A, revision=2)

    result = compare(
        _report(_state(document=DOC_B), _state(variant="var-9", document=DOC_A)),
        _report(edited, untouched, documents={DOC_A: "rev-1", DOC_B: "rev-2"}),
    )

    assert not result.unchanged
    assert [move.expected for move in result.movements] == [True, False]
    assert result.violations and "document_id=doc-actuator" in result.violations[0]


def test_a_spent_approval_is_named_even_where_it_was_expected() -> None:
    # On an edited document a lost approval is the design working — a changed
    # rule must be looked at again — but somebody still has to act on it, so it
    # is never silent.
    result = compare(
        _report(_state(document=DOC_B)),
        _report(
            _state(document=DOC_B, revision=2, review="needs_review"),
            documents={DOC_A: "rev-1", DOC_B: "rev-2"},
        ),
    )

    assert result.unchanged
    assert any(move.kind == "approval spent" for move in result.movements)
    assert any("approval spent" in item for item in result.expected)


def test_a_retirement_inside_an_edited_document_is_expected() -> None:
    result = compare(
        _report(_state(document=DOC_B)),
        _report(_state(document=DOC_B, retired=True), documents={DOC_A: "rev-1", DOC_B: "rev-2"}),
    )

    assert result.unchanged
    assert [move.kind for move in result.movements] == ["assertion retired"]


def test_a_retirement_in_an_unedited_document_is_drift() -> None:
    # Knowledge taken back from a page nobody touched is the worst outcome the
    # report can carry, and it must never be filed under expected.
    result = compare(_report(_state()), _report(_state(retired=True)))

    assert not result.unchanged
    assert result.movements[0].expected is False


def test_an_assertion_that_simply_disappeared_is_always_a_violation() -> None:
    """Rows are retired here, never deleted.

    One that is merely gone was deleted by something, and nothing in this design
    deletes — so it is a violation whatever else the run did.
    """
    result = compare(
        _report(_state(document=DOC_B)),
        _report(documents={DOC_A: "rev-1", DOC_B: "rev-2"}),
    )

    assert not result.unchanged
    assert [move.kind for move in result.movements] == ["assertion disappeared"]


def test_a_claim_with_no_document_is_never_excused() -> None:
    # It cannot be attributed, and calling it expected would hide exactly the
    # case that needs seeing.
    result = compare(
        _report(_state(document="")),
        _report(_state(document="", revision=2), documents={DOC_B: "rev-2"}),
    )

    assert not result.unchanged
    assert result.movements[0].expected is False


# ---------------------------------------------------------------------------
# What the counts still answer
# ---------------------------------------------------------------------------

def test_more_graph_rows_the_lineage_cannot_reach_is_a_violation() -> None:
    result = compare(_report(_state()), _report(_state(), graph_without_lineage=9))

    assert not result.unchanged
    assert any("cannot reach" in item for item in result.violations)


def test_fewer_unreachable_graph_rows_is_not_a_violation() -> None:
    assert compare(
        _report(_state(), graph_without_lineage=4), _report(_state())
    ).unchanged


def test_two_key_schemes_at_once_is_a_violation() -> None:
    # Nothing should be written under a newer scheme until the older one is
    # migrated, and the report is where that becomes visible.
    result = compare(
        _report(_state()), _report(_state(), entity_key_versions={"1": 10, "2": 3})
    )

    assert not result.unchanged
    assert any("key schemes" in item for item in result.violations)


def test_evidence_without_a_document_is_a_violation() -> None:
    result = compare(_report(_state()), _report(_state(), evidence_without_document=2))

    assert not result.unchanged
    assert any("without a document" in item for item in result.violations)


def test_a_movement_names_where_it_happened() -> None:
    """What the operator actually reads.

    The old output said a variant gained a revision and left finding out where
    to a database session. Provenance in the line is the difference between a
    report somebody acts on and one somebody scrolls past.
    """
    result = compare(
        _report(_state(document=DOC_B, slot="unit:aot#conditions")),
        _report(
            _state(document=DOC_B, slot="unit:aot#conditions", revision=2),
            documents={DOC_A: "rev-1", DOC_B: "rev-2"},
        ),
    )

    printed = str(result.movements[0])
    assert "document_id=doc-aot" in printed
    assert "slot_id=unit:aot#conditions" in printed
    assert "source_revision rev-1 -> rev-2" in printed
    assert "expected_change=true" in printed


def test_the_round_trip_through_json_keeps_the_provenance() -> None:
    # The baseline is written to a file between runs, so anything that does not
    # survive `as_dict`/`from_dict` may as well not have been recorded.
    original = _report(_state(document=DOC_B, slot="s", revision=3, review="needs_review"))

    restored = LineageReport.from_dict(original.as_dict())

    assert restored.states == original.states
    assert restored.documents == original.documents


# ---------------------------------------------------------------------------
# The document is the coarse gate; the section is the one that knows
# ---------------------------------------------------------------------------

def test_a_movement_in_a_section_nobody_touched_is_drift(  # noqa: D103
) -> None:
    """The gap the document gate leaves, and the reason for the finer one.

    Edit one paragraph and every claim in the file becomes "expected" — including
    the ones in sections nobody went near. That is what waved through three
    retirements and three additions in a live run: they were in an edited
    document, so they were allowed, and nothing looked closer.

    The corpus already knows better. Each slot records the parser input it was
    read with, and a section whose input did not move explains nothing about a
    claim that did.
    """
    before = _report(
        _state(variant="var-1"),
        documents={DOC_A: "rev-1"},
        slot_inputs={"slot-1": "fp-unchanged"},
        slot_observed_at={"slot-1": "rev-1"},
    )
    after = _report(
        _state(variant="var-1", revision=2),
        documents={DOC_A: "rev-2"},          # the file was edited
        slot_inputs={"slot-1": "fp-unchanged"},  # this section was not
        # Read again at the new revision: the section is still there, and it
        # says the same thing. That is what makes the movement drift.
        slot_observed_at={"slot-1": "rev-2"},
    )

    result = compare(before, after)

    assert not result.unchanged
    [move] = [m for m in result.movements if m.kind == "revision added"]
    assert move.expected is False
    assert move.because == "slot_input_unchanged"


def test_a_movement_in_the_section_that_changed_is_expected_and_says_so() -> None:
    before = _report(
        _state(variant="var-1"),
        documents={DOC_A: "rev-1"},
        slot_inputs={"slot-1": "fp-before"},
        slot_observed_at={"slot-1": "rev-1"},
    )
    after = _report(
        _state(variant="var-1", revision=2),
        documents={DOC_A: "rev-2"},
        slot_inputs={"slot-1": "fp-after"},
        slot_observed_at={"slot-1": "rev-2"},
    )

    result = compare(before, after)

    [move] = [m for m in result.movements if m.kind == "revision added"]
    assert move.expected is True
    assert move.because == "slot_input_changed"
    assert move.gate == "slot"
    assert move.slot_input == ("fp-before", "fp-after")
    rendered = str(move)
    assert "slot_input_changed=true" in rendered
    assert "old_content_fingerprint=fp-before" in rendered
    assert "new_content_fingerprint=fp-after" in rendered
    assert "because=slot_input_changed" in rendered


def test_a_slot_nobody_has_read_falls_back_to_the_document() -> None:
    """Absence of a reading is not evidence that a section stood still.

    Nothing was backfilled, so every slot observed before the readings existed
    looks exactly like an unchanged one. Calling that drift would report the
    migration rather than the corpus.
    """
    before = _report(_state(variant="var-1"), documents={DOC_A: "rev-1"})
    after = _report(_state(variant="var-1", revision=2), documents={DOC_A: "rev-2"})

    result = compare(before, after)

    [move] = [m for m in result.movements if m.kind == "revision added"]
    assert move.expected is True
    assert move.because == "slot_observation_unavailable"
    # Named, because it is the weaker answer: a transitional run right after the
    # migration must not read like a real slot comparison.
    assert move.gate == "document_fallback"
    assert "gate=document_fallback" in str(move)


def test_an_unobserved_slot_in_an_unedited_document_is_still_drift() -> None:
    # The fallback loosens nothing: with no reading and no edit, the old gate's
    # answer stands.
    before = _report(_state(variant="var-1"), documents={DOC_A: "rev-1"})
    after = _report(_state(variant="var-1", revision=2), documents={DOC_A: "rev-1"})

    result = compare(before, after)

    [move] = [m for m in result.movements if m.kind == "revision added"]
    assert move.expected is False
    assert move.because == "document_unchanged"


# ---------------------------------------------------------------------------
# Classification drift: its own channel, and nothing is merged
# ---------------------------------------------------------------------------

def test_a_kind_exchanged_in_one_slot_is_reported_separately() -> None:
    """What the slot gate cannot catch, from the run that showed it.

    Three rules were retired and a decision and two facts arrived — in a slot
    whose text really had been edited, so every one of the six read as expected
    and the report said nothing more. `kind` is part of the identity namespace,
    so the model calling one sentence a `rule` and then a `decision` retires the
    claim and creates another whether or not anything moved.

    Reported, and only reported. Whether the two are one claim is an equivalence
    judgement over whole propositions, and this has neither.
    """
    before = _report(
        _state(variant="var-1", kind="rule"),
        documents={DOC_A: "rev-1"},
        slot_inputs={"slot-1": "fp-before"},
    )
    after = _report(
        _state(variant="var-2", kind="decision"),
        documents={DOC_A: "rev-2"},
        slot_inputs={"slot-1": "fp-after"},
    )
    # The retirement, as the run would record it.
    after = LineageReport.from_dict({
        **after.as_dict(),
        "states": [
            _state(variant="var-1", kind="rule", retired=True).as_dict(),
            _state(variant="var-2", kind="decision").as_dict(),
        ],
    })

    result = compare(before, after)

    [finding] = result.classification_drift
    assert "slot-1" in finding
    assert "rule left" in finding
    assert "decision arrived" in finding
    # Reported beside the movements, never instead of them, and never as a merge.
    assert any(m.kind == "assertion retired" for m in result.movements)
    assert any(m.kind == "assertion added" for m in result.movements)


def test_the_same_kind_leaving_and_arriving_is_not_classification_drift() -> None:
    # A claim replaced by another of its own kind is ordinary movement; only an
    # exchange of kinds is the signal.
    before = _report(
        _state(variant="var-1", kind="rule"),
        documents={DOC_A: "rev-1"},
        slot_inputs={"slot-1": "fp-before"},
    )
    after = LineageReport.from_dict({
        **_report(documents={DOC_A: "rev-2"}, slot_inputs={"slot-1": "fp-after"}).as_dict(),
        "states": [
            _state(variant="var-1", kind="rule", retired=True).as_dict(),
            _state(variant="var-2", kind="rule").as_dict(),
        ],
    })

    result = compare(before, after)

    assert result.classification_drift == ()


# ---------------------------------------------------------------------------
# Historical debt is not a contract breach
# ---------------------------------------------------------------------------

def test_a_run_that_preserves_old_gaps_is_not_a_violation() -> None:
    """The measurement that would fail a perfect run if it were an equality.

    Revisions written before every kind had a proposition keep their `NULL`
    deliberately — a backfill from the payload would be a guess wearing the
    clothes of an observation. So a correct run leaves that number exactly where
    it was, and comparing it against zero would report debt the design is
    choosing to preserve as a fresh breach.
    """
    before = _report(_state(), revisions_without_proposition=19)
    after = _report(_state(), revisions_without_proposition=19)

    assert compare(before, after).unchanged


def test_a_new_revision_with_no_proposition_is_a_violation() -> None:
    # Growth is the breach. The repository refuses to write one, so more of them
    # means the refusal was bypassed — which is worth failing a run over.
    before = _report(_state(), revisions_without_proposition=19)
    after = _report(_state(), revisions_without_proposition=20)

    result = compare(before, after)

    assert not result.unchanged
    assert "1 new revision(s) written with no proposition" in str(result)


def test_a_sighting_with_nothing_to_judge_it_by_is_a_violation() -> None:
    before = _report(_state(), observations_without_proposition=0)
    after = _report(_state(), observations_without_proposition=2)

    assert "nothing to judge it by" in str(compare(before, after))


def test_unreferenced_propositions_are_reported_and_not_a_violation() -> None:
    """A retention question, deliberately not a failure.

    Propositions are content-addressed and never deleted, so abandoned
    extractions accumulate. Whether that matters is a decision to take on real
    numbers, and failing a run for it now would force the decision by accident.
    """
    before = _report(_state(), propositions=52, unreferenced_propositions=19)
    after = _report(_state(), propositions=60, unreferenced_propositions=27)

    assert compare(before, after).unchanged


# ---------------------------------------------------------------------------
# Presence before content
#
# A fingerprint may only be compared where the section was read on both sides. A
# deleted section is never read again and keeps its last reading forever, so it
# compares equal to itself — and a live run reported a removed section as the
# corpus drifting for exactly that reason, on the one document written to
# demonstrate a removed section.
# ---------------------------------------------------------------------------

def test_a_section_that_was_removed_is_not_a_section_that_stood_still() -> None:
    """The live failure, in miniature.

    The slot's stored reading is identical on both sides — because nothing read
    it again. Comparing those two equal fingerprints said "the section was read
    and had not moved", and the retirement of the claim it held became drift.
    """
    before = _report(
        _state(variant="var-1"),
        documents={DOC_A: "rev-1"},
        slot_inputs={"slot-1": "fp-1"},
        slot_observed_at={"slot-1": "rev-1"},
    )
    after = _report(
        _state(variant="var-1", retired=True),
        documents={DOC_A: "rev-2"},
        # Unchanged, and meaningless: the section is gone, so nothing read it.
        slot_inputs={"slot-1": "fp-1"},
        slot_observed_at={"slot-1": "rev-1"},
    )

    result = compare(before, after)

    [move] = result.movements
    assert move.because == "slot_removed"
    assert move.gate == "slot"
    assert move.expected is True
    assert result.unchanged, str(result)


def test_a_section_that_arrived_is_named_as_one() -> None:
    before = _report(
        _state(variant="var-1"),
        documents={DOC_A: "rev-1"},
        slot_inputs={"slot-1": "fp-1"},
        slot_observed_at={"slot-1": "rev-1"},
    )
    after = _report(
        _state(variant="var-1"),
        _state(variant="var-2", slot="slot-2"),
        documents={DOC_A: "rev-2"},
        slot_inputs={"slot-1": "fp-1", "slot-2": "fp-new"},
        slot_observed_at={"slot-1": "rev-2", "slot-2": "rev-2"},
    )

    result = compare(before, after)

    [move] = [m for m in result.movements if m.slot_id == "slot-2"]
    assert move.because == "slot_added"
    assert move.expected is True


def test_a_reading_from_neither_revision_says_nothing_and_the_document_answers() -> None:
    """The legacy case, kept and narrowed.

    A reading older than both pictures cannot speak about either, so the
    document decides — as it did before readings existed. What no longer happens
    is treating a slot that was deliberately not read again as one that stood
    still.
    """
    before = _report(
        _state(variant="var-1"),
        documents={DOC_A: "rev-5"},
        slot_inputs={"slot-1": "fp-old"},
        slot_observed_at={"slot-1": "rev-1"},
    )
    after = _report(
        _state(variant="var-1", revision=2),
        documents={DOC_A: "rev-6"},
        slot_inputs={"slot-1": "fp-old"},
        slot_observed_at={"slot-1": "rev-1"},
    )

    [move] = [m for m in compare(before, after).movements if m.kind == "revision added"]

    assert move.gate == "document_fallback"
    assert move.because == "slot_observation_unavailable"


def test_a_report_with_no_readings_at_all_still_falls_back() -> None:
    # Snapshots taken before the revision was recorded carry no presence, and
    # must not be read as "the section is gone".
    before = _report(_state(variant="var-1"), documents={DOC_A: "rev-1"})
    after = _report(
        _state(variant="var-1", revision=2), documents={DOC_A: "rev-2"}
    )

    [move] = compare(before, after).movements

    assert move.gate == "document_fallback"


# ---------------------------------------------------------------------------
# What the headline is allowed to say
# ---------------------------------------------------------------------------

def test_an_accepted_run_over_an_edited_source_does_not_claim_nothing_moved() -> None:
    """`unchanged` is a verdict, and it used to be printed as a description.

    Every acceptable comparison was headed "corpus unchanged", so a report
    listing a revision, a retirement and an addition announced that nothing had
    moved directly above them — the exact misreading `expected_change` and `gate`
    were added to prevent.
    """
    before = _report(_state())
    after = _report(_state(revision=2), documents={DOC_A: "rev-2", DOC_B: "rev-1"})

    result = compare(before, after)
    rendered = str(result)

    # Still acceptable: the document was edited, so the movement is expected.
    assert result.unchanged
    assert result.movements
    assert rendered.startswith("corpus changes accepted")
    assert "corpus unchanged" not in rendered


def test_a_comparison_where_nothing_moved_still_says_unchanged() -> None:
    # Reserved for exactly this: no violations *and* no movements.
    assert str(compare(_report(_state()), _report(_state()))).startswith("corpus unchanged")


def test_a_run_that_drifted_says_so_however_busy_the_source_was() -> None:
    before = _report(_state())
    after = _report(_state(variant="var-2"))

    assert str(compare(before, after)).startswith("CORPUS DRIFTED")
