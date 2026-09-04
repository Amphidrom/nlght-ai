# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""A run is held to what the corpus said it should do — by value, not by count.

The gap this closes: two snapshots say what moved, and "one revision added" is
equally true of a threshold moving from 30 seconds to 60 and of it moving to
6000. Only the wording separates them, so the wording is what a case asserts.
"""

from __future__ import annotations

import pytest

from nlght.core.knowledge import AssertionState, LineageReport, compare
from nlght.core.knowledge.expectations import (
    CaseExpectation,
    ClaimExpectation,
    check_expectations,
    parse_expectations,
)

DOC = "doc-1"
PATH = "corpus/timeouts.adoc"


def _state(**overrides: object) -> AssertionState:
    values: dict[str, object] = {
        "assertion_id": "a_1", "entity_key": "k1", "variant_id": "v1",
        "revision": 1, "review_state": "approved", "document_id": DOC,
        "slot_id": "slot-1", "retired": False, "kind": "rule",
        "text": "The grace period must not exceed 30 seconds.",
    }
    values.update(overrides)
    return AssertionState(**values)  # type: ignore[arg-type]


def _report(*states: AssertionState, revision: str = "rev-1", **extra: object) -> LineageReport:
    values: dict[str, object] = {
        "states": [state.as_dict() for state in states],
        "documents": {DOC: revision},
        "document_paths": {DOC: PATH},
        "slot_inputs": {"slot-1": f"fp-{revision}"},
    }
    values.update(extra)
    return LineageReport.from_dict(values)


def _run(before: LineageReport, after: LineageReport, case: CaseExpectation):  # noqa: ANN201
    return check_expectations(before, after, compare(before, after), [case])


def _raised(**overrides: object) -> tuple[AssertionState, ...]:
    """The snapshot after the grace period was raised — one row per revision.

    Both rows, because that is what `lineage_report` emits: a variant with two
    states has two of them, and the comparison takes the deepest. A fixture with
    only the latest would make `revisions_added` unmeasurable and quietly weaken
    every case built on it.
    """
    values: dict[str, object] = {
        "revision": 2, "review_state": "needs_review",
        "text": "The grace period must not exceed 60 seconds.",
    }
    values.update(overrides)
    return (_state(), _state(**values))


def _moved() -> tuple[LineageReport, LineageReport]:
    """The corpus before and after the grace period was raised to 60 seconds."""
    return _report(_state()), _report(*_raised(), revision="rev-2")


def test_the_value_change_that_was_wanted_is_met() -> None:
    before, after = _moved()

    outcome = _run(before, after, CaseExpectation(
        document="timeouts.adoc",
        case="a normative change",
        revisions_added=1,
        claims=(ClaimExpectation(
            match="grace period",
            revision=(1, 2),
            review_state=("approved", "needs_review"),
            text_loses=("30 seconds",),
            text_gains=("60 seconds",),
        ),),
    ))

    assert outcome.met, str(outcome)
    assert outcome.checked == 1


def test_a_different_value_is_caught_though_every_count_agrees() -> None:
    """The whole point.

    One revision, same assertion, approval spent — every number is what the case
    asked for. The threshold went to 6000, and only the wording says so.
    """
    before = _report(_state())
    after = _report(
        *_raised(text="The grace period must not exceed 6000 seconds."), revision="rev-2"
    )

    outcome = _run(before, after, CaseExpectation(
        document="timeouts.adoc",
        revisions_added=1,
        claims=(ClaimExpectation(
            match="grace period",
            revision=(1, 2),
            review_state=("approved", "needs_review"),
            text_gains=("60 seconds",),
        ),),
    ))

    assert not outcome.met
    assert "60 seconds" in str(outcome)


def test_an_approval_that_survived_a_moved_quantity_is_caught() -> None:
    # The failure that matters most: the same claim, the new value, and the old
    # approval still standing.
    before = _report(_state())
    after = _report(
        *_raised(review_state="approved"), revision="rev-2"
    )

    outcome = _run(before, after, CaseExpectation(
        document="timeouts.adoc",
        claims=(ClaimExpectation(
            match="grace period", review_state=("approved", "needs_review")
        ),),
    ))

    assert not outcome.met
    assert "review approved -> approved" in str(outcome)


def test_a_claim_that_opened_a_new_assertion_is_caught() -> None:
    before = _report(_state())
    after = _report(
        _state(assertion_id="a_2", variant_id="v2",
               text="The grace period must not exceed 60 seconds."),
        revision="rev-2",
    )

    outcome = _run(before, after, CaseExpectation(
        document="timeouts.adoc",
        claims=(ClaimExpectation(match="grace period", assertion="same"),),
    ))

    assert not outcome.met
    assert "instead of continuing" in str(outcome)


def test_a_claim_that_moved_slot_is_caught() -> None:
    before = _report(_state())
    after = _report(_state(), _state(revision=2, slot_id="slot-9"), revision="rev-2")

    outcome = _run(before, after, CaseExpectation(
        document="timeouts.adoc",
        claims=(ClaimExpectation(match="grace period", slot="same"),),
    ))

    assert not outcome.met
    assert "moved slot" in str(outcome)


def test_a_claim_the_model_never_extracted_fails_rather_than_passes() -> None:
    """An expectation that cannot be evaluated is not met.

    The usual cause is that the model produced nothing for that sentence, and
    reporting it as success is how a green run comes to mean nothing.
    """
    before = _report(_state(text="Something else entirely."))
    after = _report(_state(text="Something else entirely."), revision="rev-2")

    outcome = _run(before, after, CaseExpectation(
        document="timeouts.adoc",
        claims=(ClaimExpectation(match="grace period"),),
    ))

    assert not outcome.met
    assert "nothing can be checked" in str(outcome)


def test_a_case_naming_a_file_nothing_knows_is_reported_apart() -> None:
    # The corpus and the expectations having drifted apart is a different fault
    # from the pipeline misbehaving, and would otherwise read as a cascade.
    before, after = _moved()

    outcome = check_expectations(
        before, after, compare(before, after),
        [CaseExpectation(document="nowhere.adoc")],
    )

    assert not outcome.met
    assert outcome.unknown_documents == ("nowhere.adoc",)
    assert outcome.failures == ()


def test_the_untouched_control_is_checked_for_stillness() -> None:
    before = _report(_state())
    after = _report(_state())

    outcome = _run(before, after, CaseExpectation(
        document="timeouts.adoc",
        edited=False,
        revisions_added=0,
        assertions_added=0,
        claims=(ClaimExpectation(match="grace period", revision=(1, 1)),),
    ))

    assert outcome.met, str(outcome)


# ---------------------------------------------------------------------------
# The file the corpus is actually held to
# ---------------------------------------------------------------------------

def test_a_misspelled_expectation_is_refused_rather_than_ignored() -> None:
    """A silently dropped key is a case that passes without checking anything."""
    with pytest.raises(ValueError, match="unknown expectation key"):
        parse_expectations({"cases": [{"document": "a.adoc", "revisons_added": 1}]})


def _corpus_cases() -> tuple[CaseExpectation, ...]:
    """The shipped expectation file, parsed.

    Skipped rather than failed where the corpus is not checked out: it is test
    data and may live elsewhere, and a missing fixture is not a defect in the
    checker.
    """
    import pathlib  # noqa: PLC0415

    import yaml  # noqa: PLC0415

    path = (
        pathlib.Path(__file__).resolve().parents[3]
        / "test-data" / "spring-boot-docs" / "test-run-2.expected.yaml"
    )
    if not path.exists():  # pragma: no cover - the corpus is not in the repository
        pytest.skip("the corpus lives outside the repository")
    return parse_expectations(yaml.safe_load(path.read_text(encoding="utf-8")))


@pytest.mark.external_corpus
def test_the_corpus_expectations_parse_and_say_something() -> None:
    """The shipped file is exercised, so a typo in it fails here and not live."""
    cases = _corpus_cases()

    assert len(cases) >= 18
    assert all(case.case for case in cases), "every case says what it demonstrates"
    # The one that would be worthless if it only counted.
    [timeouts] = [case for case in cases if case.document == "timeouts.adoc"]
    [claim] = timeouts.claims
    assert claim.text_loses == ("30 seconds",)
    assert claim.text_gains == ("60 seconds",)
    assert claim.review_state == ("approved", "needs_review")


@pytest.mark.external_corpus
def test_every_claim_the_corpus_names_pins_its_kind() -> None:
    """`kind` is part of the identity namespace, so an unpinned case is blind.

    A sentence classified `rule` on one pass and `decision` on the next retires
    and recreates the claim whether or not anything about it moved, and the slot
    gate cannot catch it — the section may genuinely have been edited at the same
    time. Only the expectation can say which kind was meant.
    """
    cases = _corpus_cases()

    unpinned = [
        f"{case.document}: {claim.match}"
        for case in cases
        for claim in case.claims
        if not claim.kind
    ]

    assert unpinned == []


@pytest.mark.external_corpus
def test_the_corpus_exercises_all_four_kinds() -> None:
    """Facts and rules were the whole live corpus; two kinds went untested.

    The unit tests parametrise over all four, so the code paths were covered
    while the pipeline that actually runs against a model never produced a
    `decision` or a `pattern` on purpose. Anything true only of the two kinds it
    did produce would have gone unnoticed.
    """
    kinds = {claim.kind for case in _corpus_cases() for claim in case.claims}

    assert kinds == {"fact", "rule", "decision", "pattern"}


def test_a_slot_that_stopped_carrying_a_claim_is_counted() -> None:
    """Evidence is never deleted, so appearance is not support.

    A section that was removed keeps every sighting it ever had, so its slot goes
    on appearing in both pictures. Counting appearances made `slots_retired`
    unable to fire at all: a live run reported 0 for a section that had genuinely
    gone, and the pipeline was right while this check was wrong.
    """
    before = _report(_state(slot_id="s1"), _state(variant_id="v2", slot_id="s2"))
    after = _report(
        _state(slot_id="s1"),
        # The removed section's claim: retired, and its evidence — and therefore
        # its slot — still in the picture.
        _state(variant_id="v2", slot_id="s2", retired=True),
        revision="rev-2",
    )

    outcome = _run(before, after, CaseExpectation(
        document="timeouts.adoc", case="a section removed", slots_retired=1,
    ))

    assert outcome.met, str(outcome)


def test_a_slot_that_still_holds_something_is_not_counted_as_gone() -> None:
    # One claim of two retired leaves the slot carrying knowledge, so the
    # section did not go anywhere.
    before = _report(_state(slot_id="s1"), _state(variant_id="v2", slot_id="s1"))
    after = _report(
        _state(slot_id="s1"),
        _state(variant_id="v2", slot_id="s1", retired=True),
        revision="rev-2",
    )

    outcome = _run(before, after, CaseExpectation(
        document="timeouts.adoc", slots_retired=0,
    ))

    assert outcome.met, str(outcome)


# ---------------------------------------------------------------------------
# A claim that arrives or leaves is checked against the side it has
#
# `new` and `gone` used to return before the value checks ran, so every
# expectation hung on one of them was dropped without a word: a `new` claim could
# assert `review_state: [approved, needs_review]` — a statement about a state
# that does not exist — and the run went green. That is the failure this whole
# module was written to remove, sitting inside the module itself.
#
# Which side each key needs follows from what it says, and nothing is paired up
# to invent a side that is missing: no attempt is made to work out that a retired
# claim and a new one are two halves of one change. That question is about
# identity, and answering it here by comparing text would put a second, private
# notion of equivalence in the test checker.
# ---------------------------------------------------------------------------

def _replaced() -> tuple[LineageReport, LineageReport]:
    """A structural change: the old claim retires and a different one arrives.

    What a fact's change looks like whenever the moved field takes part in the
    entity key — and what a kind change looks like for any kind at all.
    """
    before = _report(_state(text="Spring Boot requires Java 17."))
    after = _report(
        _state(text="Spring Boot requires Java 17.", retired=True),
        _state(assertion_id="a_2", variant_id="v2", entity_key="k2",
               text="Spring Boot requires Java 21."),
        revision="rev-2",
    )
    return before, after


def test_a_new_claim_cannot_assert_a_state_it_never_had() -> None:
    """The case the user named: `new` + `review_state` must be red.

    There is no earlier state to compare against, so the expectation asserts
    nothing — and an expectation that cannot be evaluated is a failure.
    """
    before, after = _replaced()

    outcome = _run(before, after, CaseExpectation(
        document="timeouts.adoc",
        claims=(ClaimExpectation(
            match="Java 21", assertion="new", review_state=("approved", "needs_review"),
        ),),
    ))

    assert not outcome.met
    assert "review_state cannot be evaluated" in str(outcome)
    assert "has no baseline state" in str(outcome)


def test_a_new_claim_is_held_to_the_wording_that_arrived() -> None:
    before, after = _replaced()

    wrong = _run(before, after, CaseExpectation(
        document="timeouts.adoc",
        claims=(ClaimExpectation(match="Java 21", assertion="new", text_gains=("Java 99",)),),
    ))
    right = _run(before, after, CaseExpectation(
        document="timeouts.adoc",
        claims=(ClaimExpectation(
            match="Java 21", assertion="new", text_gains=("Java 21",), kind="rule",
        ),),
    ))

    assert not wrong.met, "a false wording assertion on a new claim used to pass"
    assert "does not contain 'Java 99'" in str(wrong)
    assert right.met, str(right)


def test_a_new_claim_cannot_assert_what_a_wording_lost() -> None:
    # A loss is a statement about the baseline, and a claim that was not there
    # has no baseline wording to have lost anything from.
    before, after = _replaced()

    outcome = _run(before, after, CaseExpectation(
        document="timeouts.adoc",
        claims=(ClaimExpectation(match="Java 21", assertion="new", text_loses=("Java 17",)),),
    ))

    assert not outcome.met
    assert "text_loses cannot be evaluated" in str(outcome)


def test_a_retired_claim_is_held_to_what_it_said_before_it_went() -> None:
    """`gone` + `text_loses`, checked against the last state before retirement.

    This is how the other half of a structural change is asserted: the claim that
    left is named by what it used to say, without anything deciding that it and
    the new claim are two halves of one movement.
    """
    before, after = _replaced()

    right = _run(before, after, CaseExpectation(
        document="timeouts.adoc",
        claims=(ClaimExpectation(
            match="Java 17", assertion="gone", text_loses=("Java 17",), kind="rule",
        ),),
    ))
    wrong = _run(before, after, CaseExpectation(
        document="timeouts.adoc",
        claims=(ClaimExpectation(
            match="Java 17", assertion="gone", text_loses=("Java 99",),
        ),),
    ))

    assert right.met, str(right)
    assert not wrong.met, "a phrase the claim never carried used to pass as lost"
    assert "never contained 'Java 99'" in str(wrong)


def test_a_retired_claim_cannot_assert_a_wording_that_arrived() -> None:
    before, after = _replaced()

    outcome = _run(before, after, CaseExpectation(
        document="timeouts.adoc",
        claims=(ClaimExpectation(match="Java 17", assertion="gone", text_gains=("Java 21",)),),
    ))

    assert not outcome.met
    assert "text_gains cannot be evaluated" in str(outcome)
    assert "has no current state" in str(outcome)


def test_a_retired_claim_cannot_assert_a_revision_pair() -> None:
    before, after = _replaced()

    outcome = _run(before, after, CaseExpectation(
        document="timeouts.adoc",
        claims=(ClaimExpectation(match="Java 17", assertion="gone", revision=(1, 2)),),
    ))

    assert not outcome.met
    assert "revision cannot be evaluated" in str(outcome)


def test_the_kind_of_a_claim_that_left_is_still_checkable() -> None:
    """A property of one state, not a comparison — so `gone` can pin it too.

    Reading `kind` off the current state would make it unpinnable on exactly the
    movements where kinds drift, since a kind change retires the claim and
    creates another.
    """
    before = _report(_state(kind="rule", text="Applications must shut down gracefully."))
    after = _report(
        _state(kind="rule", text="Applications must shut down gracefully.", retired=True),
        revision="rev-2",
    )

    right = _run(before, after, CaseExpectation(
        document="timeouts.adoc",
        claims=(ClaimExpectation(match="shut down", assertion="gone", kind="rule"),),
    ))
    wrong = _run(before, after, CaseExpectation(
        document="timeouts.adoc",
        claims=(ClaimExpectation(match="shut down", assertion="gone", kind="decision"),),
    ))

    assert right.met, str(right)
    assert not wrong.met
    assert "kind is 'rule', expected 'decision'" in str(wrong)


def test_a_loss_is_two_statements_and_both_are_checked() -> None:
    """It was there, and it is not any more.

    Absence alone let a phrase that had never been in the wording pass as
    successfully removed — a vacuous expectation reading as a met one.
    """
    before, after = _moved()

    outcome = _run(before, after, CaseExpectation(
        document="timeouts.adoc",
        claims=(ClaimExpectation(match="grace period", text_loses=("90 seconds",)),),
    ))

    assert not outcome.met
    assert "never contained '90 seconds'" in str(outcome)


def test_both_halves_of_a_structural_change_are_asserted_independently() -> None:
    # Two expectations over one movement, neither of which knows about the other.
    before, after = _replaced()

    outcome = _run(before, after, CaseExpectation(
        document="timeouts.adoc",
        case="a structural change — the claim that left, and the one that came",
        assertions_added=1,
        assertions_retired=1,
        claims=(
            ClaimExpectation(match="Java 17", assertion="gone",
                             text_loses=("Java 17",), kind="rule"),
            ClaimExpectation(match="Java 21", assertion="new",
                             text_gains=("Java 21",), kind="rule"),
        ),
    ))

    assert outcome.met, str(outcome)


# ---------------------------------------------------------------------------
# A phrase is not a claim identity
#
# One sentence may state several assertions, and they share their `observed_text`
# by design (ADR-0049). A live run proved it: `Access tokens must not be logged
# and sessions expire after one hour.` produced a `rule` and a `fact`, both with
# that wording, and the checker reported the case as ambiguous — correctly, and
# uselessly, since the corpus had said which of the two each expectation meant.
# ---------------------------------------------------------------------------

def _one_sentence_two_claims() -> tuple[LineageReport, LineageReport]:
    sentence = "Access tokens must not be logged and sessions expire after one hour."
    states = (
        _state(assertion_id="a_1", variant_id="v1", kind="rule", text=sentence),
        _state(assertion_id="a_2", variant_id="v2", kind="fact", text=sentence),
    )
    return _report(*states), _report(*states)


def test_kind_narrows_the_selection_rather_than_judging_after_it() -> None:
    before, after = _one_sentence_two_claims()

    outcome = _run(before, after, CaseExpectation(
        document="timeouts.adoc",
        case="one sentence, two claims, two kinds",
        claims=(
            ClaimExpectation(match="Access tokens", kind="rule", revision=(1, 1)),
            ClaimExpectation(match="Access tokens", kind="fact", revision=(1, 1)),
        ),
    ))

    assert outcome.met, str(outcome)


def test_a_phrase_naming_two_claims_of_one_kind_still_has_to_name_one() -> None:
    # Kind has already narrowed it, so reaching here means the language cannot
    # separate them and a structural selector is what it would take.
    sentence = "Two things are true of the cache and of the queue."
    both = (
        _state(assertion_id="a_1", variant_id="v1", kind="fact", text=sentence),
        _state(assertion_id="a_2", variant_id="v2", kind="fact", text=sentence),
    )
    before, after = _report(*both), _report(*both)

    outcome = _run(before, after, CaseExpectation(
        document="timeouts.adoc",
        claims=(ClaimExpectation(match="Two things are true", kind="fact"),),
    ))

    assert not outcome.met
    assert "matches 2 baseline claims of kind fact" in str(outcome)


def test_a_claim_of_the_wrong_kind_says_so_rather_than_going_missing() -> None:
    """Narrowing must not turn a wrong kind into "no such claim".

    The live corpus pins `fact` on a sentence run 1 read as a `rule`; that has to
    read as the mismatch it is, or the one finding worth having disappears into a
    message about the corpus and the expectations having drifted apart.
    """
    before = _report(_state(kind="rule", text="The management server binds to the same port."))
    after = _report(_state(kind="rule", text="The management server binds to the same port."))

    outcome = _run(before, after, CaseExpectation(
        document="timeouts.adoc",
        claims=(ClaimExpectation(match="binds to the same port", kind="fact"),),
    ))

    assert not outcome.met
    assert "kind is 'rule', expected 'fact'" in str(outcome)


def test_a_claim_named_by_wording_that_moved_is_the_checkers_problem_to_avoid() -> None:
    """The aot case, in miniature.

    The claim continued — one assertion, revision 1 to 2, approval carried — and
    the checker reported it as disappeared, because the phrase naming it was in
    the old wording and not in the new one. Nothing is inferred about which
    claim replaced which; the expectation simply has to name the claim by
    something that survives the edit.
    """
    before = _report(_state(kind="rule", text="You have to set up the environment when building."))
    after = _report(
        _state(kind="rule", text="You have to set up the environment when building."),
        _state(kind="rule", revision=2, text="The environment must be set up at build time."),
        revision="rev-2",
    )

    moved = _run(before, after, CaseExpectation(
        document="timeouts.adoc",
        claims=(ClaimExpectation(match="set up the environment when", kind="rule"),),
    ))
    stable = _run(before, after, CaseExpectation(
        document="timeouts.adoc",
        claims=(ClaimExpectation(
            match="environment", kind="rule", revision=(1, 2),
            review_state=("approved", "approved"),
        ),),
    ))

    assert not moved.met
    assert "disappeared instead of continuing" in str(moved)
    assert stable.met, str(stable)


# ---------------------------------------------------------------------------
# Reinforcement is one claim seen twice
#
# Two documents each holding a claim that matches a phrase is equally true of
# duplication. A live run passed a reinforcement case that way, while the two
# documents held two separate assertions of one sentence, because nothing asked
# whether they were the same assertion.
# ---------------------------------------------------------------------------

DOC_B = "doc-2"
PATH_B = "corpus/management-b.adoc"


def _two_documents(*, shared: bool) -> LineageReport:
    """One claim in two files — as one assertion, or as two.

    Shared means **one row**, carried by both documents. That is what the report
    emits: a state is one per revision, and the documents that sighted it travel
    with it. A fixture with two rows would describe duplication whatever the ids
    said.
    """
    sentence = "The management server uses the application's port."
    if shared:
        states = [_state(
            assertion_id="a_1", variant_id="v1", kind="fact", text=sentence,
            document_id=DOC_B, document_ids=(DOC, DOC_B),
        ).as_dict()]
    else:
        states = [
            _state(assertion_id="a_1", variant_id="v1", kind="fact", text=sentence,
                   document_ids=(DOC,)).as_dict(),
            _state(assertion_id="a_2", variant_id="v2", kind="fact", text=sentence,
                   document_id=DOC_B, document_ids=(DOC_B,)).as_dict(),
        ]
    return LineageReport.from_dict({
        "states": states,
        "documents": {DOC: "rev-1", DOC_B: "rev-1"},
        "document_paths": {DOC: PATH, DOC_B: PATH_B},
    })


def _support_case() -> CaseExpectation:
    return CaseExpectation(
        document="timeouts.adoc",
        case="one claim, two documents",
        claims=(ClaimExpectation(
            match="uses the application's port", kind="fact",
            supported_by=("timeouts.adoc", "management-b.adoc"),
        ),),
    )


def test_a_claim_two_documents_carry_is_visible_from_both() -> None:
    """The report keeps the latest sighting, and that is not the whole truth.

    A claim asserted by two documents appeared under one of them, so an
    expectation on the other reported "in no baseline claim" — which reads as the
    model having failed to extract it, and is the opposite of what happened.
    """
    report = _two_documents(shared=True)

    outcome = _run(report, report, CaseExpectation(
        document="management-b.adoc",
        claims=(ClaimExpectation(match="uses the application's port", kind="fact"),),
    ))
    from_the_quiet_one = check_expectations(
        report, report, compare(report, report),
        [CaseExpectation(
            document="timeouts.adoc",
            claims=(ClaimExpectation(match="uses the application's port", kind="fact"),),
        )],
    )

    assert outcome.met, str(outcome)
    assert from_the_quiet_one.met, str(from_the_quiet_one)


def test_two_documents_carrying_one_assertion_is_reinforcement() -> None:
    report = _two_documents(shared=True)

    outcome = check_expectations(report, report, compare(report, report), [_support_case()])

    assert outcome.met, str(outcome)


def test_two_documents_carrying_their_own_assertions_is_duplication_and_says_so() -> None:
    """The case that used to pass while not holding.

    Each document has a claim matching the phrase and each continued, so every
    per-document expectation was satisfied — and there were two assertions.
    """
    report = _two_documents(shared=False)

    outcome = check_expectations(report, report, compare(report, report), [_support_case()])

    assert not outcome.met
    assert "that is duplication, not reinforcement" in str(outcome)


def test_a_supporting_document_that_does_not_carry_the_claim_at_all_is_named() -> None:
    report = LineageReport.from_dict({
        "states": [_state(
            kind="fact", document_ids=(DOC,),
            text="The management server uses the application's port.",
        ).as_dict()],
        "documents": {DOC: "rev-1", DOC_B: "rev-1"},
        "document_paths": {DOC: PATH, DOC_B: PATH_B},
    })

    outcome = check_expectations(report, report, compare(report, report), [_support_case()])

    assert not outcome.met
    assert "is not in management-b.adoc at all" in str(outcome)


def test_support_names_documents_and_never_an_assertion_id() -> None:
    # The id is a surrogate minted per corpus, so a file pinning one could only
    # be written after the run it describes.
    assert "assertion_id" not in {
        field for field in ClaimExpectation.__dataclass_fields__
    }
    with pytest.raises(ValueError, match="unknown claim key"):
        parse_expectations({"cases": [{"document": "a.adoc",
                                       "claims": [{"match": "x", "assertion_id": "a_1"}]}]})


def test_a_failed_edit_expectation_says_both_what_happened_and_what_was_wanted() -> None:
    # The `+` bound inside the else branch, so this direction lost its second
    # half and read as a statement of fact rather than as a failure.
    before, after = _moved()

    outcome = _run(before, after, CaseExpectation(document="timeouts.adoc", edited=False))

    assert not outcome.met
    assert "the document was edited, expected untouched" in str(outcome)
