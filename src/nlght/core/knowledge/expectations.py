# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""What a run was supposed to do, checked against what it did.

Two snapshots say what *moved*. They cannot say whether that was what anybody
wanted: "two revisions" is equally consistent with a threshold moving from 30
seconds to 60 and with it moving to 6000, and "one assertion retired" is as true
of a claim the source dropped as of one the model lost. So the corpus that
exercises the pipeline states its expectations, and they are checked.

**Values, not counts.** A case names the claim it is about by a phrase in its
observed wording — the source's own words, which are authoritative (ADR-0048) and
are therefore the one handle that does not move when the model rephrases its
fields. Then it says what must be true of it:

    same assertion, revision 1 -> 2, approved -> needs_review,
    wording no longer contains "30 seconds" and now contains "60 seconds"

Each of those is checkable, and a failure names which one broke rather than
reporting that some number is off by one.

An expectation that cannot be evaluated is a **failure**, never a pass. A case
whose claim is not in the baseline at all usually means the model did not extract
it, and reporting that as success is how a green run comes to mean nothing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from nlght.core.knowledge.report import AssertionState, LineageReport, ReportComparison

#: What a claim's identity did between the two pictures.
SAME = "same"
DIFFERENT = "different"
GONE = "gone"
NEW = "new"

_IDENTITIES = (SAME, DIFFERENT, GONE, NEW)


@dataclass(slots=True, frozen=True)
class ClaimExpectation:
    """One claim, named by its wording, and what must have become of it."""

    #: A phrase from the claim's observed wording. Not an `assertion_id`: ids are
    #: minted per run, so a file that named them could only ever be written after
    #: the run it describes.
    match: str
    #: `same` — the assertion continued. `different` — a new one opened and the
    #: matched claim is not it. `gone` — retired. `new` — absent before, present
    #: after.
    assertion: str = SAME
    revision: tuple[int, int] | None = None
    review_state: tuple[str, str] | None = None
    #: Substrings the wording must have lost and gained. This is where a value
    #: change is actually asserted: `30 seconds` out, `60 seconds` in.
    #:
    #: Each needs the side it is about. `text_loses` is a statement about the
    #: baseline wording — it was there, and it is not any more — so a `new` claim
    #: cannot carry one. `text_gains` is about the wording that arrived, so a
    #: `gone` claim cannot. Naming one anyway is a failure, not a skipped check.
    text_loses: tuple[str, ...] = ()
    text_gains: tuple[str, ...] = ()
    #: Whether the claim stayed in the slot it stood in. A comparison, so it
    #: needs both sides.
    slot: str = ""
    #: The kind this claim must be. A property of a single state rather than a
    #: comparison, so it holds for a retired claim as well as for a live one —
    #: which is the case that matters, since a kind change retires and recreates.
    kind: str = ""
    #: Documents that must carry **this same assertion**, not merely a claim that
    #: matches. Reinforcement is one claim seen twice, and the only thing that
    #: distinguishes it from two duplicates is that both sightings resolve to one
    #: `assertion_id`. A case saying "each file has a matching claim" is true of
    #: duplication too, and a live run passed that way while two documents held
    #: two separate assertions of the same sentence.
    #:
    #: Named by document, never by id: `assertion_id` is a surrogate minted per
    #: corpus, so a file pinning one could only be written after the run it
    #: describes. Each named document is selected with this same expectation —
    #: kind and phrase — and must arrive at the id the home document did.
    supported_by: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.assertion not in _IDENTITIES:
            raise ValueError(
                f"unknown assertion expectation '{self.assertion}'; "
                f"expected one of {', '.join(_IDENTITIES)}"
            )
        if not self.match.strip():
            raise ValueError("a claim expectation needs a phrase to find its claim by")


@dataclass(slots=True, frozen=True)
class CaseExpectation:
    """One file of the corpus, and the one thing it is there to demonstrate."""

    #: The file, as its source names it. Matched by suffix so a case can say
    #: `timeouts.adoc` without knowing the root the corpus was mounted at.
    document: str
    #: What this case exists to show, quoted in the result so a failure reads as
    #: a sentence rather than as a row.
    case: str = ""
    edited: bool | None = None
    assertions_added: int | None = None
    assertions_retired: int | None = None
    revisions_added: int | None = None
    slots_added: int | None = None
    slots_retired: int | None = None
    #: Every movement attributed to this document must carry these, which is how
    #: a case asserts *why* something was allowed rather than only that it was.
    because: str = ""
    gate: str = ""
    movements_expected: bool | None = None
    classification_drift: bool | None = None
    claims: tuple[ClaimExpectation, ...] = ()


@dataclass(slots=True, frozen=True)
class Failure:
    """One expectation that did not hold, in the words of what was expected."""

    document: str
    case: str
    detail: str

    def __str__(self) -> str:
        case = f" [{self.case}]" if self.case else ""
        return f"{self.document}{case}: {self.detail}"


@dataclass(slots=True, frozen=True)
class ExpectationOutcome:
    """Every case, and every way it did not hold."""

    checked: int = 0
    failures: tuple[Failure, ...] = ()
    #: Cases naming a file no snapshot contains. Their own list, because it means
    #: the corpus and the expectations have drifted apart — a different fault
    #: from the pipeline misbehaving, and one that would otherwise read as a
    #: cascade of unrelated failures.
    unknown_documents: tuple[str, ...] = ()

    @property
    def met(self) -> bool:
        return not self.failures and not self.unknown_documents

    def __str__(self) -> str:
        head = (
            f"expectations met ({self.checked} cases)"
            if self.met
            else f"EXPECTATIONS NOT MET ({len(self.failures)} of {self.checked} cases)"
        )
        lines = [head]
        lines += [f"  ! {failure}" for failure in self.failures]
        lines += [
            f"  ? {document}: named by an expectation and in neither snapshot"
            for document in self.unknown_documents
        ]
        return "\n".join(lines)


def _resolve(report: LineageReport, wanted: str) -> str:
    """The `document_id` a case's file name refers to.

    By suffix, so a case says `timeouts.adoc` and does not have to know whether
    the corpus was mounted at `/corpus`, `test-run-2/` or anywhere else.
    """
    for document_id, path in report.document_paths.items():
        if path == wanted or path.endswith(f"/{wanted}") or path.endswith(f"\\{wanted}"):
            return document_id
    return ""


def _carried_by(states: Sequence[AssertionState], document_id: str) -> list[AssertionState]:
    """The claims a document carries — all of them, not just the ones it saw last.

    `document_id` on a state is its most recent sighting, so a claim two
    documents assert appeared under one of them and was invisible to the other.
    An expectation on the quiet document then reported "in no baseline claim",
    which reads as the model having failed to extract it and is the opposite of
    what happened: the claim was there, once, carried by both.
    """
    return [
        state
        for state in states
        if document_id in state.document_ids or state.document_id == document_id
    ]


def _current(states: Sequence[AssertionState]) -> list[AssertionState]:
    """The deepest state of each variant, which is the one that holds now.

    A snapshot carries one row per revision, so a claim that gained a state
    appears twice. An expectation is about what the claim *is* after the run, and
    matching against every revision would report it as ambiguous the moment it
    was reworded — which is precisely the case these files exist to describe.
    """
    latest: dict[str, AssertionState] = {}
    for state in states:
        seen = latest.get(state.variant_id)
        if seen is None or state.revision > seen.revision:
            latest[state.variant_id] = state
    return list(latest.values())


def _by_match(states: Sequence[AssertionState], phrase: str) -> list[AssertionState]:
    needle = " ".join(phrase.lower().split())
    return [
        state
        for state in _current(states)
        if needle in " ".join(state.text.lower().split())
    ]


def _select(
    states: Sequence[AssertionState], expectation: ClaimExpectation
) -> tuple[list[AssertionState], str]:
    """The claim an expectation is about — by wording **and** kind.

    A phrase is not a claim identity. One sentence may state several assertions
    and they share their `observed_text` by design (ADR-0049), so a phrase from
    it names all of them:

        "Access tokens must not be logged and sessions expire after one hour."
            -> rule  (the prohibition)
            -> fact  (the expiry)

    Both carry that wording, and a checker that selected on the phrase alone
    reported the case as ambiguous — correctly, and uselessly, because the corpus
    had said which of the two it meant. So `kind` narrows the selection rather
    than being checked after one was already chosen.

    Where narrowing finds nothing and the phrase names exactly one claim of
    another kind, that is a kind mismatch and it is said plainly. Reporting "no
    such claim" there would hide the one thing that went wrong.

    `observed_text` stays the authority on the wording. This only says that one
    source may hold more than one atomic assertion.
    """
    hits = _by_match(states, expectation.match)
    if not expectation.kind:
        return hits, ""
    narrowed = [state for state in hits if state.kind == expectation.kind]
    if narrowed:
        return narrowed, ""
    if len(hits) == 1:
        return [], f"kind is {hits[0].kind!r}, expected {expectation.kind!r}"
    return [], ""


def _unevaluable(what: str, side: str, expectation: ClaimExpectation) -> str:
    """One expectation that had nothing to be checked against.

    A failure and never a pass, which is the rule this module opens with. It
    used to be neither: `new` and `gone` returned before the value checks ran, so
    a `new` claim could carry `review_state: [approved, needs_review]` — an
    assertion about a state that does not exist — and the run went green.
    """
    return (
        f"{what} cannot be evaluated: a '{expectation.assertion}' claim has no "
        f"{side} state, so this expectation asserts nothing"
    )


def _check_claim(  # noqa: PLR0912, PLR0915 - one branch per expectation, and each names itself
    expectation: ClaimExpectation,
    before: Sequence[AssertionState],
    after: Sequence[AssertionState],
) -> list[str]:
    """Every way this claim failed to become what it was supposed to.

    Each expectation is checked against the sides this claim actually has, and
    nothing is paired up to manufacture a side that is missing:

        new             old = none      new = the claim
        same/different  old = the claim new = the claim
        gone            old = the claim new = none

    Which side each key needs then follows mechanically. `text_gains` is about
    the wording that arrived, so it needs the new side. `text_loses` is about
    wording that was there and went, so it needs the old one. `revision` and
    `review_state` are pairs and need both. A key whose side is absent is
    reported as unevaluable rather than skipped.

    What is deliberately *not* done is guessing that a retired claim and a new
    one are two halves of the same change. Which new claim replaced which old one
    is a question about identity, and answering it by comparing text or
    fingerprints here would put a second, private notion of equivalence in the
    test checker.
    """
    was, was_mismatch = _select(before, expectation)
    now, now_mismatch = _select(after, expectation)
    surviving = [state for state in now if not state.retired]
    problems: list[str] = []

    # A kind that names no claim where the phrase names exactly one. Said here
    # rather than as "no such claim", which would hide the only thing wrong.
    if was_mismatch or now_mismatch:
        return [f"'{expectation.match}': {was_mismatch or now_mismatch}"]

    # Ambiguity next: with more than one candidate on either side, nothing that
    # follows would be about a claim anybody named. Kind has already narrowed the
    # selection, so reaching here means two claims of *one* kind share the
    # wording — which the expectation language cannot yet separate, and a
    # structural selector is what it would take.
    if len(was) > 1:
        return [
            f"'{expectation.match}' matches {len(was)} baseline claims of kind "
            f"{expectation.kind or 'any'}; the phrase has to name one"
        ]
    if len(surviving) > 1:
        return [f"'{expectation.match}' matches {len(surviving)} claims after the run"]

    earlier = was[0] if was else None
    later = surviving[0] if surviving else None

    # --- what the claim's identity was supposed to do -----------------------
    if expectation.assertion == NEW:
        if earlier is not None:
            problems.append(f"'{expectation.match}' was expected to be new and was already there")
        if later is None:
            # Nothing arrived, so nothing about it can be checked either.
            return [*problems, f"'{expectation.match}' was expected to appear and did not"]
    elif expectation.assertion == GONE:
        if earlier is None:
            # The usual cause is that the model did not extract the claim at all,
            # and calling that "met" is how a green run stops meaning anything.
            return [f"'{expectation.match}' is in no baseline claim, so nothing can be checked"]
        if later is not None:
            problems.append(f"'{expectation.match}' was expected to be retired and is still live")
    else:
        if earlier is None:
            return [f"'{expectation.match}' is in no baseline claim, so nothing can be checked"]
        if later is None:
            return [*problems, f"'{expectation.match}' disappeared instead of continuing"]
        if expectation.assertion == SAME and later.assertion_id != earlier.assertion_id:
            problems.append(
                f"'{expectation.match}' opened a new assertion "
                f"({earlier.assertion_id} -> {later.assertion_id}) instead of continuing"
            )
        if expectation.assertion == DIFFERENT and later.assertion_id == earlier.assertion_id:
            problems.append(
                f"'{expectation.match}' continued assertion {earlier.assertion_id} "
                f"where a different claim was expected"
            )

    # --- expectations that compare two states -------------------------------
    if expectation.revision is not None:
        if earlier is None or later is None:
            problems.append(_unevaluable("revision", "baseline" if earlier is None else "current",
                                         expectation))
        else:
            wanted = tuple(expectation.revision)
            actual = (earlier.revision, later.revision)
            if actual != wanted:
                problems.append(
                    f"revision {actual[0]} -> {actual[1]}, expected {wanted[0]} -> {wanted[1]}"
                )

    if expectation.review_state is not None:
        if earlier is None or later is None:
            problems.append(_unevaluable("review_state",
                                         "baseline" if earlier is None else "current", expectation))
        else:
            wanted_states = tuple(expectation.review_state)
            actual_states = (earlier.review_state, later.review_state)
            if actual_states != wanted_states:
                problems.append(
                    f"review {actual_states[0]} -> {actual_states[1]}, "
                    f"expected {wanted_states[0]} -> {wanted_states[1]}"
                )

    if expectation.slot == SAME:
        if earlier is None or later is None:
            problems.append(_unevaluable("slot", "baseline" if earlier is None else "current",
                                         expectation))
        elif later.slot_id != earlier.slot_id:
            problems.append(
                f"the claim moved slot ({earlier.slot_id} -> {later.slot_id}); "
                f"its section was expected to keep its identity"
            )

    # --- expectations about one wording -------------------------------------
    if expectation.text_gains and later is None:
        problems.append(_unevaluable("text_gains", "current", expectation))
    elif later is not None:
        for phrase in expectation.text_gains:
            if phrase.lower() not in later.text.lower():
                problems.append(f"the wording does not contain {phrase!r}: {later.text!r}")

    if expectation.text_loses:
        if earlier is None:
            problems.append(_unevaluable("text_loses", "baseline", expectation))
        else:
            for phrase in expectation.text_loses:
                # A loss is two statements — it was there, and it is not any
                # more — and only the first can be made when the claim itself is
                # gone. Checking absence alone let a phrase that was never in the
                # wording pass as successfully removed.
                if phrase.lower() not in earlier.text.lower():
                    problems.append(
                        f"the baseline wording never contained {phrase!r}, "
                        f"so nothing was lost: {earlier.text!r}"
                    )
                elif later is not None and phrase.lower() in later.text.lower():
                    problems.append(f"the wording still contains {phrase!r}")

    # --- expectations about the claim itself --------------------------------
    if expectation.kind:
        # A property of one state rather than a comparison, so it is read off
        # whichever side this claim has — the current one where there is one, and
        # the retired one for a claim that is gone. Requiring the current state
        # would make `kind` unpinnable on exactly the movements where kinds drift.
        subject = later if later is not None else earlier
        if subject is None:
            problems.append(_unevaluable("kind", "current", expectation))
        elif subject.kind != expectation.kind:
            problems.append(f"kind is {subject.kind!r}, expected {expectation.kind!r}")

    return problems


def _check_support(
    expectation: ClaimExpectation, current: LineageReport, home: str
) -> list[str]:
    """Whether the named documents carry this claim, or one that merely matches.

    Reinforcement is one claim seen twice. Two documents each holding a claim
    that matches the phrase is equally true of duplication, so the only thing
    that separates them is that both sightings resolve to the same
    `assertion_id` — and a live run passed a reinforcement case while the two
    documents held two separate assertions of one sentence, because nothing
    asked.

    Ids are read, never pinned: `assertion_id` is a surrogate minted per corpus,
    so the file says which documents must agree and the run says on what.
    """
    if not expectation.supported_by:
        return []

    here, _ = _select(_carried_by(current.states, home), expectation)
    if len(here) != 1:
        return [
            f"'{expectation.match}' names {len(here)} claims in its own document, "
            f"so there is nothing for {len(expectation.supported_by)} document(s) to agree with"
        ]
    wanted = here[0].assertion_id

    problems: list[str] = []
    for document in expectation.supported_by:
        document_id = _resolve(current, document)
        if not document_id:
            problems.append(f"'{expectation.match}' names {document}, which no snapshot contains")
            continue
        found, mismatch = _select(_carried_by(current.states, document_id), expectation)
        if mismatch:
            problems.append(f"'{expectation.match}' in {document}: {mismatch}")
        elif not found:
            problems.append(f"'{expectation.match}' is not in {document} at all")
        elif len(found) > 1:
            problems.append(f"'{expectation.match}' names {len(found)} claims in {document}")
        elif found[0].assertion_id != wanted:
            problems.append(
                f"{document} carries its own assertion for '{expectation.match}' "
                f"({found[0].assertion_id} rather than {wanted}); "
                f"that is duplication, not reinforcement"
            )
    return problems


def check_expectations(
    baseline: LineageReport,
    current: LineageReport,
    comparison: ReportComparison,
    cases: Sequence[CaseExpectation],
) -> ExpectationOutcome:
    """Hold the run against what the corpus said it should do."""
    failures: list[Failure] = []
    unknown: list[str] = []

    for case in cases:
        document_id = _resolve(current, case.document) or _resolve(baseline, case.document)
        if not document_id:
            unknown.append(case.document)
            continue

        def fail(detail: str, case: CaseExpectation = case) -> None:
            failures.append(Failure(document=case.document, case=case.case, detail=detail))

        before = _carried_by(baseline.states, document_id)
        after = _carried_by(current.states, document_id)
        moved = [move for move in comparison.movements if move.document_id == document_id]

        if case.edited is not None:
            edited = document_id in comparison.changed_documents
            if edited != case.edited:
                # Both halves, always. The `+` used to bind inside the else
                # branch, so a document that was edited and should not have been
                # reported "the document was edited" — a statement of fact that
                # never said what was wanted, and so did not read as a failure.
                actual = "the document was edited" if edited else "the document was not edited"
                wanted = "edited" if case.edited else "untouched"
                fail(f"{actual}, expected {wanted}")

        def _live(states: Sequence[AssertionState]) -> set[str]:
            return {state.assertion_id for state in states if not state.retired}

        if case.assertions_added is not None:
            added = len(_live(after) - {state.assertion_id for state in before})
            if added != case.assertions_added:
                fail(f"{added} assertion(s) added, expected {case.assertions_added}")
        if case.assertions_retired is not None:
            retired = len(
                {state.assertion_id for state in after if state.retired}
                - {state.assertion_id for state in before if state.retired}
            )
            if retired != case.assertions_retired:
                fail(f"{retired} assertion(s) retired, expected {case.assertions_retired}")
        if case.revisions_added is not None:
            added = len(after) - len(before)
            if added != case.revisions_added:
                fail(f"{added} revision(s) added, expected {case.revisions_added}")

        def _carrying(states: Sequence[AssertionState]) -> set[str]:
            """Slots that hold a live claim.

            Not "slots that appear": evidence is never deleted, so a slot whose
            section was removed keeps every sighting it ever had and goes on
            appearing in both pictures. Comparing appearances made
            `slots_retired` unable to fire at all — it reported 0 for a section
            that had genuinely gone, and the pipeline was right while the check
            was wrong.

            Historic evidence is history. Support is a live claim.
            """
            return {
                state.slot_id
                for state in states
                if state.slot_id and not state.retired
            }

        if case.slots_added is not None:
            added = len(_carrying(after) - _carrying(before))
            if added != case.slots_added:
                fail(f"{added} slot(s) began carrying a claim, expected {case.slots_added}")
        if case.slots_retired is not None:
            gone = len(_carrying(before) - _carrying(after))
            if gone != case.slots_retired:
                fail(f"{gone} slot(s) no longer carry a claim, expected {case.slots_retired}")

        if case.movements_expected is not None:
            unexpected = [move for move in moved if move.expected != case.movements_expected]
            if unexpected:
                fail(
                    f"{len(unexpected)} movement(s) with expected="
                    f"{not case.movements_expected}: "
                    + "; ".join(f"{m.kind} ({m.because})" for m in unexpected[:3])
                )
        if case.because:
            wrong = [move for move in moved if move.because != case.because]
            if wrong:
                fail(
                    f"movement(s) decided by {sorted({m.because for m in wrong})}, "
                    f"expected {case.because!r}"
                )
        if case.gate:
            wrong = [move for move in moved if move.gate != case.gate]
            if wrong:
                fail(
                    f"movement(s) judged at gate {sorted({m.gate for m in wrong})}, "
                    f"expected {case.gate!r}"
                )

        if case.classification_drift is not None:
            drifted = any(
                state.slot_id and state.slot_id in finding
                for finding in comparison.classification_drift
                for state in after
            )
            if drifted != case.classification_drift:
                fail(
                    "kinds were exchanged in one of its slots"
                    if drifted
                    else "no classification drift, and some was expected"
                )

        for claim in case.claims:
            for problem in _check_claim(claim, before, after):
                fail(problem)
            for problem in _check_support(claim, current, document_id):
                fail(problem)

    return ExpectationOutcome(
        checked=len(cases), failures=tuple(failures), unknown_documents=tuple(unknown)
    )


def parse_expectations(document: Mapping[str, Any]) -> tuple[CaseExpectation, ...]:
    """Read the expectations from a plain mapping — a YAML or JSON file.

    Unknown keys are refused rather than ignored. A misspelled expectation that
    is silently dropped turns into a case that passes without checking anything,
    which is the failure mode this whole module exists to remove.
    """
    cases: list[CaseExpectation] = []
    for raw in document.get("cases") or ():
        claims = tuple(
            ClaimExpectation(
                match=str(claim["match"]),
                assertion=str(claim.get("assertion", SAME)),
                revision=_pair_of_int(claim.get("revision")),
                review_state=_pair_of_str(claim.get("review_state")),
                text_loses=_phrases(claim.get("text_loses")),
                text_gains=_phrases(claim.get("text_gains")),
                slot=str(claim.get("slot", "")),
                kind=str(claim.get("kind", "")),
                supported_by=_phrases(claim.get("supported_by")),
            )
            for claim in raw.get("claims") or ()
        )
        for claim in raw.get("claims") or ():
            unknown_claim = set(claim) - {
                "match", "assertion", "revision", "review_state", "text_loses",
                "text_gains", "slot", "kind", "supported_by",
            }
            if unknown_claim:
                raise ValueError(
                    f"unknown claim key(s) for '{raw.get('document', '?')}': "
                    f"{', '.join(sorted(unknown_claim))}"
                )
        known = {
            "document", "case", "edited", "assertions_added", "assertions_retired",
            "revisions_added", "slots_added", "slots_retired", "because", "gate",
            "movements_expected", "classification_drift", "claims",
        }
        unknown = set(raw) - known
        if unknown:
            raise ValueError(
                f"unknown expectation key(s) for '{raw.get('document', '?')}': "
                f"{', '.join(sorted(unknown))}"
            )
        cases.append(
            CaseExpectation(
                document=str(raw["document"]),
                case=str(raw.get("case", "")),
                edited=raw.get("edited"),
                assertions_added=raw.get("assertions_added"),
                assertions_retired=raw.get("assertions_retired"),
                revisions_added=raw.get("revisions_added"),
                slots_added=raw.get("slots_added"),
                slots_retired=raw.get("slots_retired"),
                because=str(raw.get("because", "")),
                gate=str(raw.get("gate", "")),
                movements_expected=raw.get("movements_expected"),
                classification_drift=raw.get("classification_drift"),
                claims=claims,
            )
        )
    return tuple(cases)


def _pair_of_int(value: Any) -> tuple[int, int] | None:  # noqa: ANN401
    if value is None:
        return None
    before, after = value
    return int(before), int(after)


def _pair_of_str(value: Any) -> tuple[str, str] | None:  # noqa: ANN401
    if value is None:
        return None
    before, after = value
    return str(before), str(after)


def _phrases(value: Any) -> tuple[str, ...]:  # noqa: ANN401
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)
