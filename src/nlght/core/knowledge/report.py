# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""What the corpus looks like, so a second run can be held against the first.

The acceptance metric of the incremental design is a comparison, not a number:
two runs over an unchanged source must not change the corpus — no retracted
assertion, no new assertion, no lost approval. That can only be checked by
taking a picture before the second run and holding the second picture against
it.

One distinction runs through everything here. **Evidence is expected to grow**:
a second run is a second sighting, and recording that document D still asserts
something is exactly what licenses not retracting it later. Everything else is
expected to stand still. A report that treated the two alike would either raise
a false alarm on every run or miss the drift it exists to catch.

`state_digest` is the single number that answers the metric. It folds every
assertion's key, case and state into one value, so an unchanged digest means
literally nothing moved — and a changed one means something did.

**A changed corpus is not the same as a drifting one**, and for a long time this
could not tell them apart. It reported "a variant gained a revision over an
unchanged source" on runs where the source had very much been edited, which made
every deliberate change look like a failure and trained the eye to skip the line.

So each assertion is now recorded with where it stands, and each document with
the revision it was last seen at. A document whose revision moved is a document
somebody edited — nothing has to be told which ones those were — and movement
inside it is expected. Movement anywhere else is drift, which is the only thing
worth an alarm.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from nlght.core.ingestion.document import stable_digest
from nlght.core.knowledge.lineage import APPROVED


@dataclass(slots=True, frozen=True)
class AssertionState:
    """One claim, where it stands and what state it is in."""

    assertion_id: str
    entity_key: str
    variant_id: str
    revision: int
    review_state: str
    #: Where it was last seen. Movement is only readable against provenance:
    #: "a revision appeared" is a fact about the corpus, and "a revision appeared
    #: in a document nobody edited" is a fact about the pipeline.
    document_id: str = ""
    #: Every document this state was sighted in. A claim two documents carry is
    #: one claim, and `document_id` keeps only the latest sighting — so a report
    #: filtered by it showed such a claim under one document and hid it from the
    #: other, which is exactly the case a reinforcement expectation is about.
    document_ids: tuple[str, ...] = ()
    slot_id: str = ""
    retired: bool = False
    #: Which kind of claim this is. Part of the identity namespace, so a claim
    #: the model called a `rule` on one pass and a `decision` on the next is two
    #: assertions however identical its wording — which is a distinct failure
    #: from the wording drift the surrogate id removed, and needs to be visible
    #: as one.
    kind: str = ""
    #: What this state says, in the source's own words (ADR-0048). Carried in the
    #: snapshot because counts cannot answer the question anyone actually asks
    #: after a run: *did the value change the way it was supposed to*. "Two
    #: revisions" is compatible with the threshold moving to 60 and with it
    #: moving to 6000; only the wording tells them apart.
    text: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "assertion_id": self.assertion_id,
            "entity_key": self.entity_key,
            "variant_id": self.variant_id,
            "revision": self.revision,
            "review_state": self.review_state,
            "document_id": self.document_id,
            "document_ids": list(self.document_ids),
            "slot_id": self.slot_id,
            "retired": self.retired,
            "kind": self.kind,
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> AssertionState:
        return cls(
            assertion_id=str(values.get("assertion_id", "")),
            entity_key=str(values.get("entity_key", "")),
            variant_id=str(values.get("variant_id", "")),
            revision=int(values.get("revision", 0) or 0),
            review_state=str(values.get("review_state", "")),
            document_id=str(values.get("document_id", "")),
            document_ids=tuple(str(item) for item in values.get("document_ids") or ()),
            slot_id=str(values.get("slot_id", "")),
            retired=bool(values.get("retired", False)),
            kind=str(values.get("kind", "")),
            text=str(values.get("text", "")),
        )


@dataclass(slots=True, frozen=True)
class LineageReport:
    """One picture of the corpus, taken between runs."""

    assertions: int = 0
    variants: int = 0
    revisions: int = 0
    evidence: int = 0

    #: Variants that have more than one state. After a run over an unchanged
    #: source this is the drift the design exists to remove: the model was asked
    #: again and answered differently, so a claim nobody edited moved.
    variants_beyond_first_revision: int = 0
    #: The deepest history in the corpus, which says how far any one claim has
    #: been pushed around.
    deepest_revision: int = 0

    #: Assertions in the graph with no lineage beside them. Every one is a
    #: candidate that could not name a business question — a rule extracted
    #: before `subject` and `rule_property` were asked for, or one the model
    #: left them out of, or evidence with no document. Broken down by kind,
    #: because which kinds fail says whether the prompt or the schema is at
    #: fault.
    graph_assertions: int = 0
    graph_without_lineage: int = 0
    graph_without_lineage_by_kind: Mapping[str, int] = field(default_factory=dict)
    #: Nodes migration `0018` could not point at a revision, though lineage does
    #: describe them. They are deliberately **not** canonical — withheld rather
    #: than guessed at — so without a number here nothing says whether that is
    #: three rows or thirty thousand, and an operator cannot tell a migrated
    #: corpus that is fine from one that is silently missing a third of itself.
    #:
    #: It falls to zero as re-ingestion sets the pointers, and reaching zero is
    #: what says the migration is finished. Distinct from the count above: those
    #: nodes have no lineage at all, which is a different and permanent condition.
    graph_with_unresolved_revision: int = 0
    #: Canonical claims no document currently supports (ADR-0053). They resolve
    #: without a citable source, which is the honest state after migration `0019`
    #: — which of the recorded sightings were current was never written down, so
    #: nothing was backfilled — and after any document that was only ever read in
    #: part.
    #:
    #: It falls as documents are observed in full again. A number that stays put
    #: says a source is never being re-ingested, not that the claims are wrong.
    claims_without_current_support: int = 0

    review_states: Mapping[str, int] = field(default_factory=dict)
    #: More than one entry means two key schemes are in the corpus at once, and
    #: nothing should be written under the newer one until the older is migrated.
    entity_key_versions: Mapping[str, int] = field(default_factory=dict)
    #: Evidence rows carrying no document cannot answer what document D at
    #: revision R asserted, which is the question the diff is built on.
    evidence_without_document: int = 0

    #: Revisions carrying no proposition.
    #:
    #: **Historical technical debt, not a contract breach** — the two must not be
    #: confused, which is why this is a count and never an equality. Rows written
    #: before every kind had a proposition keep their `NULL` deliberately: a
    #: backfill from the payload would be a guess wearing the clothes of an
    #: observation.
    #:
    #: So a correct run does not drive this to zero. What a correct run does is
    #: leave it exactly where it was, and `compare` treats any *growth* as the
    #: violation — that is a new revision written without a proposition, which
    #: the repository refuses, so growth means the refusal was bypassed.
    revisions_without_proposition: int = 0
    #: The same, for sightings. An observation with no proposition is judged
    #: against nothing, so this measures a judgement that could not be made.
    observations: int = 0
    observations_without_proposition: int = 0
    #: Propositions no revision and no observation points at. Content-addressed
    #: and never deleted, so this grows with abandoned extractions; it is a
    #: retention question and deliberately not a violation.
    propositions: int = 0
    unreferenced_propositions: int = 0
    #: How much of the corpus was actually judged, so "the judge is never asked"
    #: and "the judge always agrees" stop looking alike.
    equivalence_assessments: int = 0

    #: Every assertion's key, case and state, folded into one value.
    state_digest: str = ""

    #: Every claim with its provenance, so a later comparison can say *where*
    #: something moved and not only that something did.
    states: tuple[AssertionState, ...] = ()
    #: Each document at the revision it was last seen. A document whose revision
    #: moved between two pictures is one somebody edited — which is how movement
    #: gets classified without anybody having to say what was changed.
    documents: Mapping[str, str] = field(default_factory=dict)
    #: Each document by the name its source knows it under — a path, a page
    #: title, a URL. `document_id` is a hash, so without this an expectation
    #: cannot name the file it is about and a failure cannot say which one moved.
    document_paths: Mapping[str, str] = field(default_factory=dict)
    #: Each slot at the parser input it was last read with. A document is a
    #: coarse gate: edit one paragraph and every claim in the file becomes
    #: "expected", including the ones in sections nobody touched. A slot whose
    #: input is unchanged between two pictures says the section itself did not
    #: move, and a claim that moved anyway is drift however busy the file was.
    #:
    #: Absent for a slot nothing has read since these were first recorded, and
    #: absence is not evidence: a comparison falls back to the document there
    #: rather than calling an unobserved slot drift.
    slot_inputs: Mapping[str, str] = field(default_factory=dict)
    #: The document revision each slot was last read at. Presence, not content —
    #: and it has to be asked first, because equal fingerprints mean "the section
    #: did not move" only where the section was read on both sides.
    #:
    #: A deleted section is never read again and keeps its last reading forever,
    #: so it compares equal to itself and reads as an untouched section whose
    #: claim vanished anyway — a live run reported exactly that as the corpus
    #: drifting, when a section had simply been removed. Comparing this against
    #: the document's own revision says whether the slot was still there.
    slot_observed_at: Mapping[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "assertions": self.assertions,
            "variants": self.variants,
            "revisions": self.revisions,
            "evidence": self.evidence,
            "variants_beyond_first_revision": self.variants_beyond_first_revision,
            "deepest_revision": self.deepest_revision,
            "graph_assertions": self.graph_assertions,
            "graph_without_lineage": self.graph_without_lineage,
            "graph_without_lineage_by_kind": dict(self.graph_without_lineage_by_kind),
            "graph_with_unresolved_revision": self.graph_with_unresolved_revision,
            "claims_without_current_support": self.claims_without_current_support,
            "review_states": dict(self.review_states),
            "entity_key_versions": dict(self.entity_key_versions),
            "evidence_without_document": self.evidence_without_document,
            "revisions_without_proposition": self.revisions_without_proposition,
            "observations": self.observations,
            "observations_without_proposition": self.observations_without_proposition,
            "propositions": self.propositions,
            "unreferenced_propositions": self.unreferenced_propositions,
            "equivalence_assessments": self.equivalence_assessments,
            "state_digest": self.state_digest,
            "states": [state.as_dict() for state in self.states],
            "documents": dict(self.documents),
            "document_paths": dict(self.document_paths),
            "slot_inputs": dict(self.slot_inputs),
            "slot_observed_at": dict(self.slot_observed_at),
        }

    def as_summary(self) -> dict[str, Any]:
        """The picture without the per-claim detail, for reading.

        `states` carries one row per revision, which is what a comparison needs
        and what nobody wants printed: a corpus of any size would bury the counts
        under its own inventory. The file keeps everything; the screen keeps the
        numbers.
        """
        summary = self.as_dict()
        summary.pop("states", None)
        summary["documents"] = len(self.documents)
        # Same reason as `states`: one entry per section is inventory, and the
        # count is what says whether the sharper gate can be applied at all.
        summary["slot_inputs"] = len(self.slot_inputs)
        summary["slots_observed_at_current_revision"] = len(self.slot_observed_at)
        summary.pop("document_paths", None)
        return summary

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> LineageReport:
        return cls(
            assertions=int(values.get("assertions", 0) or 0),
            variants=int(values.get("variants", 0) or 0),
            revisions=int(values.get("revisions", 0) or 0),
            evidence=int(values.get("evidence", 0) or 0),
            variants_beyond_first_revision=int(
                values.get("variants_beyond_first_revision", 0) or 0
            ),
            deepest_revision=int(values.get("deepest_revision", 0) or 0),
            graph_assertions=int(values.get("graph_assertions", 0) or 0),
            graph_without_lineage=int(values.get("graph_without_lineage", 0) or 0),
            graph_without_lineage_by_kind=dict(
                values.get("graph_without_lineage_by_kind") or {}
            ),
            graph_with_unresolved_revision=int(
                values.get("graph_with_unresolved_revision", 0) or 0
            ),
            claims_without_current_support=int(
                values.get("claims_without_current_support", 0) or 0
            ),
            review_states=dict(values.get("review_states") or {}),
            entity_key_versions=dict(values.get("entity_key_versions") or {}),
            evidence_without_document=int(values.get("evidence_without_document", 0) or 0),
            revisions_without_proposition=int(
                values.get("revisions_without_proposition", 0) or 0
            ),
            observations=int(values.get("observations", 0) or 0),
            observations_without_proposition=int(
                values.get("observations_without_proposition", 0) or 0
            ),
            propositions=int(values.get("propositions", 0) or 0),
            unreferenced_propositions=int(values.get("unreferenced_propositions", 0) or 0),
            equivalence_assessments=int(values.get("equivalence_assessments", 0) or 0),
            state_digest=str(values.get("state_digest", "") or ""),
            states=tuple(
                AssertionState.from_dict(state) for state in values.get("states") or ()
            ),
            documents=dict(values.get("documents") or {}),
            document_paths=dict(values.get("document_paths") or {}),
            slot_inputs=dict(values.get("slot_inputs") or {}),
            slot_observed_at=dict(values.get("slot_observed_at") or {}),
        )


def state_digest(states: Iterable[tuple[str, str, int, str]]) -> str:
    """Fold every assertion's key, case, state and review into one value.

    Sorted first, so the digest describes the corpus rather than the order a
    query happened to return it in.
    """
    parts = sorted(
        f"{entity_key}|{variant_id}|{revision}|{review_state}"
        for entity_key, variant_id, revision, review_state in states
    )
    return stable_digest(*parts) if parts else ""


UNCHANGED_SOURCE = "unchanged-source"
CHANGED_SOURCE = "changed-source"

#: Which gate decided a movement, and they are not equally strong.
#:
#:     slot                the section's own parser input was read on both sides
#:     document_fallback   it was not, so the verdict rests on "this file was
#:                         edited somewhere" — the coarse answer, named so a
#:                         transitional run does not read like a real comparison
#:     none                the claim has no document, so nothing can excuse it
SLOT_GATE = "slot"
DOCUMENT_FALLBACK = "document_fallback"
NO_GATE = "none"


@dataclass(slots=True, frozen=True)
class Movement:
    """One thing that moved, and whether it was allowed to."""

    kind: str
    assertion_id: str
    entity_key: str
    document_id: str
    slot_id: str
    #: The document revision this claim was last seen at, before and after.
    source_revision: tuple[str, str] = ("", "")
    expected: bool = False
    detail: str = ""
    #: What made this expected, or what leaves it unexplained. A bare
    #: `expected=true` only ever meant "somebody edited this file somewhere",
    #: which waves through a claim that moved in a section nobody touched — and
    #: the reader could not tell those apart, so the flag was worth less than it
    #: looked.
    because: str = ""
    #: Which gate answered, printed beside the answer because the two are not
    #: equally strong. `document_fallback` means no reading of this slot existed
    #: on either side, so the verdict rests on "this file was edited somewhere" —
    #: a transitional run right after the migration that introduced the readings
    #: would otherwise look exactly as trustworthy as a real slot comparison.
    gate: str = ""
    #: The parser input this claim's slot was read with, before and after. Empty
    #: where the slot was never observed, which is not the same as unchanged.
    slot_input: tuple[str, str] = ("", "")
    #: What kind of claim moved — `fact`, `rule`, `pattern`, `decision`. Not the
    #: same as `kind`, which names the *movement*; the two sat under one word and
    #: the collision hid the whole classification-drift channel.
    claim_kind: str = ""

    def __str__(self) -> str:
        old, new = self.source_revision
        detail = f": {self.detail}" if self.detail else ""
        lines = [
            f"{self.kind}{detail}",
            f"    document_id={self.document_id or 'none'}",
            f"    slot_id={self.slot_id or 'none'}",
            f"    source_revision {old or '-'} -> {new or '-'}",
        ]
        before, after = self.slot_input
        if before or after:
            lines.append(f"    slot_input_changed={str(before != after).lower()}")
            lines.append(f"    old_content_fingerprint={before or '-'}")
            lines.append(f"    new_content_fingerprint={after or '-'}")
        lines.append(f"    expected_change={str(self.expected).lower()}")
        if self.because:
            lines.append(f"    because={self.because}")
        if self.gate:
            lines.append(f"    gate={self.gate}")
        return "\n".join(lines)


@dataclass(slots=True, frozen=True)
class ReportComparison:
    """What moved between two pictures, and whether that was allowed.

    Two gates, and which one applies is read off the pictures rather than
    declared. No document revision moved means nothing was edited, and then
    *nothing* may move but the evidence. A document revision that did move means
    somebody edited it, and movement inside that document is the point of the
    run — while movement anywhere else is still drift.
    """

    #: Whether the run is **acceptable** — that is, nothing moved that should
    #: have stood still. Not "nothing moved": in `changed-source` mode a run is
    #: supposed to produce revisions, retirements and additions, and those are
    #: expected rather than violations. The exit code reads this field.
    unchanged: bool
    mode: str = UNCHANGED_SOURCE
    #: Differences that mean the corpus moved where it should have stood still.
    violations: tuple[str, ...] = ()
    #: Differences that are the expected consequence of a run happening at all.
    expected: tuple[str, ...] = ()
    #: Every movement with its provenance, expected or not.
    movements: tuple[Movement, ...] = ()
    #: The documents whose revision moved between the two pictures.
    changed_documents: tuple[str, ...] = ()
    #: Slots where a claim was retired and another arrived under a *different
    #: kind*. Its own channel, because it is its own failure: `kind` is part of
    #: the identity namespace, so the model calling one sentence a `rule` on one
    #: pass and a `decision` on the next retires and recreates the claim whether
    #: or not anything about it moved — and the slot gate cannot catch it, since
    #: the section may genuinely have been edited at the same time.
    #:
    #: Reported and never acted on. What a kind change should *mean* is undecided,
    #: and joining the two quietly would move review and schema semantics with it.
    classification_drift: tuple[str, ...] = ()

    def __str__(self) -> str:
        # "unchanged" is reserved for a comparison where nothing actually moved.
        # It used to head every acceptable run, so a report listing a retirement,
        # an addition and two new revisions announced "corpus unchanged" above
        # them — inviting exactly the misreading that `expected_change` and
        # `gate` exist to prevent.
        if not self.unchanged:
            headline = "CORPUS DRIFTED"
        elif self.movements:
            headline = "corpus changes accepted"
        else:
            headline = "corpus unchanged"
        lines = [f"{headline}  [{self.mode}]"]
        if self.changed_documents:
            lines.append(f"  edited documents: {len(self.changed_documents)}")
        lines += [f"  ! {item}" for item in self.violations]
        lines += [f"  . {item}" for item in self.expected]
        lines += [f"  ? {item}" for item in self.classification_drift]
        return "\n".join(lines)


def _by_variant(report: LineageReport) -> dict[str, AssertionState]:
    """The deepest state of each variant, which is the one that is current."""
    latest: dict[str, AssertionState] = {}
    for state in report.states:
        seen = latest.get(state.variant_id)
        if seen is None or state.revision > seen.revision:
            latest[state.variant_id] = state
    return latest


def compare(baseline: LineageReport, current: LineageReport) -> ReportComparison:
    """Hold a second picture against the first.

    Evidence growing is the run happening; anything else growing is the run
    changing something — and whether that was allowed depends on where it
    happened. A document whose revision moved was edited, and a claim moving
    inside it is the result being looked for. The same movement in a document
    nobody touched is the drift this design exists to remove.

    Nothing has to be told which documents were edited. Two pictures and the
    revision each document was last seen at say it, which is what makes this
    usable on a corpus nobody can enumerate by hand.
    """
    changed_documents = {
        document
        for document, revision in current.documents.items()
        if baseline.documents.get(document) not in (None, revision)
    }

    def _slot_input(slot_id: str) -> tuple[str, str]:
        """What the claim's section was read with, before and after.

        Two empty strings mean the slot was never observed on either side, which
        is a different statement from "unchanged" — nothing was backfilled, so
        every slot read before these were recorded looks exactly like that.
        """
        if not slot_id:
            return "", ""
        return baseline.slot_inputs.get(slot_id, ""), current.slot_inputs.get(slot_id, "")

    def _observed(report: LineageReport, slot_id: str, document_id: str) -> bool:
        """Whether that section was still there when the document was last read.

        Presence, asked before content. A slot is read once per run it appears
        in, so a reading taken at the document's current revision means the
        section was among its sections then; a reading left at an older revision
        means the document has been read since without it.

        Absence of the record entirely is a third thing and not a "no": slots
        read before these were recorded have none, and that is the legacy case
        the document fallback exists for.
        """
        seen_at = report.slot_observed_at.get(slot_id)
        return bool(seen_at) and seen_at == report.documents.get(document_id)

    def _verdict(state: AssertionState) -> tuple[bool, str, str]:
        """Whether this movement was allowed, and what decides it.

        The document is the coarse gate and was the only one: edit one paragraph
        and every claim in the file becomes expected, including the ones in
        sections nobody touched. The corpus already knows better — each slot
        records the parser input it was read with — so the document is used only
        where the finer answer is unavailable.

            slot input changed      → expected, and it says so
            slot input unchanged    → drift, however busy the file was
            slot never observed     → fall back to the document

        The fallback is the honest direction for the third case. Absence of a
        reading is not evidence that a section is unchanged, and treating it as
        drift would report the migration that introduced the readings rather than
        anything about the corpus.
        """
        if not state.document_id:
            # A claim with no document cannot be attributed, and calling it
            # expected would hide exactly the case that needs seeing.
            return False, NO_GATE, "no_document"

        edited = state.document_id in changed_documents
        before, after = _slot_input(state.slot_id)
        if before or after:
            # Presence first. Comparing fingerprints is only meaningful where the
            # section was read on *both* sides: a deleted section is never read
            # again and keeps its last reading forever, so it compares equal to
            # itself and used to read as an untouched section whose claim
            # vanished anyway. A live run reported a removed section as the
            # corpus drifting for exactly that reason.
            was_there = _observed(baseline, state.slot_id, state.document_id)
            still_there = _observed(current, state.slot_id, state.document_id)
            if was_there and not still_there:
                return True, SLOT_GATE, "slot_removed"
            if still_there and not was_there:
                return True, SLOT_GATE, "slot_added"
            if was_there and still_there:
                if before != after:
                    return True, SLOT_GATE, "slot_input_changed"
                # The section this claim stood in was read on both sides and had
                # not moved. That the file was edited elsewhere explains nothing
                # about this claim.
                return False, SLOT_GATE, "slot_input_unchanged"
            # Read at neither revision: a reading exists, but from further back
            # than either picture. Nothing can be concluded from it, so the
            # document answers.

        if not edited:
            return False, DOCUMENT_FALLBACK, "document_unchanged"
        return True, DOCUMENT_FALLBACK, "slot_observation_unavailable"

    def _revisions(document_id: str) -> tuple[str, str]:
        return baseline.documents.get(document_id, ""), current.documents.get(document_id, "")

    def _move(kind: str, state: AssertionState, detail: str = "") -> Movement:
        expected, gate, because = _verdict(state)
        return Movement(
            kind=kind,
            assertion_id=state.assertion_id,
            entity_key=state.entity_key,
            document_id=state.document_id,
            slot_id=state.slot_id,
            source_revision=_revisions(state.document_id),
            expected=expected,
            gate=gate,
            because=because,
            slot_input=_slot_input(state.slot_id),
            claim_kind=state.kind,
            detail=detail,
        )

    movements: list[Movement] = []
    before, after = _by_variant(baseline), _by_variant(current)

    for variant_id, state in after.items():
        was = before.get(variant_id)
        if was is None:
            movements.append(_move("assertion added", state))
            continue
        if state.revision != was.revision:
            movements.append(
                _move("revision added", state, f"revision {was.revision} -> {state.revision}")
            )
        if was.review_state == APPROVED and state.review_state != APPROVED:
            # Losing an approval is never merely expected. On an edited document
            # it is the design working — a changed rule must be looked at again —
            # but it is always worth naming, because somebody has to act on it.
            movements.append(
                _move("approval spent", state, f"{was.review_state} -> {state.review_state}")
            )
        if state.retired and not was.retired:
            movements.append(_move("assertion retired", state))

    for variant_id, was in before.items():
        if variant_id not in after:
            # Rows do not vanish here; they are retired. One that is simply gone
            # was deleted, and nothing in this design deletes.
            movements.append(
                Movement(
                    kind="assertion disappeared",
                    assertion_id=was.assertion_id,
                    entity_key=was.entity_key,
                    document_id=was.document_id,
                    slot_id=was.slot_id,
                    claim_kind=was.kind,
                    source_revision=_revisions(was.document_id),
                    expected=False,
                )
            )

    violations = [str(move) for move in movements if not move.expected]
    expected: list[str] = [str(move) for move in movements if move.expected]

    if current.graph_without_lineage > baseline.graph_without_lineage:
        violations.append(
            f"{current.graph_without_lineage - baseline.graph_without_lineage} more graph "
            f"assertion(s) the lineage cannot reach"
        )
    if current.evidence_without_document > baseline.evidence_without_document:
        violations.append("evidence appeared without a document")
    if current.revisions_without_proposition > baseline.revisions_without_proposition:
        # Growth, never the count. Rows written before every kind had a
        # proposition keep their `NULL` on purpose, so a correct run leaves this
        # number exactly where it was — measuring equality would fail a perfect
        # run for debt it is deliberately preserving. Growth is different: a new
        # revision without a proposition is refused, so more of them means the
        # refusal was bypassed.
        violations.append(
            f"{current.revisions_without_proposition - baseline.revisions_without_proposition}"
            f" new revision(s) written with no proposition"
        )
    if current.observations_without_proposition > baseline.observations_without_proposition:
        violations.append("a sighting was recorded with nothing to judge it by")
    if len(current.entity_key_versions) > 1:
        violations.append(
            f"two key schemes in the corpus at once: {sorted(current.entity_key_versions)}"
        )

    if current.evidence != baseline.evidence:
        expected.append(
            f"evidence: {baseline.evidence} -> {current.evidence} "
            f"(a second run is a second sighting)"
        )

    return ReportComparison(
        unchanged=not violations,
        mode=CHANGED_SOURCE if changed_documents else UNCHANGED_SOURCE,
        violations=tuple(violations),
        expected=tuple(expected),
        movements=tuple(movements),
        changed_documents=tuple(sorted(changed_documents)),
        classification_drift=_classification_drift(movements),
    )


def _classification_drift(movements: Sequence[Movement]) -> tuple[str, ...]:
    """Slots where a claim left and another arrived under a different kind.

    Its own channel, and deliberately not folded into the expected/violation
    split. `kind` is part of the identity namespace, so a model calling one
    sentence a `rule` on one pass and a `decision` on the next retires the claim
    and creates a new one whether or not anything about it moved — and the slot
    gate cannot catch that, because the section may have been genuinely edited at
    the same moment. It was, in the run this was written for: three rules left
    and one decision and two facts arrived in a slot whose text really had
    changed, so every one of the six read as expected.

    Structural, and it does not claim the claims are the same. Whether they are
    is an equivalence judgement over whole propositions, and this has neither the
    propositions nor a judge. It says only: here is a slot where kinds were
    exchanged, which is worth a person's attention.

    Reported and never acted on. What a kind change *should* mean — a new claim, a
    reclassification of the same one, something to quarantine — is undecided, and
    joining them quietly would move review and schema semantics with it.
    """
    left: dict[str, set[str]] = {}
    arrived: dict[str, set[str]] = {}
    for move in movements:
        if not move.slot_id or not move.kind:
            continue
        if move.kind in ("assertion retired", "assertion disappeared"):
            left.setdefault(move.slot_id, set()).add(move.claim_kind)
        elif move.kind == "assertion added":
            arrived.setdefault(move.slot_id, set()).add(move.claim_kind)

    findings = []
    for slot_id in sorted(left.keys() & arrived.keys()):
        gone, came = left[slot_id] - {""}, arrived[slot_id] - {""}
        if gone and came and gone != came:
            findings.append(
                f"classification drift in {slot_id}: "
                f"{'/'.join(sorted(gone))} left, {'/'.join(sorted(came))} arrived "
                f"(kind is part of identity, so this retires and recreates the claim; "
                f"whether they are one claim is not decided here)"
            )
    return tuple(findings)
