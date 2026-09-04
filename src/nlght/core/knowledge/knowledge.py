# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Knowledge graph domain types.

Modelled on the validated prototype schema in
``next-integration/knowledge-indexer/knowledge.sql``: one ``knowledge`` node
carries the shared lifecycle (kind, type, confidence, review state, product and
version scope), and one kind-specific payload carries the assertion itself.

``kind`` is an open identifier rather than a closed enum — ``fact``, ``rule``,
``pattern``, and ``decision`` are the built-in kinds, and a later kind must not
require changing the lifecycle tables (ADR-0032).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from datetime import datetime
from enum import StrEnum
from typing import Any

from nlght.core.errors.errors import PermanentError
from nlght.core.knowledge.knowledge_proposition import Proposition

FACT = "fact"
RULE = "rule"
PATTERN = "pattern"
DECISION = "decision"

BUILTIN_KINDS = (FACT, RULE, PATTERN, DECISION)


class ReviewDecision(StrEnum):
    APPROVED = "approved"
    REJECTED = "rejected"
    EDITED = "edited"
    #: Two assertions say the same thing in different words. The source keeps
    #: its history and points at the target, which inherits its evidence.
    MERGED = "merged"


class KnowledgeKindConflictError(PermanentError):
    """A sighting tried to continue an assertion of a different kind.

    `kind` is part of the identity namespace, so a `rule` continuing a `fact`
    would merge two namespaces and lose a claim. The graph refuses the write, and
    that refusal is *correct* — it means the matcher answered wrongly, which is a
    fault in the run rather than a condition of the world.

    Permanent, and that is the point of it having its own type. Nothing about a
    second attempt makes the same input resolve differently, so retrying would
    repeat the failure until the attempts ran out. Worse, the answer comes from a
    model: a retry can produce a different extraction, succeed by accident, and
    leave a matcher fault invisible until it recurs on a corpus somebody cares
    about.
    """


class ReviewConflictError(ValueError):
    """Somebody already ruled on this exact version of the assertion.

    Two decisions on one version do not merely duplicate work: the second
    silently overwrites the first, so approve-then-reject and
    reject-then-approve leave opposite states behind and the history holds two
    verdicts that contradict each other. Which one wins is decided by timing.

    A ValueError so that a caller written before this existed still catches it;
    a distinct type so that one written after can tell "this was already
    judged" from "this input is wrong" and offer the reviewer the choice.
    """


_TERM_SEPARATORS = re.compile(r"[\s\-_./]+")
"""What counts as one separator inside a short field."""


def _require(value: str, name: str) -> str:
    if not value.strip():
        raise ValueError(f"knowledge {name} must not be empty")
    return value


def canonical_term(value: str) -> str:
    """Fold a short schema field to one spelling.

    A model asked for a short field still varies its casing and its spacing, so
    a raw one would carry exactly the drift it was introduced to remove, only
    shorter. Folded to lower case with runs of separators collapsed to a single
    underscore — enough that "Approval Threshold" and "approval-threshold" are
    one value, and little enough that "approval_threshold" and
    "prohibition_threshold" stay two.
    """
    folded = _TERM_SEPARATORS.sub("_", value.strip().lower()).strip("_")
    if not folded:
        raise ValueError("knowledge term must not be empty")
    return folded


def _canonical_pair(
    first: str | None, second: str | None, names: tuple[str, str]
) -> tuple[str | None, str | None]:
    """Both fields or neither, canonically.

    Half the pair identifies nothing: a subject cannot say which question about
    it this is, and a property cannot say what it is a property of. Allowing
    half would let an entity key later be built from something that is not one.

    Absent is a legitimate state — assertions written before these fields
    existed have neither, and backfilling by guessing at their wording is the
    drift all of this is trying to remove.
    """
    if first is None and second is None:
        return None, None
    if first is None:
        raise ValueError(f"knowledge {names[0]} is required when {names[1]} is given")
    if second is None:
        raise ValueError(f"knowledge {names[1]} is required when {names[0]} is given")
    return canonical_term(first), canonical_term(second)


@dataclass(slots=True, frozen=True)
class FactPayload:
    subject: str
    predicate: str
    object: str

    def __post_init__(self) -> None:
        _require(self.subject, "fact subject")
        _require(self.predicate, "fact predicate")
        _require(self.object, "fact object")


@dataclass(slots=True, frozen=True)
class RulePayload:
    """A rule: what it is about, which question about it, and how it reads.

    `subject` and `rule_property` are the short, canonical pair; `rule_text` is
    the body a citation quotes. Both short fields are optional because a corpus
    written before they existed has neither.
    """

    rule_text: str
    subject: str | None = None
    rule_property: str | None = None

    def __post_init__(self) -> None:
        _require(self.rule_text, "rule text")
        subject, rule_property = _canonical_pair(
            self.subject, self.rule_property, ("subject", "rule_property")
        )
        object.__setattr__(self, "subject", subject)
        object.__setattr__(self, "rule_property", rule_property)


@dataclass(slots=True, frozen=True)
class PatternPayload:
    pattern_name: str
    description: str

    def __post_init__(self) -> None:
        _require(self.pattern_name, "pattern name")
        _require(self.description, "pattern description")


@dataclass(slots=True, frozen=True)
class DecisionPayload:
    """A decision: what it is about, which kind of decision, and what it said."""

    decision: str
    effect: str
    subject: str | None = None
    decision_type: str | None = None

    def __post_init__(self) -> None:
        _require(self.decision, "decision")
        _require(self.effect, "decision effect")
        subject, decision_type = _canonical_pair(
            self.subject, self.decision_type, ("subject", "decision_type")
        )
        object.__setattr__(self, "subject", subject)
        object.__setattr__(self, "decision_type", decision_type)


KnowledgePayload = FactPayload | RulePayload | PatternPayload | DecisionPayload

_PAYLOAD_KINDS: dict[type[Any], str] = {
    FactPayload: FACT,
    RulePayload: RULE,
    PatternPayload: PATTERN,
    DecisionPayload: DECISION,
}


def kind_of(payload: KnowledgePayload) -> str:
    """The built-in kind identifier for a payload type."""
    return _PAYLOAD_KINDS[type(payload)]


_KIND_PAYLOADS: dict[str, type[Any]] = {
    FACT: FactPayload,
    RULE: RulePayload,
    PATTERN: PatternPayload,
    DECISION: DecisionPayload,
}


def payload_for(kind: str, fields: dict[str, Any]) -> KnowledgePayload:
    """Build a payload of ``kind`` from named fields, validating as it goes.

    A reviewer correcting an assertion by hand submits loose strings; this is
    where they become a checked payload, so an empty predicate is refused at the
    same boundary that refuses one from an extractor.
    """
    payload_type = _KIND_PAYLOADS.get(kind)
    if payload_type is None:
        raise ValueError(f"unknown knowledge kind '{kind}'")
    expected = {field_.name for field_ in dataclass_fields(payload_type)}
    unknown = set(fields) - expected
    if unknown:
        raise ValueError(
            f"unknown field(s) for kind '{kind}': {', '.join(sorted(unknown))}"
        )
    missing = expected - set(fields)
    if missing:
        raise ValueError(
            f"missing field(s) for kind '{kind}': {', '.join(sorted(missing))}"
        )
    built: KnowledgePayload = payload_type(**fields)
    return built


@dataclass(slots=True, frozen=True)
class KnowledgeWrite:
    """One atomic, independently reviewable assertion plus its evidence.

    ``identity`` is the caller-computed stable identity used for deduplication
    and reinforcement, so re-extracting the same assertion from another document
    strengthens it instead of duplicating it.

    A fact carries its ``proposition``, and that is what it says. The
    ``payload`` is the legacy row derived from it, present only while the
    derivation loses nothing — so "Alice transfers CHF 500 to Bob" arrives with a
    proposition and no payload rather than being cut down to fit three columns or
    thrown away as malformed. Every other kind still arrives as a payload and is
    untouched by this.
    """

    identity: str
    payload: KnowledgePayload | None
    type: str
    confidence: float
    source_id: str
    run_id: str
    product: str | None = None
    version: str | None = None
    review_required: bool = False
    review_reason: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    #: What this claim says, structured — for every kind.
    proposition: Proposition | None = None
    #: Which kind, where the payload cannot say. A projection may be absent, so
    #: the kind cannot be read off it; `kind` falls back to the payload only for
    #: callers that predate this.
    _kind: str = ""

    def __post_init__(self) -> None:
        _require(self.identity, "identity")
        _require(self.type, "type")
        _require(self.source_id, "source_id")
        _require(self.run_id, "run_id")
        if self.payload is None and self.proposition is None:
            raise ValueError("a knowledge write must carry a payload or a proposition")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("knowledge confidence must be within [0.0, 1.0]")
        if self.review_required and not (self.review_reason or "").strip():
            raise ValueError("a review-required assertion must carry a review reason")

    @property
    def kind(self) -> str:
        """What kind of claim this is — stated, or read off the payload.

        It used to fall back to `fact` whenever there was no payload, which was
        true only while facts were the one kind that could lack one. Now every
        kind's payload is a projection that may legitimately be absent, so a rule
        carrying four fields its row cannot hold would have been written into the
        graph as a *fact*. The caller says which kind it has.
        """
        if self._kind:
            return self._kind
        if self.payload is not None:
            return kind_of(self.payload)
        raise ValueError(
            f"knowledge write '{self.identity}' has no payload to read a kind from "
            f"and none was given; a projection may be absent, a kind may not"
        )


@dataclass(slots=True, frozen=True)
class Observation:
    """One sighting of an assertion; a retry must not count twice."""

    identity: str
    source_id: str
    run_id: str
    evidence: dict[str, Any]
    observed_at: datetime


@dataclass(slots=True, frozen=True)
class Review:
    identity: str
    reviewer_id: str
    decision: ReviewDecision
    reviewed_at: datetime
    comment: str | None = None
    changes: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class KnowledgeStatistics:
    """What the graph looks like from a reviewer's chair.

    ``pending`` counts the queue as the queue defines it — quarantined and
    undecided — so it matches what a reviewer is actually shown rather than the
    raw quarantine flag, which rejection also leaves set.
    """

    total: int
    pending: int
    retrievable: int
    merged: int
    average_pending_confidence: float
    average_retrievable_confidence: float
    kinds: int
    types: int


@dataclass(slots=True, frozen=True)
class SupportingEvidence:
    """One document that carries a claim now, and where it was last seen to.

    A claim is not *in* a document. It is supported by however many sources
    currently assert it, and a second document is what keeps it alive when the
    first stops (ADR-0044). So this is always plural at the claim, and none of
    its entries is the claim's home: naming one `document_id` on an assertion
    would make whichever source happened to be read first look like the origin.

    It is what a citation is built from and never a citation itself.

    **Two different statements live here, and the field names keep them apart.**
    The row's existence says the document carries this state *now*, decided by
    the last complete observation of it. `observed_document_revision` says where
    it was last actually read saying so, which is not the same thing and can be
    older — see that field.

    Deliberately no `source_span`. The column exists on the evidence row and no
    producer sets it, so it is `None` for every sighting the platform has ever
    written — offering it here would advertise line coordinates the corpus
    cannot supply.
    """

    document_id: str
    #: The document revision this state was last **extracted** at — not
    #: necessarily the revision the document is at now.
    #:
    #: The two come apart whenever a section the stable-slot guard protected was
    #: not re-extracted: the document moved on, that section did not change, and
    #: nobody read the claim out of the newer revision. Writing the current
    #: revision here would fabricate a sighting; leaving it is the truthful
    #: answer to "where was this last seen to be said".
    #:
    #: Named for what it is, because `document_revision` on a table called
    #: *current support* reads as "the revision that currently carries it" and
    #: that is a claim this cannot make.
    observed_document_revision: str = ""
    #: As the sighting saw it. A renamed document is the same document, so this
    #: is provenance and never identity.
    document_path: str = ""
    #: Empty where the section had nothing to be found by — a format with no
    #: headings — which is a real and common answer, not a missing one.
    slot_id: str = ""


@dataclass(slots=True, frozen=True)
class KnowledgeObject:
    """A resolved assertion as returned by the knowledge store."""

    identity: str
    kind: str
    type: str
    confidence: float
    payload: dict[str, Any]
    review_required: bool
    review_reason: str | None
    product: str | None
    version: str | None
    metadata: dict[str, Any]
    created_at: datetime
    updated_at: datetime
    #: Set when a reviewer folded this assertion into another one. It keeps its
    #: history but is no longer canonical and never reaches a retrieval path.
    #: Last and defaulted so that adding it did not break existing construction.
    merged_into: str | None = None
    #: What the source actually says, in its own words (ADR-0048).
    #:
    #: Read from the revision this node was written by, never rebuilt from the
    #: payload. It is the field a lexical index should rank on: real language
    #: carries phrases, grammar and synonyms, and a claim found through it can be
    #: quoted back to a reader. An assembled bag of field values carries none of
    #: that and is not quotable.
    #:
    #: Empty for a node with no lineage — written before revisions existed, or by
    #: a path that keeps none — which is why nothing may require it.
    observed_text: str = ""

    #: The durable claim behind this node. `identity` is a content address and
    #: changes when the structure does; this does not, which is what a citation
    #: to "the claim" has to name.
    assertion_id: str = ""
    #: The state this node represents (ADR-0051). Named so a reader can say
    #: *which* state was cited, and so `observed_text` and `support` are
    #: verifiably about the same one.
    revision_id: str = ""
    #: Every sighting that supports **this state** — plural, unordered by
    #: importance, and with no owner among them.
    #:
    #: Scoped to the revision rather than to the assertion, so a wording the
    #: corpus has moved past does not turn up as current support. What it cannot
    #: yet exclude is a document that supported this exact state and has since
    #: stopped carrying the claim while another document keeps it alive: evidence
    #: is never deleted (ADR-0044) and nothing records that a document dropped a
    #: claim, so that sighting stays visible here.
    support: tuple[SupportingEvidence, ...] = ()


def reinforce_confidence(previous: float, evidence: float) -> float:
    """Bayesian odds-multiplication merge, as used by the prototype.

    Independent sightings reinforce one another: two mediocre observations
    combine into a stronger belief than either alone, while a single weak one
    cannot by itself push an assertion to certainty.
    """
    clamped = [min(max(value, 1e-6), 1 - 1e-6) for value in (previous, evidence)]
    odds = (clamped[0] / (1 - clamped[0])) * (clamped[1] / (1 - clamped[1]))
    return odds / (1 + odds)
