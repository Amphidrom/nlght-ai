# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Knowledge graph schema — a database of its own.

Deliberately on a separate declarative base from the platform models: the
knowledge graph is not platform metadata and does not live in the runtime's
database. It has its own connection URL, its own migration chain, and can sit on
a different server entirely.

Shape follows next-integration/knowledge-indexer/knowledge.sql: one lifecycle
node per assertion plus one kind-specific payload row, so a later kind adds a
payload table without touching the lifecycle.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    TIMESTAMP,
    Boolean,
    CheckConstraint,
    Float,
    ForeignKey,
    Index,
    Integer,
    Text,
    UniqueConstraint,
    false,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

_JSON = JSON().with_variant(JSONB(), "postgresql")


class KnowledgeBase(DeclarativeBase):
    """Metadata root for the knowledge database, separate from the platform."""


class Knowledge(KnowledgeBase):
    __tablename__ = "knowledge"
    __table_args__ = (
        CheckConstraint("confidence >= 0.0 AND confidence <= 1.0", name="ck_knowledge_confidence"),
    )

    identity: Mapped[str] = mapped_column(Text, primary_key=True)
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    type: Mapped[str] = mapped_column(Text, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    review_required: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    review_reason: Mapped[str | None] = mapped_column(Text)
    # Set when a reviewer merged this assertion into another. The row stays —
    # deleting it would cascade away its observations and its decision history,
    # which is the trail a later disagreement needs — but it is no longer
    # canonical and never reaches a retrieval path. RESTRICT, so a merge target
    # cannot be deleted out from under the assertions pointing at it.
    merged_into: Mapped[str | None] = mapped_column(
        Text, ForeignKey("knowledge.identity", ondelete="RESTRICT")
    )
    #: Which state of the claim this node represents — a relationship, and the
    #: only one there is between the graph and the lineage.
    #:
    #: It used to be inferred from `KnowledgeRevision.fingerprint == identity`,
    #: which is not a relationship: a fingerprint is a content address, several
    #: revisions may legitimately carry the same one — a revision is new when the
    #: *observed wording* moves, and the structured claim may not have — and
    #: nothing said which of them the node stood for. Every read that needed one
    #: therefore quantified over the set and hoped it answered.
    #:
    #: **Which node's pointer may move, and when.** A write reaches the node its
    #: own state addresses — the arriving revision's fingerprint *is* the
    #: identity looked up — so the two cases stay apart by construction:
    #:
    #:     R1, R2 hash to one P    one node P, and its pointer moves R1 → R2:
    #:                             the same structured state, in its latest
    #:                             recorded form (a corrected full stop is the
    #:                             plain case)
    #:
    #:     R1→P1, R2→P2            two nodes. P1 keeps R1 and is never re-aimed:
    #:                             a record whose content is P1 claiming to
    #:                             represent R2 would be a lie about itself. It
    #:                             stops being canonical because R1 is no longer
    #:                             the latest revision of its variant, which is a
    #:                             fact about the lineage and not about the node.
    #:
    #: NULL means this node names no state, and there are two ways to arrive at
    #: it — see `revision_unresolved`, which is what tells them apart.
    revision_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("knowledge_revisions.revision_id", ondelete="RESTRICT"),
    )
    #: Set by migration `0018` on a node that **has** lineage whose exact state
    #: could not be proved — several revisions carried its fingerprint, and
    #: picking the deepest is the set logic `revision_id` exists to remove.
    #:
    #: Without this, one NULL would mean two different things:
    #:
    #:     no revision, none ever      not a withdrawal, not a supersession —
    #:                                 absence of a record is not evidence, and
    #:                                 this node stays canonical as it always was
    #:
    #:     lineage exists, state       unknown, and it may well be a state the
    #:     unresolved                  corpus has moved past. Reading it as the
    #:                                 first would hand a superseded wording back
    #:                                 to retrieval as though it were current
    #:
    #: So an unresolved node is not canonical: it is withheld rather than
    #: guessed at. The next sighting of the claim sets `revision_id` and clears
    #: this, which is the only thing that can settle it honestly.
    revision_unresolved: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default=false()
    )
    product: Mapped[str | None] = mapped_column(Text)
    version: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)


# Retrieval must never surface quarantined assertions, so the common lookup is
# "visible knowledge of a kind".
Index("knowledge_visible_by_kind", Knowledge.kind, Knowledge.review_required)
Index("knowledge_review_queue", Knowledge.review_required, Knowledge.updated_at)
Index("knowledge_merged_into", Knowledge.merged_into)
# Every canonical read joins the node to the state it stands for.
Index("knowledge_by_revision", Knowledge.revision_id)


class KnowledgeFact(KnowledgeBase):
    __tablename__ = "knowledge_facts"

    identity: Mapped[str] = mapped_column(Text, ForeignKey("knowledge.identity", ondelete="CASCADE"), primary_key=True)
    subject: Mapped[str] = mapped_column(Text, nullable=False)
    predicate: Mapped[str] = mapped_column(Text, nullable=False)
    object: Mapped[str] = mapped_column(Text, nullable=False)


Index("knowledge_facts_subject", KnowledgeFact.subject)
Index("knowledge_facts_predicate", KnowledgeFact.predicate)
Index("knowledge_facts_object", KnowledgeFact.object)


class KnowledgeRule(KnowledgeBase):
    __tablename__ = "knowledge_rules"

    identity: Mapped[str] = mapped_column(Text, ForeignKey("knowledge.identity", ondelete="CASCADE"), primary_key=True)
    rule_text: Mapped[str] = mapped_column(Text, nullable=False)
    # Nullable, because a corpus written before these existed has neither, and
    # backfilling by guessing at the wording is the drift they exist to remove.
    subject: Mapped[str | None] = mapped_column(Text)
    rule_property: Mapped[str | None] = mapped_column(Text)


class KnowledgePattern(KnowledgeBase):
    __tablename__ = "knowledge_patterns"

    identity: Mapped[str] = mapped_column(Text, ForeignKey("knowledge.identity", ondelete="CASCADE"), primary_key=True)
    pattern_name: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False)


class KnowledgeDecision(KnowledgeBase):
    __tablename__ = "knowledge_decisions"

    identity: Mapped[str] = mapped_column(Text, ForeignKey("knowledge.identity", ondelete="CASCADE"), primary_key=True)
    decision: Mapped[str] = mapped_column(Text, nullable=False)
    effect: Mapped[str] = mapped_column(Text, nullable=False)
    subject: Mapped[str | None] = mapped_column(Text)
    decision_type: Mapped[str | None] = mapped_column(Text)


class KnowledgeObservation(KnowledgeBase):
    """One sighting. A unique (identity, run_id, source_id) keeps a retried
    workflow step from counting the same evidence twice."""

    __tablename__ = "knowledge_observations"
    __table_args__ = (
        UniqueConstraint("identity", "run_id", "source_id", name="uq_knowledge_observation"),
    )

    observation_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    identity: Mapped[str] = mapped_column(Text, ForeignKey("knowledge.identity", ondelete="CASCADE"), nullable=False)
    source_id: Mapped[str] = mapped_column(Text, nullable=False)
    run_id: Mapped[str] = mapped_column(Text, nullable=False)
    evidence: Mapped[dict[str, Any] | None] = mapped_column(_JSON)
    observed_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)


class KnowledgeAssertion(KnowledgeBase):
    """One business question, over its whole life.

    The surrogate the whole design rests on. `assertion_id` is minted once and
    never computed from content, so a rewording changes what the assertion says
    without changing which assertion it is — and a faulty matcher cannot
    retroactively decide what this database considers identity.

    `entity_key` says which quantity it is, and `entity_key_version` says which
    scheme produced that key. The version is a column of its own rather than
    part of the key: folded in, a key of an older scheme would merely differ
    from a new one and nothing could tell that from a key naming something else.
    """

    __tablename__ = "knowledge_assertions"
    #: Indexed, not unique. The key describes an assertion and does not identify
    #: one — two may legitimately carry the same description, because an
    #: ambiguous match mints a new assertion rather than choosing between equally
    #: plausible candidates, and a claim asserted again after being dropped is a
    #: new decision rather than an undo.
    __table_args__ = (
        Index(
            "knowledge_assertions_by_entity_key", "entity_key_version", "entity_key"
        ),
    )

    assertion_id: Mapped[str] = mapped_column(Text, primary_key=True)
    entity_key_version: Mapped[str] = mapped_column(Text, nullable=False)
    entity_key: Mapped[str] = mapped_column(Text, nullable=False)
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )
    retired_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))


class KnowledgeAssertionLineage(KnowledgeBase):
    """One assertion took another's place.

    An edge rather than a column, because succession is not one-to-one. A rule
    splits into two, or two collapse into one, and a `superseded_by` field can
    say neither:

        A → B          a claim replaced
        A → B, C       one split into two
        A, B → C       two folded into one

    A rewording is *not* recorded here — that keeps its assertion and adds a
    revision. This is the other case: propositions that say different things,
    where one took the other's place rather than arriving beside it. Whether it
    did cannot be read off the words, so it is written down with who or what
    decided it, and never inferred.
    """

    __tablename__ = "knowledge_assertion_lineage"
    __table_args__ = (
        UniqueConstraint(
            "predecessor_id", "successor_id", "relation", name="uq_knowledge_lineage_edge"
        ),
        Index("knowledge_lineage_by_successor", "successor_id"),
    )

    lineage_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    predecessor_id: Mapped[str] = mapped_column(
        Text, ForeignKey("knowledge_assertions.assertion_id", ondelete="CASCADE"),
        nullable=False,
    )
    successor_id: Mapped[str] = mapped_column(
        Text, ForeignKey("knowledge_assertions.assertion_id", ondelete="CASCADE"),
        nullable=False,
    )
    relation: Mapped[str] = mapped_column(Text, nullable=False, default="supersedes")
    #: Why, in whatever terms the recorder had. A succession nobody can account
    #: for later is one nobody can undo either.
    reason: Mapped[str | None] = mapped_column(Text)
    #: Who or what recorded it — a reviewer, a run, a repair.
    recorded_by: Mapped[str | None] = mapped_column(Text)
    recorded_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )


class KnowledgeAssertionObservation(KnowledgeBase):
    """One sighting of one assertion, kept before anything is decided about it.

    The comparison unit of the equivalence audit, and the reason it exists at
    all: a judgement has to point at the two things it was about, and both must
    be durable *before* the decision they justify. Every other candidate for
    that role is downstream of the decision — `knowledge_evidence` hangs off a
    revision, and the revision is created only once the matcher and the judge
    have already answered.

    For every kind, which is the other reason. `knowledge_propositions` can hold
    a fact and nothing else, so an audit keyed on it leaves `rule`, `decision`
    and `pattern` unauditable.

        kind             which identity namespace this belongs to
        observed_text    what the source says, word for word (ADR-0048)
        representation   which assertion inside that sentence this is

    **A sighting, not an identity.** Nothing here is deduplicated: two runs that
    read one sentence saw it twice, and collapsing them would lose which run saw
    what — the question an audit is asked. That is also why this is not
    content-addressed the way `knowledge_propositions` is: that table holds what
    a claim *is*, this one holds that somebody looked.
    """

    __tablename__ = "knowledge_assertion_observations"

    observation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    kind: Mapped[str] = mapped_column(Text, nullable=False)
    observed_text: Mapped[str] = mapped_column(Text, nullable=False)
    #: The kind's own structured shape — a `Proposition`'s fields for a
    #: fact, the payload for the others. Stored as it arrived rather than folded
    #: into one universal form, because no invariant asks for one.
    representation: Mapped[dict[str, Any]] = mapped_column(_JSON, nullable=False)
    #: Which scheme produced it, for the reason `entity_key_version` is a column:
    #: representations will change, and an old one must not look like it lives in
    #: the new namespace.
    representation_version: Mapped[str] = mapped_column(Text, nullable=False, default="1")
    run_id: Mapped[str] = mapped_column(Text, nullable=False)
    document_id: Mapped[str] = mapped_column(Text, nullable=False, default="")
    slot_id: Mapped[str] = mapped_column(Text, nullable=False, default="")
    observed_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )


class KnowledgeEquivalenceAssessment(KnowledgeBase):
    """One judgement about one pair of observations, and who reached it.

    Append-only, with **no unique key on the pair**. The same judge asked twice
    can answer differently, and a constraint permitting one row would delete
    exactly the evidence that a model is unstable — the thing anyone auditing
    this would most want to see. Reusing an answer rather than asking again is a
    cache: a different table, with a different key, added when there is a reason
    to.

    It replaces `knowledge_proposition_equivalence`, which compared the wrong
    unit. That table was keyed on fact propositions, so three of the four kinds
    could never be audited at all.

    A decision about a **pair**, so it is a record of its own rather than a field
    on either side: `equivalent(A, B)` says nothing about A, and hanging it on A
    would make a statement about a relationship look like a property of a thing.
    """

    __tablename__ = "knowledge_equivalence_assessments"
    __table_args__ = (
        Index(
            "knowledge_equivalence_by_pair",
            "left_observation_id",
            "right_observation_id",
        ),
    )

    assessment_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    left_observation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("knowledge_assertion_observations.observation_id", ondelete="CASCADE"),
        nullable=False,
    )
    right_observation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("knowledge_assertion_observations.observation_id", ondelete="CASCADE"),
        nullable=False,
    )
    verdict: Mapped[str] = mapped_column(Text, nullable=False)
    #: `None` where nothing decided — which is what `unjudged` means, and is not
    #: the same as a judge having looked and found them different.
    classifier: Mapped[str | None] = mapped_column(Text)
    model: Mapped[str | None] = mapped_column(Text)
    classifier_version: Mapped[str | None] = mapped_column(Text)
    run_id: Mapped[str | None] = mapped_column(Text)
    decided_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )


class KnowledgeProposition(KnowledgeBase):
    """One extracted claim, kept exactly as it arrived.

    `proposition_id` is a surrogate and **not** an assertion's identity. Whether
    two propositions are one claim is a judgement recorded in
    `knowledge_proposition_equivalence`, and reading it off this table would be
    the content-derived identity the whole design removed.

    `fingerprint` addresses content and nothing more: same fingerprint means
    byte-identical after normalisation, and *different* fingerprints say nothing
    about whether the claims are one — that is the question being asked.
    `fingerprint_version` says which scheme produced it, because normalisation
    will change and an old value must not look like it lives in the new
    namespace.
    """

    __tablename__ = "knowledge_propositions"
    __table_args__ = (
        # Content addressing, so one claim seen twice is one row. Not identity:
        # the same claim said differently has a different fingerprint and is a
        # second row, which is exactly what equivalence exists to judge.
        UniqueConstraint(
            "fingerprint_version", "fingerprint", name="uq_knowledge_proposition_content"
        ),
    )

    proposition_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    fingerprint: Mapped[str] = mapped_column(Text, nullable=False)
    fingerprint_version: Mapped[str] = mapped_column(Text, nullable=False)
    #: The claim, whole. Nothing about its shape is interpreted here.
    fields: Mapped[dict[str, Any]] = mapped_column(_JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )


class KnowledgeSlot(KnowledgeBase):
    """Where a claim stands: a section of a document, identified once.

    The third surrogate in the model, for the same reason as the other two. Two
    runs compare one slot's assertions against the same slot's, so a slot that
    loses its identity retracts and recreates everything inside it and takes the
    approvals with them. The id is minted and never derived — a slot keyed on its
    position renumbered every section below an inserted paragraph, and a slot
    keyed on its heading would move whenever somebody renamed one.

    `document_id` is part of the effective identity: matching never crosses a
    document, so the same heading on two pages is two slots.
    """

    __tablename__ = "knowledge_slots"

    slot_id: Mapped[str] = mapped_column(Text, primary_key=True)
    document_id: Mapped[str] = mapped_column(Text, nullable=False)
    #: `live` or `retired`. Retired rather than deleted: a section that comes
    #: back must be distinguishable from one that was never there, or the
    #: decision whether to continue its lineage cannot be taken at all.
    status: Mapped[str] = mapped_column(Text, nullable=False, default="live")
    #: A hash of what this section last said. The recovery rung of the resolver
    #: compares against it, and without it that rung could never fire from the
    #: database — which is the rung that catches a renamed heading, so a rename
    #: would mint a new slot and retract everything the old one held.
    #:
    #: A hash rather than the text: the registry is about where a claim stands,
    #: and keeping a second copy of every section here would make it a second
    #: corpus that can disagree with the first.
    content_fingerprint: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )
    retired_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))


class KnowledgeSlotObservation(KnowledgeBase):
    """What one run saw in one slot: the parser input it was given.

    `KnowledgeSlot.content_fingerprint` is a *current state* — the resolver
    overwrites it every run so the recovery rung can match against what the
    section says now. That makes it useless for the question a report has to
    answer: whether the section a claim stood in was actually edited between two
    pictures. The value readable after a run is already that run's, and the
    previous one is gone.

    So the fingerprint is kept per sighting as well. One is what the slot says
    now, the other is what it said each time it was read, and only the second can
    show *why* a movement was expected:

        expected_change=true
        because=slot_input_changed

    rather than the document-wide "something in this file was edited", which
    waves through a claim that moved in a section nobody touched.

    Written per run and per slot, so it is a log of readings rather than of
    changes — a run that changed nothing still records that it looked, which is
    what tells "unchanged" from "not observed".
    """

    __tablename__ = "knowledge_slot_observations"
    __table_args__ = (
        UniqueConstraint("run_id", "slot_id", name="uq_knowledge_slot_observation"),
    )

    observation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    run_id: Mapped[str] = mapped_column(Text, nullable=False)
    document_id: Mapped[str] = mapped_column(Text, nullable=False)
    slot_id: Mapped[str] = mapped_column(Text, nullable=False)
    #: What the parser handed the model for this section, hashed exactly as it
    #: stood. Emphatically *not* the value the resolver's recovery rung matches
    #: on: that one is normalised, because a reflowed section is still the same
    #: section, and this one must be able to say that a section was edited at
    #: all — a sentence given a full stop is a different question to the model.
    parser_input_fingerprint: Mapped[str] = mapped_column(Text, nullable=False)
    #: Which revision of the document this reading came from, so a comparison can
    #: be anchored to the same provenance the rest of the report uses.
    document_revision: Mapped[str] = mapped_column(Text, nullable=False, default="")
    observed_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )


class KnowledgeSlotAnchor(KnowledgeBase):
    """One path a slot has been found by, and how much that path was worth.

    The strength belongs **here** and not on the slot, because it changes over a
    slot's life and the change is the interesting part:

        authored → authored    ordinary
        derived  → authored    an upgrade; very likely the same section
        derived  → derived     the fallback rungs are doing the work
        authored → derived     a degradation — do not continue blindly

    The last row is why one strength per slot would not do. When `[[expenses]]`
    disappears and only the heading path is left, a resolver that still believed
    it held an authored anchor would carry a lineage across a change nobody could
    see. Recorded per anchor, the transition is visible and can be treated as
    what it is.

    Temporal rather than a flag plus a history: `valid_to IS NULL` *is* current,
    so there is one fact rather than two that can disagree.
    """

    __tablename__ = "knowledge_slot_anchors"
    __table_args__ = (
        Index("knowledge_slot_anchors_by_slot", "slot_id", "valid_to"),
        Index("knowledge_slot_anchors_by_anchor", "anchor"),
    )

    anchor_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    slot_id: Mapped[str] = mapped_column(
        Text, ForeignKey("knowledge_slots.slot_id", ondelete="CASCADE"), nullable=False
    )
    #: What the source gave: an authored id taken unchanged, or a heading path.
    anchor: Mapped[str] = mapped_column(Text, nullable=False)
    #: `authored` or `derived`. `none` never reaches here — a section nothing can
    #: find again gets no slot, because an id that cannot be resolved back to is
    #: a number claiming to be an identity.
    strength: Mapped[str] = mapped_column(Text, nullable=False)
    valid_from: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )
    valid_to: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))

    @property
    def is_current(self) -> bool:
        return self.valid_to is None


class KnowledgeVariant(KnowledgeBase):
    """One simultaneously valid case of an assertion.

    CHF 500 for employees and CHF 5000 for executives are both true, and
    revisions are a sequence of states — so two values true at once cannot be
    revisions of one thing. They are variants.

    `scope` is stored but is not the identity: a variant id is a surrogate,
    because widening a scope continues the variant it replaced and a derived key
    would change with the scope and retire it instead. `resolution` records how
    the variant was arrived at, so an `ambiguous` one can be found and looked at.
    """

    __tablename__ = "knowledge_variants"

    variant_id: Mapped[str] = mapped_column(Text, primary_key=True)
    assertion_id: Mapped[str] = mapped_column(
        Text, ForeignKey("knowledge_assertions.assertion_id", ondelete="CASCADE"), nullable=False
    )
    scope: Mapped[list[str]] = mapped_column(_JSON, nullable=False, default=list)
    resolution: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )
    retired_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))


class KnowledgeRevision(KnowledgeBase):
    """One state of one variant, and whether a person has approved it.

    `review_state` is separate from the lineage on purpose: whether a claim is
    still the same claim and whether an approval of it still holds are two
    questions. An expense limit moving from CHF 500 to 550 continues the
    assertion and invalidates the approval.

    A revision is also where the complete extracted structure hangs, and that
    placement is the point. One assertion has many revisions and each records the
    form the claim took at the time, so the proposition belongs to the *state*
    and not to the continuity:

        Assertion A
          Revision 1 → Proposition P1
          Revision 2 → Proposition P2

    Two propositions with different fingerprints can continue one assertion when
    an equivalence judgement says they are one claim. Hanging a single
    proposition on the assertion — or on the graph node — would flatten that
    history into "one proposition per assertion" and lose the earlier form of
    every claim that was ever reworded.
    """

    __tablename__ = "knowledge_revisions"
    __table_args__ = (
        UniqueConstraint("variant_id", "revision", name="uq_knowledge_revision"),
    )

    revision_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    variant_id: Mapped[str] = mapped_column(
        Text, ForeignKey("knowledge_variants.variant_id", ondelete="CASCADE"), nullable=False
    )
    revision: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    fingerprint: Mapped[str] = mapped_column(Text, nullable=False)
    extraction_version: Mapped[str] = mapped_column(Text, nullable=False)
    review_state: Mapped[str] = mapped_column(Text, nullable=False)
    supersedes: Mapped[int | None] = mapped_column(Integer)
    #: The sighting this revision came from. Without it there is no way back
    #: from a claim in the corpus to the observation that recorded it, and an
    #: equivalence judgement would have to synthesise the candidate's side —
    #: making a sighting into a sighting per comparison, stamped with the run
    #: that compared rather than the run that saw.
    observation_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "knowledge_assertion_observations.observation_id", ondelete="RESTRICT"
        ),
    )
    #: The complete extracted structure this revision recorded. Null for the
    #: kinds that have no proposition and for everything extracted before
    #: propositions existed — nothing is backfilled, because reconstructing one
    #: from a payload that could not hold it is guessing.
    proposition_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("knowledge_propositions.proposition_id", ondelete="RESTRICT"),
    )
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )


class KnowledgeEvidence(KnowledgeBase):
    """Where one revision was found: which document, which revision of it, which slot.

    Document and document revision are columns rather than keys inside a JSON
    blob, because the incremental diff asks "which assertions came from document
    D at revision R" and that question needs them to be queryable.

    `document_path` is what the source called that document when it was read.
    Provenance and never identity: a renamed document is the same document, so
    the path may move without opening an assertion or writing a revision, and no
    slot resolves by it. It is here rather than looked up because the obvious
    place to look — `ingestion_documents` — is empty for a knowledge-only flow,
    and joining the two lifecycles to borrow a string is a decision of its own.
    """

    __tablename__ = "knowledge_evidence"
    __table_args__ = (
        # The slot is part of what makes a sighting distinct. Two sections of one
        # document saying the same thing are two sightings, and without the slot
        # only the first was recorded — which left the support view unable to say
        # which section stopped carrying a claim, because it never knew both had.
        # A retry still records nothing twice: run, document and slot are all the
        # same on a retry.
        UniqueConstraint(
            "revision_id", "document_id", "slot_id", "run_id", name="uq_knowledge_evidence"
        ),
        Index("knowledge_evidence_by_document", "document_id", "document_revision"),
    )

    evidence_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    revision_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("knowledge_revisions.revision_id", ondelete="CASCADE"),
        nullable=False,
    )
    document_id: Mapped[str] = mapped_column(Text, nullable=False)
    document_revision: Mapped[str] = mapped_column(Text, nullable=False)
    #: Stored exactly as observed, never reconstructed from `document_id`. Two
    #: sightings of one document under two names each keep the name they saw.
    document_path: Mapped[str] = mapped_column(Text, nullable=False, default="")
    #: Empty rather than null where a section has nothing to be found by, so the
    #: uniqueness above holds: PostgreSQL treats nulls as distinct, and a null
    #: here would let a retry record the same sighting twice.
    slot_id: Mapped[str] = mapped_column(Text, nullable=False, default="")
    source_span: Mapped[dict[str, Any] | None] = mapped_column(_JSON)
    run_id: Mapped[str] = mapped_column(Text, nullable=False)
    observed_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )


class KnowledgeDocumentSupport(KnowledgeBase):
    """Which documents carry a state **now**, as opposed to ever having.

    `KnowledgeEvidence` is append-only and answers "was observed here". That is
    the right shape for an audit and the wrong one for a citation: a document
    that has since removed the sentence is still in it forever, so an answer
    built from evidence would send a reader to a file that no longer says it —
    while the claim itself is perfectly alive, held up by a second document.

    So currency is recorded rather than inferred from absence. Nothing here
    reinterprets or deletes a sighting; this is a second, smaller relation that
    says what is true today, and it is replaced from what a run saw rather than
    accumulated.

    **The document is the unit of the decision, and the slot is not.** A slot is
    provenance and lifecycle (ADR-0044): a section can be renamed, split or lose
    its heading without the document ceasing to assert anything. So a row
    disappears only when a *complete* observation of its document no longer
    produces its state — never because a section moved. The `slot_id` on the row
    is where it was seen, and takes part in no removal decision of its own.

    Maintained only by `record_document`, which is only called for a document
    observed in full. A partial run cannot say what a document currently carries
    — it did not see all of it — so it writes evidence and leaves this alone.
    That is the same gate retraction uses, deliberately: two mechanisms deciding
    currency differently is how a corpus starts contradicting itself.
    """

    __tablename__ = "knowledge_document_support"
    __table_args__ = (
        UniqueConstraint(
            "document_id", "revision_id", "slot_id",
            name="uq_knowledge_document_support",
        ),
    )

    support_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    document_id: Mapped[str] = mapped_column(Text, nullable=False)
    #: The state this document carries. A claim whose structure moved has a new
    #: revision, and a document supports the one it actually produced.
    revision_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("knowledge_revisions.revision_id", ondelete="CASCADE"),
        nullable=False,
    )
    #: The document revision this state was last **extracted** at — not
    #: necessarily the revision the document is at now.
    #:
    #: The two come apart whenever a section the stable-slot guard protected was
    #: not re-extracted: the document moved on, that section did not change, and
    #: nobody read the claim out of the newer revision. Writing the current
    #: revision here would fabricate a sighting.
    #:
    #: Named for what it is. `document_revision` on a table called *current
    #: support* reads as "the revision that currently carries it", which is a
    #: claim this column cannot make and a citation must not repeat.
    observed_document_revision: Mapped[str] = mapped_column(
        Text, nullable=False, default=""
    )
    document_path: Mapped[str] = mapped_column(Text, nullable=False, default="")
    #: Where in the document, for provenance. Never a reason to remove a row.
    slot_id: Mapped[str] = mapped_column(Text, nullable=False, default="")
    observed_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )


# A citation asks "which documents carry this state", so that is the shape.
Index(
    "knowledge_document_support_by_revision",
    KnowledgeDocumentSupport.revision_id,
    KnowledgeDocumentSupport.document_id,
)


class KnowledgeExtractionState(KnowledgeBase):
    """That one document was extracted, under one extraction, at one content.

    The knowledge counterpart to the data layer's `ingestion_index_state`, and
    it exists for the same reason: without it every run redoes every document.
    On the data side that costs an embedding; here it costs a model call whose
    answer is phrased slightly differently each time, so redoing an unchanged
    document is what makes assertions drift.

    Keyed on the document alone: a document has one extraction state, and the
    version it was last extracted under is what the row records rather than
    what it is keyed by. Two rows for one document would mean two answers to
    "may this be skipped".
    """

    __tablename__ = "knowledge_extraction_state"

    document_id: Mapped[str] = mapped_column(Text, primary_key=True)
    content_hash: Mapped[str] = mapped_column(Text, nullable=False)
    extraction_version: Mapped[str] = mapped_column(Text, nullable=False)
    run_id: Mapped[str] = mapped_column(Text, nullable=False)
    assertion_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    extracted_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )


class KnowledgeReview(KnowledgeBase):
    __tablename__ = "knowledge_reviews"
    __table_args__ = (
        CheckConstraint(
            "decision IN ('approved', 'rejected', 'edited', 'merged')",
            name="ck_knowledge_reviews_decision",
        ),
    )

    review_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    identity: Mapped[str] = mapped_column(Text, ForeignKey("knowledge.identity", ondelete="CASCADE"), nullable=False)
    reviewer_id: Mapped[str] = mapped_column(Text, nullable=False)
    decision: Mapped[str] = mapped_column(Text, nullable=False)
    comment: Mapped[str | None] = mapped_column(Text)
    changes: Mapped[dict[str, Any] | None] = mapped_column(_JSON)
    reviewed_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)


Index("knowledge_reviews_by_identity", KnowledgeReview.identity, KnowledgeReview.reviewed_at)


class KnowledgeMetadata(KnowledgeBase):
    __tablename__ = "knowledge_metadata"

    identity: Mapped[str] = mapped_column(Text, ForeignKey("knowledge.identity", ondelete="CASCADE"), primary_key=True)
    attributes: Mapped[dict[str, Any]] = mapped_column(_JSON, nullable=False, default=dict)
    schema_version: Mapped[str] = mapped_column(Text, nullable=False, default="v1")
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
