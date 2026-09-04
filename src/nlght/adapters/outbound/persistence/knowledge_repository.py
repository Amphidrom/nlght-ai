# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""PostgreSQL knowledge graph adapter.

Short transactions only; no external backend call happens here. Confidence
merging follows the prototype's Bayesian odds multiplication so independent
sightings reinforce one another (see ``reinforce_confidence``).
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import and_, delete, func, or_, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from sqlalchemy.orm import aliased
from sqlalchemy.sql.elements import ColumnElement

from nlght.adapters.outbound.persistence.knowledge_models import (
    Knowledge,
    KnowledgeAssertion,
    KnowledgeAssertionLineage,
    KnowledgeAssertionObservation,
    KnowledgeDecision,
    KnowledgeDocumentSupport,
    KnowledgeEquivalenceAssessment,
    KnowledgeEvidence,
    KnowledgeExtractionState,
    KnowledgeFact,
    KnowledgeMetadata,
    KnowledgeObservation,
    KnowledgePattern,
    KnowledgeProposition,
    KnowledgeReview,
    KnowledgeRevision,
    KnowledgeRule,
    KnowledgeSlot,
    KnowledgeSlotAnchor,
    KnowledgeSlotObservation,
    KnowledgeVariant,
)
from nlght.core.knowledge import (
    APPROVED,
    DECISION,
    FACT,
    FINGERPRINT_VERSION,
    NEEDS_REVIEW,
    NORMATIVE,
    PATTERN,
    RULE,
    SUPERSEDES,
    AssertionState,
    Candidate,
    DecisionPayload,
    DocumentOutcome,
    EntityKeyVersionMismatch,
    ExtractionState,
    ExtractionStateWrite,
    FactPayload,
    Judge,
    KnowledgeKindConflictError,
    KnowledgeObject,
    KnowledgePayload,
    KnowledgeStatistics,
    KnowledgeWrite,
    LineageReport,
    LineageWrite,
    Observation,
    ObservedSection,
    PatternPayload,
    Proposition,
    RecordedLineage,
    RetiredAssertion,
    Review,
    ReviewConflictError,
    ReviewDecision,
    RulePayload,
    SlotAnchor,
    SlotSupport,
    SupportingEvidence,
    canonical_term,
    current_anchor,
    degraded,
    fold_whitespace,
    match_proposition,
    material_content,
    new_assertion_id,
    payload_for,
    reinforce_confidence,
    resolve_slot,
    resolve_variant,
    semantic_change,
    settled,
    state_digest,
)
from nlght.core.knowledge.equivalence import (
    UNJUDGED,
    AssertionObservation,
    EquivalenceJudge,
    continuation,
    equivalent,
)

# Two functions named `settled` answer the same question about different
# subjects: one compares two wordings, the other two whole propositions. The
# package exports the first, so the second is named at the import.
from nlght.core.knowledge.equivalence import settled as settled_equivalence
from nlght.ports.outbound.knowledge_repository import KnowledgeRepository

#: Verdicts no judge produced. Recording a judge against one would name somebody
#: for a decision they never made.
_UNASKED = frozenset({UNJUDGED})

logger = logging.getLogger(__name__)

_PAYLOAD_TABLES = {
    FACT: KnowledgeFact,
    RULE: KnowledgeRule,
    PATTERN: KnowledgePattern,
    DECISION: KnowledgeDecision,
}


def _utcnow() -> datetime:
    return datetime.now(UTC)


# The review queue's order: least confident first, because the extractor's own
# doubt is the best available guide to where a reviewer's attention is worth
# most; newest first among equals; identity last so the order is total. Paging
# and prev/next navigation both depend on there being no ties left over.
_QUEUE_ORDER = (
    Knowledge.confidence.asc(),
    Knowledge.created_at.desc(),
    Knowledge.identity.asc(),
)


def _payload_columns(write: KnowledgeWrite) -> dict[str, Any]:
    payload = write.payload
    if payload is None:
        raise TypeError("a legacy payload row requires a payload")
    return _payload_fields(payload)


def _payload_fields(payload: KnowledgePayload) -> dict[str, Any]:
    if isinstance(payload, FactPayload):
        return {"subject": payload.subject, "predicate": payload.predicate, "object": payload.object}
    if isinstance(payload, RulePayload):
        return {
            "rule_text": payload.rule_text,
            "subject": payload.subject,
            "rule_property": payload.rule_property,
        }
    if isinstance(payload, PatternPayload):
        return {"pattern_name": payload.pattern_name, "description": payload.description}
    if isinstance(payload, DecisionPayload):
        return {
            "decision": payload.decision,
            "effect": payload.effect,
            "subject": payload.subject,
            "decision_type": payload.decision_type,
        }
    raise TypeError(f"unsupported knowledge payload: {type(payload).__name__}")


def _extraction_of(section: ObservedSection) -> str:
    """The extraction version this section's claims were produced under.

    One run asks one question, so every claim in a section carries the same
    version; the first that has one answers for the section. Empty where the
    section produced nothing, which is a section that cannot have gained a claim
    anyway.
    """
    for _, lineage in section.assertions:
        if lineage is not None and lineage.extraction_version:
            return str(lineage.extraction_version)
    return ""


def _wordings_of(held: Mapping[str, set[tuple[str, str]]], slot_id: str) -> set[str]:
    """The folded wordings a slot already carries, whatever version read them."""
    return {wording for wording, _ in held.get(slot_id, set())}


def _versions_of(held: Mapping[str, set[tuple[str, str]]], slot_id: str) -> set[str]:
    """The extraction versions a slot has been read under."""
    return {version for _, version in held.get(slot_id, set())}


#: A later revision of the same variant, aliased so a correlated subquery can ask
#: about the node's own revision without colliding with it.
orm_aliased_revision = aliased(KnowledgeRevision)


def _retired() -> ColumnElement[bool]:
    """Whether the claim this node stands for has been retired.

    One hop: the node names its revision, the revision its variant, the variant
    its assertion — and retirement is recorded on the assertion (ADR-0044).

    It used to be "every assertion carrying this identity", quantified over the
    revisions that happened to hash to the node's address, because there was no
    edge to follow. The plural was never real: a claim two documents assert is
    *one* assertion, and whether a second document still carries it is decided
    inside retraction, before `retired_at` is ever set.

    A node that names no revision is not retired — one written before lineage
    existed, one written by a path that keeps none, or one whose pointer the
    migration could not prove. Absence of a record is not evidence of withdrawal.
    """
    return (
        select(KnowledgeAssertion.assertion_id)
        .join(KnowledgeVariant, KnowledgeVariant.assertion_id == KnowledgeAssertion.assertion_id)
        .join(KnowledgeRevision, KnowledgeRevision.variant_id == KnowledgeVariant.variant_id)
        .where(
            KnowledgeRevision.revision_id == Knowledge.revision_id,
            KnowledgeAssertion.retired_at.is_not(None),
        )
        .exists()
    )


def _superseded() -> ColumnElement[bool]:
    """Whether this node records a state the claim has since moved past.

    A claim whose structure moves leaves a second node — the first keeps the
    state the source used to be in. Both were canonical, and a retrieval could
    return "must not exceed 30 seconds" beside "must not exceed 60 seconds" as
    two live answers to one question. That is the same failure as returning a
    retired claim: a state the corpus has moved past is not what the corpus says.

    The node names its state, so the question is exactly "is this state the
    latest of its variant" — one hop, and an exact answer.

    It used to be "no revision carrying this node's fingerprint is the deepest of
    its variant", which is a set standing in for the edge that did not exist.
    Being a *set* it could only ever be conservative, and it had to be: two
    revisions may legitimately carry one fingerprint — a revision is new when the
    observed wording moves, and the structure it hashes may not have — so the
    hash could not say which of them the node was.

    A node that names no revision is not superseded, for the same reason it is
    not retired.
    """
    return (
        select(orm_aliased_revision.revision_id)
        .join(
            KnowledgeRevision,
            KnowledgeRevision.variant_id == orm_aliased_revision.variant_id,
        )
        .where(
            orm_aliased_revision.revision_id == Knowledge.revision_id,
            KnowledgeRevision.revision > orm_aliased_revision.revision,
        )
        .exists()
    )


class SqlAlchemyKnowledgeRepository(KnowledgeRepository):
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    # -- reads ---------------------------------------------------------------

    async def _payload_of(self, session: AsyncSession, node: Knowledge) -> dict[str, Any]:
        """What this claim says, whole where the corpus has it whole.

        Reached through the revision this node names, and not through a column on
        the node. A node has one current state; an assertion has many revisions
        and each recorded the form the claim took at the time, so the structure
        belongs to the state. A `proposition_id` on the node would have read as
        "one proposition per assertion" and quietly flattened that history — and
        history is what this whole design is for.

        The join is `revision_id`, which is a relationship. It used to be the
        fingerprint and a `deepest first` ordering over everything hashing to it,
        which is a guess dressed as a lookup.

        The legacy table stays the answer for the kinds that have no proposition,
        for everything extracted before propositions existed, and for a candidate
        with no lineage at all — that last one has no revision to hang a
        structure on, so an n-ary claim written that way is stored complete in
        `knowledge_propositions` and read back through the payload it does not
        fit. That gap is real and narrow, and closing it means giving those
        candidates lineage rather than giving the node a column.
        """
        proposition_id = (
            (
                await session.execute(
                    select(KnowledgeRevision.proposition_id).where(
                        KnowledgeRevision.revision_id == node.revision_id
                    )
                )
            ).scalars().first()
            if node.revision_id is not None
            else None
        )
        if proposition_id is not None:
            proposition = await session.get(KnowledgeProposition, proposition_id)
            if proposition is not None:
                return dict(proposition.fields)

        table = _PAYLOAD_TABLES.get(node.kind)
        if table is None:
            return {}
        row = await session.get(table, node.identity)
        if row is None:
            return {}
        return {
            column.name: getattr(row, column.name)
            for column in table.__table__.columns
            if column.name != "identity"
        }

    async def _observed_text(self, session: AsyncSession, node: Knowledge) -> str:
        """What the source said, in the state this node stands for.

        The node names its revision, so the wording is one hop away and never has
        to be reassembled from the payload — which is the whole of ADR-0048
        applied to the read side.

        It used to be the deepest revision carrying the node's fingerprint, and
        that reached the right row for the reason a set reaches it: usually. A
        claim re-observed with the same structure keeps one node and gains
        revisions, and which of them the node stood for was not recorded
        anywhere. Now it is.
        """
        if node.revision_id is None:
            return ""
        return str(
            (
                await session.execute(
                    select(KnowledgeRevision.text).where(
                        KnowledgeRevision.revision_id == node.revision_id
                    )
                )
            ).scalar()
            or ""
        )

    async def _support(
        self, session: AsyncSession, node: Knowledge
    ) -> tuple[str, tuple[SupportingEvidence, ...]]:
        """The claim behind this node, and the documents that carry its state now.

        One hop each, because the node names its revision (ADR-0051): the
        revision's variant names the assertion, and support rows point straight
        at the revision. Before that relationship existed this could only have
        been asked as "evidence of some revision hashing to this address", which
        is the set logic that made the question unanswerable.

        Read from `knowledge_document_support` and **not** from the evidence.
        Evidence is append-only and answers "was observed here", which is the
        right shape for an audit and the wrong one for a citation: a document
        that has since removed the sentence stays in it forever, so an answer
        built from evidence would send a reader to a file that no longer says it
        while the claim is alive through a second document (ADR-0053).

        Ordered by document and section so two runs cite in the same order. It is
        an order for reading, not a ranking: no sighting here outranks another,
        and a claim two documents assert has two supports and no owner.
        """
        if node.revision_id is None:
            return "", ()
        assertion_id = (
            await session.execute(
                select(KnowledgeVariant.assertion_id)
                .join(
                    KnowledgeRevision,
                    KnowledgeRevision.variant_id == KnowledgeVariant.variant_id,
                )
                .where(KnowledgeRevision.revision_id == node.revision_id)
            )
        ).scalars().first()
        rows = (
            await session.execute(
                select(
                    KnowledgeDocumentSupport.document_id,
                    KnowledgeDocumentSupport.observed_document_revision,
                    KnowledgeDocumentSupport.document_path,
                    KnowledgeDocumentSupport.slot_id,
                )
                .where(KnowledgeDocumentSupport.revision_id == node.revision_id)
                .order_by(
                    KnowledgeDocumentSupport.document_id.asc(),
                    KnowledgeDocumentSupport.slot_id.asc(),
                )
            )
        ).all()
        return str(assertion_id or ""), tuple(
            SupportingEvidence(
                document_id=str(document_id),
                observed_document_revision=str(document_revision or ""),
                document_path=str(document_path or ""),
                slot_id=str(slot_id or ""),
            )
            for document_id, document_revision, document_path, slot_id in rows
        )

    async def _to_object(self, session: AsyncSession, node: Knowledge) -> KnowledgeObject:
        meta = await session.get(KnowledgeMetadata, node.identity)
        assertion_id, support = await self._support(session, node)
        return KnowledgeObject(
            identity=node.identity,
            kind=node.kind,
            type=node.type,
            confidence=node.confidence,
            payload=await self._payload_of(session, node),
            review_required=node.review_required,
            review_reason=node.review_reason,
            merged_into=node.merged_into,
            product=node.product,
            version=node.version,
            metadata=dict(meta.attributes) if meta is not None else {},
            created_at=node.created_at,
            updated_at=node.updated_at,
            observed_text=await self._observed_text(session, node),
            assertion_id=assertion_id,
            revision_id=str(node.revision_id) if node.revision_id is not None else "",
            support=support,
        )

    async def resolve(
        self,
        identities: Sequence[str],
        *,
        include_quarantined: bool = False,
    ) -> list[KnowledgeObject]:
        if not identities:
            return []
        async with AsyncSession(self._engine) as session:
            statement = select(Knowledge).where(Knowledge.identity.in_(list(identities)))
            if not include_quarantined:
                # Three ways to stop being canonical: still awaiting or refused
                # by review, merged away into another assertion, or retired
                # because the source stopped saying it. Retrieval must see none
                # of them; a review surface must see all three.
                #
                # The third does not live on this row. Retirement is recorded on
                # the assertion, and the graph node has no flag for it — so a
                # claim a document had dropped went on being handed out by the
                # knowledge store, which a retrieval visibility test found. It
                # is answered by asking the lineage rather than by adding a
                # column, because the assertion is where the fact lives and two
                # places recording it would eventually disagree.
                #
                # And the fourth is not a state of the claim but a gap in the
                # record: migration `0018` could not prove which revision a
                # lineage-backed node stands for. It may be a state the corpus
                # moved past years ago, so it is withheld rather than guessed
                # at — unlike a node with no lineage at all, which has always
                # been canonical and stays so, because absence of a record is
                # not evidence of anything.
                statement = statement.where(
                    Knowledge.review_required.is_(False),
                    Knowledge.merged_into.is_(None),
                    Knowledge.revision_unresolved.is_(False),
                    ~_retired(),
                    ~_superseded(),
                )
            rows = (await session.execute(statement)).scalars().all()
            by_identity = {row.identity: row for row in rows}
            # Preserve caller order — the ranking comes from the search backend.
            return [
                await self._to_object(session, by_identity[identity])
                for identity in identities
                if identity in by_identity
            ]

    @staticmethod
    def _undecided() -> ColumnElement[bool]:
        """Quarantined, and not reviewed since it last changed.

        Rejection keeps ``review_required`` set — that is what keeps the
        assertion out of retrieval — so the flag alone cannot tell an unseen
        assertion from a rejected one, and a queue built on it can never be
        emptied. The decision history answers it instead: a review recorded at
        or after the node's last change means somebody has ruled on this exact
        version. Later evidence bumps ``updated_at`` past that review and
        deliberately reopens the item.
        """
        return ~(
            select(KnowledgeReview.review_id)
            .where(
                KnowledgeReview.identity == Knowledge.identity,
                KnowledgeReview.reviewed_at >= Knowledge.updated_at,
            )
            .exists()
        )

    async def pending_review(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> list[KnowledgeObject]:
        if limit < 1:
            raise ValueError("pending_review limit must be positive")
        if offset < 0:
            raise ValueError("pending_review offset must not be negative")
        async with AsyncSession(self._engine) as session:
            rows = (
                await session.execute(
                    select(Knowledge)
                    .where(
                        Knowledge.review_required.is_(True),
                        Knowledge.merged_into.is_(None),
                        self._undecided(),
                    )
                    .order_by(*_QUEUE_ORDER)
                    .limit(limit)
                    .offset(offset)
                )
            ).scalars().all()
            return [await self._to_object(session, row) for row in rows]

    async def pending_review_count(self) -> int:
        async with AsyncSession(self._engine) as session:
            total = await session.scalar(
                select(func.count())
                .select_from(Knowledge)
                .where(
                    Knowledge.review_required.is_(True),
                    Knowledge.merged_into.is_(None),
                    self._undecided(),
                )
            )
            return int(total or 0)

    async def queue_neighbours(self, identity: str) -> tuple[str | None, str | None]:
        """The assertions before and after this one in the queue order.

        A keyset comparison rather than an offset, so a reviewer's position
        survives other reviewers draining the queue underneath them. Deciding
        an item removes it, which is why "next" is resolved from the item's own
        sort key and not from a remembered index.
        """
        async with AsyncSession(self._engine) as session:
            node = await session.get(Knowledge, identity)
            if node is None:
                return (None, None)
            queued = (
                Knowledge.review_required.is_(True),
                Knowledge.merged_into.is_(None),
                self._undecided(),
            )
            confidence, created, ident = node.confidence, node.created_at, node.identity

            after = await session.scalar(
                select(Knowledge.identity)
                .where(
                    *queued,
                    or_(
                        Knowledge.confidence > confidence,
                        and_(Knowledge.confidence == confidence, Knowledge.created_at < created),
                        and_(
                            Knowledge.confidence == confidence,
                            Knowledge.created_at == created,
                            Knowledge.identity > ident,
                        ),
                    ),
                )
                .order_by(*_QUEUE_ORDER)
                .limit(1)
            )
            before = await session.scalar(
                select(Knowledge.identity)
                .where(
                    *queued,
                    or_(
                        Knowledge.confidence < confidence,
                        and_(Knowledge.confidence == confidence, Knowledge.created_at > created),
                        and_(
                            Knowledge.confidence == confidence,
                            Knowledge.created_at == created,
                            Knowledge.identity < ident,
                        ),
                    ),
                )
                .order_by(
                    Knowledge.confidence.desc(),
                    Knowledge.created_at.asc(),
                    Knowledge.identity.desc(),
                )
                .limit(1)
            )
            return (before, after)

    async def statistics(self) -> KnowledgeStatistics:
        async with AsyncSession(self._engine) as session:
            row = (
                await session.execute(
                    select(
                        func.count(),
                        func.count().filter(
                            Knowledge.review_required.is_(True), self._undecided()
                        ),
                        func.count().filter(
                            Knowledge.review_required.is_(False),
                            Knowledge.merged_into.is_(None),
                        ),
                        func.count().filter(Knowledge.merged_into.is_not(None)),
                        func.avg(Knowledge.confidence).filter(
                            Knowledge.review_required.is_(True), self._undecided()
                        ),
                        func.avg(Knowledge.confidence).filter(
                            Knowledge.review_required.is_(False),
                            Knowledge.merged_into.is_(None),
                        ),
                        func.count(func.distinct(Knowledge.kind)),
                        func.count(func.distinct(Knowledge.type)),
                    ).select_from(Knowledge)
                )
            ).one()
            return KnowledgeStatistics(
                total=int(row[0] or 0),
                pending=int(row[1] or 0),
                retrievable=int(row[2] or 0),
                merged=int(row[3] or 0),
                average_pending_confidence=float(row[4] or 0.0),
                average_retrievable_confidence=float(row[5] or 0.0),
                kinds=int(row[6] or 0),
                types=int(row[7] or 0),
            )

    # -- publication ---------------------------------------------------------
    #
    # What belongs in the search index and what must be taken out of it. Ordered
    # by identity, which is total, so paging is stable while the graph changes
    # under it.

    async def canonical_page(
        self, *, limit: int = 200, offset: int = 0
    ) -> list[KnowledgeObject]:
        """Assertions that may be found: reviewed, approved, not merged away."""
        if limit < 1:
            raise ValueError("canonical_page limit must be positive")
        if offset < 0:
            raise ValueError("canonical_page offset must not be negative")
        async with AsyncSession(self._engine) as session:
            rows = (
                await session.execute(
                    select(Knowledge)
                    .where(
                        Knowledge.review_required.is_(False),
                        Knowledge.merged_into.is_(None),
                    )
                    .order_by(Knowledge.identity.asc())
                    .limit(limit)
                    .offset(offset)
                )
            ).scalars().all()
            return [await self._to_object(session, row) for row in rows]

    async def withdrawn_page(
        self, *, limit: int = 200, offset: int = 0
    ) -> list[tuple[str, str]]:
        """Identity and kind of assertions that must not be findable.

        Only the two fields the removal needs. Loading a full object — payload,
        metadata, one query each — to then delete it by id would be work done
        for nothing, and this is the larger of the two sets while a review
        backlog exists.
        """
        if limit < 1:
            raise ValueError("withdrawn_page limit must be positive")
        if offset < 0:
            raise ValueError("withdrawn_page offset must not be negative")
        async with AsyncSession(self._engine) as session:
            rows = await session.execute(
                select(Knowledge.identity, Knowledge.kind)
                .where(
                    or_(
                        Knowledge.review_required.is_(True),
                        Knowledge.merged_into.is_not(None),
                    )
                )
                .order_by(Knowledge.identity.asc())
                .limit(limit)
                .offset(offset)
            )
            return [(identity, kind) for identity, kind in rows.all()]

    async def record_supersession(
        self,
        *,
        predecessors: Sequence[str],
        successors: Sequence[str],
        reason: str | None = None,
        recorded_by: str | None = None,
    ) -> int:
        """Record that one set of assertions took another's place.

        Sets rather than a pair, because succession is not one-to-one: a rule
        splits into two, or two fold into one, and both happen routinely in a
        corpus of policies. Every predecessor is linked to every successor and
        retired.

        A rewording does not come here — that keeps its assertion and adds a
        revision. This is the other case: propositions that say different
        things, where one took the other's place rather than arriving beside it.
        Whether it did cannot be read off the words, so it is written down with
        who decided it and why.

        The predecessors stay. Deleting one would take its evidence and its
        review history with it, and a citation naming it would find nothing.
        """
        if not predecessors or not successors:
            raise ValueError("a supersession needs at least one assertion on each side")
        overlap = set(predecessors) & set(successors)
        if overlap:
            raise ValueError(
                f"{', '.join(sorted(overlap))} cannot supersede itself"
            )

        now = _utcnow()
        async with AsyncSession(self._engine) as session, session.begin():
            known = {
                row.assertion_id: row
                for row in (
                    await session.execute(
                        select(KnowledgeAssertion).where(
                            KnowledgeAssertion.assertion_id.in_(
                                {*predecessors, *successors}
                            )
                        )
                    )
                ).scalars().all()
            }
            missing = [
                key for key in (*predecessors, *successors) if key not in known
            ]
            if missing:
                raise ValueError(f"no assertion {', '.join(sorted(set(missing)))}")

            existing = {
                (row.predecessor_id, row.successor_id)
                for row in (
                    await session.execute(
                        select(KnowledgeAssertionLineage).where(
                            KnowledgeAssertionLineage.predecessor_id.in_(predecessors),
                            KnowledgeAssertionLineage.successor_id.in_(successors),
                            KnowledgeAssertionLineage.relation == SUPERSEDES,
                        )
                    )
                ).scalars().all()
            }

            written = 0
            for predecessor in predecessors:
                for successor in successors:
                    if (predecessor, successor) in existing:
                        # Recording the same succession twice is a retry, not a
                        # second event.
                        continue
                    session.add(
                        KnowledgeAssertionLineage(
                            predecessor_id=predecessor,
                            successor_id=successor,
                            relation=SUPERSEDES,
                            reason=reason,
                            recorded_by=recorded_by,
                            recorded_at=now,
                        )
                    )
                    written += 1
                known[predecessor].retired_at = (
                    known[predecessor].retired_at or now
                )
            return written

    async def supersessions_of(self, assertion_id: str) -> tuple[str, ...]:
        """What took this assertion's place, in the order it was recorded."""
        async with AsyncSession(self._engine) as session:
            rows = (
                await session.execute(
                    select(KnowledgeAssertionLineage.successor_id)
                    .where(
                        KnowledgeAssertionLineage.predecessor_id == assertion_id,
                        KnowledgeAssertionLineage.relation == SUPERSEDES,
                    )
                    .order_by(KnowledgeAssertionLineage.recorded_at)
                )
            ).scalars().all()
        return tuple(str(row) for row in rows)

    async def lineage_report(self) -> LineageReport:
        """One picture of the corpus, for holding a later run against.

        Read-only and cheap: counts plus one digest over every assertion's key,
        case and state. It exists to be taken before a rerun and compared after,
        which is the only way the design's acceptance metric — two runs over an
        unchanged source change nothing — can actually be checked.

        The graph and the lineage are keyed to one another now (ADR-0051), so
        "how much of the graph names no state" is a lookup rather than a
        difference between two tallies. It is reported as two numbers, because
        the two causes are unrelated: a node with no lineage at all cannot name a
        business question and never will, while a node marked
        `revision_unresolved` is a migration that has not finished and falls to
        zero as the corpus is re-ingested.
        """
        async with AsyncSession(self._engine) as session:
            # Every state with where it stands. The provenance is what lets a
            # later comparison say whether a revision appeared in a document
            # somebody edited or in one nobody touched — the difference between
            # the run working and the corpus drifting, which counts alone could
            # never tell apart.
            rows = (
                await session.execute(
                    select(
                        KnowledgeAssertion.assertion_id,
                        KnowledgeAssertion.entity_key,
                        KnowledgeAssertion.retired_at,
                        KnowledgeAssertion.kind,
                        KnowledgeVariant.variant_id,
                        KnowledgeRevision.revision,
                        KnowledgeRevision.review_state,
                        KnowledgeRevision.text,
                        KnowledgeEvidence.document_id,
                        KnowledgeEvidence.slot_id,
                        KnowledgeEvidence.observed_at,
                    )
                    .join(
                        KnowledgeVariant,
                        KnowledgeVariant.assertion_id == KnowledgeAssertion.assertion_id,
                    )
                    .join(
                        KnowledgeRevision,
                        KnowledgeRevision.variant_id == KnowledgeVariant.variant_id,
                    )
                    .outerjoin(
                        KnowledgeEvidence,
                        KnowledgeEvidence.revision_id == KnowledgeRevision.revision_id,
                    )
                )
            ).all()

            # One row per revision, keeping its most recent sighting: a claim
            # asserted by several documents has one state and several places it
            # was seen, and the latest is the one a comparison is about.
            latest: dict[tuple[str, int], tuple[Any, AssertionState]] = {}
            # And every document each revision was seen in, because "the latest"
            # is the wrong answer to "which documents carry this claim".
            seen_in: dict[tuple[str, int], set[str]] = {}
            for (
                assertion_id, entity_key, retired_at, kind, variant_id,
                revision, review_state, text, document_id, slot_id, observed_at,
            ) in rows:
                state = AssertionState(
                    assertion_id=str(assertion_id),
                    entity_key=str(entity_key),
                    variant_id=str(variant_id),
                    revision=int(revision),
                    review_state=str(review_state),
                    document_id=str(document_id or ""),
                    slot_id=str(slot_id or ""),
                    retired=retired_at is not None,
                    kind=str(kind or ""),
                    text=str(text or ""),
                )
                key = (str(variant_id), int(revision))
                if document_id:
                    seen_in.setdefault(key, set()).add(str(document_id))
                seen = latest.get(key)
                if seen is None or (observed_at is not None and observed_at > seen[0]):
                    latest[key] = (observed_at, state)
            assertion_states = tuple(
                replace(state, document_ids=tuple(sorted(seen_in.get(key, set()))))
                for key, (_, state) in latest.items()
            )
            states = [
                (state.entity_key, state.variant_id, state.revision, state.review_state)
                for state in assertion_states
            ]

            # Each document at the revision it was last seen at. A document whose
            # revision moves between two pictures is one somebody edited, and
            # that is how movement gets attributed without anybody saying what
            # they changed.
            documents = {
                str(document_id): str(document_revision)
                for document_id, document_revision in (
                    await session.execute(
                        select(
                            KnowledgeEvidence.document_id,
                            KnowledgeEvidence.document_revision,
                        )
                        .order_by(KnowledgeEvidence.observed_at.asc())
                    )
                ).all()
                if document_id
            }

            async def _count(model: Any, *where: Any) -> int:  # noqa: ANN401
                statement = select(func.count()).select_from(model)
                for clause in where:
                    statement = statement.where(clause)
                return int((await session.execute(statement)).scalar_one())

            async def _by_kind(column: Any, model: Any) -> dict[str, int]:  # noqa: ANN401
                return {
                    str(kind): int(count)
                    for kind, count in (
                        await session.execute(
                            select(column, func.count()).select_from(model).group_by(column)
                        )
                    ).all()
                }

            async def _by_kind_where(
                column: Any,  # noqa: ANN401
                model: Any,  # noqa: ANN401
                *where: Any,  # noqa: ANN401
            ) -> dict[str, int]:
                statement = select(column, func.count()).select_from(model)
                for clause in where:
                    statement = statement.where(clause)
                return {
                    str(kind): int(count)
                    for kind, count in (await session.execute(statement.group_by(column))).all()
                }

            graph_by_kind = await _by_kind(Knowledge.kind, Knowledge)
            # Graph nodes that name no state. Asked of the relationship, not of
            # the content hash: it was "no revision hashes to this address",
            # which is the set logic ADR-0051 removed and which could not
            # distinguish a node with no lineage from one whose state is
            # unrecorded. Those two need separate numbers — the first is a
            # candidate that could never name a business question, the second is
            # a migration that has not finished — so they get them.
            #
            # (It was also once "graph count minus assertion count", which stopped
            # being the same question the moment identity left the content hash: a
            # live run reported seven such nodes while the persist step reported
            # `no_entity_key=0`, and the two cannot both be right.)
            unreachable = await _by_kind_where(
                Knowledge.kind,
                Knowledge,
                Knowledge.revision_id.is_(None),
                Knowledge.revision_unresolved.is_(False),
            )
            unresolved_revision = await _count(
                Knowledge, Knowledge.revision_unresolved.is_(True)
            )
            # Canonical nodes nothing currently carries. Not the same as "no
            # lineage": these name a state, and no document has been read in
            # full since migration 0019 to say it still asserts it. A citation
            # for them would have nowhere to point.
            uncited = await _count(
                Knowledge,
                Knowledge.revision_id.is_not(None),
                ~select(KnowledgeDocumentSupport.support_id)
                .where(KnowledgeDocumentSupport.revision_id == Knowledge.revision_id)
                .exists(),
            )
            versions = {
                str(version): int(count)
                for version, count in (
                    await session.execute(
                        select(KnowledgeAssertion.entity_key_version, func.count())
                        .group_by(KnowledgeAssertion.entity_key_version)
                    )
                ).all()
            }

            # The latest reading of each slot. This is what lets a comparison
            # say *why* a movement was expected: the slot's own column holds
            # what the section says now and is overwritten every run, so only
            # these readings can show that the section a claim stood in was
            # actually the one that was edited.
            #
            # The document revision each was read at comes with it, and it is
            # what makes the reading *usable*. A fingerprint that is equal on
            # both sides of a comparison means "the section did not move" only
            # where the section was read on both sides; a slot whose section was
            # deleted is never read again and keeps its last fingerprint
            # unchanged forever, which read identically and meant the opposite.
            readings = (
                await session.execute(
                    select(
                        KnowledgeSlotObservation.slot_id,
                        KnowledgeSlotObservation.parser_input_fingerprint,
                        KnowledgeSlotObservation.document_revision,
                    ).order_by(KnowledgeSlotObservation.observed_at.asc())
                )
            ).all()
            slot_inputs = {slot_id: fingerprint for slot_id, fingerprint, _ in readings}
            slot_observed_at = {slot_id: revision for slot_id, _, revision in readings}

            # What each document is called, from the sighting that saw it.
            #
            # `document_id` is a hash, so a report naming only that could say a
            # document moved and never which one. The obvious source was
            # `ingestion_documents` and it is the wrong one: a knowledge-only
            # flow leaves that table empty, so every name came back missing —
            # and joining the two lifecycles to borrow a string is a decision of
            # its own rather than a lookup.
            #
            # The latest sighting wins, so a renamed document is named by what it
            # is called now while the older evidence keeps what it saw.
            document_paths = {
                document_id: path
                for document_id, path in (
                    await session.execute(
                        select(
                            KnowledgeEvidence.document_id,
                            KnowledgeEvidence.document_path,
                        )
                        .where(KnowledgeEvidence.document_path != "")
                        .order_by(KnowledgeEvidence.observed_at.asc())
                    )
                ).all()
            }

            # The five numbers that separate historical debt from a new breach.
            revisions_without_proposition = await _count(
                KnowledgeRevision, KnowledgeRevision.proposition_id.is_(None)
            )
            observation_rows = await _count(KnowledgeAssertionObservation)
            observations_without_proposition = await _count(
                KnowledgeAssertionObservation,
                KnowledgeAssertionObservation.representation == {},
            )
            proposition_rows = await _count(KnowledgeProposition)
            referenced = set(
                (
                    await session.execute(
                        select(KnowledgeRevision.proposition_id).where(
                            KnowledgeRevision.proposition_id.is_not(None)
                        )
                    )
                ).scalars()
            )
            assessments = await _count(KnowledgeEquivalenceAssessment)

            assertions = await _count(KnowledgeAssertion)
            variants = await _count(KnowledgeVariant)
            evidence = await _count(KnowledgeEvidence)
            evidence_without_document = await _count(
                KnowledgeEvidence, KnowledgeEvidence.document_id == ""
            )

        deepest: dict[str, int] = {}
        review_states: dict[str, int] = {}
        for _, variant_id, revision, review_state in states:
            deepest[variant_id] = max(deepest.get(variant_id, 0), revision)
            review_states[review_state] = review_states.get(review_state, 0) + 1

        missing = {kind: count for kind, count in unreachable.items() if count > 0}

        return LineageReport(
            assertions=assertions,
            variants=variants,
            revisions=len(states),
            evidence=evidence,
            variants_beyond_first_revision=sum(1 for depth in deepest.values() if depth > 1),
            deepest_revision=max(deepest.values(), default=0),
            states=assertion_states,
            documents=documents,
            document_paths=document_paths,
            slot_inputs=slot_inputs,
            slot_observed_at=slot_observed_at,
            graph_assertions=sum(graph_by_kind.values()),
            graph_without_lineage=sum(missing.values()),
            graph_without_lineage_by_kind=missing,
            graph_with_unresolved_revision=unresolved_revision,
            claims_without_current_support=uncited,
            review_states=review_states,
            entity_key_versions=versions,
            evidence_without_document=evidence_without_document,
            revisions_without_proposition=revisions_without_proposition,
            observations=observation_rows,
            observations_without_proposition=observations_without_proposition,
            propositions=proposition_rows,
            unreferenced_propositions=max(proposition_rows - len(referenced), 0),
            equivalence_assessments=assessments,
            state_digest=state_digest(states),
        )

    async def record_lineage(self, write: LineageWrite) -> RecordedLineage:
        """Place one sighting: which assertion, which case, which state.

        Four decisions in one short transaction, each answered before the next:

        1. **Which assertion.** The entity key resolves deterministically. A hit
           is that assertion; no score is consulted, because a similarity
           deciding identity is what the surrogate exists to prevent.
        2. **Which case.** The scope resolves against the assertion's existing
           variants by the ladder in `resolve_variant` — exact, then a single
           unambiguous widening or narrowing, then a new one marked ambiguous.
        3. **Which state.** A fingerprint that matches the variant's latest
           revision is the same state seen again: no revision is written, which
           is what makes a second run over an unchanged source leave the corpus
           alone.
        4. **Whether the approval survives.** Only when a revision is written,
           and only from the caller's `normative_change` — a rewording keeps it,
           a moved value does not.

        Evidence is recorded either way and is unique per revision, document and
        run, so a retried step that re-writes the same sighting does not record
        it twice.
        """
        async with AsyncSession(self._engine, expire_on_commit=False) as session, session.begin():
            return await self._record_lineage(session, write, _utcnow())

    @staticmethod
    async def _candidates(
        session: AsyncSession, write: LineageWrite
    ) -> list[Candidate]:
        """What this proposition could be continuing.

        Two populations, because the ladder needs both: everything already
        asserted by this document — which is where a claim that merely moved
        between sections is found — and the assertions carrying this entity key
        anywhere, which is how the same claim asserted by a second document
        reinforces one node instead of duplicating it.

        Each candidate is described by its *current* wording, not its first: the
        question is what the corpus says now.
        """
        rows = (
            await session.execute(
                select(
                    KnowledgeAssertion.assertion_id,
                    KnowledgeAssertion.entity_key,
                    KnowledgeAssertion.entity_key_version,
                    KnowledgeAssertion.kind,
                    KnowledgeRevision.revision,
                    KnowledgeRevision.text,
                    KnowledgeEvidence.document_id,
                    KnowledgeEvidence.slot_id,
                )
                .join(KnowledgeVariant, KnowledgeVariant.assertion_id == KnowledgeAssertion.assertion_id)
                .join(KnowledgeRevision, KnowledgeRevision.variant_id == KnowledgeVariant.variant_id)
                .join(KnowledgeEvidence, KnowledgeEvidence.revision_id == KnowledgeRevision.revision_id)
                .where(
                    KnowledgeAssertion.retired_at.is_(None),
                    or_(
                        KnowledgeEvidence.document_id == write.document_id,
                        KnowledgeAssertion.entity_key == write.entity.key,
                    ),
                )
            )
        ).all()

        latest: dict[str, tuple[int, str, str, str, str, str]] = {}
        slots: dict[str, set[str]] = {}
        documents: dict[str, str] = {}
        for (
            assertion_id, entity_key, key_version, kind, revision, text, document_id, slot_id
        ) in rows:
            seen = latest.get(assertion_id)
            if seen is None or revision > seen[0]:
                latest[assertion_id] = (
                    revision, entity_key, key_version, text, document_id, kind
                )
            if slot_id:
                slots.setdefault(assertion_id, set()).add(slot_id)
            # An assertion asserted by several documents is still one assertion;
            # the document recorded here is only what lets the ladder tell "this
            # document's claims" from the rest.
            if document_id == write.document_id:
                documents[assertion_id] = document_id

        return [
            Candidate(
                assertion_id=assertion_id,
                entity_key=entity_key,
                entity_key_version=key_version,
                text=text,
                document_id=documents.get(assertion_id, document_id),
                slot_ids=frozenset(slots.get(assertion_id, ())),
                kind=kind,
            )
            for assertion_id, (
                _, entity_key, key_version, text, document_id, kind
            ) in latest.items()
        ]

    async def _record_lineage(
        self, session: AsyncSession, write: LineageWrite, now: datetime
    ) -> RecordedLineage:
        """The lineage write, without its transaction — see `_upsert`."""
        # A key the corpus already holds under a different scheme stops the write
        # before anything is decided. Both silent readings are wrong: treating it
        # as a match reads an old scheme's key as if the new one had produced it,
        # and treating it as new leaves the corpus holding one business question
        # twice. This sits ahead of the ladder rather than behind it, because the
        # ladder would often continue the claim on its wording alone and a corpus
        # half-migrated one sentence at a time is what the version exists to
        # prevent. Migration is a deliberate act; this is what makes somebody
        # perform it.
        other_scheme = (
            await session.execute(
                select(KnowledgeAssertion.entity_key_version).where(
                    KnowledgeAssertion.entity_key == write.entity.key,
                    KnowledgeAssertion.entity_key_version != write.entity.version,
                )
            )
        ).scalars().first()
        if other_scheme is not None:
            raise EntityKeyVersionMismatch(
                f"entity key '{write.entity.key}' is already recorded under key "
                f"scheme '{other_scheme}' and arrived under '{write.entity.version}'; "
                f"migrate the existing assertions before writing the new scheme"
            )

        # Which assertion this is, decided by where it stands and what it says —
        # never by the fields the model chose to describe it with. Those change
        # between runs over an unedited sentence, and identity followed them.
        candidates = await self._candidates(session, write)
        decision = match_proposition(
            entity_key=write.entity.key,
            entity_key_version=write.entity.version,
            text=write.text,
            slot_id=write.slot_id,
            document_id=write.document_id,
            candidates=candidates,
            kind=write.kind,
        )
        # The ladder first, always. An equivalence judgement is consulted only
        # where exact agreement found nothing, so a model can never override what
        # the source plainly said — and it was reached before this transaction
        # opened, because asking a model here would hold row locks across the
        # network.
        #
        # Which is exactly why it is checked again here. The judgement was formed
        # against the slot this section was *expected* to resolve to; the
        # authoritative placement happens inside this transaction and can land
        # elsewhere — the registry moved, or recovery answered differently.
        # Applying it anyway would let a verdict reached about slot A continue an
        # assertion standing in slot B, which is the one thing the slot scope
        # exists to prevent.
        #
        # Judged yes is not applied yes. The assessment stays in the audit either
        # way, because the judge really was asked and really did answer; what it
        # does not get to do is outlive the premise it was asked under.
        continues = decision.assertion_id
        if continues is None and write.continues is not None:
            # Revalidated against the premise it was actually judged under, and
            # there are two. A candidate found in this slot was judged as a claim
            # standing here, so it has to still be standing here. One found
            # across documents was judged as a live claim sharing enough
            # structure to be worth asking about — slot membership was never part
            # of that, and demanding it would discard every cross-document
            # verdict as stale, which is how the widened reach first appeared to
            # do nothing at all.
            still_here = {
                candidate.assertion_id
                for candidate in candidates
                if write.slot_id is not None and write.slot_id in candidate.slot_ids
            }
            if write.continues_scope != "corpus" and write.continues in still_here:
                continues = write.continues
            elif write.continues_scope == "corpus" and write.continues in {
                candidate.assertion_id
                for candidate in await self._plausible(session, write, exclude=set())
            }:
                # Re-derived here rather than trusted from before the
                # transaction: the claim may have been retired or moved out of
                # reach while the model was being asked.
                continues = write.continues
            else:
                logger.info(
                    "knowledge.equivalence.stale | assertion=%s slot=%s document=%s run=%s",
                    write.continues, write.slot_id, write.document_id, write.run_id,
                )
        assertion = (
            (
                await session.execute(
                    select(KnowledgeAssertion).where(
                        KnowledgeAssertion.assertion_id == continues
                    )
                )
            ).scalar_one_or_none()
            if continues is not None
            else None
        )
        if assertion is None:
            assertion = KnowledgeAssertion(
                assertion_id=new_assertion_id(),
                entity_key_version=write.entity.version,
                entity_key=write.entity.key,
                kind=write.kind,
                created_at=now,
            )
            session.add(assertion)
            await session.flush()

        existing_variants = (
            await session.execute(
                select(KnowledgeVariant).where(
                    KnowledgeVariant.assertion_id == assertion.assertion_id,
                    KnowledgeVariant.retired_at.is_(None),
                )
            )
        ).scalars().all()
        match = resolve_variant(
            write.entity,
            write.scope,
            {row.variant_id: row.scope for row in existing_variants},
        )
        variant = next(
            (row for row in existing_variants if row.variant_id == match.variant_id), None
        )
        if variant is None:
            variant = KnowledgeVariant(
                variant_id=match.variant_id,
                assertion_id=assertion.assertion_id,
                scope=list(write.scope),
                resolution=match.resolution,
                created_at=now,
            )
            session.add(variant)
            await session.flush()
        else:
            # A widening or narrowing keeps the variant and moves its scope:
            # the case is the same case, now covering different people.
            variant.scope = list(write.scope)

        latest = (
            await session.execute(
                select(KnowledgeRevision)
                .where(KnowledgeRevision.variant_id == variant.variant_id)
                .order_by(KnowledgeRevision.revision.desc())
                .limit(1)
            )
        ).scalar_one_or_none()

        # A revision is the state that was **observed**, so the observed wording
        # is what decides whether there is a new one (ADR-0048).
        #
        # It used to be "the same fingerprint *or* the same normalised text", and
        # the first half was wrong once the wording became authoritative. A fact's
        # fingerprint is its structured claim, which does not contain the
        # sentence: an edit that reworded a claim without moving its
        # subject/predicate/object hashed identically, so no revision was written
        # and the corpus went on quoting a sentence the document no longer
        # contained — measured live, on `defaults to the same as` becoming `is by
        # default identical to`.
        #
        # And the fingerprint may not decide it in the other direction either. A
        # model that describes an unedited sentence with different fields has
        # changed its description, not the state; writing a revision for that is
        # the drift ADR-0043 removed, arriving through the history instead of
        # through identity.
        #
        # So: the wording alone, folded for whitespace, because a parser may
        # rewrap a line and that changes no word. Anything a reader would see is
        # a new state — which is what a revision is for, and what keeps
        # `semantic_change` able to say whether the approval survives it.
        unchanged = latest is not None and (
            fold_whitespace(latest.text) == fold_whitespace(write.text)
        )
        # Every sighting is an observation, not only the ones a judgement was
        # asked about. Without this a candidate has no historical row to point
        # at, and the comparison would have to invent one — which is how a
        # sighting becomes a sighting per comparison.
        #
        # Reused where `_assess` already recorded it: the same sighting, seen
        # once, whatever was decided about it afterwards.
        observation_id = write.observation_id or await self._record_observation(
            session, write, write.slot_id or ""
        )

        # The complete structure this sighting carried, written before the
        # revision that names it. Content-addressed, so the same structure seen
        # in two documents is one row — storage economy and not identity: whether
        # two *different* structures are one claim is a judgement with a table of
        # its own.
        proposition_id = (
            await self._record_proposition(session, write.proposition)
            if write.proposition is not None
            else None
        )

        if latest is not None and unchanged:
            revision = latest
            if revision.observation_id is None and observation_id is not None:
                revision.observation_id = observation_id
            if revision.proposition_id is None and proposition_id is not None:
                # A revision written before propositions existed, seen again
                # unchanged. Attaching the structure it recorded is not a
                # rewrite: `unchanged` means this sighting says what that
                # revision already said, now in a form that holds all of it.
                revision.proposition_id = proposition_id
        else:
            # Whether the approval given to the previous state still covers this
            # one. The caller may say so outright; beyond that, only here is the
            # previous state known, so only here can the change be seen at all.
            #
            # A quantity that moved is a normative change: CHF 500 becoming CHF
            # 550 continues the same rule and a person approved five hundred, not
            # five fifty. A rewording that leaves the quantities alone is not, and
            # its approval carries — which is the whole reason the assertion id is
            # a surrogate. Compared as quantities rather than by resemblance
            # because resemblance inverts the two: the reworded rule scores 0.46
            # against its predecessor and the changed threshold 0.97.
            normative = write.normative_change or (
                latest is not None
                and material_content(latest.text) != material_content(write.text)
            )
            if proposition_id is None:
                # Every new revision has a proposition, for every kind. A
                # revision is a state of a claim, and a state with no structured
                # claim in it is a row that says only that something happened.
                #
                # Historical revisions predating this keep their `NULL` and stay
                # readable through their legacy payload. What must never happen
                # is *backfilling* one from that payload — and not because the
                # payload lost something: a stored projection is lossless, which
                # is what `Represented | NotRepresentable` guarantees. What is not
                # guaranteed is the inverse. Reading a payload back does not
                # reproduce the proposition a past run produced, because the same
                # columns can be reached from more than one structure and the run
                # that wrote them is gone.
                raise ValueError(
                    f"a new revision of '{write.entity.key}' has no proposition; "
                    f"a state of a claim must say what the claim is"
                )
            revision = KnowledgeRevision(
                variant_id=variant.variant_id,
                revision=(latest.revision + 1) if latest is not None else 1,
                text=write.text,
                fingerprint=write.fingerprint,
                extraction_version=write.extraction_version,
                # A first sighting is approved-by-policy like any other new
                # assertion; the review policy upstream is what quarantines it,
                # not the lineage.
                review_state=NEEDS_REVIEW if normative else APPROVED,
                supersedes=latest.revision if latest is not None else None,
                observation_id=observation_id,
                proposition_id=proposition_id,
                created_at=now,
            )
            session.add(revision)
            await session.flush()

        # The slot is part of what makes a sighting distinct. Two sections of one
        # document saying the same thing are two sightings, and a key without the
        # slot recorded only the first — which left the support view unable to say
        # which section had stopped carrying a claim, because it never knew both
        # had. A retried step still records nothing twice: the run, the document
        # and the slot are all the same on a retry.
        seen = (
            await session.execute(
                select(KnowledgeEvidence.evidence_id).where(
                    KnowledgeEvidence.revision_id == revision.revision_id,
                    KnowledgeEvidence.document_id == write.document_id,
                    KnowledgeEvidence.slot_id == (write.slot_id or ""),
                    KnowledgeEvidence.run_id == write.run_id,
                )
            )
        ).scalars().first()
        if seen is None:
            session.add(
                KnowledgeEvidence(
                    revision_id=revision.revision_id,
                    document_id=write.document_id,
                    document_revision=write.document_revision,
                    document_path=write.document_path,
                    slot_id=write.slot_id or "",
                    source_span=write.source_span,
                    run_id=write.run_id,
                    observed_at=now,
                )
            )

        return RecordedLineage(
            assertion_id=assertion.assertion_id,
            variant_id=variant.variant_id,
            revision=revision.revision,
            review_state=revision.review_state,
            supersedes=revision.supersedes,
            resolution=decision.resolution,
            fingerprint=revision.fingerprint,
            revision_id=str(revision.revision_id),
        )

    async def extraction_state(self, document_id: str) -> ExtractionState | None:
        async with AsyncSession(self._engine) as session:
            row = await session.get(KnowledgeExtractionState, document_id)
            if row is None:
                return None
            return ExtractionState(
                document_id=row.document_id,
                content_hash=row.content_hash,
                extraction_version=row.extraction_version,
                extracted_at=row.extracted_at,
            )

    async def record_extraction(self, write: ExtractionStateWrite) -> ExtractionState:
        now = datetime.now(UTC)
        async with AsyncSession(self._engine, expire_on_commit=False) as session:
            async with session.begin():
                row = await session.get(KnowledgeExtractionState, write.document_id)
                if row is None:
                    row = KnowledgeExtractionState(document_id=write.document_id)
                    session.add(row)
                # Overwritten rather than appended: a document has one answer to
                # "may this be skipped", and it is the latest one.
                row.content_hash = write.content_hash
                row.extraction_version = write.extraction_version
                row.run_id = write.run_id
                row.assertion_count = write.assertion_count
                row.extracted_at = now
        return ExtractionState(
            document_id=write.document_id,
            content_hash=write.content_hash,
            extraction_version=write.extraction_version,
            extracted_at=now,
        )

    async def similar(self, identity: str, *, limit: int = 5) -> list[KnowledgeObject]:
        """Other assertions of the same kind and type.

        A reviewer deciding whether a claim is a duplicate or a contradiction
        needs to see its neighbours; quarantined ones are included, because a
        pair of near-identical pending assertions is exactly what a reviewer
        must be shown before approving either.
        """
        async with AsyncSession(self._engine) as session:
            node = await session.get(Knowledge, identity)
            if node is None:
                return []
            rows = (
                await session.execute(
                    select(Knowledge)
                    .where(
                        Knowledge.identity != identity,
                        Knowledge.kind == node.kind,
                        Knowledge.type == node.type,
                        # A merged assertion is no longer a thing to merge into.
                        Knowledge.merged_into.is_(None),
                    )
                    .order_by(Knowledge.confidence.desc(), Knowledge.identity)
                    .limit(limit)
                )
            ).scalars().all()
            return [await self._to_object(session, row) for row in rows]

    async def observations(self, identity: str) -> list[Observation]:
        async with AsyncSession(self._engine) as session:
            rows = (
                await session.execute(
                    select(KnowledgeObservation)
                    .where(KnowledgeObservation.identity == identity)
                    .order_by(KnowledgeObservation.observed_at.desc())
                )
            ).scalars().all()
            return [
                Observation(
                    identity=row.identity,
                    source_id=row.source_id,
                    run_id=row.run_id,
                    evidence=dict(row.evidence or {}),
                    observed_at=row.observed_at,
                )
                for row in rows
            ]

    async def reviews(self, identity: str) -> list[Review]:
        async with AsyncSession(self._engine) as session:
            rows = (
                await session.execute(
                    select(KnowledgeReview)
                    .where(KnowledgeReview.identity == identity)
                    .order_by(KnowledgeReview.reviewed_at.desc())
                )
            ).scalars().all()
            return [
                Review(
                    identity=row.identity,
                    reviewer_id=row.reviewer_id,
                    decision=ReviewDecision(row.decision),
                    reviewed_at=row.reviewed_at,
                    comment=row.comment,
                    changes=dict(row.changes or {}),
                )
                for row in rows
            ]

    # -- writes --------------------------------------------------------------

    async def persist(
        self, write: KnowledgeWrite, lineage: LineageWrite | None = None
    ) -> tuple[KnowledgeObject, RecordedLineage | None]:
        """The graph and the lineage, in one transaction.

        During the transition both records of an assertion are written, and they
        must not be able to disagree. Two transactions would let one land
        without the other — a split brain arriving precisely while identity is
        being migrated, which is the worst moment for the two to diverge. A
        failure on either side therefore rolls back both.

        ``lineage`` is optional and its absence is the caller's decision, not a
        fallback taken here. An assertion whose kind cannot yet say which
        business question it is has no entity key, and the honest record of that
        is no lineage row — never a substitute key invented from its wording,
        which a later run would then fail to match.

        Optional for a payload, not for a claim. A fact keeps its structure on
        the revision a sighting produces, so a write carrying a proposition and
        no lineage has nowhere to put what it says: it produced a graph row with
        no payload, no proposition and no revision — a node asserting nothing,
        which is a quieter failure than dropping the claim rather than a milder
        one. By the time it reaches here that is a programming error and not a
        data condition, because extraction rejects a fact that cannot name what
        it claims and the persist step counts and skips one whose evidence names
        no document. So it raises, and the invariant holds by construction:

            every stored fact proposition is reachable through a revision
        """
        if lineage is None and write.proposition is not None:
            raise ValueError(
                f"fact '{write.identity}' carries a proposition and no lineage, so it "
                f"has no revision to keep it on; reject it at extraction or give it "
                f"the evidence a lineage row needs"
            )
        async with AsyncSession(self._engine, expire_on_commit=False) as session, session.begin():
            now = _utcnow()
            # Lineage first where there is one: it decides which state this
            # sighting is, and the graph node is addressed by that state so the
            # two records cannot describe different things. See `record_slot`.
            recorded = (
                await self._record_lineage(session, lineage, now)
                if lineage is not None
                else None
            )
            if recorded is not None and recorded.fingerprint:
                write = replace(write, identity=recorded.fingerprint)
            node = await self._upsert(
                session, write, now,
                revision_id=recorded.revision_id if recorded is not None else "",
            )
            return node, recorded

    async def _assertions_in(self, session: AsyncSession, document_id: str) -> set[str]:
        """The live assertions this document currently carries.

        The whole document rather than one of its sections. A section's id is its
        position, so scoping this to a slot made an inserted paragraph retire
        every unchanged claim below it.
        """
        rows = (
            await session.execute(
                select(KnowledgeAssertion.assertion_id)
                .join(KnowledgeVariant, KnowledgeVariant.assertion_id == KnowledgeAssertion.assertion_id)
                .join(KnowledgeRevision, KnowledgeRevision.variant_id == KnowledgeVariant.variant_id)
                .join(KnowledgeEvidence, KnowledgeEvidence.revision_id == KnowledgeRevision.revision_id)
                .where(
                    KnowledgeAssertion.retired_at.is_(None),
                    KnowledgeEvidence.document_id == document_id,
                )
            )
        ).scalars().all()
        return set(rows)

    async def _carried_elsewhere(
        self, session: AsyncSession, assertion_id: str, document_id: str
    ) -> bool:
        """Whether another document still asserts this claim.

        Deliberately conservative: any evidence from a different document holds
        the claim up. Erring towards keeping a claim that no longer has a source
        costs a stale row a reviewer can retire; erring the other way deletes
        knowledge a second document was carrying all along.
        """
        return (
            await session.execute(
                select(KnowledgeEvidence.evidence_id)
                .join(KnowledgeRevision, KnowledgeRevision.revision_id == KnowledgeEvidence.revision_id)
                .join(KnowledgeVariant, KnowledgeVariant.variant_id == KnowledgeRevision.variant_id)
                .where(
                    KnowledgeVariant.assertion_id == assertion_id,
                    KnowledgeEvidence.document_id != document_id,
                )
                .limit(1)
            )
        ).scalar_one_or_none() is not None

    async def _latest_fingerprint(self, session: AsyncSession, assertion_id: str) -> str:
        row = (
            await session.execute(
                select(KnowledgeRevision.fingerprint)
                .join(KnowledgeVariant, KnowledgeVariant.variant_id == KnowledgeRevision.variant_id)
                .where(KnowledgeVariant.assertion_id == assertion_id)
                .order_by(KnowledgeRevision.revision.desc())
                .limit(1)
            )
        ).scalars().first()
        return str(row or "")

    async def _propositions_of(
        self, session: AsyncSession, assertion_ids: Sequence[str]
    ) -> dict[str, tuple[uuid.UUID, Proposition, str]]:
        """What each of these assertions currently says, whole.

        The latest revision that recorded a structure — not the first, because
        the question is what the corpus says now, and not every revision has one:
        anything written before propositions existed has none.
        """
        if not assertion_ids:
            return {}
        rows = (
            await session.execute(
                select(
                    KnowledgeVariant.assertion_id,
                    KnowledgeRevision.revision,
                    KnowledgeProposition.proposition_id,
                    KnowledgeProposition.fields,
                    # The sentence the candidate was read from. Context for the
                    # judge, so it can tell what a field means — never the thing
                    # compared, since one sentence can carry two claims.
                    KnowledgeRevision.text,
                )
                .join(KnowledgeRevision, KnowledgeRevision.variant_id == KnowledgeVariant.variant_id)
                .join(
                    KnowledgeProposition,
                    KnowledgeProposition.proposition_id == KnowledgeRevision.proposition_id,
                )
                .where(KnowledgeVariant.assertion_id.in_(list(assertion_ids)))
            )
        ).all()

        latest: dict[str, tuple[int, uuid.UUID, dict[str, Any], str]] = {}
        for assertion_id, revision, proposition_id, fields, text in rows:
            seen = latest.get(assertion_id)
            if seen is None or revision > seen[0]:
                latest[assertion_id] = (revision, proposition_id, fields, text)
        return {
            assertion_id: (proposition_id, Proposition.from_dict(fields), text or "")
            for assertion_id, (_, proposition_id, fields, text) in latest.items()
        }

    @staticmethod
    def _observed(write: LineageWrite) -> AssertionObservation | None:
        """This sighting as the judgement receives it.

        One contract for every kind: what it is, what the source said, and the
        structured claim. The representation used to be kind-specific — a fact's
        proposition, and for the others a pair synthesised out of the wording and
        the entity key — which meant a rule was compared on something no
        extraction produced.

        `None` where there is nothing to compare: no wording, or no proposition.
        Both are counted elsewhere, and neither is invented here.
        """
        if not write.text.strip() or write.proposition is None:
            return None
        return AssertionObservation(
            kind=write.kind,
            observed_text=write.text,
            representation=dict(write.proposition.fields),
        )

    async def _sightings_of(
        self, session: AsyncSession, assertion_ids: Sequence[str]
    ) -> dict[str, uuid.UUID]:
        """The observation each candidate was last recorded as.

        The candidate's side of an assessment must point at the row that
        actually recorded it. Building one at comparison time would make a
        sighting into a sighting per comparison, and stamp a claim seen last
        week with the run that merely looked at it today — an audit that is
        complete and historically false, which is worse than incomplete because
        it reads as evidence.

        Absent for anything written before revisions named their sighting.
        Nothing is invented for those: a fabricated event in an audit is the one
        thing worse than a missing one.
        """
        if not assertion_ids:
            return {}
        rows = (
            await session.execute(
                select(
                    KnowledgeVariant.assertion_id,
                    KnowledgeRevision.revision,
                    KnowledgeRevision.observation_id,
                )
                .join(
                    KnowledgeRevision,
                    KnowledgeRevision.variant_id == KnowledgeVariant.variant_id,
                )
                .where(
                    KnowledgeVariant.assertion_id.in_(list(assertion_ids)),
                    KnowledgeRevision.observation_id.is_not(None),
                )
            )
        ).all()
        latest: dict[str, tuple[int, uuid.UUID]] = {}
        for assertion_id, revision, observation_id in rows:
            seen = latest.get(assertion_id)
            if seen is None or revision > seen[0]:
                latest[assertion_id] = (revision, observation_id)
        return {
            assertion_id: observation_id
            for assertion_id, (_, observation_id) in latest.items()
        }

    async def _observations_of(
        self, session: AsyncSession, candidates: Sequence[Candidate]
    ) -> dict[str, AssertionObservation]:
        """What each candidate currently says, as an observation.

        A fact reads its representation from the proposition its revision names;
        every other kind has none, so its structured fields are what the matcher
        already carries. Both end as the same contract, which is the point —
        one question, asked the same way for every kind.
        """
        if not candidates:
            return {}
        by_id = {candidate.assertion_id: candidate for candidate in candidates}
        propositions = await self._propositions_of(session, list(by_id))

        observed: dict[str, AssertionObservation] = {}
        for assertion_id, candidate in by_id.items():
            found = propositions.get(assertion_id)
            if found is not None:
                _, proposition, text = found
                observed[assertion_id] = AssertionObservation(
                    kind=candidate.kind,
                    observed_text=text or candidate.text,
                    representation=dict(proposition.fields),
                )
            elif candidate.text.strip():
                observed[assertion_id] = AssertionObservation(
                    kind=candidate.kind,
                    observed_text=candidate.text,
                    representation={
                        "text": candidate.text,
                        "entity_key": candidate.entity_key,
                    },
                )
        return observed

    #: How far a plausible-candidate search may reach, and how little it may
    #: accept. Two shared terms is the smallest overlap that is a signal rather
    #: than a coincidence — one shared subject is true of half a corpus — and the
    #: cap is what keeps "not global" a property of the code rather than of the
    #: corpus size.
    PLAUSIBLE_MINIMUM_TERMS = 2
    PLAUSIBLE_LIMIT = 8

    async def _plausible(
        self,
        session: AsyncSession,
        write: LineageWrite,
        *,
        exclude: set[str],
    ) -> list[Candidate]:
        """Claims elsewhere in the corpus this sighting might be continuing.

        Retrieval, and nothing more. It decides which pairs are worth a question
        and never what the answer is: two claims reaching each other here are
        still two assertions until the judge says otherwise, and the terms they
        share confer no identity of their own.

        The guardrail this widens used to read "equivalence only within one
        slot". That was too narrow for the case it broke on — one sentence in two
        documents, read as `uses` in one and `uses_port` in the other, which are
        two entity keys and so never met. Model vocabulary decided identity,
        which is exactly what `Proposition` and the judge exist to prevent.

        What the guardrail actually protects against is an *unbounded* search: a
        judgement ranging over every proposition would be a merge engine looking
        for pairs. So the reach is bounded twice — same kind, and at least
        `PLAUSIBLE_MINIMUM_TERMS` shared field values — and capped.

        Uniform across kinds on purpose. The per-kind projection tables carry the
        same signal in indexed columns, and reaching for them would put a
        fifth kind-specific path in the resolver.
        """
        if write.proposition is None or not write.kind:
            return []
        ours = {
            canonical_term(str(value))
            for value in write.proposition.fields.values()
            if str(value).strip()
        }
        if len(ours) < self.PLAUSIBLE_MINIMUM_TERMS:
            # Too little to be plausible about. A claim with one term would reach
            # everything that mentions it.
            return []

        rows = (
            await session.execute(
                select(
                    KnowledgeAssertion.assertion_id,
                    KnowledgeAssertion.entity_key,
                    KnowledgeAssertion.entity_key_version,
                    KnowledgeAssertion.kind,
                    KnowledgeRevision.revision,
                    KnowledgeRevision.text,
                    KnowledgeProposition.fields,
                    KnowledgeEvidence.document_id,
                    KnowledgeEvidence.slot_id,
                )
                .join(
                    KnowledgeVariant,
                    KnowledgeVariant.assertion_id == KnowledgeAssertion.assertion_id,
                )
                .join(
                    KnowledgeRevision,
                    KnowledgeRevision.variant_id == KnowledgeVariant.variant_id,
                )
                .join(
                    KnowledgeProposition,
                    KnowledgeProposition.proposition_id == KnowledgeRevision.proposition_id,
                )
                .join(
                    KnowledgeEvidence,
                    KnowledgeEvidence.revision_id == KnowledgeRevision.revision_id,
                )
                .where(
                    KnowledgeAssertion.retired_at.is_(None),
                    KnowledgeAssertion.kind == write.kind,
                    KnowledgeAssertion.entity_key != write.entity.key,
                )
            )
        ).all()

        latest: dict[str, tuple[int, str, str, str, str, set[str]]] = {}
        slots: dict[str, set[str]] = {}
        for (
            assertion_id, entity_key, key_version, kind, revision, text,
            fields, _document_id, slot_id,
        ) in rows:
            if assertion_id in exclude:
                continue
            theirs = {
                canonical_term(str(value))
                for value in (fields or {}).values()
                if str(value).strip()
            }
            if len(ours & theirs) < self.PLAUSIBLE_MINIMUM_TERMS:
                continue
            seen = latest.get(assertion_id)
            if seen is None or revision > seen[0]:
                latest[assertion_id] = (
                    revision, entity_key, key_version, text, kind, theirs
                )
            if slot_id:
                slots.setdefault(assertion_id, set()).add(slot_id)

        # Strongest overlap first, then by id so two runs agree on which eight.
        ordered = sorted(
            latest.items(),
            key=lambda item: (-len(ours & item[1][5]), item[0]),
        )[: self.PLAUSIBLE_LIMIT]
        return [
            Candidate(
                assertion_id=assertion_id,
                entity_key=entity_key,
                entity_key_version=key_version,
                text=text,
                document_id="",
                kind=kind,
                slot_ids=frozenset(slots.get(assertion_id, set())),
            )
            for assertion_id, (_, entity_key, key_version, text, kind, _) in ordered
        ]

    async def _assess(
        self,
        session: AsyncSession,
        write: LineageWrite,
        candidates: Sequence[Candidate],
        assess: EquivalenceJudge | None,
        slot_id: str | None,
    ) -> tuple[str | None, list[tuple[uuid.UUID, uuid.UUID, str]], uuid.UUID | None, str]:
        """Whether some claim already in this slot is this claim said differently.

        Asked only where the matching ladder found nothing, and only of the
        candidates standing **in the same slot**. Never over the corpus: a claim
        is continued because the source keeps saying it in the place it stood,
        and a judgement ranging over every proposition would be a merge engine
        looking for pairs rather than a resolver placing one sighting.

        Two further limits on when the model is asked at all. `settled` answers
        first — identical structures are the same claim and a moved quantity is a
        different one, neither of which needs to know what words mean — so a
        differing fingerprint is not by itself a question. And the whole of this
        runs outside any transaction, because a model round trip while holding
        row locks is how one slow answer becomes a stalled corpus.

        Returns the assertion the judgements permit continuing, and every
        assessment reached on the way. Both, because the assessments are recorded
        whatever the answer: "these were judged different" is a finding, and a
        trail that kept only the joins would show a model that never disagreed.
        """
        if slot_id is None:
            return None, [], None, ""
        ours = self._observed(write)
        if ours is None:
            return None, [], None, ""
        # Same kind only. `kind` is part of the identity namespace, so the
        # answer for a different one is already `no` and the judge is never
        # asked a question nothing could change.
        in_slot = [
            c for c in candidates if slot_id in c.slot_ids and c.kind == write.kind
        ]
        # And, beyond this slot, the few claims elsewhere that share enough
        # structure to be worth a question. Reinforcement across documents is
        # continuity over several sightings, not identity of two propositions:
        # one sentence read as `uses` in one document and `uses_port` in another
        # is one claim, and leaving it as two would put assertion identity back
        # in the model's choice of label.
        pool = in_slot + await self._plausible(
            session, write, exclude={c.assertion_id for c in in_slot}
        )
        known = await self._observations_of(session, pool)
        if not known:
            return None, [], None, ""

        # Durable before anything is decided, because an assessment written
        # after the merge could only describe it. **Once** for this sighting,
        # however many candidates it is compared against: an observation is an
        # event, and comparing it five times does not make it five events.
        mine = write.observation_id or await self.record_observation(
            ours, run_id=write.run_id, document_id=write.document_id, slot_id=slot_id
        )
        # And the candidates bring theirs. A candidate was seen in an earlier
        # run and already has a row; re-recording it here would stamp it with
        # the run that compared rather than the run that saw.
        sightings = await self._sightings_of(session, list(known))
        verdicts: list[tuple[str, str]] = []
        assessments: list[tuple[uuid.UUID, uuid.UUID, str]] = []
        for assertion_id, candidate in sorted(known.items()):
            theirs = sightings.get(assertion_id)
            # The model is asked only for the band nothing else can answer.
            # Identical structures are the same claim and a moved quantity is a
            # different one — neither needs to know what words mean, and asking
            # anyway would spend a round trip per differing fingerprint.
            answer = (
                await assess.classify(ours, candidate)
                if assess is not None and settled_equivalence(ours, candidate) is None
                else None
            )
            # Through `equivalent` even when a judge answered, so an unreadable
            # reply lands on `unjudged` rather than being stored as a decision.
            # Nobody was asked and a judge found them different are both "do not
            # join", and only one of them is evidence.
            verdict = equivalent(ours, candidate, verdict=answer)
            verdicts.append((assertion_id, verdict))
            if theirs is not None:
                assessments.append((mine, theirs, verdict))
            else:
                # Judged, and not recordable: this candidate predates revisions
                # naming their sighting, so there is no row to point at. The
                # verdict still counts — refusing to judge an older claim would
                # make the corpus behave differently on either side of a
                # migration — but nothing fabricates an event to file it under.
                logger.info(
                    "knowledge.equivalence.unrecordable | assertion=%s verdict=%s run=%s",
                    assertion_id, verdict, write.run_id,
                )
        joined = continuation(verdicts)
        # Which pool the joined candidate came from, so the transaction can
        # revalidate against the premise the verdict was actually formed under.
        scope = ""
        if joined is not None:
            scope = "slot" if joined in {c.assertion_id for c in in_slot} else "corpus"
        return joined, assessments, mine, scope

    async def _judged(
        self,
        writes: Sequence[tuple[KnowledgeWrite, LineageWrite | None]],
        classify: Callable[[str, str], Awaitable[str]] | None,
        assess: EquivalenceJudge | None = None,
        run_id: str | None = None,
        slot_id: str | None = None,
    ) -> list[tuple[KnowledgeWrite, LineageWrite | None]]:
        """Decide, for each changed body, whether the approval survives it.

        Read-only and outside any transaction. For every write that continues an
        assertion whose current wording differs, the change is put to
        `semantic_change` — which settles what it can from the wording and the
        quantities and asks the classifier only for the band where meaning is the
        only difference left.

        A write of a claim the corpus has not seen carries no approval, so there
        is nothing to spend and nothing to ask.
        """
        judged: list[tuple[KnowledgeWrite, LineageWrite | None]] = []
        assessments: list[tuple[uuid.UUID, uuid.UUID, str]] = []
        async with AsyncSession(self._engine) as session:
            for graph, lineage in writes:
                if lineage is None:
                    judged.append((graph, lineage))
                    continue
                candidates = await self._candidates(session, lineage)
                decision = match_proposition(
                    entity_key=lineage.entity.key,
                    entity_key_version=lineage.entity.version,
                    text=lineage.text,
                    slot_id=lineage.slot_id,
                    document_id=lineage.document_id,
                    candidates=candidates,
                    kind=lineage.kind,
                )
                # Only where exact agreement found nothing. A ladder that
                # answered is an answer, and asking a model to review it would
                # let a judgement overrule what the source plainly said.
                continues, reached, recorded_as, scope = (
                    (None, [], None, "")
                    if decision.assertion_id is not None
                    else await self._assess(session, lineage, candidates, assess, slot_id)
                )
                assessments.extend(reached)
                # The sighting was recorded once, here. Carried forward so the
                # transaction that writes the claim reuses it instead of
                # recording the same event a second time.
                lineage = replace(
                    lineage, continues=continues, observation_id=recorded_as,
                    continues_scope=scope,
                )

                if lineage.normative_change:
                    judged.append((graph, lineage))
                    continue
                # Against whichever assertion this sighting will continue —
                # including one reached by equivalence, which carries an approval
                # like any other and must be able to lose it the same way.
                previous = next(
                    (
                        c.text
                        for c in candidates
                        if c.assertion_id == (decision.assertion_id or continues)
                    ),
                    None,
                )
                if previous is None:
                    judged.append((graph, lineage))
                    continue
                verdict = None
                if classify is not None and settled(previous, lineage.text) is None:
                    verdict = await classify(previous, lineage.text)
                normative = semantic_change(previous, lineage.text, verdict=verdict) == NORMATIVE
                judged.append((graph, replace(lineage, normative_change=normative)))

        # Written before anything is joined, and written whatever the answer.
        # The record of what was decided may not depend on the decision going one
        # way: a trail holding only the joins would show a judge that never
        # disagreed and never hesitated.
        for left, right, verdict in assessments:
            await self.record_equivalence(
                left=left,
                right=right,
                verdict=verdict,
                judge=assess.judge if assess is not None and verdict not in _UNASKED else None,
                run_id=run_id,
            )
        return judged

    async def record_proposition(self, proposition: Proposition) -> uuid.UUID:
        """Keep one extracted claim, exactly as it arrived.

        Content-addressed, so one claim seen twice is one row. The id it returns
        is a surrogate and **not** an assertion's identity — whether two
        propositions are one claim is a judgement, recorded separately, and
        reading it off this table would be the content-derived identity this
        design removed.
        """
        async with AsyncSession(self._engine) as session, session.begin():
            fingerprint = proposition.fingerprint
            existing = (
                await session.execute(
                    select(KnowledgeProposition).where(
                        KnowledgeProposition.fingerprint == fingerprint,
                        KnowledgeProposition.fingerprint_version == FINGERPRINT_VERSION,
                    )
                )
            ).scalars().first()
            if existing is not None:
                return existing.proposition_id
            row = KnowledgeProposition(
                fingerprint=fingerprint,
                fingerprint_version=FINGERPRINT_VERSION,
                fields=proposition.as_dict(),
            )
            session.add(row)
            await session.flush()
            return row.proposition_id

    async def _record_observation(
        self, session: AsyncSession, write: LineageWrite, slot_id: str
    ) -> uuid.UUID | None:
        """This sighting, inside the caller's transaction.

        The public `record_observation` opens its own session, which is right
        when the assessment must land before the claim and wrong here — the
        graph write already holds one, and a second on the same engine would
        contend with it.
        """
        observed = self._observed(write)
        if observed is None:
            return None
        row = KnowledgeAssertionObservation(
            kind=observed.kind,
            observed_text=observed.observed_text,
            representation=dict(observed.representation),
            run_id=write.run_id,
            document_id=write.document_id,
            slot_id=slot_id,
        )
        session.add(row)
        await session.flush()
        return row.observation_id

    async def record_observation(
        self,
        observation: AssertionObservation,
        *,
        run_id: str,
        document_id: str = "",
        slot_id: str = "",
    ) -> uuid.UUID:
        """Keep one sighting of one assertion, before anything is decided.

        The unit an equivalence judgement compares, and it has to be durable
        first: an assessment written after the merge can only *describe* what
        happened, and a reader cannot tell a judgement that caused a
        continuation from one reconstructed to explain it.

        Nothing downstream could serve. `knowledge_evidence` hangs off a
        revision, and the revision exists only once the matcher and the judge
        have already answered — the assessment would have to wait for the thing
        it justifies.

        For every kind, which is the other half. `knowledge_propositions` holds a
        fact and nothing else, so an audit keyed on it leaves `rule`, `decision`
        and `pattern` unauditable.

        **Not deduplicated, and deliberately.** An observation is an event: two
        runs that read one sentence saw it twice, and collapsing them would lose
        which run saw what — the question an audit is asked. That is also what
        separates this from `knowledge_propositions`, which is content-addressed
        because it holds what a claim *is* rather than that somebody looked.
        """
        async with AsyncSession(self._engine) as session, session.begin():
            row = KnowledgeAssertionObservation(
                kind=observation.kind,
                observed_text=observation.observed_text,
                representation=dict(observation.representation),
                run_id=run_id,
                document_id=document_id,
                slot_id=slot_id,
            )
            session.add(row)
            await session.flush()
            return row.observation_id

    async def record_equivalence(
        self,
        *,
        left: uuid.UUID,
        right: uuid.UUID,
        verdict: str,
        judge: Judge | None = None,
        run_id: str | None = None,
    ) -> uuid.UUID:
        """Append one judgement about one pair, as it was actually made.

        Append-only, and nothing is checked against an earlier answer. The same
        judge asked twice can answer differently, and that is a finding worth
        keeping: refusing the second row would delete the evidence that a model
        is unstable, which is what anyone auditing this would most want to see.

        Reusing a decision instead of asking again is a cache, and a cache is a
        different table. This one answers what was decided.
        """
        async with AsyncSession(self._engine) as session, session.begin():
            row = KnowledgeEquivalenceAssessment(
                # Canonical, because "X is equivalent to Y" is the same question
                # as "Y is equivalent to X". The assertions themselves stay
                # directed; only the *relation between them* is symmetric.
                left_observation_id=min(left, right, key=str),
                right_observation_id=max(left, right, key=str),
                verdict=verdict,
                classifier=judge.classifier if judge else None,
                model=judge.model if judge else None,
                classifier_version=judge.version if judge else None,
                run_id=run_id,
            )
            session.add(row)
            await session.flush()
            return row.assessment_id

    async def _known_slots(
        self, session: AsyncSession, document_id: str
    ) -> dict[str, dict[str, Any]]:
        """This document's slots, as the resolver reads them."""
        slots = (
            await session.execute(
                select(KnowledgeSlot).where(KnowledgeSlot.document_id == document_id)
            )
        ).scalars().all()
        if not slots:
            return {}
        rows = (
            await session.execute(
                select(KnowledgeSlotAnchor)
                .where(KnowledgeSlotAnchor.slot_id.in_([slot.slot_id for slot in slots]))
                .order_by(KnowledgeSlotAnchor.valid_from)
            )
        ).scalars().all()
        history: dict[str, list[SlotAnchor]] = {}
        for row in rows:
            history.setdefault(row.slot_id, []).append(
                SlotAnchor(
                    anchor=row.anchor,
                    strength=row.strength,
                    valid_from=row.valid_from,
                    valid_to=row.valid_to,
                )
            )
        return {
            slot.slot_id: {
                "document_id": slot.document_id,
                "anchors": tuple(history.get(slot.slot_id, ())),
                "content": slot.content_fingerprint,
                "status": slot.status,
            }
            for slot in slots
        }

    async def _slot_support(
        self, session: AsyncSession, document_id: str
    ) -> set[tuple[str, str]]:
        """Which claims this document currently supports, and from which slot."""
        rows = (
            await session.execute(
                select(KnowledgeVariant.assertion_id, KnowledgeEvidence.slot_id)
                .join(
                    KnowledgeRevision,
                    KnowledgeRevision.variant_id == KnowledgeVariant.variant_id,
                )
                .join(
                    KnowledgeEvidence,
                    KnowledgeEvidence.revision_id == KnowledgeRevision.revision_id,
                )
                .where(KnowledgeEvidence.document_id == document_id)
            )
        ).all()
        return {(str(assertion_id), str(slot_id or "")) for assertion_id, slot_id in rows}

    async def _slot_readings(
        self, session: AsyncSession, document_id: str
    ) -> dict[str, str]:
        """What each of this document's sections was, exactly, when last read.

        The exact parser input and not the normalised section fingerprint —
        these readings decide whether the model was asked the same question, and
        a hash that folds punctuation cannot answer that.

        The slot's own `content_fingerprint` cannot answer this either: `_place`
        overwrites it with the current value every run, so by the time a
        retraction is decided it already holds what this run saw. The readings
        are a log rather than a state, and the latest of them is the previous
        run's.
        """
        rows = (
            await session.execute(
                select(
                    KnowledgeSlotObservation.slot_id,
                    KnowledgeSlotObservation.parser_input_fingerprint,
                )
                .where(KnowledgeSlotObservation.document_id == document_id)
                .order_by(KnowledgeSlotObservation.observed_at.asc())
            )
        ).all()
        return {str(slot_id): str(fingerprint) for slot_id, fingerprint in rows}

    async def _slot_texts(
        self, session: AsyncSession, document_id: str
    ) -> dict[str, set[tuple[str, str]]]:
        """What each of this document's sections already says, and under which
        extraction version it was read.

        The wording is the observation (ADR-0048), so it is what says whether a
        claim arriving now is one the section was already known to carry. The
        extraction version travels with it because an unchanged section asked a
        *different* question may legitimately answer differently — refusing that
        would freeze the corpus against every improvement to the contract.
        """
        rows = (
            await session.execute(
                select(
                    KnowledgeEvidence.slot_id,
                    KnowledgeRevision.text,
                    KnowledgeRevision.extraction_version,
                )
                .join(
                    KnowledgeRevision,
                    KnowledgeRevision.revision_id == KnowledgeEvidence.revision_id,
                )
                .where(KnowledgeEvidence.document_id == document_id)
            )
        ).all()
        held: dict[str, set[tuple[str, str]]] = {}
        for slot_id, text, extraction_version in rows:
            if slot_id:
                held.setdefault(str(slot_id), set()).add(
                    (fold_whitespace(str(text or "")), str(extraction_version or ""))
                )
        return held

    async def _sync_support(
        self,
        session: AsyncSession,
        document_id: str,
        *,
        carries: Mapping[tuple[str, str], LineageWrite],
        unchanged: set[str],
        now: datetime,
    ) -> None:
        """Replace what this document currently carries with what this run saw.

        Called only from `record_document`, and therefore only for a document
        observed in full. A partial run cannot say what a document currently
        carries — it did not see all of it — so it writes evidence and leaves
        this untouched. That is the same gate retraction uses, which is the
        point: two mechanisms deciding currency by different rules is how a
        corpus starts contradicting itself.

        **A section the stable-slot guard protected keeps its support.** The
        guard exists because a model asked twice about byte-identical text
        answered differently (ADR-0044), and it refuses to retire what came back
        missing. Support has to agree with that: dropping the row for a claim
        the same run refused to retire would say the document no longer carries
        something the corpus insists it still asserts, and a citation would
        vanish over a model's nondeterminism.

        Those kept rows keep the document revision they were last *extracted* at
        rather than the one the document has now. That is the truthful answer —
        it is where this was last actually seen to be said — and inventing a
        newer one would claim an observation nobody made.

        Removal is decided per document and never per section. A slot that was
        renamed, split, or lost its heading has not stopped the document
        asserting anything (ADR-0044); what removes a row is the document being
        read in full and no longer producing that state.
        """
        existing = (
            await session.execute(
                select(
                    KnowledgeDocumentSupport.revision_id,
                    KnowledgeDocumentSupport.slot_id,
                ).where(KnowledgeDocumentSupport.document_id == document_id)
            )
        ).all()
        held = {
            (str(revision_id), str(slot_id))
            for revision_id, slot_id in existing
            if slot_id and slot_id in unchanged
        }
        # Removed as a statement rather than row by row: the unit of work orders
        # its inserts before its deletes, so replacing a row through the ORM
        # collides with the row it is replacing.
        removal = delete(KnowledgeDocumentSupport).where(
            KnowledgeDocumentSupport.document_id == document_id
        )
        if unchanged:
            removal = removal.where(
                KnowledgeDocumentSupport.slot_id.notin_(sorted(unchanged))
            )
        await session.execute(removal)
        for (revision_id, slot_id), lineage in carries.items():
            if (revision_id, slot_id) in held:
                continue
            session.add(
                KnowledgeDocumentSupport(
                    document_id=document_id,
                    revision_id=uuid.UUID(revision_id),
                    observed_document_revision=lineage.document_revision,
                    document_path=lineage.document_path,
                    slot_id=slot_id,
                    observed_at=now,
                )
            )

    async def _retire_slots(
        self, session: AsyncSession, document_id: str, resolved: set[str], now: datetime
    ) -> list[str]:
        """Slots this document no longer has a section for.

        A slot's life and a claim's life are separate, and keeping them separate
        is the point. A section that lost its heading has no slot any more and
        everything it said is still asserted — so this retires the slot and never
        touches the assertions, which are decided document-wide and afterwards.

        Only for a document observed in full, like everything else here: a
        partial run would read as a document that lost every section it did not
        get to.

        Retired rather than deleted, and its anchors closed rather than dropped:
        a section that comes back must be distinguishable from one that was never
        there, and it is not reactivated automatically — that decision is
        deliberate or it is not taken.
        """
        rows = (
            await session.execute(
                select(KnowledgeSlot).where(
                    KnowledgeSlot.document_id == document_id,
                    KnowledgeSlot.status == "live",
                )
            )
        ).scalars().all()
        gone = [row for row in rows if row.slot_id not in resolved]
        for row in gone:
            row.status = "retired"
            row.retired_at = now
            current = (
                await session.execute(
                    select(KnowledgeSlotAnchor).where(
                        KnowledgeSlotAnchor.slot_id == row.slot_id,
                        KnowledgeSlotAnchor.valid_to.is_(None),
                    )
                )
            ).scalars().first()
            if current is not None:
                # The path it had when it left is the one a returning section
                # arrives under, so it is closed and kept rather than removed.
                current.valid_to = now
        return [row.slot_id for row in gone]

    async def _place(
        self,
        session: AsyncSession,
        document_id: str,
        section: ObservedSection,
        known: dict[str, dict[str, Any]],
        claimed: set[str],
        now: datetime,
    ) -> str:
        """Which slot this section is, minting one and moving the history if so.

        The history is advanced **only on a real transition**. A run over an
        unchanged document resolves every section exactly and writes nothing; a
        rename resolves through an alias or the content, and *then* the current
        anchor is closed and the new one opened. Writing a row per run would turn
        an audit trail into a log of runs.

        A section that comes back under a path it once had gets a third row
        rather than the first one reopened:

            A  [t1, t2)
            B  [t2, t3)
            A  [t3, ∞)

        which is what makes "which anchor was valid in March" answerable.

        In the same transaction as the claims, deliberately. A slot continued
        while its history still described the previous state is a corpus that
        disagrees with itself, and the disagreement would be invisible.
        """
        # A slot another section of *this* run already took is not a candidate.
        # Recovery exists to find a slot from a previous run; offering it a
        # sibling would let two sections of one document, alike in text and
        # different in heading, collapse into one — and which of them won would
        # depend on the order they happened to be written in.
        available = {
            slot_id: value for slot_id, value in known.items() if slot_id not in claimed
        }
        match = resolve_slot(
            document_id, section.anchor, section.content_fingerprint, available
        )
        claimed.add(match.slot_id)
        existing = known.get(match.slot_id)
        if existing is None:
            session.add(
                KnowledgeSlot(
                    slot_id=match.slot_id,
                    document_id=document_id,
                    status="live",
                    content_fingerprint=section.content_fingerprint,
                )
            )
            session.add(
                KnowledgeSlotAnchor(
                    slot_id=match.slot_id,
                    anchor=section.anchor,
                    strength=section.anchor_strength,
                    valid_from=now,
                )
            )
            known[match.slot_id] = {
                "document_id": document_id,
                "anchors": (SlotAnchor(section.anchor, section.anchor_strength, valid_from=now),),
                "content": section.content_fingerprint,
                "status": "live",
            }
            return match.slot_id

        if not match.is_continuation:
            return match.slot_id

        row_for_slot = await session.get(KnowledgeSlot, match.slot_id)
        if row_for_slot is not None:
            # The section's text moves with every edit; the anchor usually does
            # not. Keeping it current is what lets the next rename be recovered.
            row_for_slot.content_fingerprint = section.content_fingerprint

        current = current_anchor(existing)
        if current is not None and (current.anchor, current.strength) == (
            section.anchor,
            section.anchor_strength,
        ):
            # Nothing moved. The commonest case by far, and the one that decides
            # whether this table stays readable.
            return match.slot_id

        if current is not None:
            row = (
                await session.execute(
                    select(KnowledgeSlotAnchor).where(
                        KnowledgeSlotAnchor.slot_id == match.slot_id,
                        KnowledgeSlotAnchor.valid_to.is_(None),
                    )
                )
            ).scalars().first()
            if row is not None:
                row.valid_to = now
        session.add(
            KnowledgeSlotAnchor(
                slot_id=match.slot_id,
                anchor=section.anchor,
                strength=section.anchor_strength,
                valid_from=now,
            )
        )
        if degraded(existing, section.anchor_strength):
            # The slot continues — it is the same section — but it stopped being
            # anchored on something the source wrote, and continuing without
            # noticing is how a lineage gets carried across a change nobody sees.
            logger.info(
                "knowledge.slot.degraded | document=%s slot=%s anchor=%s resolution=%s",
                document_id, match.slot_id, section.anchor, match.resolution,
            )
        known[match.slot_id] = {
            **existing,
            "anchors": (
                *(
                    SlotAnchor(a.anchor, a.strength, a.valid_from, a.valid_to or now)
                    for a in existing["anchors"]
                ),
                SlotAnchor(section.anchor, section.anchor_strength, valid_from=now),
            ),
        }
        return match.slot_id

    async def record_document(
        self,
        *,
        document_id: str,
        sections: Sequence[ObservedSection],
        run_id: str,
        classify: Callable[[str, str], Awaitable[str]] | None = None,
        assess: EquivalenceJudge | None = None,
    ) -> DocumentOutcome:
        """Everything one fully observed document says, and what it stopped saying.

        **Only call this for a document observed in full.** A claim is gone
        because the source no longer contains it, which is knowable only once
        everything the source *does* contain has arrived. A partial observation —
        the normal case, since an unchanged document is skipped whole and yields
        nothing — would read as a document that lost everything.

        The container is the document and not the slot, and that distinction was
        paid for. A slot id is a section's position, so inserting a paragraph
        renumbers every section below it: slot-scoped retraction then retired
        each unchanged claim from the slot it used to occupy and re-created it in
        the one it had moved to, taking its approval and its history with it. An
        edit that touched none of those claims destroyed all of them. The slot
        remains what *matches* a claim; the document is what decides one is gone.

        One transaction for the whole document. Writes landing without their
        retractions, or the reverse, leave the corpus asserting and denying the
        same thing with nothing able to say which half was right — and deciding
        retraction per slot made the outcome depend on the order the slots
        happened to be written in, which is not a property anything should rest
        on.

        Evidence is never deleted. It answers what document D at revision R
        asserted, which is the question the whole diff is built on, and a claim
        that stopped being asserted did not stop having been asserted. What is
        recorded is that the assertion is retired.

        A section that has something to be found by resolves to a slot, and its
        claims are written against that slot rather than against the ordinal the
        parser produced. One that has not — a format with no headings — never
        gets a registry row and keeps the label it arrived with: an id nothing
        can resolve back to would be new on every run.
        """
        # Asked before the transaction opens, never inside it. This module keeps
        # its transactions short and free of external calls, and a model round
        # trip per changed body would hold row locks across the network.
        # Which slot each section probably is, resolved read-only so the
        # equivalence question can be scoped to the claims already standing
        # *there*. The parser's ordinal renumbers whenever a paragraph is
        # inserted above it, so scoping by that would have compared this
        # section's claims against whatever now happens to sit at its old
        # number — and usually against nothing at all.
        #
        # A guess, deliberately, and one nothing rests on: the authoritative
        # placement still happens inside the transaction, and being wrong here
        # only means a judgement was asked about the wrong neighbours or not
        # asked at all, never that a claim lands in the wrong slot.
        async with AsyncSession(self._engine) as session:
            known = await self._known_slots(session, document_id)
        judged = [
            (
                section,
                await self._judged(
                    section.assertions,
                    classify,
                    assess,
                    run_id,
                    resolve_slot(
                        document_id,
                        section.anchor,
                        section.content_fingerprint,
                        known,
                    ).slot_id
                    if section.anchorable
                    else None,
                ),
            )
            for section in sections
        ]

        async with AsyncSession(self._engine) as session, session.begin():
            now = _utcnow()
            before = await self._assertions_in(session, document_id)
            supported_before = await self._slot_support(session, document_id)
            # What each section said the last time anybody read it, captured
            # before this run records its own readings. It is what lets a
            # retraction tell "the source dropped this claim" from "the model
            # did not mention it this time".
            read_before = await self._slot_readings(session, document_id)
            # And what it said, and under which extraction. A section only counts
            # as unchanged where the same question was put to it: a new
            # extraction version is a different question, so what comes back is
            # not the model contradicting itself.
            said_before = await self._slot_texts(session, document_id)
            # Resolution and the anchor history live in this transaction with the
            # claims. A slot continued while its history still described the
            # previous state is a corpus disagreeing with itself, and nothing
            # afterwards could say which half was right.
            known = await self._known_slots(session, document_id)
            placements: list[str | None] = []
            claimed: set[str] = set()
            #: Sections that say exactly what they said last time. A claim
            #: standing in one of these may not be retired for being absent.
            still_saying_the_same: set[str] = set()
            for section, _ in judged:
                slot_id = (
                    await self._place(session, document_id, section, known, claimed, now)
                    if section.anchorable
                    else None
                )
                placements.append(slot_id)
                if slot_id is not None:
                    # The *exact* input, never the section fingerprint. That one
                    # is normalised on purpose — a reflowed section is still the
                    # same section — so using it here answered "is this probably
                    # the same section" to a question that asked "was the model
                    # shown exactly this", and a sentence given a full stop
                    # froze the slot instead of producing a revision.
                    #
                    # An empty fingerprint never matches: a section whose input
                    # nobody could name is not a section known to be unchanged,
                    # and two unknowns comparing equal would freeze the slot for
                    # the worst possible reason.
                    reading = section.parser_input_fingerprint
                    if (
                        reading
                        and read_before.get(slot_id) == reading
                        and _extraction_of(section) in _versions_of(said_before, slot_id)
                    ):
                        still_saying_the_same.add(slot_id)
                    # What this run was given to read, kept rather than
                    # overwritten. `_place` has just moved the slot's
                    # `content_fingerprint` to the current value so the recovery
                    # rung can match against what the section says now — which is
                    # exactly why that column cannot answer "was this section
                    # edited between two pictures": after the run it is already
                    # the new value.
                    #
                    # Recorded whether or not anything changed, because a run that
                    # looked and found nothing is what tells "unchanged" from
                    # "never observed", and only the first of those is evidence.
                    session.add(
                        KnowledgeSlotObservation(
                            run_id=run_id,
                            document_id=document_id,
                            slot_id=slot_id,
                            parser_input_fingerprint=reading,
                            # From the sighting rather than from the section:
                            # a section knows its text, a lineage write knows
                            # which revision of the document that text came from.
                            document_revision=next(
                                (
                                    lineage.document_revision
                                    for _, lineage in section.assertions
                                    if lineage is not None
                                ),
                                "",
                            ),
                        )
                    )

            stored: list[KnowledgeObject] = []
            recorded: list[RecordedLineage] = []
            matched: set[str] = set()
            supported_now: set[tuple[str, str]] = set()
            #: What this document carries now: (revision, slot) → the sighting
            #: whose provenance a citation will quote.
            carries: dict[tuple[str, str], LineageWrite] = {}
            for (_, writes), slot_id in zip(judged, placements, strict=True):
                for write, lineage in writes:
                    if lineage is not None and slot_id is not None:
                        # The ordinal the parser produced is position metadata.
                        # What the evidence records is where the claim stands.
                        lineage = replace(lineage, slot_id=slot_id)
                    if lineage is None:
                        # The same guard as `persist`, on the other route in. A
                        # claim with no revision has nowhere to keep what it says.
                        if write.proposition is not None:
                            raise ValueError(
                                f"fact '{write.identity}' carries a proposition and no "
                                f"lineage, so it has no revision to keep it on"
                            )
                        stored.append(await self._upsert(session, write, now))
                        continue
                    # Lineage first, because it decides which state this sighting
                    # is — and the graph node is then addressed by that state.
                    #
                    # It used to be addressed by the candidate's own content
                    # hash, which covers the whole content block while the state
                    # is judged on the body. So a model that reworded a `subject`
                    # while leaving the rule alone produced no revision,
                    # correctly, and a graph row nothing pointed at, silently:
                    # unreachable from the lineage, therefore invisible to the
                    # diff and impossible to retract.
                    # A section that says exactly what it said, asked exactly
                    # the same question, cannot have gained a claim either. The
                    # retirement guard closed one direction of this and a live
                    # run promptly showed the other: an unedited paragraph was
                    # read twice with an identical fingerprint and came back with
                    # a third claim. Absence and arrival are the same fact about
                    # the model, and only one of them was being refused.
                    #
                    # Reinforcement still lands — a sighting of something the
                    # slot already holds is evidence, and welcome. What is
                    # refused is wording the slot has never carried.
                    if (
                        slot_id in still_saying_the_same
                        and fold_whitespace(lineage.text)
                        not in _wordings_of(said_before, slot_id)
                    ):
                        logger.info(
                            "knowledge.document.unchanged_section_gained_nothing | "
                            "document=%s slot=%s run=%s text=%.60s",
                            document_id, slot_id, run_id, lineage.text,
                        )
                        continue
                    placed = await self._record_lineage(session, lineage, now)
                    recorded.append(placed)
                    matched.add(placed.assertion_id)
                    supported_now.add((placed.assertion_id, slot_id or ""))
                    # What this document carries *now*, keyed by the state and
                    # the section it was seen in. Collected here because this is
                    # the one place that has both halves: the revision this
                    # sighting landed on, and the document provenance it came
                    # with.
                    carries[(placed.revision_id, lineage.slot_id or "")] = lineage
                    stored.append(
                        await self._upsert(
                            session,
                            replace(write, identity=placed.fingerprint or write.identity),
                            now,
                            revision_id=placed.revision_id,
                        )
                    )

            await self._sync_support(
                session, document_id,
                carries=carries, unchanged=still_saying_the_same, now=now,
            )

            retired_slots = await self._retire_slots(
                session, document_id, {slot for slot in placements if slot}, now
            )

            retired: list[RetiredAssertion] = []
            for assertion_id in sorted(before - matched):
                if await self._carried_elsewhere(session, assertion_id, document_id):
                    continue
                # The section this claim stood in says exactly what it said
                # before, so the source did not drop it — the model did not
                # mention it this time. Retracting on that turns the model's
                # nondeterminism into lost knowledge, and a live run did exactly
                # that: an unedited paragraph was read twice with an identical
                # fingerprint, one of its two claims came back and the other was
                # retired.
                #
                # Retraction is a set operation over what the source says
                # (ADR-0044), and this is what keeps it that: absence is only
                # evidence of removal where the section could have changed.
                if any(
                    slot in still_saying_the_same
                    for held, slot in supported_before
                    if held == assertion_id and slot
                ):
                    continue
                assertion = await session.get(KnowledgeAssertion, assertion_id)
                if assertion is None or assertion.retired_at is not None:
                    continue
                assertion.retired_at = now
                retired.append(
                    RetiredAssertion(
                        assertion_id=assertion_id,
                        retired_at=now,
                        kind=assertion.kind,
                        fingerprint=await self._latest_fingerprint(session, assertion_id),
                    )
                )

            if retired:
                logger.info(
                    "knowledge.document.retired | document=%s sections=%d retired=%d run=%s",
                    document_id, len(sections), len(retired), run_id,
                )
            supports = tuple(
                SlotSupport(
                    assertion_id=assertion_id,
                    document_id=document_id,
                    slot_id=slot_id,
                    active=(assertion_id, slot_id) in supported_now,
                )
                for assertion_id, slot_id in sorted(supported_before | supported_now)
            )
            return DocumentOutcome(
                stored=tuple(stored),
                recorded=tuple(recorded),
                retired=tuple(retired),
                retired_slots=tuple(retired_slots),
                supports=supports,
            )

    async def upsert(self, write: KnowledgeWrite) -> KnowledgeObject:
        async with AsyncSession(self._engine) as session, session.begin():
            return await self._upsert(session, write, _utcnow())

    async def _record_proposition(
        self, session: AsyncSession, proposition: Proposition
    ) -> uuid.UUID:
        """Content-addressed, inside a caller's transaction."""
        existing = (
            await session.execute(
                select(KnowledgeProposition).where(
                    KnowledgeProposition.fingerprint == proposition.fingerprint,
                    KnowledgeProposition.fingerprint_version == FINGERPRINT_VERSION,
                )
            )
        ).scalars().first()
        if existing is not None:
            return existing.proposition_id
        row = KnowledgeProposition(
            fingerprint=proposition.fingerprint,
            fingerprint_version=FINGERPRINT_VERSION,
            fields=proposition.as_dict(),
        )
        session.add(row)
        await session.flush()
        return row.proposition_id

    async def _upsert(
        self,
        session: AsyncSession,
        write: KnowledgeWrite,
        now: datetime,
        revision_id: str = "",
    ) -> KnowledgeObject:
        """The graph write, without its transaction.

        Split out so it can share one with the lineage write. During the
        transition both are written, and two transactions would let one land
        without the other — a split brain arriving precisely while identity is
        being migrated, which is the worst possible moment for the two records
        of one assertion to disagree.

        `revision_id` is the state this write landed on, and the node is pointed
        at it. Empty where the caller has no lineage — a candidate that could not
        name its business question, or the administrative `upsert` — and then the
        pointer is left exactly as it was: a write that knows nothing about the
        state must not be able to say the state is unknown.
        """
        node = await session.get(Knowledge, write.identity, with_for_update=True)

        # Whether this source has said this before, asked before anything is
        # changed because it decides whether this is *evidence* at all.
        #
        # Keyed on the source and not on the run. A later run over the same
        # document is the same document saying the same thing again — not
        # independent corroboration, and not new information about a claim a
        # reviewer already judged. Treating a rerun as fresh evidence made a
        # claim grow more certain for being re-read and handed a rejected
        # assertion back to a reviewer on every pass.
        #
        # The observation row is still written per run, because the audit trail
        # wants to know which run saw what. Only the *belief* is guarded here.
        source_seen_before = (
            await session.execute(
                select(KnowledgeObservation.observation_id).where(
                    KnowledgeObservation.identity == write.identity,
                    KnowledgeObservation.source_id == write.source_id,
                )
            )
        ).scalars().first() is not None

        # The same run and the same source writing twice: a retry after a crash.
        # Never a sighting, and never a new observation row.
        seen_before = (
            await session.execute(
                select(KnowledgeObservation.observation_id).where(
                    KnowledgeObservation.identity == write.identity,
                    KnowledgeObservation.run_id == write.run_id,
                    KnowledgeObservation.source_id == write.source_id,
                )
            )
        ).scalar_one_or_none() is not None

        if node is None:
            node = Knowledge(
                identity=write.identity,
                kind=write.kind,
                type=write.type,
                confidence=write.confidence,
                review_required=write.review_required,
                review_reason=write.review_reason,
                product=write.product,
                version=write.version,
                revision_id=uuid.UUID(revision_id) if revision_id else None,
                created_at=now,
                updated_at=now,
            )
            session.add(node)
            await session.flush()

            # No legacy row where the derivation would have lost something. The
            # claim is already stored whole above, and retrieval reads it from
            # there — so this is a view that is sometimes absent, never a claim
            # that is sometimes missing.
            if write.payload is not None:
                table = _PAYLOAD_TABLES[write.kind]
                session.add(table(identity=write.identity, **_payload_columns(write)))
        else:
            if node.kind != write.kind:
                raise KnowledgeKindConflictError(
                    f"knowledge '{write.identity}' already exists as kind '{node.kind}'; "
                    f"refusing to rewrite it as '{write.kind}'"
                )
            # Independent sightings reinforce; the payload itself is immutable
            # because identity is derived from it. A repeat of one this run
            # already recorded is not independent and leaves belief alone —
            # otherwise a claim grows more certain for having been
            # interrupted.
            if not source_seen_before:
                # A different document asserting the same thing is independent
                # corroboration, and it is new information about a claim a
                # reviewer has already refused — so the queue predicate has to
                # see the node as changed. The same document again is neither.
                node.confidence = reinforce_confidence(node.confidence, write.confidence)
                node.updated_at = now

            # `review_required` is deliberately untouched, and `updated_at`
            # moves only above.
            #
            # Ingestion says what a source asserts. It does not get to say
            # whether a person has ruled on it, and it used to say both. Raising
            # the flag again undid an approval on the next run over an unchanged
            # page — the review policy flags every candidate, so an approved
            # assertion was re-quarantined and left retrieval for a document
            # nobody had touched.
            #
            # Stamping `updated_at` did the same through a quieter door. A
            # rejection keeps the flag set, so the queue tells an unseen
            # assertion from a judged one by comparing the review against the
            # node's last change; moving that timestamp on every sighting made
            # the node look changed and handed the rejected assertion back to a
            # reviewer, on every run, forever.
            #
            # Neither is a change to what the assertion says. Identity is
            # derived from content, so an existing node with this identity says
            # exactly what it said before; only the belief in it moved, and
            # belief is not a claim.
            #
            # The state it stands for is the one thing that does move here, and
            # it is not a change to the claim either.
            #
            # This node is the one the *arriving* state addresses — its identity
            # is that revision's fingerprint — so the pointer only ever moves
            # between revisions of one structured state: a wording corrected by
            # a full stop is the plain case. A claim whose structure moved
            # addresses a different node and never reaches this line, which is
            # what keeps the older node truthfully on the older revision.
            #
            # Leaving the pointer behind in the first case would have retrieval
            # quote a sentence the source has replaced; moving it in the second
            # would have a record whose content is P1 claim to represent R2.
            if revision_id:
                node.revision_id = uuid.UUID(revision_id)
                # A sighting is the only honest way out of an unresolved
                # migration: the state is now recorded rather than inferred.
                node.revision_unresolved = False

        if write.metadata:
            meta = await session.get(KnowledgeMetadata, write.identity)
            if meta is None:
                session.add(
                    KnowledgeMetadata(
                        identity=write.identity,
                        attributes=dict(write.metadata),
                        created_at=now,
                        updated_at=now,
                    )
                )
            else:
                meta.attributes = {**meta.attributes, **write.metadata}
                meta.updated_at = now

        if not seen_before:
            session.add(
                KnowledgeObservation(
                    observation_id=uuid.uuid4(),
                    identity=write.identity,
                    source_id=write.source_id,
                    run_id=write.run_id,
                    evidence=dict(write.evidence) or None,
                    observed_at=now,
                )
            )

        await session.flush()
        return await self._to_object(session, node)

    async def _merge(
        self,
        session: AsyncSession,
        source: Knowledge,
        target_identity: str,
        now: datetime,
    ) -> dict[str, Any]:
        """Fold one assertion into another, evidence and all.

        The source row survives. Deleting it would cascade away its observations
        and its decision history, and any index entry still naming it would find
        nothing — so it stays, points at the target, and stops being canonical.

        Its observations move. They are why the reviewer judged the two claims
        equivalent, and leaving them behind would strip the target of the
        evidence that justified the merge. The unique key is
        ``(identity, run_id, source_id)``, so a sighting the target already has
        from the same run and source is dropped rather than counted twice.

        The target's confidence then reinforces against the source's: two
        independently extracted wordings of one claim are exactly the case the
        Bayesian merge exists for, and it is what an identical re-extraction
        would have done automatically.
        """
        if target_identity == source.identity:
            raise ValueError("an assertion cannot be merged into itself")
        target = await session.get(Knowledge, target_identity, with_for_update=True)
        if target is None:
            raise ValueError(f"unknown merge target '{target_identity}'")
        if target.merged_into is not None:
            # Chains would eventually point at something that is itself not
            # canonical, and resolving them at read time turns every lookup into
            # a walk. The reviewer picks the surviving assertion directly.
            raise ValueError(
                f"merge target '{target_identity}' was itself merged into "
                f"'{target.merged_into}'; merge into that instead"
            )
        if source.merged_into is not None:
            raise ValueError(
                f"knowledge '{source.identity}' was already merged into '{source.merged_into}'"
            )

        existing = {
            (row.run_id, row.source_id)
            for row in (
                await session.execute(
                    select(KnowledgeObservation).where(
                        KnowledgeObservation.identity == target_identity
                    )
                )
            ).scalars()
        }
        moved = 0
        for observation in (
            await session.execute(
                select(KnowledgeObservation).where(
                    KnowledgeObservation.identity == source.identity
                )
            )
        ).scalars():
            if (observation.run_id, observation.source_id) in existing:
                continue
            observation.identity = target_identity
            moved += 1

        before = target.confidence
        target.confidence = reinforce_confidence(before, source.confidence)
        target.updated_at = now
        source.merged_into = target_identity
        await session.flush()

        return {
            "target": target_identity,
            "observations_moved": moved,
            "target_confidence": {"from": before, "to": target.confidence},
        }

    async def record_review(
        self,
        *,
        identity: str,
        reviewer_id: str,
        decision: ReviewDecision,
        comment: str | None = None,
        changes: dict[str, object] | None = None,
        payload: dict[str, Any] | None = None,
        confidence: float | None = None,
        merge_into: str | None = None,
        supersede: bool = False,
    ) -> Review:
        """Audit a decision, and apply a reviewer's corrections with it.

        ``payload`` and ``confidence`` belong to ``EDITED``; ``merge_into``
        belongs to ``MERGED``. Each correction and the record of who made it are
        one transaction, so the graph can never hold a changed assertion with no
        account of who changed it.

        The identity does not move. It was derived from the extracted wording,
        and rederiving it here would orphan the assertion's observations,
        decision history, and every index entry pointing at it — so after an
        edit the identity is a stable handle rather than a hash of the current
        content. The audit row keeps the previous values.

        A second decision on a version somebody already judged is refused with
        ``ReviewConflictError`` unless ``supersede`` says otherwise. Refusing is
        not pedantry: the write below is unconditional, so without it the last
        writer wins and approve-then-reject and reject-then-approve end
        differently for no reason a reviewer could see. ``supersede`` exists so
        that correcting one's own misclick stays possible — deliberately, and
        with both decisions in the history rather than one of them lost.
        """
        if not reviewer_id.strip():
            raise ValueError("reviewer_id must not be empty")
        if (payload is not None or confidence is not None) and decision is not ReviewDecision.EDITED:
            raise ValueError("only an 'edited' decision may change content or confidence")
        if confidence is not None and not 0.0 <= confidence <= 1.0:
            raise ValueError("knowledge confidence must be within [0.0, 1.0]")
        if (merge_into is None) != (decision is not ReviewDecision.MERGED):
            raise ValueError("a 'merged' decision requires a merge target, and only it may have one")

        async with AsyncSession(self._engine) as session, session.begin():
            node = await session.get(Knowledge, identity, with_for_update=True)
            if node is None:
                raise ValueError(f"unknown knowledge identity '{identity}'")

            # Before anything is mutated: has this version already been judged?
            # The row is held FOR UPDATE, so two reviewers pressing at once
            # queue here and the second sees the first's decision rather than
            # overwriting it. The predicate is the one the queue already uses —
            # a review at or after the node's last change — so an assertion that
            # gained evidence since is deliberately open again.
            if not supersede:
                existing = (
                    await session.execute(
                        select(KnowledgeReview)
                        .where(
                            KnowledgeReview.identity == identity,
                            KnowledgeReview.reviewed_at >= node.updated_at,
                        )
                        .order_by(KnowledgeReview.reviewed_at.desc())
                        .limit(1)
                    )
                ).scalar_one_or_none()
                if existing is not None:
                    raise ReviewConflictError(
                        f"'{existing.reviewer_id}' already recorded '{existing.decision}' for this "
                        f"version of '{identity}' at "
                        f"{existing.reviewed_at.isoformat(timespec='seconds')}. Deciding again "
                        f"would replace it silently; pass supersede to do it deliberately."
                    )

            recorded = dict(changes) if changes else {}
            if payload is not None:
                table = _PAYLOAD_TABLES.get(node.kind)
                if table is None:
                    raise ValueError(f"cannot edit unknown knowledge kind '{node.kind}'")
                row = await session.get(table, identity)
                if row is None:
                    raise ValueError(f"knowledge '{identity}' has no {node.kind} payload")
                # Validate through the domain payload so a hand-typed correction
                # meets the same bar as an extracted one.
                validated = payload_for(node.kind, dict(payload))
                fields = _payload_fields(validated)
                recorded["payload"] = {
                    key: {"from": getattr(row, key), "to": value}
                    for key, value in fields.items()
                    if getattr(row, key) != value
                }
                for key, value in fields.items():
                    setattr(row, key, value)
            if confidence is not None and confidence != node.confidence:
                recorded["confidence"] = {"from": node.confidence, "to": confidence}
                node.confidence = confidence

            now = _utcnow()
            if merge_into is not None:
                recorded["merged_into"] = await self._merge(session, node, merge_into, now)
            review = KnowledgeReview(
                review_id=uuid.uuid4(),
                identity=identity,
                reviewer_id=reviewer_id,
                decision=decision.value,
                comment=comment,
                changes=recorded or None,
                reviewed_at=now,
            )
            session.add(review)

            # Approving or editing lifts the quarantine; rejecting keeps the
            # assertion out of every retrieval path. Merging needs neither: the
            # source is excluded because it points at a target, not because it
            # is quarantined.
            node.review_required = decision is ReviewDecision.REJECTED
            node.review_reason = None if decision is not ReviewDecision.REJECTED else node.review_reason
            node.updated_at = now
            await session.flush()

            return Review(
                identity=identity,
                reviewer_id=reviewer_id,
                decision=decision,
                reviewed_at=now,
                comment=comment,
                changes=recorded,
            )
