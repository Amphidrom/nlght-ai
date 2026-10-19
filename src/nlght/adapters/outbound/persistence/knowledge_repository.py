# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Persist the knowledge graph through PostgreSQL transactions.

Transactions stay local to PostgreSQL and make no external backend calls.
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

from nlght.adapters.outbound.persistence.ingestion_models import IngestionDocument
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
# most. Newest first among equals. Identity last so the order is total. Paging
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
    """Read the extraction version shared by a section's claims.

    Args:
        section (ObservedSection): Observed section with assertion-lineage pairs.

    Returns:
        str: First nonempty extraction version, or an empty string.
    """
    for _, lineage in section.assertions:
        if lineage is not None and lineage.extraction_version:
            return str(lineage.extraction_version)
    return ""


def _wordings_of(held: Mapping[str, set[tuple[str, str]]], slot_id: str) -> set[str]:
    """Collect wording variants already held by a slot.

    Args:
        held (Mapping[str, set[tuple[str, str]]]): Wording and extraction-version pairs
            keyed by slot ID.
        slot_id (str): Slot whose wordings are requested.

    Returns:
        set[str]: Folded wording variants across extraction versions.
    """
    return {wording for wording, _ in held.get(slot_id, set())}


def _versions_of(held: Mapping[str, set[tuple[str, str]]], slot_id: str) -> set[str]:
    """Collect extraction versions already held by a slot.

    Args:
        held (Mapping[str, set[tuple[str, str]]]): Wording and extraction-version pairs
            keyed by slot ID.
        slot_id (str): Slot whose extraction versions are requested.

    Returns:
        set[str]: Extraction versions already held by the slot.
    """
    return {version for _, version in held.get(slot_id, set())}


#: A later revision of the same variant, aliased so a correlated subquery can ask
#: about the node's own revision without colliding with it.
orm_aliased_revision = aliased(KnowledgeRevision)


def _retired() -> ColumnElement[bool]:
    """Test retirement through a node's exact revision lineage.

    Returns:
        ColumnElement[bool]: Correlated SQL existence expression for a retired assertion.

    Side Effects:
        None. Missing revision links do not imply retirement.

    Notes:
        Uses the Knowledge row in the surrounding SQL query.
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
    """Test whether a node's revision precedes its variant's latest.

    Returns:
        ColumnElement[bool]: Correlated SQL existence expression for a later revision.

    Side Effects:
        None. Missing revision links do not imply supersession.

    Notes:
        Uses the Knowledge row in the surrounding SQL query.
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
        """Resolve current claim fields from its revision or legacy payload.

        Args:
            session (AsyncSession): Active database session.
            node (Knowledge): Knowledge node whose current fields are needed.

        Returns:
            dict[str, Any]: Proposition fields, legacy payload fields, or an empty mapping.

        Side Effects:
            Reads revision, proposition, and legacy payload rows.
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
        """Read wording from the exact revision named by a node.

        Args:
            session (AsyncSession): Active database session.
            node (Knowledge): Knowledge node naming the revision to inspect.

        Returns:
            str: Wording from that revision, or empty text without a revision.

        Side Effects:
            Reads a revision row.
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
        """Resolve a node's assertion and current document support.

        Args:
            session (AsyncSession): Active database session.
            node (Knowledge): Knowledge node whose support is requested.

        Returns:
            tuple[str, tuple[SupportingEvidence, ...]]: Assertion ID and ordered current
                supporting evidence.

        Side Effects:
            Reads revision, variant, and document-support rows.
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
                    IngestionDocument.source_id,
                )
                .outerjoin(
                    IngestionDocument,
                    IngestionDocument.document_id == KnowledgeDocumentSupport.document_id,
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
                source_id=str(source_id or ""),
            )
            for document_id, document_revision, document_path, slot_id, source_id in rows
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
                # of them. A review surface must see all three.
                #
                # The third does not live on this row. Retirement is recorded on
                # the assertion, and the graph node has no flag for it, so a
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
                # at. A node with no lineage has always
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
            # Preserve caller order because ranking comes from the search backend.
            return [
                await self._to_object(session, by_identity[identity])
                for identity in identities
                if identity in by_identity
            ]

    @staticmethod
    def _undecided() -> ColumnElement[bool]:
        """Find quarantined assertions not reviewed since their last change.

        Returns:
            ColumnElement[bool]: SQL predicate for assertions awaiting review.

        Notes:
            Decision time is compared with the node's last update. The review
            flag alone cannot distinguish unseen from rejected.
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
        """Find the assertions adjacent to one item in the review queue.

        Args:
            identity (str): Assertion identity.

        Returns:
            tuple[str | None, str | None]: Previous and next assertion IDs, or None at
                queue boundaries.

        Side Effects:
            Reads PostgreSQL.

        Notes:
            Keyset navigation survives other reviewers draining the queue.
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
        """Page assertions that may appear in search.

        Args:
            limit (int): Maximum page size.
            offset (int): Number of rows to skip.

        Returns:
            list[KnowledgeObject]: Page of searchable knowledge objects.

        Side Effects:
            Reads PostgreSQL.
        """
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
        """Page assertions that must leave the search index.

        Args:
            limit (int): Maximum page size.
            offset (int): Number of rows to skip.

        Returns:
            list[tuple[str, str]]: Page of withdrawn assertion identity and kind pairs.

        Side Effects:
            Reads PostgreSQL.
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
        """Record which assertions replace a predecessor set.

        Args:
            predecessors (Sequence[str]): Assertions being replaced.
            successors (Sequence[str]): Assertions taking their place.
            reason (str | None): Optional explanation of supersession.
            recorded_by (str | None): Optional identity of the decision maker.

        Returns:
            int: Number of recorded predecessor-successor links.

        Side Effects:
            Writes supersession and retirement rows in PostgreSQL.

        Notes:
            Every predecessor links to every successor. Rewordings keep their
            assertion identity.
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
        """List successors of one assertion in recorded order.

        Args:
            assertion_id (str): Assertion identity.

        Returns:
            tuple[str, ...]: Successor assertion IDs in recorded order.

        Side Effects:
            Reads PostgreSQL.
        """
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
        """Summarize corpus state for comparison with a later run.

        Returns:
            LineageReport: Corpus counts and lineage digest.

        Side Effects:
            Reads PostgreSQL.

        Notes:
            The digest detects changes after a rerun of an unchanged source.
        """
        async with AsyncSession(self._engine) as session:
            # Every state with where it stands. The provenance is what lets a
            # later comparison say whether a revision appeared in a document
            # somebody edited or in one nobody touched. That distinguishes an
            # expected update from drift, which counts alone cannot do.
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
            # unrecorded. Those two need separate numbers. The first is a
            # candidate that could never name a business question, the second is
            # a migration that has not finished, so they get them.
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
            # where the section was read on both sides. A slot whose section was
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
            # flow leaves that table empty, so every name came back missing.
            # Joining the two lifecycles to borrow a string is a decision of
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
        """Record one assertion sighting and its lineage.

        Args:
            write (LineageWrite): Lineage sighting to persist.

        Returns:
            RecordedLineage: The recorded assertion, variant, revision, and evidence linkage.

        Side Effects:
            Writes lineage and evidence in one PostgreSQL transaction.

        Notes:
            Entity keys decide identity. Matching fingerprints reuse the current
                revision and evidence is idempotent.
        """
        async with AsyncSession(self._engine, expire_on_commit=False) as session, session.begin():
            return await self._record_lineage(session, write, _utcnow())

    @staticmethod
    async def _candidates(
        session: AsyncSession, write: LineageWrite
    ) -> list[Candidate]:
        """Gather assertions that an arriving sighting may continue.

        Args:
            session (AsyncSession): Active PostgreSQL session.
            write (LineageWrite): Arriving lineage sighting.

        Returns:
            list[Candidate]: Current candidate assertions for the lineage write.

        Side Effects:
            Reads through the supplied PostgreSQL session.

        Notes:
            Candidates include claims from this document and claims sharing the entity
                key. Current wording is compared.
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
        # An assertion supported by several documents remains one assertion.
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
        """Apply a lineage write inside the caller transaction.

        Args:
            session (AsyncSession): Active PostgreSQL session.
            write (LineageWrite): Lineage sighting to persist.
            now (datetime): Timestamp for this write.

        Returns:
            RecordedLineage: Recorded lineage for this sighting.

        Side Effects:
            Writes lineage through the supplied transaction.
        """
        # A key the corpus already holds under a different scheme stops the write
        # before anything is decided. Both silent readings are wrong: treating it
        # as a match reads an old scheme's key as if the new one had produced it,
        # and treating it as new leaves the corpus holding one business question
        # twice. This sits ahead of the ladder rather than behind it, because the
        # ladder would often continue the claim on its wording alone and a corpus
        # half-migrated one sentence at a time is what the version exists to
        # prevent. Migration is a deliberate act. This is what makes somebody
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

        # Decide the assertion from its location and wording, never from fields
        # chosen by the model. Those fields change between runs over an unedited
        # sentence, and identity used to follow them.
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
        # the source plainly said, and it was reached before this transaction
        # opened, because asking a model here would hold row locks across the
        # network.
        #
        # Which is exactly why it is checked again here. The judgement was formed
        # against the slot this section was *expected* to resolve to. The
        # authoritative placement happens inside this transaction and can land
        # elsewhere if the registry moved or recovery answered differently.
        # Applying it anyway would let a verdict reached about slot A continue an
        # assertion standing in slot B, which is the one thing the slot scope
        # exists to prevent.
        #
        # Judged yes is not applied yes. The assessment stays in the audit either
        # way, because the judge really was asked and really did answer. What it
        # does not get to do is outlive the premise it was asked under.
        continues = decision.assertion_id
        if continues is None and write.continues is not None:
            # Revalidated against the premise it was actually judged under, and
            # there are two. A candidate found in this slot was judged as a claim
            # standing here, so it has to still be standing here. One found
            # across documents was judged as a live claim sharing enough
            # structure to be worth asking about. Slot membership was never part
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
        # contained. A live case changed `defaults to the same as` into `is by
        # default identical to`.
        #
        # And the fingerprint may not decide it in the other direction either. A
        # model that describes an unedited sentence with different fields has
        # changed its description, not the state. Writing a revision for that is
        # the drift ADR-0043 removed, arriving through the history instead of
        # through identity.
        #
        # So: the wording alone, folded for whitespace, because a parser may
        # rewrap a line and that changes no word. Anything a reader would see is
        # a new state, which is what a revision is for, and what keeps
        # `semantic_change` able to say whether the approval survives it.
        unchanged = latest is not None and (
            fold_whitespace(latest.text) == fold_whitespace(write.text)
        )
        # Every sighting is an observation, not only the ones a judgement was
        # asked about. Without this a candidate has no historical row to point
        # at, and the comparison would have to invent one, which is how a
        # sighting becomes a sighting per comparison.
        #
        # Reused where `_assess` already recorded it: the same sighting, seen
        # once, whatever was decided about it afterwards.
        observation_id = write.observation_id or await self._record_observation(
            session, write, write.slot_id or ""
        )

        # The complete structure this sighting carried, written before the
        # revision that names it. Content-addressed, so the same structure seen
        # in two documents is one row for storage economy, not identity. Whether
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
            # one. The caller may say so outright. Beyond that, only here is the
            # previous state known, so only here can the change be seen at all.
            #
            # A quantity that moved is a normative change: CHF 500 becoming CHF
            # 550 continues the same rule and a person approved five hundred, not
            # five fifty. A rewording that leaves the quantities alone is not, and
            # its approval carries, which is the whole reason the assertion id is
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
                # is *backfilling* one from that payload, and not because the
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
                # assertion. The review policy upstream is what quarantines it,
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
        # slot recorded only the first, which left the support view unable to say
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
        """Find same-kind assertions for a review decision.

        Args:
            identity (str): Assertion identity.
            limit (int): Maximum page size.

        Returns:
            list[KnowledgeObject]: Nearby assertions of the same kind and type.

        Side Effects:
            Reads PostgreSQL.
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
        """Persist an assertion and its optional lineage atomically.

        Args:
            write (KnowledgeWrite): Knowledge graph write to persist.
            lineage (LineageWrite | None): Optional lineage write accompanying a graph write.

        Returns:
            tuple[KnowledgeObject, RecordedLineage | None]: Stored knowledge object and
                optional recorded lineage.

        Side Effects:
            Writes graph and lineage in one PostgreSQL transaction.

        Notes:
            Graph and lineage roll back together. A fact proposition requires a lineage record.
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
        """List live assertions carried by one document.

        Args:
            session (AsyncSession): Active PostgreSQL session.
            document_id (str): Source document identity.

        Returns:
            set[str]: IDs of live assertions supported by this document.

        Side Effects:
            Reads through the supplied PostgreSQL session.

        Notes:
            Retraction is document-wide because inserting a section can renumber slots.
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
        """Check whether another document still supports a claim.

        Args:
            session (AsyncSession): Active PostgreSQL session.
            assertion_id (str): Assertion identity.
            document_id (str): Source document identity.

        Returns:
            bool: Whether another document supplies evidence for this assertion.

        Side Effects:
            Reads through the supplied PostgreSQL session.

        Notes:
            Any evidence from another document conservatively keeps the assertion live.
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
        """Read the current structured claim for each assertion.

        Args:
            session (AsyncSession): Active PostgreSQL session.
            assertion_ids (Sequence[str]): Assertion identities to inspect.

        Returns:
            dict[str, tuple[uuid.UUID, Proposition, str]]: Current revision,
                proposition, and kind by assertion ID.

        Side Effects:
            Reads through the supplied PostgreSQL session.
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
                    # judge, so it can tell what a field means, never the thing
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
        """Build the judge's observation from extracted source content.

        Args:
            write (LineageWrite): A lineage write with kind, observed text, and proposition.

        Returns:
            AssertionObservation | None: AssertionObservation or None when wording or
                structure is absent.
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
        """Read each assertion candidate's latest recorded sighting.

        Args:
            session (AsyncSession): Active PostgreSQL session.
            assertion_ids (Sequence[str]): Assertion identities to inspect.

        Returns:
            dict[str, uuid.UUID]: Latest observation ID by assertion ID.

        Side Effects:
            Reads through the supplied PostgreSQL session.

        Notes:
            Missing legacy sightings remain absent. Comparison must not invent audit events.
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
        """Read each candidate as a current assertion observation.

        Args:
            session (AsyncSession): Active PostgreSQL session.
            candidates (Sequence[Candidate]): Assertions eligible for this comparison.

        Returns:
            dict[str, AssertionObservation]: Current observation by candidate assertion ID.

        Side Effects:
            Reads through the supplied PostgreSQL session.
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
    #: than a coincidence. One shared subject is true of half a corpus, and the
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
        """Retrieve bounded same-kind candidates beyond the current slot.

        Args:
            session (AsyncSession): Active database session.
            write (LineageWrite): Arriving assertion sighting.
            exclude (set[str]): Assertion IDs already considered.

        Returns:
            list[Candidate]: At most eight live same-kind candidates with shared field values.

        Side Effects:
            Reads assertion, revision, proposition, and evidence rows.
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
        """Assess same-slot candidates when deterministic matching fails.

        Args:
            session (AsyncSession): Active PostgreSQL session.
            write (LineageWrite): Arriving lineage sighting.
            candidates (Sequence[Candidate]): Assertions eligible for this comparison.
            assess (EquivalenceJudge | None): Optional equivalence judge.
            slot_id (str | None): Section slot identity.

        Returns:
            tuple[str | None, list[tuple[uuid.UUID, uuid.UUID, str]], uuid.UUID | None, str]:
                Matching assertion ID, assessment links, observation ID, and verdict.

        Side Effects:
            May call the equivalence judge. Reads through the supplied session.

        Notes:
            Only candidates in the same slot are judged, outside a write transaction.
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
        # run and already has a row. Re-recording it here would stamp it with
        # the run that compared rather than the run that saw.
        sightings = await self._sightings_of(session, list(known))
        verdicts: list[tuple[str, str]] = []
        assessments: list[tuple[uuid.UUID, uuid.UUID, str]] = []
        for assertion_id, candidate in sorted(known.items()):
            theirs = sightings.get(assertion_id)
            # The model is asked only for the band nothing else can answer.
            # Identical structures are the same claim and a moved quantity is a
            # different one. Neither needs semantic interpretation, and asking
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
                # verdict still counts. Refusing to judge an older claim would
                # make the corpus behave differently on either side of a
                # migration, but nothing fabricates an event to file it under.
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
        """Resolve changed claim bodies before persistence.

        Args:
            writes (Sequence[tuple[KnowledgeWrite, LineageWrite | None]]): Graph writes
                paired with optional lineage writes.
            classify (Callable[[str, str], Awaitable[str]] | None): Optional classifier
                for changed wording.
            assess (EquivalenceJudge | None): Optional equivalence judge.
            run_id (str | None): Observation run identity.
            slot_id (str | None): Section slot for the changed claims, when known.

        Returns:
            list[tuple[KnowledgeWrite, LineageWrite | None]]: Writes paired with their
                resolved lineage decisions.

        Side Effects:
            May call the classifier and equivalence judge. Reads PostgreSQL.

        Notes:
            Classifier and judge calls finish before write transactions begin.
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
                # Against whichever assertion this sighting continues,
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
        """Store one complete structured proposition.

        Args:
            proposition (Proposition): Complete structured proposition.

        Returns:
            uuid.UUID: ID of the stored proposition.

        Side Effects:
            Writes PostgreSQL.
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
        """Record a sighting inside the caller transaction.

        Args:
            session (AsyncSession): Active PostgreSQL session.
            write (LineageWrite): Lineage sighting to record.
            slot_id (str): Section slot identity.

        Returns:
            uuid.UUID | None: ID of the recorded observation, or None when absent.

        Side Effects:
            Writes through the supplied PostgreSQL transaction.
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
        """Record each assertion sighting before an equivalence decision.

        Args:
            observation (AssertionObservation): Assertion state observed in this run.
            run_id (str): Identity of the observation run.
            document_id (str): Source document identity, if available.
            slot_id (str): Source section slot, if available.

        Returns:
            uuid.UUID: Unique ID for this sighting.

        Side Effects:
            Inserts a new observation event without deduplicating it.
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
        """Append a judgement about two observations.

        Args:
            left (uuid.UUID): First observation identity.
            right (uuid.UUID): Second observation identity.
            verdict (str): Equivalence verdict to record.
            judge (Judge | None): Judge that made the decision.
            run_id (str | None): Observation run identity.

        Returns:
            uuid.UUID: ID of the new equivalence assessment.

        Side Effects:
            Writes a judgement to PostgreSQL.

        Notes:
            Judgements are append-only, including repeated or conflicting assessments.
        """
        async with AsyncSession(self._engine) as session, session.begin():
            row = KnowledgeEquivalenceAssessment(
                # Canonical, because "X is equivalent to Y" is the same question
                # as "Y is equivalent to X". The assertions themselves stay
                # directed. Only the *relation between them* is symmetric.
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
        """Read the slots of one document for resolution.

        Args:
            session (AsyncSession): Active PostgreSQL session.
            document_id (str): Source document identity.

        Returns:
            dict[str, dict[str, Any]]: Slot attributes keyed by slot ID.

        Side Effects:
            Reads through the supplied PostgreSQL session.
        """
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
        """Read document support by assertion and slot.

        Args:
            session (AsyncSession): Active PostgreSQL session.
            document_id (str): Source document identity.

        Returns:
            set[tuple[str, str]]: Supported assertion and slot pairs.

        Side Effects:
            Reads through the supplied PostgreSQL session.
        """
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
        """Read the last exact parser input for each slot.

        Args:
            session (AsyncSession): Active PostgreSQL session.
            document_id (str): Source document identity.

        Returns:
            dict[str, str]: Latest exact section text by slot ID.

        Side Effects:
            Reads through the supplied PostgreSQL session.

        Notes:
            These are exact parser inputs, not normalized fingerprints.
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
        """Read observed wording and extraction versions by slot.

        Args:
            session (AsyncSession): Active PostgreSQL session.
            document_id (str): Source document identity.

        Returns:
            dict[str, set[tuple[str, str]]]: Observed wording and extraction-version
                pairs by slot ID.

        Side Effects:
            Reads through the supplied PostgreSQL session.
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
        """Reconcile support after a full-document observation.

        Args:
            session (AsyncSession): Active PostgreSQL session.
            document_id (str): Source document identity.
            carries (Mapping[tuple[str, str], LineageWrite]): Claims observed in the
                document, keyed by assertion and slot.
            unchanged (set[str]): Slots protected from retraction by stable input.
            now (datetime): Timestamp for this write.

        Side Effects:
            Replaces document support through the supplied transaction.

        Notes:
            Only a full-document read can replace support. Protected unchanged slots
                keep their prior evidence revision.
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
        """Retire section slots absent from a full document read.

        Args:
            session (AsyncSession): Active PostgreSQL session.
            document_id (str): Source document identity.
            resolved (set[str]): Slot IDs resolved during this run.
            now (datetime): Timestamp for this write.

        Returns:
            list[str]: IDs of slots retired by this full-document observation.

        Side Effects:
            Retires slots and closes anchors through the supplied transaction.

        Notes:
            Retiring a slot does not retire its document-wide claims.
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
        """Resolve a section to a slot and record real anchor transitions.

        Args:
            session (AsyncSession): Active PostgreSQL session.
            document_id (str): Source document identity.
            section (ObservedSection): Arriving observed section.
            known (dict[str, dict[str, Any]]): Known slot values for this document.
            claimed (set[str]): Slot IDs already claimed in this run.
            now (datetime): Timestamp for this write.

        Returns:
            str: Resolved or newly created slot ID.

        Side Effects:
            May write a slot and its anchor history through the supplied transaction.

        Notes:
            Anchor history advances only when a section actually changes place or name.
        """
        # A slot another section of *this* run already took is not a candidate.
        # Recovery exists to find a slot from a previous run. Offering it a
        # sibling would let two sections of one document, alike in text and
        # different in heading, collapse into one, and which of them won would
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
            # The section's text moves with every edit. The anchor usually does
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
            # The slot continues as the same section, but it stopped being
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
        """Reconcile every claim of one fully observed document.

        Args:
            document_id (str): Identity of the fully observed document.
            sections (Sequence[ObservedSection]): Every section produced by this document read.
            run_id (str): Observation run identity.
            classify (Callable[[str, str], Awaitable[str]] | None): Optional classifier
                for changed wording.
            assess (EquivalenceJudge | None): Optional equivalence judge.

        Returns:
            DocumentOutcome: Outcome of the complete document observation.

        Side Effects:
            Writes claims, evidence, support, and retractions in one transaction.
            May call judges.

        Notes:
            Call only after a complete document read. Claims and retractions
            share one transaction, and evidence is retained.
        """
        # Asked before the transaction opens, never inside it. This module keeps
        # its transactions short and free of external calls, and a model round
        # trip per changed body would hold row locks across the network.
        # Which slot each section probably is, resolved read-only so the
        # equivalence question can be scoped to the claims already standing
        # *there*. The parser's ordinal renumbers whenever a paragraph is
        # inserted above it, so scoping by that would have compared this
        # section's claims against whatever now happens to sit at its old
        # number, and usually against nothing at all.
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
                    # is normalised on purpose. A reflowed section is still the
                    # same section, so using it here answered "is this probably
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
                    # rung can match against what the section says now, which is
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
            #: What this document carries now: each (revision, slot) names the sighting
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
                    # is, and the graph node is then addressed by that state.
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
                    # Reinforcement still lands because a sighting of something the
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
                # before, so the source did not drop it. The model did not
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
        """Store a proposition inside the caller transaction.

        Args:
            session (AsyncSession): Active PostgreSQL session.
            proposition (Proposition): Complete structured proposition.

        Returns:
            uuid.UUID: ID of the stored proposition.

        Side Effects:
            Writes through the supplied PostgreSQL transaction.
        """
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
        """Upsert the knowledge graph row inside the caller transaction.

        Args:
            session (AsyncSession): Active PostgreSQL session.
            write (KnowledgeWrite): Knowledge graph write to apply.
            now (datetime): Timestamp for this write.
            revision_id (str): Revision reached by the lineage write, when available.

        Returns:
            KnowledgeObject: Stored knowledge object.

        Side Effects:
            Writes graph rows through the supplied transaction.

        Notes:
            A graph write without lineage leaves the existing revision pointer unchanged.
        """
        node = await session.get(Knowledge, write.identity, with_for_update=True)

        # Whether this source has said this before, asked before anything is
        # changed because it decides whether this is *evidence* at all.
        #
        # Keyed on the source and not on the run. A later run over the same
        # document is the same document saying the same thing again, not
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
            # there, so this is a view that is sometimes absent, never a claim
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
            # Independent sightings reinforce. The payload itself is immutable
            # because identity is derived from it. A repeat of one this run
            # already recorded is not independent and leaves belief alone.
            # Otherwise a claim grows more certain for having been interrupted.
            if not source_seen_before:
                # A different document asserting the same thing is independent
                # corroboration, and it is new information about a claim a
                # reviewer has already refused, so the queue predicate has to
                # see the node as changed. The same document again is neither.
                node.confidence = reinforce_confidence(node.confidence, write.confidence)
                node.updated_at = now

            # `review_required` is deliberately untouched, and `updated_at`
            # moves only above.
            #
            # Ingestion says what a source asserts. It does not get to say
            # whether a person has ruled on it, and it used to say both. Raising
            # the flag again undid an approval on the next run over an unchanged
                # page. The review policy flags every candidate, so an approved
            # assertion was re-quarantined and left retrieval for a document
            # nobody had touched.
            #
            # Stamping `updated_at` did the same through a quieter door. A
            # rejection keeps the flag set, so the queue tells an unseen
            # assertion from a judged one by comparing the review against the
            # node's last change. Moving that timestamp on every sighting made
            # the node look changed and handed the rejected assertion back to a
            # reviewer, on every run, forever.
            #
            # Neither is a change to what the assertion says. Identity is
            # derived from content, so an existing node with this identity says
            # exactly what it said before. Only the belief in it moved, and
            # belief is not a claim.
            #
            # The state it stands for is the one thing that does move here, and
            # it is not a change to the claim either.
            #
            # This node is the one the *arriving* state addresses. Its identity
            # is that revision's fingerprint, so the pointer only ever moves
            # between revisions of one structured state: a wording corrected by
            # a full stop is the plain case. A claim whose structure moved
            # addresses a different node and never reaches this line, which is
            # what keeps the older node truthfully on the older revision.
            #
            # Leaving the pointer behind in the first case would have retrieval
            # quote a sentence the source has replaced. Moving it in the second
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
        """Merge one assertion into another while retaining its history.

        Args:
            session (AsyncSession): Active PostgreSQL session.
            source (Knowledge): Assertion being merged into another.
            target_identity (str): Identity of the merge target.
            now (datetime): Timestamp for this write.

        Returns:
            dict[str, Any]: Summary of the merged source and target.

        Side Effects:
            Moves evidence and updates assertion rows through the supplied transaction.

        Notes:
            The source row survives as a noncanonical handle while its evidence
            moves to the target.
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
        """Record a version-guarded review and apply its corrections.

        Args:
            identity (str): Assertion identity to review.
            reviewer_id (str): Identity of the reviewer.
            decision (ReviewDecision): Review decision to apply.
            comment (str | None): Optional review explanation.
            changes (dict[str, object] | None): Optional field corrections.
            payload (dict[str, Any] | None): Optional replacement payload.
            confidence (float | None): Optional corrected confidence.
            merge_into (str | None): Optional target assertion for a merge.
            supersede (bool): Whether to replace a prior decision on this version.

        Returns:
            Review: Persisted review decision.

        Side Effects:
            Writes review history and corrections in one PostgreSQL transaction.

        Notes:
            A second decision on the same version needs supersede. The audit preserves
                both decisions.
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
            # overwriting it. The predicate is the one the queue already uses:
            # a review at or after the node's last change, so an assertion that
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

            # Approving or editing lifts the quarantine. Rejecting keeps the
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
