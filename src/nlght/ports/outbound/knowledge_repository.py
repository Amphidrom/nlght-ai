# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Protocol

from nlght.core.knowledge import (
    DocumentOutcome,
    ExtractionState,
    ExtractionStateWrite,
    KnowledgeObject,
    KnowledgeStatistics,
    KnowledgeWrite,
    LineageWrite,
    Observation,
    ObservedSection,
    RecordedLineage,
    Review,
    ReviewDecision,
)


class KnowledgeRepository(Protocol):
    """PostgreSQL authority for the knowledge graph.

    Writes are identity-addressed and idempotent: re-extracting the same
    assertion reinforces its confidence and records another observation rather
    than creating a duplicate. A retried workflow step contributes the same
    observation once, because observations are unique per
    ``(identity, run_id, source_id)``.

    Reads default to *canonical* knowledge only. An assertion stops being
    canonical two ways: quarantined by review — still awaiting a decision, or
    refused — or merged into another assertion. Neither may reach a retrieval
    path (ADR-0032); a review surface asks for both.
    """

    async def upsert(self, write: KnowledgeWrite) -> KnowledgeObject:
        """Insert a new assertion, or reinforce and re-observe an existing one."""
        ...

    async def resolve(
        self,
        identities: Sequence[str],
        *,
        include_quarantined: bool = False,
    ) -> list[KnowledgeObject]:
        """Resolve identities to full objects, preserving the given order."""
        ...

    async def pending_review(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> list[KnowledgeObject]:
        """The undecided part of the quarantine, least-confident first.

        An assertion is *pending* while it is quarantined and no review has been
        recorded since it last changed. A decision therefore removes it from the
        queue even when the decision was to reject — rejection keeps the
        quarantine, so ``review_required`` alone can never distinguish "nobody
        has looked at this" from "somebody looked and said no".

        Fresh evidence reopens it: ``upsert`` bumps the node past its last
        review, which is the intended behaviour — a rejected assertion seen
        again in another document deserves a second look.
        """
        ...

    async def pending_review_count(self) -> int:
        """How many assertions are awaiting a decision, for paging the queue."""
        ...

    async def queue_neighbours(self, identity: str) -> tuple[str | None, str | None]:
        """The queue entries immediately before and after this one, if any."""
        ...

    async def statistics(self) -> KnowledgeStatistics:
        """Counts and averages over the graph, for the review surface."""
        ...

    async def canonical_page(
        self, *, limit: int = 200, offset: int = 0
    ) -> list[KnowledgeObject]:
        """Assertions that may be found: reviewed, approved, not merged away.

        What ``knowledge.publish`` copies into the search index. Ordered by
        identity so paging stays stable while the graph changes under it.
        """
        ...

    async def withdrawn_page(
        self, *, limit: int = 200, offset: int = 0
    ) -> list[tuple[str, str]]:
        """Identity and kind of assertions that must not be findable.

        Quarantined, refused, or merged away. Only the two fields a removal
        needs — the full object would be work done to then delete it by id.
        """
        ...

    async def persist(
        self, write: KnowledgeWrite, lineage: LineageWrite | None = None
    ) -> tuple[KnowledgeObject, RecordedLineage | None]:
        """Write the graph and the lineage of one assertion in one transaction.

        They must not be able to disagree while identity is being migrated, so
        a failure on either side rolls back both. A `lineage` of `None` is the
        caller saying this assertion cannot name a business question yet — not a
        fallback, and never a substitute key invented from its wording.
        """
        ...

    async def record_document(
        self,
        *,
        document_id: str,
        sections: Sequence[ObservedSection],
        run_id: str,
        classify: Callable[[str, str], Awaitable[str]] | None = None,
    ) -> DocumentOutcome:
        """Everything one fully observed document says, and what it stopped saying.

        **Only for a document observed in full.** Retraction is a set operation:
        a claim is gone because the source no longer contains it, which is
        knowable only once everything it *does* contain has arrived. A partial
        observation — the normal case, since an unchanged document is skipped
        whole and yields nothing — would read as a document that lost
        everything.

        The container is the document, not the slot. A slot id was a section's
        position, so inserting a paragraph renumbered every section below it, and
        slot-scoped retraction then retired each unchanged claim from the slot it
        used to occupy and re-created it in the one it had moved to. The slot
        remains what matches a claim; the document is what decides one is gone.

        Each section resolves to a persisted slot where it has something to be
        found by, and its claims are recorded against that rather than against
        the ordinal the parser produced. A format with no headings gets no slot
        and keeps the document as its diff boundary.

        One transaction for the whole document, because writes landing without
        their retractions leave the corpus asserting and denying the same thing
        with nothing able to say which half is right — and because deciding
        retraction per slot made the outcome depend on the order the slots
        happened to be written in.

        Evidence is never deleted. It answers what document D at revision R
        asserted, and a claim that stopped being asserted did not stop having
        been asserted; what is recorded is that the assertion is retired.

        `classify` answers whether a changed body spends its approval, and is
        asked only for the band nothing else can settle — same quantities,
        different words. It is called before the transaction opens, never inside
        it. Without it the deterministic answer stands, which means a reworded
        rule keeps its approval and a moved quantity does not.
        """
        raise NotImplementedError

    async def record_supersession(
        self,
        *,
        predecessors: Sequence[str],
        successors: Sequence[str],
        reason: str | None = None,
        recorded_by: str | None = None,
    ) -> int:
        """Record that one set of assertions took another's place.

        Sets rather than a pair: a rule splits into two, or two fold into one,
        and both happen routinely. A rewording does not come here — it keeps its
        assertion and adds a revision. Returns how many edges were written, so a
        retry that records the same succession again writes none.
        """
        ...

    async def supersessions_of(self, assertion_id: str) -> tuple[str, ...]:
        """What took this assertion's place, in the order it was recorded."""
        ...

    async def record_lineage(self, write: LineageWrite) -> RecordedLineage:
        """Place one sighting of one claim: which assertion, case and state.

        Decides in that order and never by similarity: the entity key resolves
        the assertion deterministically, the scope resolves the case by the
        variant ladder, and a fingerprint matching the case's latest state means
        no new revision — which is what makes a second run over an unchanged
        source leave the corpus alone.
        """
        ...

    async def extraction_state(self, document_id: str) -> ExtractionState | None:
        """What is recorded about extracting this document, if anything.

        ``None`` means never extracted, which is not the same as current: the
        absent case licenses no skip.
        """
        ...

    async def record_extraction(self, write: ExtractionStateWrite) -> ExtractionState:
        """Record that a document has been extracted under a given extraction.

        Written after the extraction succeeded, never before — the same
        ordering the data layer's ``record_indexed`` has, so a crash in between
        redoes the work rather than losing it.
        """
        ...

    async def similar(self, identity: str, *, limit: int = 5) -> list[KnowledgeObject]:
        """Other assertions of the same kind and type, quarantined ones included."""
        ...

    async def observations(self, identity: str) -> list[Observation]:
        """Every recorded sighting of an assertion, most recent first.

        This is the evidence a reviewer judges: without it the queue only shows
        the claim, never where it came from.
        """
        ...

    async def reviews(self, identity: str) -> list[Review]:
        """The decision history of an assertion, most recent first."""
        ...

    async def record_review(
        self,
        *,
        identity: str,
        reviewer_id: str,
        decision: ReviewDecision,
        comment: str | None = None,
        changes: dict[str, object] | None = None,
        payload: dict[str, object] | None = None,
        confidence: float | None = None,
        merge_into: str | None = None,
        supersede: bool = False,
    ) -> Review:
        """Audit a review outcome and lift or keep the quarantine accordingly.

        Raises ``ReviewConflictError`` when this version of the assertion has
        already been judged, unless ``supersede`` is set. Without that guard the
        second of two decisions silently replaces the first.

        ``payload`` and ``confidence`` apply a reviewer's corrections in the
        same transaction as the audit row, and only for an ``EDITED`` decision.
        The identity does not move with the content.

        ``merge_into`` belongs to ``MERGED`` and only to it: the source keeps its
        history, hands its observations to the target, reinforces the target's
        confidence, and stops being canonical.
        """
        ...
