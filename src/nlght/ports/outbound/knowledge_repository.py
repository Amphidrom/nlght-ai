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
    """Persist identity-addressed knowledge and expose canonical graph reads.

    Deduplicate observations by ``(identity, run_id, source_id)`` so a retried
    workflow step cannot reinforce the same evidence twice. Exclude quarantined,
    refused, and merged assertions from canonical retrieval (ADR-0032).
    """

    async def upsert(self, write: KnowledgeWrite) -> KnowledgeObject:
        """Insert or reinforce an assertion and record its observation.

        Args:
            write (KnowledgeWrite): The identity-addressed knowledge write.

        Returns:
            KnowledgeObject: The persisted knowledge object.

        Side Effects:
            Writes the graph and observation records.
        """
        ...

    async def resolve(
        self,
        identities: Sequence[str],
        *,
        include_quarantined: bool = False,
    ) -> list[KnowledgeObject]:
        """Resolve identities in the requested order.

        Args:
            identities (Sequence[str]): Assertion identities in caller order.
            include_quarantined (bool): Whether pending assertions may be returned.

        Returns:
            list[KnowledgeObject]: Matching knowledge objects in input order.

        Side Effects:
            Reads the knowledge graph.
        """
        ...

    async def pending_review(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
    ) -> list[KnowledgeObject]:
        """List undecided quarantined assertions by confidence.

        Args:
            limit (int): Maximum page size.
            offset (int): Number of queue entries to skip.

        Returns:
            list[KnowledgeObject]: Page of undecided quarantined assertions.

        Side Effects:
            Reads the review queue.
        """
        ...

    async def pending_review_count(self) -> int:
        """Count assertions awaiting a review decision.

        Returns:
            int: The pending review count.

        Side Effects:
            Reads the review queue.
        """
        ...

    async def queue_neighbours(self, identity: str) -> tuple[str | None, str | None]:
        """Find the adjacent pending review entries.

        Args:
            identity (str): The current assertion identity.

        Returns:
            tuple[str | None, str | None]: Previous and next identities, or None at either boundary.

        Side Effects:
            Reads the review queue.
        """
        ...

    async def statistics(self) -> KnowledgeStatistics:
        """Summarize the graph for the review surface.

        Returns:
            KnowledgeStatistics: Knowledge counts and averages.

        Side Effects:
            Reads the knowledge graph.
        """
        ...

    async def canonical_page(
        self, *, limit: int = 200, offset: int = 0
    ) -> list[KnowledgeObject]:
        """Page canonical assertions in identity order.

        Args:
            limit (int): Maximum page size.
            offset (int): Number of canonical entries to skip.

        Returns:
            list[KnowledgeObject]: Reviewed, approved, unmerged assertions.

        Side Effects:
            Reads the knowledge graph.
        """
        ...

    async def withdrawn_page(
        self, *, limit: int = 200, offset: int = 0
    ) -> list[tuple[str, str]]:
        """Page identities that must leave the search index.

        Args:
            limit (int): Maximum page size.
            offset (int): Number of withdrawn entries to skip.

        Returns:
            list[tuple[str, str]]: Assertion identity and kind pairs to remove from search.

        Side Effects:
            Reads the knowledge graph.
        """
        ...

    async def persist(
        self, write: KnowledgeWrite, lineage: LineageWrite | None = None
    ) -> tuple[KnowledgeObject, RecordedLineage | None]:
        """Persist an assertion and optional lineage atomically.

        Args:
            write (KnowledgeWrite): Assertion graph write.
            lineage (LineageWrite | None): Optional sighting and lineage write.

        Returns:
            tuple[KnowledgeObject, RecordedLineage | None]: Persisted assertion and any
                recorded lineage.

        Side Effects:
            Writes graph and lineage atomically.
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
        """Reconcile a fully observed document and its prior claims.

        Args:
            document_id (str): Identity of a fully observed document.
            sections (Sequence[ObservedSection]): Every observed section in the document.
            run_id (str): Identity of this observation run.
            classify (Callable[[str, str], Awaitable[str]] | None): Optional classifier
                for changed claims.

        Returns:
            DocumentOutcome: Recorded changes and retractions for the document.

        Side Effects:
            May classify before writing. Records claims and retractions atomically
                without deleting evidence.
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
        """Connect predecessor and successor assertion sets.

        Args:
            predecessors (Sequence[str]): Assertions being replaced.
            successors (Sequence[str]): Assertions that replace them.
            reason (str | None): Optional explanation of the succession.
            recorded_by (str | None): Identity of the decision maker, if known.

        Returns:
            int: Number of newly recorded predecessor-successor edges.

        Side Effects:
            Writes idempotent supersession links.
        """
        ...

    async def supersessions_of(self, assertion_id: str) -> tuple[str, ...]:
        """Resolve an assertion's recorded successors.

        Args:
            assertion_id (str): Assertion identity.

        Returns:
            tuple[str, ...]: Successor identities in recording order.

        Side Effects:
            Reads supersession edges.
        """
        ...

    async def record_lineage(self, write: LineageWrite) -> RecordedLineage:
        """Record a claim sighting against its identity and case.

        Args:
            write (LineageWrite): The lineage write with source and claim details.

        Returns:
            RecordedLineage: The recorded lineage.

        Side Effects:
            Writes a revision when the case state changes.
        """
        ...

    async def extraction_state(self, document_id: str) -> ExtractionState | None:
        """Read a document's latest extraction state.

        Args:
            document_id (str): Document identity.

        Returns:
            ExtractionState | None: Recorded extraction state, or None when never extracted.

        Side Effects:
            Reads extraction records.
        """
        ...

    async def record_extraction(self, write: ExtractionStateWrite) -> ExtractionState:
        """Record a successful document extraction.

        Args:
            write (ExtractionStateWrite): The completed extraction state write.

        Returns:
            ExtractionState: The recorded extraction state.

        Side Effects:
            Writes the extraction record.
        """
        ...

    async def similar(self, identity: str, *, limit: int = 5) -> list[KnowledgeObject]:
        """Find same-kind, same-type assertion candidates.

        Args:
            identity (str): Assertion whose neighbours are requested.
            limit (int): Maximum number of candidates.

        Returns:
            list[KnowledgeObject]: Same-kind, same-type candidates, including
                quarantined assertions.

        Side Effects:
            Reads the knowledge graph.
        """
        ...

    async def observations(self, identity: str) -> list[Observation]:
        """List an assertion's recorded sightings.

        Args:
            identity (str): Assertion identity.

        Returns:
            list[Observation]: Observations in descending recency order.

        Side Effects:
            Reads observation records.
        """
        ...

    async def reviews(self, identity: str) -> list[Review]:
        """List an assertion's review decisions.

        Args:
            identity (str): Assertion identity.

        Returns:
            list[Review]: Reviews in descending recency order.

        Side Effects:
            Reads review records.
        """
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
        """Apply and audit a version-guarded review decision.

        Args:
            identity (str): Assertion to review.
            reviewer_id (str): Identity of the reviewer.
            decision (ReviewDecision): Review decision to record.
            comment (str | None): Optional decision explanation.
            changes (dict[str, object] | None): Optional field edits.
            payload (dict[str, object] | None): Optional replacement payload.
            confidence (float | None): Optional corrected confidence.
            merge_into (str | None): Optional target assertion for a merge.
            supersede (bool): Whether to replace a previous decision on this version.

        Returns:
            Review: Recorded review and applied decision.

        Raises:
            ReviewConflictError: This version was already judged and supersede is false.

        Side Effects:
            Atomically writes review history and allowed corrections. May merge evidence.
        """
        ...
