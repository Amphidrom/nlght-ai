# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from nlght.core.ingestion.state import (
    IndexState,
    IndexStateSummary,
    IndexStateWrite,
    SnapshotCommit,
    SnapshotCommitResult,
)


class IngestionRepository(Protocol):
    """PostgreSQL authority for ingested sources, documents, and index state.

    Indexing is idempotent per document rather than generation-scoped: a write
    step replaces one document's payload in one target and records the state
    below, so retrieval stays available throughout indexing without an atomic
    cutover.

    The index *generation* is a different thing from those retired copy-on-write
    generations. It does not switch a corpus over; it names which incarnation of
    a target's indexes the recorded state describes, so that recreating an index
    invalidates the record instead of silently contradicting it.
    """

    async def commit_snapshot(self, command: SnapshotCommit) -> SnapshotCommitResult: ...

    async def current_processing_revisions(
        self, document_ids: Sequence[str]
    ) -> dict[str, str]:
        """Which processed fassung of each of these documents is published.

        The currency half of the authority, and deliberately separate from
        reading content: deciding whether a candidate is still current needs
        identity only, and a search asks it of every candidate — up to twice
        `fetch_k` of them — while content is read for the handful that survive
        the ranking (ADR-0062).

        A tombstoned document is absent from the result rather than present with
        a revision. It has no published fassung, so every candidate naming it is
        stale, which is the same answer by a different route.
        """
        ...

    async def read_revision_contents(
        self, keys: Sequence[tuple[str, str]]
    ) -> dict[tuple[str, str], str]:
        """The stored text of these `(document_id, processing_revision_id)` pairs.

        The authority itself. A pair with no row, or a revision that was observed
        without usable text, is absent from the result — the caller then has a
        finding it cannot answer with, which is the honest outcome and not an
        occasion to fall back to an index payload.
        """
        ...

    async def get_index_state(self, *, document_id: str, target: str) -> IndexState | None: ...

    async def record_indexed(self, write: IndexStateWrite) -> IndexState:
        """Idempotently upsert the index state for one document and target."""
        ...

    async def clear_index_state(self, *, document_id: str, target: str) -> bool:
        """Drop the state for a document removed from a target. False if absent."""
        ...

    async def current_generation(self, *, target: str) -> str:
        """The generation of the target's indexes, minting one if none exists."""
        ...

    async def rotate_generation(self, *, target: str) -> str:
        """Declare the target's indexes a new incarnation; returns the one now in force.

        Called when the bootstrap had to create a collection or index: what is
        there is not what the recorded state describes, so everything must be
        written again.
        """
        ...

    async def index_state_summary(self, *, target: str) -> IndexStateSummary:
        """What the database expects of one target, for reconciliation."""
        ...

    async def stale_document_ids(self, *, target: str, limit: int = 100) -> tuple[str, ...]:
        """Documents recorded against an older generation of this target."""
        ...
