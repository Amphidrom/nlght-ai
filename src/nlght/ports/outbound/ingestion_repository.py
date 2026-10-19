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
    generations. It does not switch a corpus over. It names which incarnation of
    a target's indexes the recorded state describes, so that recreating an index
    invalidates the record instead of silently contradicting it.
    """

    async def commit_snapshot(self, command: SnapshotCommit) -> SnapshotCommitResult: ...

    async def current_processing_revisions(
        self, document_ids: Sequence[str]
    ) -> dict[str, str]:
        """Resolve published processing revisions without reading content.

        Args:
            document_ids (Sequence[str]): Document IDs to check.

        Returns:
            dict[str, str]: Current revision IDs keyed by document. Excludes tombstones.

        Side Effects:
            Reads document authority records.
        """
        ...

    async def read_revision_contents(
        self, keys: Sequence[tuple[str, str]]
    ) -> dict[tuple[str, str], str]:
        """Read authoritative text for selected document revisions.

        Args:
            keys (Sequence[tuple[str, str]]): Document ID and processing revision ID pairs.

        Returns:
            dict[tuple[str, str], str]: Stored text keyed by pair. Omits missing or unusable text.

        Side Effects:
            Reads revision content records.
        """
        ...

    async def get_index_state(self, *, document_id: str, target: str) -> IndexState | None: ...

    async def record_indexed(self, write: IndexStateWrite) -> IndexState:
        """Record one document's indexed state idempotently.

        Args:
            write (IndexStateWrite): Index state write for a document and target.

        Returns:
            IndexState: Persisted index state.

        Side Effects:
            Upserts an index state record.
        """
        ...

    async def clear_index_state(self, *, document_id: str, target: str) -> bool:
        """Clear index state for a removed document.

        Args:
            document_id (str): Document whose index state is cleared.
            target (str): Index target whose state is cleared.

        Returns:
            bool: True when a state record was removed.

        Side Effects:
            Deletes the matching state record when present.
        """
        ...

    async def current_generation(self, *, target: str) -> str:
        """Resolve the current index generation for a target.

        Args:
            target (str): Target name.

        Returns:
            str: Existing or newly minted generation ID.

        Side Effects:
            Creates a generation record when absent.
        """
        ...

    async def rotate_generation(self, *, target: str) -> str:
        """Invalidate recorded state after target index recreation.

        Args:
            target (str): Target name.

        Returns:
            str: Newly active generation ID.

        Side Effects:
            Rotates the target's generation record.
        """
        ...

    async def index_state_summary(self, *, target: str) -> IndexStateSummary:
        """Summarize expected index state for reconciliation.

        Args:
            target (str): Target name.

        Returns:
            IndexStateSummary: Index state summary.

        Side Effects:
            Reads index state records.
        """
        ...

    async def stale_document_ids(self, *, target: str, limit: int = 100) -> tuple[str, ...]:
        """Find documents indexed under a stale target generation.

        Args:
            target (str): Index target to inspect.
            limit (int): Maximum number of stale document IDs.

        Returns:
            tuple[str, ...]: Stale document IDs.

        Side Effects:
            Reads index state records.
        """
        ...
