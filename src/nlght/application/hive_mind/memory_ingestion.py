# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Memory ingestion pipeline.

Bridges extraction and StoreCoordinator transaction semantics.
The extractor is injected — no default, no NLP fallback.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from nlght.core.hive_mind.extraction import (
    ExtractedArtifact,
    ExtractionResult,
    MemoryCandidate,
    MemoryCandidateConfidence,
    MemoryCandidateKind,
)
from nlght.core.hive_mind.models import AtomType, WriteIntent
from nlght.ports.outbound.store_coordinator import StoreCoordinator

if TYPE_CHECKING:
    from nlght.core.workspace.workspace import WorkspaceContext


class MemoryExtractor(Protocol):
    async def extract(self, text: str) -> ExtractionResult: ...


@dataclass
class MemoryTransaction:
    extraction: ExtractionResult
    committed: list[MemoryCandidate] = field(default_factory=list)
    discarded: list[MemoryCandidate] = field(default_factory=list)
    promoted_atom_count: int = 0
    stored_result_count: int = 0

    @property
    def candidates(self) -> list[MemoryCandidate]:
        return self.extraction.candidates

    def committed_entities(self) -> list[str]:
        entities: list[str] = []
        for candidate in self.committed:
            for entity in candidate.entities:
                if entity not in entities:
                    entities.append(entity)
        return entities

    def summary(self) -> str:
        committed = "\n".join(f"- {c.content}" for c in self.committed) or "- none"
        discarded = "\n".join(f"- {c.content}" for c in self.discarded) or "- none"
        refs = ", ".join(self.extraction.referenced_labels) or "none"
        return (
            "Memory transaction report:\n"
            f"Intent: {self.extraction.intent}\n"
            f"Referenced labels: {refs}\n"
            f"Candidates staged: {len(self.candidates)}\n"
            f"Atoms promoted from turn branch: {self.promoted_atom_count}\n"
            f"Session results committed: {self.stored_result_count}\n\n"
            f"Committed facts:\n{committed}\n\n"
            f"Discarded candidates:\n{discarded}"
        )


class MemoryIngestionPipeline:
    def __init__(self, extractor: MemoryExtractor) -> None:
        self._extractor = extractor

    async def ingest(
        self,
        *,
        store: StoreCoordinator,
        correlation_id: str,
        user_text: str,
        turn_nr: int,
        workspace: WorkspaceContext | None = None,
    ) -> MemoryTransaction:
        transaction = MemoryTransaction(extraction=await self._extractor.extract(user_text))

        if workspace is not None:
            store.set_slot("workspace", workspace)

        artifact_refs: dict[str, str] = {}
        if workspace is not None and transaction.extraction.artifacts:
            artifact_refs = _promote_to_workspace(workspace, transaction.extraction.artifacts)

        session_dedup = _SessionDedup(store)
        turn_label = f"turn:{correlation_id}:{turn_nr}"

        store.open_branch(turn_label)
        try:
            self._stage_extraction_branch(store, correlation_id, transaction.candidates)
            self._stage_validation_branch(store, correlation_id, transaction, artifact_refs, session_dedup)
            self._stage_response_plan_branch(store, correlation_id, transaction)

            promoted = store.close_branch(promote=True)
            transaction.promoted_atom_count = len(promoted)
            stored = store.store_promoted_atoms(
                promoted,
                turn_nr=turn_nr,
                entities=transaction.committed_entities(),
            )
            transaction.stored_result_count = len(stored)
            return transaction
        except Exception:
            store.close_branch(promote=False)
            raise

    def _stage_extraction_branch(
        self,
        store: StoreCoordinator,
        correlation_id: str,
        candidates: list[MemoryCandidate],
    ) -> None:
        store.open_branch(f"extract:{correlation_id}")
        for candidate in candidates:
            store.write(WriteIntent(
                atom_type=AtomType.SPEC,
                content=f"candidate:{candidate.key} | {candidate.content}",
                task_id=f"extract:{correlation_id}",
                entities=candidate.entities,
                key=candidate.key,
                kind=str(candidate.kind),
                tags=["memory_candidate", candidate.confidence.value, *candidate.tags],
                promote_immediately=True,
            ))
        store.close_branch(promote=True)

    def _stage_validation_branch(
        self,
        store: StoreCoordinator,
        correlation_id: str,
        transaction: MemoryTransaction,
        artifact_refs: dict[str, str] | None = None,
        session_dedup: _SessionDedup | None = None,
    ) -> None:
        store.open_branch(f"validate:{correlation_id}")
        for candidate in transaction.candidates:
            if candidate.confidence == MemoryCandidateConfidence.VALIDATED:
                content = candidate.content
                if artifact_refs and candidate.kind == MemoryCandidateKind.ARTIFACT:
                    label = candidate.entities[0] if candidate.entities else ""
                    compact = artifact_refs.get(label)
                    if compact:
                        content = compact
                if session_dedup is not None and session_dedup.is_duplicate(content):
                    continue
                transaction.committed.append(candidate)
                if session_dedup is not None:
                    session_dedup.register(content)
                store.write(WriteIntent(
                    atom_type=AtomType.RESULT,
                    content=content,
                    task_id=f"validate:{correlation_id}",
                    entities=candidate.entities,
                    tags=["validated", "committed", *candidate.tags],
                    key=candidate.key,
                    kind=str(candidate.kind),
                    promote_immediately=True,
                ))
            else:
                transaction.discarded.append(candidate)
                store.write(WriteIntent(
                    atom_type=AtomType.EVAL,
                    content=f"discarded_candidate:{candidate.key} | {candidate.content}",
                    task_id=f"validate:{correlation_id}",
                    entities=candidate.entities,
                    key=candidate.key,
                    kind=str(candidate.kind),
                    tags=["discarded", candidate.confidence.value, *candidate.tags],
                    promote_immediately=False,
                ))
        store.close_branch(promote=True)

    def _stage_response_plan_branch(
        self,
        store: StoreCoordinator,
        correlation_id: str,
        transaction: MemoryTransaction,
    ) -> None:
        store.open_branch(f"response-plan:{correlation_id}")
        store.write(WriteIntent(
            atom_type=AtomType.PLAN,
            content=(
                "Answer using committed session facts only. "
                f"Intent={transaction.extraction.intent} "
                f"committed={len(transaction.committed)} "
                f"discarded={len(transaction.discarded)}."
            ),
            task_id=f"response-plan:{correlation_id}",
            tags=["response_plan", "transient"],
            promote_immediately=False,
        ))
        store.close_branch(promote=False)


def _promote_to_workspace(
    workspace: WorkspaceContext,
    artifacts: list[ExtractedArtifact],
) -> dict[str, str]:
    """Write each artifact's content to workspace and return label → compact-ref map."""
    artifacts_dir = workspace.root_path / "memory_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    refs: dict[str, str] = {}
    for artifact in artifacts:
        artifact_id = str(uuid.uuid4())
        ext = Path(artifact.label).suffix or ".txt"
        file_path = artifacts_dir / f"{artifact_id}{ext}"
        file_path.write_text(artifact.content, encoding="utf-8")
        size = len(artifact.content.encode("utf-8"))
        refs[artifact.label] = (
            f"[memory_artifact] id={artifact_id}"
            f" | label={artifact.label}"
            f" | type={artifact.content_type}"
            f" | size={size}"
        )
    return refs


def _parse_artifact_ref(content: str) -> dict[str, str]:
    """Parse fields from a ``[memory_artifact]`` compact reference string."""
    rest = content[len("[memory_artifact]"):].strip()
    fields: dict[str, str] = {}
    for part in rest.split(" | "):
        key, sep, value = part.partition("=")
        if sep:
            fields[key.strip()] = value.strip()
    return fields


class _SessionDedup:
    """Prevents writing candidates that are already present in known_results.

    Built once per ingest call *before* the turn branch is opened so it
    captures only pre-turn state (not the current extraction).
    """

    def __init__(self, store: StoreCoordinator) -> None:
        results = store.get_context([]).get("known_results", []) or []
        self._hashes: set[str] = set()
        self._artifact_labels: set[str] = set()
        for r in results:
            self._index(str(getattr(r, "content", "") or "").strip())

    def _index(self, content: str) -> None:
        if not content:
            return
        if content.startswith("[memory_artifact]"):
            label = _parse_artifact_ref(content).get("label", "")
            if label:
                self._artifact_labels.add(label)
        else:
            self._hashes.add(hashlib.sha1(content.encode("utf-8", errors="replace")).hexdigest()[:12])

    def is_duplicate(self, stored_content: str) -> bool:
        if stored_content.startswith("[memory_artifact]"):
            label = _parse_artifact_ref(stored_content).get("label", "")
            return bool(label) and label in self._artifact_labels
        h = hashlib.sha1(stored_content.encode("utf-8", errors="replace")).hexdigest()[:12]
        return h in self._hashes

    def register(self, stored_content: str) -> None:
        self._index(stored_content)
