# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Ingestion steps: source binding through step config, deterministic processing."""

from __future__ import annotations

import uuid
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from nlght.adapters.outbound.stores.data_writer import DataIndexWriterTool
from nlght.adapters.outbound.tools.activator import ResourceActivator
from nlght.adapters.outbound.workflow.registry import step_registry
from nlght.adapters.outbound.workflow.steps.ingestion import (
    IngestionEmbedStep,
    IngestionProcessStep,
    IngestionSourceStep,
    IngestionWriteStep,
)
from nlght.adapters.outbound.workflow.steps.ingestion.embed import SKIPPED
from nlght.adapters.outbound.workflow.steps.ingestion.process import EMPTY
from nlght.adapters.outbound.workflow.steps.ingestion.source import INCOMPLETE
from nlght.core.entry.context import RequestContext
from nlght.core.errors.errors import WorkflowConfigurationError, WorkflowExecutionError
from nlght.core.ingestion import (
    IndexState,
    IndexStateWrite,
    SnapshotCommitResult,
    SourceSnapshot,
)
from nlght.core.protocol.protocol import ProtocolKind
from nlght.core.trigger.trigger import Trigger, TriggerKind
from nlght.core.workflow.step import WorkflowStepContext
from nlght.core.workflow.workflow import WorkflowStepDef
from nlght.ports.outbound.embedding_client import EmbeddingResult


class _Emitter:
    async def emit(self, signal: object) -> None: ...


def _ctx() -> WorkflowStepContext:
    context = RequestContext(
        correlation_id="cid-1",
        request_id="rid-1",
        received_at=datetime(2026, 1, 1, tzinfo=UTC),
        path="/",
        method="POST",
        headers={},
        query_params={},
        client_host=None,
    )
    return WorkflowStepContext(
        correlation_id="cid-1",
        trigger=Trigger(
            kind=TriggerKind.INBOUND_EVENT,
            protocol=ProtocolKind.GENERIC_JSON,
            operation="ingest",
            payload={},
            context=context,
        ),
        model="",
        messages=[],
        stream=False,
        emitter=_Emitter(),
    )


async def _run_until_not_more(step, ctx: WorkflowStepContext):
    result = await step.run(ctx)
    for _ in range(20):
        if result.verdict != "more":
            return result
        result = await step.run(result.ctx)
    raise AssertionError("step kept returning more")


def _repo(tmp_path: Path) -> Path:
    (tmp_path / "readme.md").write_text("# Title\n\nSome prose about widgets.\n", encoding="utf-8")
    (tmp_path / "module.py").write_text("def widget():\n    return 1\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("ignored by the include pattern", encoding="utf-8")
    return tmp_path


def test_both_steps_are_registered_under_their_type() -> None:
    assert "ingestion.source" in step_registry._registry
    assert "ingestion.process" in step_registry._registry


async def test_a_filesystem_source_is_bound_entirely_through_step_config(tmp_path: Path) -> None:
    step = IngestionSourceStep(
        config={
            "type": "filesystem",
            "source_id": "repo",
            "roots": [{"path": str(_repo(tmp_path)), "alias": "repo"}],
            "include": ["**/*.md", "**/*.py"],
        }
    )

    result = await step.run(_ctx())

    snapshot = result.ctx.metadata["ingestion.snapshot"]
    assert isinstance(snapshot, SourceSnapshot)
    assert snapshot.source_id == "repo"
    assert sorted(document.path for document in snapshot.documents) == [
        "repo/module.py",
        "repo/readme.md",
    ]
    assert result.verdict == "DEFAULT"


async def test_an_unknown_source_type_is_a_configuration_error() -> None:
    step = IngestionSourceStep(config={"type": "carrier-pigeon", "source_id": "x"})

    with pytest.raises(WorkflowConfigurationError, match="unknown source type"):
        await step.run(_ctx())


async def test_a_missing_required_config_key_names_the_key() -> None:
    step = IngestionSourceStep(config={"type": "filesystem", "source_id": "repo"})

    with pytest.raises(WorkflowConfigurationError, match="'roots'"):
        await step.run(_ctx())


async def test_processing_classifies_enriches_and_chunks_every_document(tmp_path: Path) -> None:
    ctx = _ctx()
    await IngestionSourceStep(
        config={
            "type": "filesystem",
            "source_id": "repo",
            "roots": [{"path": str(_repo(tmp_path))}],
            "include": ["**/*.md", "**/*.py"],
        }
    ).run(ctx)

    step = IngestionProcessStep(config={"chunk_size": 64, "chunk_overlap": 8})

    first = await step.run(ctx)
    assert first.verdict == "more"
    assert first.ctx.metadata["ingestion.process.next_document"] == 1

    result = await step.run(first.ctx)

    processed = result.ctx.metadata["ingestion.processed"]
    assert len(processed) == 2
    assert all(document.processing_revision_id for document in processed)
    assert any(document.chunks for document in processed)
    assert result.verdict == "DEFAULT"


async def test_processing_is_deterministic_for_the_same_input(tmp_path: Path) -> None:
    # The reindex decision depends on this: an unstable revision id would
    # reprocess unchanged documents forever.
    async def revisions() -> list[str]:
        ctx = _ctx()
        await IngestionSourceStep(
            config={
                "type": "filesystem",
                "source_id": "repo",
                "roots": [{"path": str(tmp_path)}],
                "include": ["**/*.md"],
            }
        ).run(ctx)
        result = await _run_until_not_more(IngestionProcessStep(config={}), ctx)
        return [document.processing_revision_id for document in result.ctx.metadata["ingestion.processed"]]

    _repo(tmp_path)
    assert await revisions() == await revisions()


async def test_changed_chunking_changes_the_processing_revision(tmp_path: Path) -> None:
    async def revision(size: int) -> str:
        ctx = _ctx()
        await IngestionSourceStep(
            config={
                "type": "filesystem",
                "source_id": "repo",
                "roots": [{"path": str(tmp_path)}],
                "include": ["**/*.md"],
            }
        ).run(ctx)
        result = await _run_until_not_more(
            IngestionProcessStep(config={"chunk_size": size}), ctx
        )
        return result.ctx.metadata["ingestion.processed"][0].processing_revision_id

    _repo(tmp_path)
    assert await revision(4000) != await revision(512)


async def test_a_renamed_file_gets_its_own_revision_and_its_own_chunk_ids(
    tmp_path: Path,
) -> None:
    """Same bytes, new path — and nothing the two documents write may collide.

    A rename is the one case where two documents hold identical content, so it
    is the case an identity derived from content alone would get wrong. A
    chunk's point id in Qdrant is `uuid5` over its chunk id, and that chunk id
    is `stable_digest(processing_revision_id, position, content)`, so two documents sharing
    a revision would write to each other's points. The tombstone that follows a
    rename would then delete the file it had just written.

    The path reaches that digest twice over: `document_id` is
    `stable_digest(source, external_id)` at the source, and the processing
    revision folds in `document.path` again. Either alone is enough — verified
    by removing each and watching this still pass, then removing both and
    watching it fail. This asserts the property rather than either mechanism, so
    it survives one of them being refactored away and still catches losing both.
    """
    async def processed(pattern: str) -> tuple[str, tuple[str, ...]]:
        ctx = _ctx()
        await IngestionSourceStep(
            config={
                "type": "filesystem",
                "source_id": "repo",
                "roots": [{"path": str(tmp_path)}],
                "include": [pattern],
            }
        ).run(ctx)
        result = await _run_until_not_more(
            IngestionProcessStep(config={"chunk_size": 32, "chunk_overlap": 4}), ctx
        )
        document = result.ctx.metadata["ingestion.processed"][0]
        return document.processing_revision_id, tuple(chunk.chunk_id for chunk in document.chunks)

    (tmp_path / "docs").mkdir()
    (tmp_path / "readme.md").write_text(
        "# Title\n\nUnchanged prose about widgets and other things.\n", encoding="utf-8"
    )
    before_revision, before_chunks = await processed("readme.md")

    (tmp_path / "readme.md").rename(tmp_path / "docs" / "guide.md")
    after_revision, after_chunks = await processed("docs/guide.md")

    assert before_revision != after_revision
    assert before_chunks and after_chunks
    assert not set(before_chunks) & set(after_chunks)


async def test_processing_without_a_snapshot_names_the_missing_step() -> None:
    with pytest.raises(WorkflowConfigurationError, match="ingestion.source"):
        await IngestionProcessStep(config={}).run(_ctx())


async def test_invalid_chunking_is_rejected_as_configuration(tmp_path: Path) -> None:
    ctx = _ctx()
    ctx.metadata["ingestion.snapshot"] = SourceSnapshot(
        source_id="repo", documents=(), observed_external_ids=(), complete=True
    )

    with pytest.raises(WorkflowConfigurationError, match="invalid chunking"):
        await IngestionProcessStep(config={"chunk_size": 10, "chunk_overlap": 10}).run(ctx)


async def test_process_rejects_invalid_document_batch_size() -> None:
    ctx = _ctx()
    ctx.metadata["ingestion.snapshot"] = SourceSnapshot(
        source_id="repo", documents=(), observed_external_ids=(), complete=True
    )

    with pytest.raises(WorkflowConfigurationError, match="document_batch_size"):
        await IngestionProcessStep(config={"document_batch_size": 0}).run(ctx)


async def test_an_empty_snapshot_reports_its_own_verdict() -> None:
    ctx = _ctx()
    ctx.metadata["ingestion.snapshot"] = SourceSnapshot(
        source_id="repo", documents=(), observed_external_ids=(), complete=True
    )

    result = await IngestionProcessStep(config={}).run(ctx)

    assert result.verdict == EMPTY


async def test_an_incomplete_snapshot_is_routable_rather_than_fatal(tmp_path: Path) -> None:
    # Incomplete means "index what was seen, but never delete from it".
    missing = tmp_path / "gone"
    step = IngestionSourceStep(
        config={
            "type": "filesystem",
            "source_id": "repo",
            "roots": [{"path": str(missing)}],
        }
    )

    result = await step.run(_ctx())

    assert result.verdict == INCOMPLETE
    assert result.ctx.metadata["ingestion.snapshot"].complete is False


# ---------------------------------------------------------------------------
# ingestion.embed
# ---------------------------------------------------------------------------


class _Embedding:
    def __init__(self, *, dimension: int = 3) -> None:
        self.dimension = dimension
        self.batches: list[list[str]] = []

    async def embed(self, texts):
        self.batches.append(list(texts))
        return EmbeddingResult(
            vectors=tuple((float(len(text)),) * self.dimension for text in texts),
            provider="fake",
            model="fake-model",
            dimension=self.dimension,
        )


async def _processed(tmp_path: Path) -> WorkflowStepContext:
    ctx = _ctx()
    _repo(tmp_path)
    await IngestionSourceStep(
        config={
            "type": "filesystem",
            "source_id": "repo",
            "roots": [{"path": str(tmp_path)}],
            "include": ["**/*.md", "**/*.py"],
        }
    ).run(ctx)
    await _run_until_not_more(
        IngestionProcessStep(config={"chunk_size": 32, "chunk_overlap": 4}), ctx
    )
    return ctx


async def test_embedding_carries_the_provenance_of_every_vector(tmp_path: Path) -> None:
    ctx = await _processed(tmp_path)
    embedding = _Embedding()

    step = IngestionEmbedStep(config={}, embedding_client=embedding)
    first = await step.run(ctx)
    assert first.verdict == "more"
    assert first.ctx.metadata["ingestion.embed.next_document"] == 1

    result = await step.run(first.ctx)

    embedded = result.ctx.metadata["ingestion.embedded"]
    assert embedded
    assert all(chunk.model == "fake-model" and chunk.dimension == 3 for chunk in embedded)
    assert all(len(chunk.vector) == 3 for chunk in embedded)
    assert result.verdict == "DEFAULT"


async def test_every_chunk_is_embedded_exactly_once(tmp_path: Path) -> None:
    ctx = await _processed(tmp_path)
    embedding = _Embedding()

    result = await _run_until_not_more(
        IngestionEmbedStep(config={}, embedding_client=embedding), ctx
    )

    chunk_total = sum(len(doc.chunks) for doc in ctx.metadata["ingestion.processed"])
    assert len(result.ctx.metadata["ingestion.embedded"]) == chunk_total
    assert max(len(batch) for batch in embedding.batches) < chunk_total


async def test_a_lexical_only_deployment_skips_instead_of_failing(tmp_path: Path) -> None:
    ctx = await _processed(tmp_path)

    result = await IngestionEmbedStep(config={}, embedding_client=None).run(ctx)

    assert result.verdict == SKIPPED
    assert result.ctx.metadata["ingestion.embedded"] == ()


async def test_a_required_embedding_step_fails_loudly_when_unconfigured(tmp_path: Path) -> None:
    ctx = await _processed(tmp_path)

    with pytest.raises(WorkflowConfigurationError, match="no embedding provider"):
        await IngestionEmbedStep(config={"required": True}, embedding_client=None).run(ctx)


async def test_embed_rejects_invalid_document_batch_size() -> None:
    ctx = _ctx()
    ctx.metadata["ingestion.processed"] = ()

    with pytest.raises(WorkflowConfigurationError, match="document_batch_size"):
        await IngestionEmbedStep(
            config={"document_batch_size": 0}, embedding_client=_Embedding()
        ).run(ctx)


async def test_embedding_without_processed_documents_names_the_missing_step() -> None:
    with pytest.raises(WorkflowConfigurationError, match="ingestion.process"):
        await IngestionEmbedStep(config={}, embedding_client=_Embedding()).run(_ctx())


async def test_nothing_to_embed_reports_skipped() -> None:
    ctx = _ctx()
    ctx.metadata["ingestion.processed"] = ()

    result = await IngestionEmbedStep(config={}, embedding_client=_Embedding()).run(ctx)

    assert result.verdict == SKIPPED


# ---------------------------------------------------------------------------
# runtime dependencies reach steps through the loader
# ---------------------------------------------------------------------------


def test_a_bound_loader_injects_runtime_dependencies() -> None:
    embedding = _Embedding()
    bound = step_registry.bind(embedding_client=embedding)

    step = bound.load(
        WorkflowStepDef(
            step_id=uuid.uuid4(),
            position=0,
            name="embed",
            type="ingestion.embed",
            enabled=True,
            config={},
            transitions={},
            is_start=True,
            is_terminal=False,
            is_resume=False,
        )
    )

    assert isinstance(step, IngestionEmbedStep)
    assert step._embedding is embedding


def test_binding_does_not_leak_into_the_global_registry() -> None:
    # Two containers in one process must not see each other's wiring.
    step_registry.bind(embedding_client=_Embedding())

    assert step_registry._runtime_deps == {}


def test_a_step_that_declares_no_dependencies_ignores_them() -> None:
    bound = step_registry.bind(embedding_client=_Embedding())

    step = bound.load(
        WorkflowStepDef(
            step_id=uuid.uuid4(),
            position=0,
            name="source",
            type="ingestion.source",
            enabled=True,
            config={"type": "filesystem", "source_id": "x", "roots": [{"path": "/tmp"}]},
            transitions={},
            is_start=True,
            is_terminal=False,
            is_resume=False,
        )
    )

    assert isinstance(step, IngestionSourceStep)


# ---------------------------------------------------------------------------
# ingestion.write
# ---------------------------------------------------------------------------


class _Writer:
    target = "data:main"

    def __init__(self) -> None:
        # The writer activation owns the index state, so the step reads it from
        # here rather than from anything injected around it.
        self.repository = _IngestionRepo()
        self.deleted: list[str] = []
        self.written: list[str] = []
        self.vectors: dict[str, list] = {}
        self.prepared_dimension: int | None = None
        # What ensure_ready reports: whether it had to create an index, which is
        # what rotates the target's generation.
        self.created_index = False

    def ensure_ready(self, *, dimension: int) -> bool:
        self.prepared_dimension = dimension
        # The real writer memoises its bootstrap, so it reports "created" once
        # per process however often it is asked. A fake that kept saying yes
        # would rotate the generation on every batch.
        created, self.created_index = self.created_index, False
        return created

    def projection_fingerprint(self, *, provider, model, dimension) -> str:
        return f"projection:{provider}:{model}:{dimension}"

    async def write_document(
        self, *, document_id, path, text, source_revision_id, processing_revision_id,
        metadata, chunks,
    ) -> None:
        self.written.append(document_id)
        self.vectors[document_id] = list(chunks)

    async def delete_document(self, *, document_id) -> None:
        self.deleted.append(document_id)


class _IngestionRepo:
    def __init__(self, tombstoned: tuple[str, ...] = ()) -> None:
        self.state: dict[tuple[str, str], IndexState] = {}
        self.recorded: list[IndexStateWrite] = []
        self.committed: list[object] = []
        self.cleared: list[tuple[str, str]] = []
        self.generation = "generation-1"
        self.rotations = 0
        self._tombstoned = tombstoned

    async def get_index_state(self, *, document_id, target):
        return self.state.get((document_id, target))

    async def record_indexed(self, write: IndexStateWrite):
        self.recorded.append(write)
        state = IndexState(
            document_id=write.document_id,
            target=write.target,
            content_hash=write.content_hash,
            enricher_hash=write.enricher_hash,
            processing_revision_id=write.processing_revision_id,
            index_generation=write.index_generation,
            projection_fingerprint=write.projection_fingerprint,
            indexed_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        self.state[(write.document_id, write.target)] = state
        return state

    async def commit_snapshot(self, command):
        self.committed.append(command)
        return SnapshotCommitResult(
            run_id=uuid.uuid4(),
            created_processing_revision_ids=tuple(item.processing_revision_id for item in command.documents),
            tombstoned_document_ids=self._tombstoned,
            idempotent_replay=False,
        )

    async def clear_index_state(self, *, document_id, target):
        self.cleared.append((document_id, target))
        return True

    async def current_generation(self, *, target):
        return self.generation

    async def rotate_generation(self, *, target):
        self.rotations += 1
        self.generation = f"generation-{self.rotations + 1}"
        return self.generation


def _write_step(writer: _Writer, repo: _IngestionRepo) -> IngestionWriteStep:
    writer.repository = repo
    step = IngestionWriteStep(
        config={"writer": "data-writer"},
        resource_repository=object(),
        tool_loader=object(),
        resource_activator=ResourceActivator(resources=object(), loader=object()),
    )
    step._writer = lambda caller=None: _resolved(writer)  # type: ignore[assignment]
    return step


async def _resolved(writer: _Writer) -> _Writer:
    return writer


async def test_writing_records_index_state_for_every_document(tmp_path: Path) -> None:
    ctx = await _processed(tmp_path)
    writer, repo = _Writer(), _IngestionRepo()

    step = _write_step(writer, repo)
    first = await step.run(ctx)
    assert first.verdict == "more"
    assert first.ctx.metadata["ingestion.write.next_document"] == 1

    result = await step.run(first.ctx)

    assert len(writer.written) == 2
    assert {write.target for write in repo.recorded} == {"data:main"}
    assert result.verdict == "DEFAULT"


async def test_an_unchanged_document_is_not_rewritten(tmp_path: Path) -> None:
    ctx = await _processed(tmp_path)
    writer, repo = _Writer(), _IngestionRepo()
    await _run_until_not_more(_write_step(writer, repo), ctx)

    ctx = await _processed(tmp_path)
    second = _Writer()
    result = await _run_until_not_more(_write_step(second, repo), ctx)

    assert second.written == []
    assert result.verdict == "unchanged"


async def test_a_changed_enricher_revision_forces_a_rewrite(tmp_path: Path) -> None:
    ctx = await _processed(tmp_path)
    writer, repo = _Writer(), _IngestionRepo()
    await _run_until_not_more(_write_step(writer, repo), ctx)

    # Same content, different processing revision — the reindex rule must fire.
    for key, state in list(repo.state.items()):
        repo.state[key] = IndexState(
            document_id=state.document_id,
            target=state.target,
            content_hash=state.content_hash,
            enricher_hash="stale-enricher",
            processing_revision_id=state.processing_revision_id,
            # Carried over deliberately: the generation must stay current so the
            # rewrite is attributable to the enricher and nothing else.
            index_generation=state.index_generation,
            projection_fingerprint=state.projection_fingerprint,
            indexed_at=state.indexed_at,
        )

    second = _Writer()
    ctx = await _processed(tmp_path)
    await _run_until_not_more(_write_step(second, repo), ctx)

    assert len(second.written) == 2


async def test_vectors_are_grouped_onto_their_document(tmp_path: Path) -> None:
    ctx = await _processed(tmp_path)
    await _run_until_not_more(IngestionEmbedStep(config={}, embedding_client=_Embedding()), ctx)
    writer, repo = _Writer(), _IngestionRepo()

    await _run_until_not_more(_write_step(writer, repo), ctx)

    assert any(writer.vectors[document_id] for document_id in writer.written)


async def test_writing_without_a_way_to_activate_its_writer_is_refused(tmp_path: Path) -> None:
    ctx = await _processed(tmp_path)
    step = IngestionWriteStep(config={"writer": "data-writer"})

    with pytest.raises(WorkflowConfigurationError, match="activate its writer"):
        await step.run(ctx)


async def test_writing_without_processed_documents_names_the_missing_step() -> None:
    step = IngestionWriteStep(
        config={"writer": "w"},
        resource_repository=object(),
        tool_loader=object(),
        resource_activator=ResourceActivator(resources=object(), loader=object()),
    )

    with pytest.raises(WorkflowConfigurationError, match="ingestion.process"):
        await step.run(_ctx())


async def test_write_rejects_invalid_document_batch_size(tmp_path: Path) -> None:
    ctx = await _processed(tmp_path)
    writer, repo = _Writer(), _IngestionRepo()
    step = _write_step(writer, repo)
    step.config["document_batch_size"] = 0

    with pytest.raises(WorkflowConfigurationError, match="document_batch_size"):
        await step.run(ctx)


async def test_an_unnamed_writer_is_refused(tmp_path: Path) -> None:
    ctx = await _processed(tmp_path)
    step = IngestionWriteStep(
        config={},
        resource_repository=object(),
        tool_loader=object(),
        resource_activator=ResourceActivator(resources=object(), loader=object()),
    )

    with pytest.raises(WorkflowConfigurationError, match="'writer' config key"):
        await step.run(ctx)


def test_the_index_writer_is_never_offered_to_a_model() -> None:
    assert DataIndexWriterTool.signatures() == []
    assert DataIndexWriterTool.KIND != "data_store"


async def test_collections_are_prepared_before_the_first_write(tmp_path: Path) -> None:
    ctx = await _processed(tmp_path)
    await _run_until_not_more(IngestionEmbedStep(config={}, embedding_client=_Embedding()), ctx)
    writer, repo = _Writer(), _IngestionRepo()

    await _run_until_not_more(_write_step(writer, repo), ctx)

    # Writing into a missing collection fails; the bootstrap must run first.
    assert writer.prepared_dimension == 3


async def test_chunk_semantics_reach_the_writer(tmp_path: Path) -> None:
    ctx = await _processed(tmp_path)
    await _run_until_not_more(IngestionEmbedStep(config={}, embedding_client=_Embedding()), ctx)

    embedded = ctx.metadata["ingestion.embedded"]
    assert any(chunk.semantics for chunk in embedded)


# -- the corpus record -------------------------------------------------------


async def test_writing_records_what_the_source_contained(tmp_path: Path) -> None:
    ctx = await _processed(tmp_path)
    writer, repo = _Writer(), _IngestionRepo()

    await _run_until_not_more(_write_step(writer, repo), ctx)

    assert len(repo.committed) == 1
    command = repo.committed[0]
    assert command.source_id == "repo"
    assert command.source_type == "filesystem"
    assert command.complete is True
    # Every observed document, keyed the same way the index is.
    assert {item.document_id for item in command.documents} == {
        document.document_id for document in ctx.metadata["ingestion.processed"]
    }
    # Decision A: the reference says where to fetch the content again, not where
    # a copy of it was stored.
    revision = command.documents[0]
    assert revision.artifact_ref.startswith("repo:")
    assert revision.source_revision_id in revision.artifact_ref
    assert revision.processing["classifier_version"]


async def test_a_reconfigured_source_is_visible_in_the_record(tmp_path: Path) -> None:
    # Two different source configurations must not look like the same snapshot.
    first = await _processed(tmp_path)
    second = _ctx()
    await IngestionSourceStep(
        config={
            "type": "filesystem",
            "source_id": "repo",
            "roots": [{"path": str(tmp_path)}],
            "include": ["**/*.md"],
        }
    ).run(second)
    await _run_until_not_more(IngestionProcessStep(config={}), second)

    assert (
        first.metadata["ingestion.config_revision"]
        != second.metadata["ingestion.config_revision"]
    )


async def test_a_document_absent_from_a_complete_snapshot_leaves_the_index(
    tmp_path: Path,
) -> None:
    ctx = await _processed(tmp_path)
    writer = _Writer()
    repo = _IngestionRepo(tombstoned=("gone-1", "gone-2"))

    result = await _run_until_not_more(_write_step(writer, repo), ctx)

    assert writer.deleted == ["gone-1", "gone-2"]
    # Cleared only after the backend call returned, so a crash in between
    # re-runs the removal instead of losing it.
    assert repo.cleared == [("gone-1", writer.target), ("gone-2", writer.target)]
    assert result.ctx.metadata["ingestion.removed"] == 2


async def test_a_removal_alone_is_still_work_done(tmp_path: Path) -> None:
    ctx = await _processed(tmp_path)
    writer, repo = _Writer(), _IngestionRepo(tombstoned=("gone",))
    # Everything already indexed, so nothing is written this run.
    step = _write_step(writer, repo)
    await _run_until_not_more(step, ctx)

    second_writer = _Writer()
    second_repo = _IngestionRepo(tombstoned=("gone",))
    second_repo.state = repo.state
    ctx = await _processed(tmp_path)
    result = await _run_until_not_more(_write_step(second_writer, second_repo), ctx)

    assert result.ctx.metadata["ingestion.written"] == 0
    assert result.verdict == "DEFAULT"


async def test_a_rename_writes_the_new_document_and_removes_the_old_in_one_run(
    tmp_path: Path,
) -> None:
    """The one run that both writes and deletes, and the two must not interfere.

    Every other removal test tombstones documents that this run never touched.
    A rename is different: the same run writes the arrival and deletes the
    departure, and the deletion happens after the write. If the two documents
    shared any id — which they would if identity came from content — the delete
    would take the write with it and the file would vanish from the index while
    the run reported success.
    """
    async def acquire(pattern: str) -> WorkflowStepContext:
        ctx = _ctx()
        await IngestionSourceStep(
            config={
                "type": "filesystem",
                "source_id": "repo",
                "roots": [{"path": str(tmp_path)}],
                "include": [pattern],
            }
        ).run(ctx)
        return (
            await _run_until_not_more(
                IngestionProcessStep(config={"chunk_size": 32, "chunk_overlap": 4}), ctx
            )
        ).ctx

    (tmp_path / "docs").mkdir()
    (tmp_path / "readme.md").write_text(
        "# Title\n\nProse about widgets and how they are made.\n", encoding="utf-8"
    )
    before = await acquire("readme.md")
    old_id = before.metadata["ingestion.processed"][0].document_id

    (tmp_path / "readme.md").rename(tmp_path / "docs" / "guide.md")
    after = await acquire("docs/guide.md")
    new_id = after.metadata["ingestion.processed"][0].document_id

    writer, repo = _Writer(), _IngestionRepo(tombstoned=(old_id,))
    result = await _run_until_not_more(_write_step(writer, repo), after)

    assert writer.written == [new_id]
    assert writer.deleted == [old_id]
    assert repo.cleared == [(old_id, writer.target)]
    # The arrival survived the departure.
    assert new_id not in writer.deleted
    assert result.ctx.metadata["ingestion.removed"] == 1


async def test_writing_without_the_acquired_snapshot_is_refused(tmp_path: Path) -> None:
    ctx = await _processed(tmp_path)
    del ctx.metadata["ingestion.snapshot"]

    with pytest.raises(WorkflowConfigurationError, match="what a source contained"):
        await _write_step(_Writer(), _IngestionRepo()).run(ctx)


# -- the index generation ----------------------------------------------------


async def test_a_created_index_rotates_the_generation(tmp_path: Path) -> None:
    # An index that had to be created is not the one the recorded state
    # describes. Without the rotation, a wiped Qdrant plus an intact index state
    # produces a run that skips every document and reports success.
    ctx = await _processed(tmp_path)
    writer, repo = _Writer(), _IngestionRepo()
    writer.created_index = True

    await _run_until_not_more(_write_step(writer, repo), ctx)

    assert repo.rotations == 1
    assert {write.index_generation for write in repo.recorded} == {repo.generation}


async def test_an_existing_index_does_not_rotate(tmp_path: Path) -> None:
    ctx = await _processed(tmp_path)
    writer, repo = _Writer(), _IngestionRepo()

    await _run_until_not_more(_write_step(writer, repo), ctx)

    assert repo.rotations == 0


async def test_a_document_on_an_older_generation_is_written_again(tmp_path: Path) -> None:
    # This is what makes an interrupted reindex resumable: content and enricher
    # match, so the old check would have skipped it, but it is not in the index
    # that is there now.
    ctx = await _processed(tmp_path)
    writer, repo = _Writer(), _IngestionRepo()
    await _run_until_not_more(_write_step(writer, repo), ctx)
    assert writer.written

    repo.generation = "generation-after-a-wipe"
    replay = await _processed(tmp_path)
    rewriter = _Writer()
    await _run_until_not_more(_write_step(rewriter, repo), replay)

    assert sorted(rewriter.written) == sorted(writer.written)


async def test_an_unchanged_document_on_the_current_generation_is_skipped(tmp_path: Path) -> None:
    # The other half: without a rotation a rerun must still cost nothing.
    ctx = await _processed(tmp_path)
    writer, repo = _Writer(), _IngestionRepo()
    await _run_until_not_more(_write_step(writer, repo), ctx)

    again = await _processed(tmp_path)
    rewriter = _Writer()
    await _run_until_not_more(_write_step(rewriter, repo), again)

    assert rewriter.written == []


async def test_a_document_with_text_and_no_chunks_is_refused_not_recorded(tmp_path: Path) -> None:
    # It would be written lexically, be absent from the vector index, and then
    # be recorded as fully indexed — a divergence nothing would ever repair.
    ctx = await _processed(tmp_path)
    await _run_until_not_more(IngestionEmbedStep(config={}, embedding_client=_Embedding()), ctx)
    # An embedded run whose chunks were lost for one document.
    ctx.metadata["ingestion.embedded"] = tuple(
        chunk for chunk in ctx.metadata["ingestion.embedded"]
        if chunk.document_id != ctx.metadata["ingestion.processed"][0].document_id
    )
    writer, repo = _Writer(), _IngestionRepo()

    with pytest.raises(WorkflowExecutionError, match="produced no chunks"):
        await _run_until_not_more(_write_step(writer, repo), ctx)

    assert repo.recorded == []


async def test_a_changed_projection_rewrites_a_document_the_hashes_call_unchanged(
    tmp_path: Path,
) -> None:
    """The reindex case none of the older three values could express.

    The source has not changed, the processing has not changed, and the index
    was not recreated — so `content_hash`, `enricher_hash` and
    `index_generation` all still agree. Only the projection moved. Before the
    fingerprint existed this document was skipped, the run reported success, and
    the collection kept vectors from a model the queries no longer used.
    """
    ctx = await _processed(tmp_path)
    writer, repo = _Writer(), _IngestionRepo()
    await _run_until_not_more(_write_step(writer, repo), ctx)
    assert writer.written

    recorded = dict(repo.state)
    for key, state in recorded.items():
        assert state.projection_fingerprint
        repo.state[key] = replace(state, projection_fingerprint="a-different-projection")

    again = await _processed(tmp_path)
    rewriter = _Writer()
    await _run_until_not_more(_write_step(rewriter, repo), again)

    assert rewriter.written == writer.written, (
        "a document whose projection changed must be written again, even though "
        "its source, its processing and the index generation all still agree"
    )
