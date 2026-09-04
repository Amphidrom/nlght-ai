# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Fan-out: one acquisition's documents spread over the worker pool.

What matters here is what a child is given — the *names* of its documents and a
link to the run that fanned it out — and what fan-out refuses to do when the
deployment cannot support it.
"""

from __future__ import annotations

import dataclasses
import uuid
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from nlght.adapters.outbound.workflow.steps.ingestion import (
    IngestionFanOutStep,
    IngestionSourceStep,
)
from nlght.adapters.outbound.workflow.steps.ingestion.fan_out import NOTHING_TO_FAN_OUT
from nlght.core.entry.context import PrincipalRef, RequestContext
from nlght.core.errors.errors import WorkflowConfigurationError
from nlght.core.execution import ExecutionSubmission
from nlght.core.ingestion import SourceSnapshot
from nlght.core.ingestion.document import SourceDocument
from nlght.core.protocol.protocol import ProtocolKind
from nlght.core.trigger.trigger import Trigger, TriggerKind
from nlght.core.workflow.step import WorkflowStepContext
from nlght.core.workflow.workflow import WorkflowDef, WorkflowVersionDef


class _Emitter:
    async def emit(self, signal: object) -> None: ...


def _ctx(payload: dict[str, Any] | None = None) -> WorkflowStepContext:
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
            payload=payload or {},
            context=context,
        ),
        model="",
        messages=[],
        stream=False,
        emitter=_Emitter(),
    )


def _document(external_id: str) -> SourceDocument:
    return SourceDocument(
        source="repo",
        external_id=external_id,
        path=external_id,
        content=b"content of " + external_id.encode(),
    )


def _snapshot(*external_ids: str) -> SourceSnapshot:
    documents = tuple(_document(item) for item in external_ids)
    return SourceSnapshot(
        source_id="repo",
        documents=documents,
        observed_external_ids=external_ids,
        complete=True,
    )


class _Dispatcher:
    def __init__(self) -> None:
        self.submissions: list[ExecutionSubmission] = []

    async def submit(self, submission: ExecutionSubmission) -> SimpleNamespace:
        self.submissions.append(submission)
        return SimpleNamespace(execution_id=uuid.uuid4())


class _Workflows:
    """Resolves exactly one child workflow, by name."""

    def __init__(
        self,
        name: str | None = "ingest-data-document",
        *,
        active: bool = True,
        declared_capabilities: tuple[str, ...] = (),
    ) -> None:
        self._name = name
        self._active = active
        self.declared_capabilities = declared_capabilities
        self.workflow_id = uuid.uuid4()
        self.version_id = uuid.uuid4()

    async def find_by_name(self, name: str) -> WorkflowDef | None:
        if name != self._name:
            return None
        return WorkflowDef(
            workflow_id=self.workflow_id,
            name=name,
            enabled=True,
            capabilities=list(self.declared_capabilities),
        )

    async def find_active_version(self, workflow_id: uuid.UUID) -> WorkflowVersionDef | None:
        if not self._active:
            return None
        return WorkflowVersionDef(
            version_id=self.version_id,
            workflow_id=workflow_id,
            version=1,
            status="active",
            steps=[],
        )


def _fanning_ctx(*external_ids: str) -> tuple[WorkflowStepContext, _Dispatcher]:
    ctx = _ctx()
    ctx.metadata["ingestion.snapshot"] = _snapshot(*external_ids)
    ctx.execution_id = uuid.uuid4()
    dispatcher = _Dispatcher()
    ctx.dispatcher = dispatcher
    return ctx, dispatcher


async def test_the_work_flow_must_be_named() -> None:
    """A distribution flow that does not say where the work goes is not one."""
    ctx, _ = _fanning_ctx("a.md")

    with pytest.raises(WorkflowConfigurationError, match="requires 'workflow'"):
        await IngestionFanOutStep(config={}, workflow_repository=_Workflows()).run(ctx)


async def test_each_document_becomes_a_child_execution_of_this_run() -> None:
    ctx, dispatcher = _fanning_ctx("a.md", "b.md", "c.md")

    result = await IngestionFanOutStep(
        config={"workflow": "ingest-data-document"},
        workflow_repository=_Workflows(),
    ).run(ctx)

    assert result.verdict == "DEFAULT"
    assert len(dispatcher.submissions) == 3
    assert {s.parent_execution_id for s in dispatcher.submissions} == {ctx.execution_id}
    assert len(result.ctx.metadata["ingestion.fan_out.children"]) == 3
    assert result.ctx.metadata["ingestion.fan_out.batches"] == 3


async def test_a_child_is_told_which_documents_are_its_share_not_their_content() -> None:
    ctx, dispatcher = _fanning_ctx("a.md", "b.md")

    await IngestionFanOutStep(
        config={"workflow": "ingest-data-document"},
        workflow_repository=_Workflows(),
    ).run(ctx)

    payloads = [s.trigger.payload for s in dispatcher.submissions]
    assert [p["external_ids"] for p in payloads] == [["a.md"], ["b.md"]]
    assert all(p["source_id"] == "repo" for p in payloads)
    assert all(p["parent_execution_id"] == str(ctx.execution_id) for p in payloads)
    # Nothing carries document content: a child fetches its own from the source.
    assert not any("content" in str(p) for p in payloads)
    assert all(s.artifact_refs == () for s in dispatcher.submissions)


async def test_every_child_gets_its_own_run_identity() -> None:
    """A correlation id identifies one run, and children are separate runs.

    Sharing the parent's would collide on every key built from it — the ingestion
    snapshot record, the executor's already-running guard, the per-run workspace.
    """
    ctx, dispatcher = _fanning_ctx("a.md", "b.md", "c.md")

    await IngestionFanOutStep(
        config={"workflow": "ingest-data-document"},
        workflow_repository=_Workflows(),
    ).run(ctx)

    correlation_ids = [s.trigger.context.correlation_id for s in dispatcher.submissions]
    assert len(set(correlation_ids)) == 3
    assert ctx.trigger.context.correlation_id not in correlation_ids
    # Derived, so the lineage stays greppable and the parent is still named.
    assert all(cid.startswith(ctx.trigger.context.correlation_id) for cid in correlation_ids)
    assert {s.trigger.payload["parent_correlation_id"] for s in dispatcher.submissions} == {
        ctx.trigger.context.correlation_id
    }


async def test_a_retried_fan_out_reuses_each_child_identity() -> None:
    ctx, dispatcher = _fanning_ctx("a.md", "b.md")
    step = IngestionFanOutStep(
        config={"workflow": "ingest-data-document"},
        workflow_repository=_Workflows(),
    )

    await step.run(ctx)
    first = [s.trigger.context.correlation_id for s in dispatcher.submissions]
    dispatcher.submissions.clear()
    await step.run(ctx)

    assert [s.trigger.context.correlation_id for s in dispatcher.submissions] == first


async def test_a_batch_size_puts_several_documents_in_one_child() -> None:
    ctx, dispatcher = _fanning_ctx("a.md", "b.md", "c.md", "d.md", "e.md")

    await IngestionFanOutStep(
        config={"workflow": "ingest-data-document", "document_batch_size": 2},
        workflow_repository=_Workflows(),
    ).run(ctx)

    assert [s.trigger.payload["external_ids"] for s in dispatcher.submissions] == [
        ["a.md", "b.md"],
        ["c.md", "d.md"],
        ["e.md"],
    ]


async def test_children_keep_their_idempotency_across_a_retried_fan_out() -> None:
    """At-least-once delivery must mean a rerun, not a doubled corpus."""
    ctx, dispatcher = _fanning_ctx("a.md", "b.md")
    step = IngestionFanOutStep(
        config={"workflow": "ingest-data-document"},
        workflow_repository=_Workflows(),
    )

    await step.run(ctx)
    first = [s.idempotency_key for s in dispatcher.submissions]
    dispatcher.submissions.clear()
    await step.run(ctx)

    assert [s.idempotency_key for s in dispatcher.submissions] == first


async def test_required_capabilities_travel_to_the_children() -> None:
    ctx, dispatcher = _fanning_ctx("a.md")

    await IngestionFanOutStep(
        config={
            "workflow": "ingest-data-document",
            "capabilities": "parser:pdf, source:repo",
        },
        workflow_repository=_Workflows(),
    ).run(ctx)

    assert dispatcher.submissions[0].required_capabilities == ("parser:pdf", "source:repo")


async def test_an_empty_snapshot_fans_nothing_out() -> None:
    ctx, dispatcher = _fanning_ctx()

    result = await IngestionFanOutStep(
        config={"workflow": "ingest-data-document"},
        workflow_repository=_Workflows(),
    ).run(ctx)

    assert result.verdict == NOTHING_TO_FAN_OUT
    assert dispatcher.submissions == []


async def test_fanning_out_from_an_inline_run_is_a_configuration_error() -> None:
    """No execution to parent children to means no fan-out — not a silent serial run."""
    ctx, _ = _fanning_ctx("a.md")
    ctx.execution_id = None

    with pytest.raises(WorkflowConfigurationError, match="dispatched execution"):
        await IngestionFanOutStep(
            config={"workflow": "ingest-data-document"},
            workflow_repository=_Workflows(),
        ).run(ctx)


async def test_fanning_out_without_a_durable_queue_is_a_configuration_error() -> None:
    ctx, _ = _fanning_ctx("a.md")
    ctx.dispatcher = None

    with pytest.raises(WorkflowConfigurationError, match="durable execution queue"):
        await IngestionFanOutStep(
            config={"workflow": "ingest-data-document"},
            workflow_repository=_Workflows(),
        ).run(ctx)


async def test_an_unknown_child_workflow_is_named_in_the_error() -> None:
    ctx, _ = _fanning_ctx("a.md")

    with pytest.raises(WorkflowConfigurationError, match="no workflow named 'missing'"):
        await IngestionFanOutStep(
            config={"workflow": "missing"},
            workflow_repository=_Workflows(),
        ).run(ctx)


async def test_a_child_workflow_without_an_active_version_is_refused() -> None:
    ctx, _ = _fanning_ctx("a.md")

    with pytest.raises(WorkflowConfigurationError, match="no active version"):
        await IngestionFanOutStep(
            config={"workflow": "ingest-data-document"},
            workflow_repository=_Workflows(active=False),
        ).run(ctx)


async def test_fan_out_rejects_an_invalid_batch_size() -> None:
    ctx, _ = _fanning_ctx("a.md")

    with pytest.raises(WorkflowConfigurationError, match="document_batch_size"):
        await IngestionFanOutStep(
            config={"workflow": "ingest-data-document", "document_batch_size": 0},
            workflow_repository=_Workflows(),
        ).run(ctx)


# -- the child side: acquiring only its own share -------------------------------


def _repo(tmp_path: Path) -> Path:
    (tmp_path / "one.md").write_text("# One\n", encoding="utf-8")
    (tmp_path / "two.md").write_text("# Two\n", encoding="utf-8")
    (tmp_path / "three.md").write_text("# Three\n", encoding="utf-8")
    return tmp_path


def _source_config(tmp_path: Path) -> dict[str, Any]:
    return {
        "type": "filesystem",
        "source_id": "repo",
        "roots": [{"path": str(tmp_path), "alias": "repo"}],
        "include": ["**/*.md"],
    }


async def test_a_child_acquires_only_the_documents_it_was_given(tmp_path: Path) -> None:
    _repo(tmp_path)
    full = await IngestionSourceStep(config=_source_config(tmp_path)).run(_ctx())
    every_id = [d.external_id for d in full.ctx.metadata["ingestion.snapshot"].documents]
    mine = every_id[:1]

    child = await IngestionSourceStep(config=_source_config(tmp_path)).run(
        _ctx({"external_ids": mine, "source_id": "repo"})
    )

    snapshot = child.ctx.metadata["ingestion.snapshot"]
    assert [d.external_id for d in snapshot.documents] == mine
    assert snapshot.observed_external_ids == tuple(mine)


async def test_a_child_never_reports_a_complete_view_of_the_source(tmp_path: Path) -> None:
    """Completeness is what licenses deletion, and a share is never the whole."""
    _repo(tmp_path)
    full = await IngestionSourceStep(config=_source_config(tmp_path)).run(_ctx())
    every_id = [d.external_id for d in full.ctx.metadata["ingestion.snapshot"].documents]

    child = await IngestionSourceStep(config=_source_config(tmp_path)).run(
        _ctx({"external_ids": every_id})  # even given every id
    )

    assert full.ctx.metadata["ingestion.snapshot"].complete is True
    assert child.ctx.metadata["ingestion.snapshot"].complete is False


async def test_a_run_without_a_share_still_acquires_the_whole_source(tmp_path: Path) -> None:
    _repo(tmp_path)

    result = await IngestionSourceStep(config=_source_config(tmp_path)).run(_ctx())

    assert len(result.ctx.metadata["ingestion.snapshot"].documents) == 3
    assert result.ctx.metadata["ingestion.snapshot"].complete is True


async def test_a_child_given_an_empty_share_acquires_nothing(tmp_path: Path) -> None:
    """An empty share is not 'no restriction' — it must not index the corpus."""
    _repo(tmp_path)

    result = await IngestionSourceStep(config=_source_config(tmp_path)).run(
        _ctx({"external_ids": []})
    )

    assert result.ctx.metadata["ingestion.snapshot"].documents == ()


async def test_an_unreadable_share_is_a_configuration_error(tmp_path: Path) -> None:
    _repo(tmp_path)

    with pytest.raises(WorkflowConfigurationError, match="external_ids"):
        await IngestionSourceStep(config=_source_config(tmp_path)).run(
            _ctx({"external_ids": "one.md"})
        )


async def test_a_child_carries_the_work_flow_s_own_capabilities() -> None:
    # Every other way of submitting a workflow derives required_capabilities from
    # workflow.capabilities — the HTTP and streaming paths both do. Reading them
    # only there meant a flow restricted to particular workers was restricted
    # when a caller submitted it and open to anyone when a fan-out did, which is
    # how a machine that cannot reach the source ends up holding a share.
    ctx, dispatcher = _fanning_ctx("a.md")

    await IngestionFanOutStep(
        config={"workflow": "ingest-data-document"},
        workflow_repository=_Workflows(declared_capabilities=("ingestion",)),
    ).run(ctx)

    assert dispatcher.submissions[0].required_capabilities == ("ingestion",)


async def test_the_step_narrows_further_rather_than_replacing() -> None:
    ctx, dispatcher = _fanning_ctx("a.md")

    await IngestionFanOutStep(
        config={"workflow": "ingest-data-document", "capabilities": "gpu"},
        workflow_repository=_Workflows(declared_capabilities=("ingestion",)),
    ).run(ctx)

    # Both, deduplicated and ordered — a worker needs everything named.
    assert dispatcher.submissions[0].required_capabilities == ("gpu", "ingestion")


async def test_a_capability_named_twice_is_required_once() -> None:
    ctx, dispatcher = _fanning_ctx("a.md")

    await IngestionFanOutStep(
        config={"workflow": "ingest-data-document", "capabilities": "ingestion, gpu"},
        workflow_repository=_Workflows(declared_capabilities=("ingestion",)),
    ).run(ctx)

    assert dispatcher.submissions[0].required_capabilities == ("gpu", "ingestion")


async def test_a_flow_that_declares_nothing_still_fans_out_to_any_worker() -> None:
    # The permissive case stays permissive: this is only about not *losing* a
    # restriction that was declared.
    ctx, dispatcher = _fanning_ctx("a.md")

    await IngestionFanOutStep(
        config={"workflow": "ingest-data-document"},
        workflow_repository=_Workflows(),
    ).run(ctx)

    assert dispatcher.submissions[0].required_capabilities == ()


async def test_children_inherit_the_principal_that_started_the_run() -> None:
    """A distribution must not launder work into an identity nobody authorized.

    Every child is a separate execution with its own correlation id, and it is
    tempting to build its context field by field from what a child needs — at
    which point the principal quietly stops travelling, and the children run
    as nobody. `dataclasses.replace` carries it because it carries everything
    not named; this pins that, because the next edit here could easily not.
    """
    ctx, dispatcher = _fanning_ctx("a.md", "b.md", "c.md")
    ctx.trigger = dataclasses.replace(
        ctx.trigger,
        context=dataclasses.replace(ctx.trigger.context, principal=PrincipalRef("alice")),
    )

    await IngestionFanOutStep(
        config={"workflow": "ingest-data-document"},
        workflow_repository=_Workflows(),
    ).run(ctx)

    assert dispatcher.submissions, "nothing was distributed"
    owners = {s.trigger.context.principal for s in dispatcher.submissions}
    assert owners == {PrincipalRef("alice")}
