# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from nlght.application.execution.worker import ExecutionWorker, ExecutionWorkerSettings
from nlght.core.entry.context import RequestContext
from nlght.core.execution import (
    ExecutionClaim,
    ExecutionRecord,
    ExecutionStatus,
)
from nlght.core.protocol.protocol import ProtocolKind
from nlght.core.signals.signal import Signal
from nlght.core.trigger.trigger import Trigger, TriggerKind
from nlght.core.workflow.workflow import WorkflowDef, WorkflowInvocation, WorkflowVersionDef


def _claim(
    number: int,
    workflow_id: uuid.UUID,
    version_id: uuid.UUID,
    *,
    stream: bool = False,
) -> ExecutionClaim:
    now = datetime.now(UTC)
    worker_id = uuid.uuid4()
    boot_token = uuid.uuid4()
    record = ExecutionRecord(
        execution_id=uuid.uuid4(),
        workflow_id=workflow_id,
        workflow_version_id=version_id,
        trigger=Trigger(
            kind=TriggerKind.HTTP_REQUEST,
            protocol=ProtocolKind.GENERIC_JSON,
            operation="ingest",
            payload={"number": number},
            stream=stream,
            context=RequestContext(
                correlation_id=f"corr-{number}",
                request_id=f"request-{number}",
                received_at=now,
                path="/ingest",
                method="POST",
                headers={},
                query_params={},
                client_host=None,
            ),
        ),
        idempotency_key=f"job-{number}",
        required_capabilities=(),
        priority=0,
        artifact_refs=(),
        source_config_revision=None,
        max_attempts=3,
        attempt_count=number,
        fencing_token=number,
        status=ExecutionStatus.LEASED,
        owner_worker_id=worker_id,
        owner_boot_token=boot_token,
        lease_expires_at=now + timedelta(seconds=30),
        next_attempt_at=now,
        cancel_requested=False,
        progress={},
        result=None,
        diagnostics=None,
        created_at=now,
        updated_at=now,
        started_at=None,
        completed_at=None,
    )
    return ExecutionClaim(
        execution=record,
        attempt_id=uuid.uuid4(),
        attempt_number=number,
        worker_id=worker_id,
        boot_token=boot_token,
        fencing_token=number,
        lease_expires_at=now + timedelta(seconds=30),
    )


class _Repository:
    def __init__(self, claims: list[ExecutionClaim]) -> None:
        self.claims = claims
        self.claim_count = 0
        self.in_state_transition = False
        self.completed = asyncio.Event()
        self.success_count = 0

    async def register_worker(self, *_args: object, **_kwargs: object) -> None:
        return None

    async def heartbeat_worker(self, *_args: object, **_kwargs: object) -> bool:
        return True

    async def stop_worker(self, *_args: object, **_kwargs: object) -> None:
        return None

    async def claim(self, *_args: object, **_kwargs: object) -> ExecutionClaim | None:
        self.in_state_transition = True
        try:
            if not self.claims:
                return None
            self.claim_count += 1
            return self.claims.pop(0)
        finally:
            self.in_state_transition = False

    async def mark_running(self, *_args: object, **_kwargs: object) -> bool:
        self.in_state_transition = True
        try:
            return True
        finally:
            self.in_state_transition = False

    async def heartbeat_execution(self, *_args: object, **_kwargs: object) -> bool:
        return True

    async def succeed(self, _claim: ExecutionClaim, _result: dict[str, Any]) -> bool:
        self.success_count += 1
        if self.success_count == 2:
            self.completed.set()
        return True

    async def fail(self, *_args: object, **_kwargs: object) -> bool:
        return True


class _WorkflowRepository:
    def __init__(self, workflow_id: uuid.UUID, version_id: uuid.UUID) -> None:
        self.workflow = WorkflowDef(workflow_id, "ingestion", True, [])
        self.version = WorkflowVersionDef(version_id, workflow_id, 1, "active", [])

    async def find_by_id(self, _workflow_id: uuid.UUID) -> WorkflowDef:
        return self.workflow

    async def find_version(
        self,
        _workflow_id: uuid.UUID,
        _workflow_version_id: uuid.UUID,
    ) -> WorkflowVersionDef:
        return self.version


class _Executor:
    def __init__(self, repository: _Repository) -> None:
        self.repository = repository
        self.first_started = asyncio.Event()
        self.release_first = asyncio.Event()
        self.calls = 0

    async def execute(self, _invocation: WorkflowInvocation) -> dict[str, Any]:
        assert not self.repository.in_state_transition
        self.calls += 1
        if self.calls == 1:
            self.first_started.set()
            await self.release_first.wait()
        return {"call": self.calls}


async def test_worker_enforces_capacity_and_executes_outside_state_transition() -> None:
    workflow_id = uuid.uuid4()
    version_id = uuid.uuid4()
    repository = _Repository(
        [_claim(1, workflow_id, version_id), _claim(2, workflow_id, version_id)]
    )
    executor = _Executor(repository)
    worker = ExecutionWorker(
        repository=repository,  # type: ignore[arg-type]
        workflow_repository=_WorkflowRepository(workflow_id, version_id),  # type: ignore[arg-type]
        executor=executor,  # type: ignore[arg-type]
        settings=ExecutionWorkerSettings(
            concurrency=1,
            poll_interval_seconds=0.01,
            lease_seconds=1,
            heartbeat_seconds=0.2,
        ),
    )

    await worker.start()
    await asyncio.wait_for(executor.first_started.wait(), timeout=1)
    assert repository.claim_count == 1
    assert worker.active_count == 1

    executor.release_first.set()
    await asyncio.wait_for(repository.completed.wait(), timeout=1)
    await worker.stop()

    assert executor.calls == 2
    assert repository.success_count == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"concurrency": 0},
        {"poll_interval_seconds": 0},
        {"lease_seconds": 0},
        {"lease_seconds": 10, "heartbeat_seconds": 10},
        {"retry_base_seconds": 2, "retry_max_seconds": 1},
    ],
)
def test_worker_settings_reject_invalid_values(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        ExecutionWorkerSettings(**kwargs)


class _StreamingExecutor:
    def __init__(self, signals: list[Signal]) -> None:
        self._signals = signals
        self.execute_calls = 0

    async def execute(self, _invocation: WorkflowInvocation) -> dict[str, Any]:
        self.execute_calls += 1
        return {"buffered": True}

    async def stream_signals(self, _invocation: WorkflowInvocation):
        for signal in self._signals:
            yield signal


class _RecordingBroker:
    def __init__(self) -> None:
        self.published: list[tuple[uuid.UUID, Signal]] = []
        self.closed: list[uuid.UUID] = []

    async def publish(self, execution_id: uuid.UUID, signal: Signal) -> None:
        self.published.append((execution_id, signal))

    async def close(self, execution_id: uuid.UUID) -> None:
        self.closed.append(execution_id)


class _OneShotRepository(_Repository):
    def __init__(self, claims: list[ExecutionClaim]) -> None:
        super().__init__(claims)
        self.results: list[dict[str, Any]] = []

    async def succeed(self, _claim: ExecutionClaim, result: dict[str, Any]) -> bool:
        self.results.append(result)
        self.completed.set()
        return True


async def test_worker_streams_a_streaming_execution_through_the_broker() -> None:
    workflow_id = uuid.uuid4()
    version_id = uuid.uuid4()
    claim = _claim(1, workflow_id, version_id, stream=True)
    signals = [
        Signal(role="assistant", content="he", kind="token"),
        Signal(role="assistant", content="llo", kind="token"),
        Signal(role="assistant", content="", kind="done"),
    ]
    executor = _StreamingExecutor(signals)
    broker = _RecordingBroker()
    repository = _OneShotRepository([claim])
    worker = ExecutionWorker(
        repository=repository,  # type: ignore[arg-type]
        workflow_repository=_WorkflowRepository(workflow_id, version_id),  # type: ignore[arg-type]
        executor=executor,  # type: ignore[arg-type]
        stream_broker=broker,  # type: ignore[arg-type]
        settings=ExecutionWorkerSettings(
            concurrency=1, poll_interval_seconds=0.01, lease_seconds=1, heartbeat_seconds=0.2
        ),
    )

    await worker.start()
    await asyncio.wait_for(repository.completed.wait(), timeout=1)
    await worker.stop()

    # The streaming path published every signal to the broker and closed it,
    # without taking the buffered execute() path.
    assert [s.content for _, s in broker.published] == ["he", "llo", ""]
    assert broker.closed == [claim.execution.execution_id]
    assert executor.execute_calls == 0
    # The durable record keeps the assembled content.
    assert repository.results[0]["choices"][0]["message"]["content"] == "hello"


# ---------------------------------------------------------------------------
# What is worth attempting again, and what is not
# ---------------------------------------------------------------------------

def test_a_permanent_failure_is_classified_by_type_and_not_by_message() -> None:
    """The rule, asserted where it would be undone.

    A timeout, a refused connection and a busy backend are worth another
    attempt. A misconfigured step and a write the domain refuses are not:
    retrying burns attempts on an outcome that cannot change, and where a model
    produced the input a retry can succeed *by accident* and hide the fault
    instead of surfacing it — a kind conflict caught that way stays caught until
    it recurs on a corpus somebody cares about.

    By type, because classifying on message text is a guess about wording that
    breaks when somebody rephrases an error and cannot separate two failures
    that read alike.
    """
    import inspect

    from nlght.application.execution import worker as worker_module
    from nlght.core.errors.errors import PermanentError, WorkflowConfigurationError
    from nlght.core.knowledge import KnowledgeKindConflictError

    source = inspect.getsource(worker_module.ExecutionWorker._execute)  # noqa: SLF001
    [line] = [row.strip() for row in source.splitlines() if "retry_at =" in row]

    assert "isinstance(exc, PermanentError)" in line
    assert "str(exc)" not in line and ".message" not in line

    # Both kinds of permanent failure answer to it.
    assert issubclass(WorkflowConfigurationError, PermanentError)
    assert issubclass(KnowledgeKindConflictError, PermanentError)
    # And an ordinary failure still gets another attempt.
    assert not issubclass(TimeoutError, PermanentError)
