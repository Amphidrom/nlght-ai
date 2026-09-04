# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""The filesystem watcher as an InboundAdapter.

It decides *when* a workflow runs and nothing else: resolution and execution are
the ordinary ones, through the executor and repository the port hands it.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest

from nlght.adapters.inbound.ingestion.filesystem_watcher import (
    DebouncedFilesystemWatcher,
    FilesystemWatcherSettings,
)
from nlght.core.errors.errors import WorkflowNotFoundError
from nlght.core.ingestion import SourceDocument, SourceSnapshot
from nlght.core.trigger.trigger import TriggerKind


class _Executor:
    def __init__(self, *, failures: int = 0) -> None:
        self.invocations = []
        self.failures = failures

    async def execute(self, invocation):
        self.invocations.append(invocation)
        if self.failures:
            self.failures -= 1
            raise RuntimeError("temporary execution failure")
        return {"ok": True}


class _Repository:
    def __init__(self, *, name: str = "ingest-data", enabled: bool = True, version=True) -> None:
        self.workflow = SimpleNamespace(
            workflow_id=uuid.uuid4(), name=name, enabled=enabled, capabilities=[],
            is_blocking=False,
        )
        # `version_id`, as WorkflowVersionDef names it. The fake said
        # `workflow_version_id` and nothing noticed, because until now no test
        # read the field off a version.
        self._version = SimpleNamespace(version_id=uuid.uuid4()) if version else None

    async def find_by_name(self, name):
        return self.workflow if name == self.workflow.name else None

    async def find_active_version(self, workflow_id):
        return self._version


class _SnapshotSource:
    source_id = "repo-source"

    def __init__(self, snapshots: list[SourceSnapshot]) -> None:
        self._snapshots = snapshots
        self._position = 0

    async def acquire(self) -> SourceSnapshot:
        snapshot = self._snapshots[min(self._position, len(self._snapshots) - 1)]
        self._position += 1
        return snapshot


def _snapshot(*documents: SourceDocument, complete: bool = True) -> SourceSnapshot:
    return SourceSnapshot(
        source_id="repo-source",
        documents=documents,
        observed_external_ids=tuple(document.external_id for document in documents),
        complete=complete,
    )


class _Dispatcher:
    """Stands in for the durable queue."""

    def __init__(self) -> None:
        self.submissions = []

    async def submit(self, submission):
        self.submissions.append(submission)
        return SimpleNamespace(execution_id=uuid.uuid4())


def _watcher(source, executor, repository, dispatcher=None, **settings) -> DebouncedFilesystemWatcher:
    watcher = DebouncedFilesystemWatcher(
        source=source,
        settings=FilesystemWatcherSettings(workflow="ingest-data", **settings),
        dispatcher=dispatcher,
        event_id_factory=settings.pop("event_id_factory", lambda: "watch-event-1"),
    )
    # What start() would inject; the polling loop itself is not under test here.
    watcher._executor = executor
    watcher._repository = repository
    return watcher


@pytest.mark.asyncio
async def test_the_watcher_runs_an_ordinary_workflow(tmp_path) -> None:
    first = SourceDocument("repo-source", "fs:repo/a.py", "repo/a.py", b"one")
    changed = SourceDocument("repo-source", "fs:repo/a.py", "repo/a.py", b"two")
    executor, repository = _Executor(), _Repository()
    watcher = _watcher(
        _SnapshotSource([_snapshot(first), _snapshot(changed), _snapshot(changed)]),
        executor,
        repository,
        debounce_seconds=1,
    )

    await watcher.poll_once(now=0)
    await watcher.poll_once(now=1)
    await watcher.poll_once(now=2)

    assert len(executor.invocations) == 1
    invocation = executor.invocations[0]
    # The same WorkflowInvocation an HTTP trigger produces: resolved workflow,
    # pinned active version, a trigger describing where it came from.
    assert invocation.workflow is repository.workflow
    assert invocation.version is repository._version
    assert invocation.trigger.kind is TriggerKind.INBOUND_EVENT
    assert invocation.trigger.operation == "ingest-data"
    assert invocation.trigger.payload["changed_paths"] == ["fs:repo/a.py"]
    assert invocation.trigger.payload["source_id"] == "repo-source"


@pytest.mark.asyncio
async def test_debounces_updates_and_does_not_infer_deletion_from_a_partial_scan() -> None:
    a1 = SourceDocument("repo-source", "fs:repo/a.py", "repo/a.py", b"one")
    a2 = SourceDocument("repo-source", "fs:repo/a.py", "repo/a.py", b"two")
    b1 = SourceDocument("repo-source", "fs:repo/b.py", "repo/b.py", b"one")
    executor = _Executor()
    watcher = _watcher(
        _SnapshotSource(
            [
                _snapshot(a1, b1),
                _snapshot(a2, b1),
                # b.py missing from an *incomplete* scan proves nothing.
                _snapshot(a2, complete=False),
                _snapshot(a2, b1),
                _snapshot(a2, b1),
            ]
        ),
        executor,
        _Repository(),
        debounce_seconds=2,
    )

    await watcher.poll_once(now=0)
    await watcher.poll_once(now=1)
    await watcher.poll_once(now=2)
    await watcher.poll_once(now=3)

    assert len(executor.invocations) == 1
    assert executor.invocations[0].trigger.payload["changed_paths"] == ["fs:repo/a.py"]


@pytest.mark.asyncio
async def test_a_failed_run_is_retried_under_the_same_event_identity() -> None:
    first = SourceDocument("repo-source", "fs:repo/a.py", "repo/a.py", b"one")
    changed = SourceDocument("repo-source", "fs:repo/a.py", "repo/a.py", b"two")
    executor = _Executor(failures=1)
    generated = 0

    def event_id() -> str:
        nonlocal generated
        generated += 1
        return "stable-watch-event"

    watcher = DebouncedFilesystemWatcher(
        source=_SnapshotSource([_snapshot(first), _snapshot(changed), _snapshot(changed)]),
        settings=FilesystemWatcherSettings(workflow="ingest-data", debounce_seconds=1),
        event_id_factory=event_id,
    )
    watcher._executor = executor
    watcher._repository = _Repository()

    await watcher.poll_once(now=0)
    await watcher.poll_once(now=1)
    with pytest.raises(RuntimeError, match="temporary execution failure"):
        await watcher.poll_once(now=2)
    await watcher.poll_once(now=3)

    # One event, attempted twice — not two events for one change.
    assert generated == 1
    assert len(executor.invocations) == 2
    assert {invocation.trigger.context.correlation_id for invocation in executor.invocations} == {
        "stable-watch-event"
    }


@pytest.mark.asyncio
async def test_an_unknown_or_inactive_workflow_is_named_in_the_error() -> None:
    first = SourceDocument("repo-source", "fs:repo/a.py", "repo/a.py", b"one")
    changed = SourceDocument("repo-source", "fs:repo/a.py", "repo/a.py", b"two")
    source = [_snapshot(first), _snapshot(changed), _snapshot(changed)]

    watcher = _watcher(_SnapshotSource(source), _Executor(), _Repository(name="other"),
                       debounce_seconds=1)
    await watcher.poll_once(now=0)
    await watcher.poll_once(now=1)
    with pytest.raises(WorkflowNotFoundError, match="ingest-data"):
        await watcher.poll_once(now=2)

    watcher = _watcher(_SnapshotSource(source), _Executor(), _Repository(version=False),
                       debounce_seconds=1)
    await watcher.poll_once(now=0)
    await watcher.poll_once(now=1)
    with pytest.raises(WorkflowNotFoundError, match="no active version"):
        await watcher.poll_once(now=2)


@pytest.mark.asyncio
async def test_starting_up_is_not_a_change() -> None:
    only = SourceDocument("repo-source", "fs:repo/a.py", "repo/a.py", b"one")
    executor = _Executor()
    watcher = _watcher(_SnapshotSource([_snapshot(only)]), executor, _Repository(),
                       debounce_seconds=0)

    await watcher.poll_once(now=0)
    await watcher.poll_once(now=1)

    assert executor.invocations == []


def test_the_watcher_needs_a_workflow_to_run() -> None:
    with pytest.raises(ValueError, match="workflow name must not be empty"):
        FilesystemWatcherSettings(workflow="  ")


@pytest.mark.asyncio
async def test_a_watched_change_becomes_a_durable_execution() -> None:
    """Noticing is this adapter's job; running the corpus is the queue's.

    Executing here would tie a whole ingestion to the process that happened to
    see the file change: no lease, no retry, no fan-out across workers, and
    nothing about it in the executions view. A watch is a trigger like any
    other, so it submits like any other.
    """
    before = SourceDocument("repo-source", "fs:repo/a.py", "repo/a.py", b"one")
    after = SourceDocument("repo-source", "fs:repo/a.py", "repo/a.py", b"two")
    source = _SnapshotSource([_snapshot(before), _snapshot(after)])
    executor, repository, dispatcher = _Executor(), _Repository(), _Dispatcher()
    watcher = _watcher(source, executor, repository, dispatcher=dispatcher, debounce_seconds=0)

    await watcher.poll_once(now=0.0)
    await watcher.poll_once(now=1.0)

    assert executor.invocations == []
    assert len(dispatcher.submissions) == 1
    submitted = dispatcher.submissions[0]
    assert submitted.workflow_id == repository.workflow.workflow_id
    assert submitted.trigger.payload["changed_paths"] == ["fs:repo/a.py"]
    # The event identity, so the retry the watcher already does coalesces onto
    # the run it started rather than queueing a second one for the same edit.
    assert submitted.idempotency_key == "watch:repo-source:watch-event-1"


@pytest.mark.asyncio
async def test_without_a_queue_the_watcher_still_runs_the_workflow() -> None:
    # A single process with no persistence has no queue to submit to. It keeps
    # working exactly as it did, which is the same fallback the HTTP trigger
    # makes rather than a special case for watching.
    before = SourceDocument("repo-source", "fs:repo/a.py", "repo/a.py", b"one")
    after = SourceDocument("repo-source", "fs:repo/a.py", "repo/a.py", b"two")
    source = _SnapshotSource([_snapshot(before), _snapshot(after)])
    executor, repository = _Executor(), _Repository()
    watcher = _watcher(source, executor, repository, debounce_seconds=0)

    await watcher.poll_once(now=0.0)
    await watcher.poll_once(now=1.0)

    assert len(executor.invocations) == 1


@pytest.mark.asyncio
async def test_a_submitted_watch_carries_the_workflow_requirements() -> None:
    # A watcher must not decide where a run happens. The workflow says what it
    # needs and the queue matches that against what workers advertise.
    before = SourceDocument("repo-source", "fs:repo/a.py", "repo/a.py", b"one")
    after = SourceDocument("repo-source", "fs:repo/a.py", "repo/a.py", b"two")
    source = _SnapshotSource([_snapshot(before), _snapshot(after)])
    repository = _Repository()
    repository.workflow.capabilities = ["ingestion:data"]
    repository.workflow.is_blocking = True
    dispatcher = _Dispatcher()
    watcher = _watcher(source, _Executor(), repository, dispatcher=dispatcher, debounce_seconds=0)

    await watcher.poll_once(now=0.0)
    await watcher.poll_once(now=1.0)

    submitted = dispatcher.submissions[0]
    assert submitted.required_capabilities == ("ingestion:data",)
    assert submitted.exclusive is True
