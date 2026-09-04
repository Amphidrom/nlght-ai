# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""A source that runs a workflow when its documents change.

An ``InboundAdapter``, which is the port that already exists for exactly this:
"non-HTTP trigger sources — polling loops, message queues, cron schedulers". It
receives the executor and the workflow repository on ``start()`` and drives an
ordinary workflow with them.

What it drives is an ordinary *execution*: given a dispatcher it submits to the
durable queue exactly as an HTTP trigger does, so a watched corpus is claimable,
retried, visible in the executions view, and able to fan out across workers.
Running it here instead would tie a whole ingestion to the lifetime of the
process that happened to notice the change.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from nlght.core.entry.context import RequestContext
from nlght.core.errors.errors import WorkflowNotFoundError
from nlght.core.execution import ExecutionSubmission
from nlght.core.ingestion import SourceSnapshot
from nlght.core.protocol.protocol import ProtocolKind
from nlght.core.trigger.trigger import Trigger, TriggerKind
from nlght.core.workflow.workflow import WorkflowInvocation
from nlght.ports.outbound.document_source import DocumentSource
from nlght.ports.outbound.execution_dispatcher import ExecutionDispatcher
from nlght.ports.outbound.workflow_executor import WorkflowExecutor
from nlght.ports.outbound.workflow_repository import WorkflowRepository

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class FilesystemWatcherSettings:
    """What to run, and how long to wait before running it.

    ``workflow`` is a workflow name like any other. The watcher decides *when*,
    never *what happens next* — that is the workflow's business.
    """

    workflow: str
    poll_interval_seconds: float = 1.0
    debounce_seconds: float = 2.0

    def __post_init__(self) -> None:
        if not self.workflow.strip():
            raise ValueError("watcher workflow name must not be empty")
        if self.poll_interval_seconds <= 0 or self.debounce_seconds < 0:
            raise ValueError("watcher intervals are invalid")


class DebouncedFilesystemWatcher:
    def __init__(
        self,
        *,
        source: DocumentSource,
        settings: FilesystemWatcherSettings,
        dispatcher: ExecutionDispatcher | None = None,
        event_id_factory: Callable[[], str] = lambda: str(uuid.uuid4()),
    ) -> None:
        self._source = source
        self._settings = settings
        # Given one, a fired watch becomes an ordinary durable execution:
        # claimable, retried, visible in the executions view, and able to fan a
        # corpus out across workers. Without one there is no queue to submit to
        # and the run happens here — the same fallback `generic_json` makes, so
        # a single process with no persistence behaves as it always did.
        self._dispatcher = dispatcher
        self._event_id_factory = event_id_factory
        self._executor: WorkflowExecutor | None = None
        self._repository: WorkflowRepository | None = None
        self._state: dict[str, str] | None = None
        self._pending_paths: set[str] = set()
        self._pending_event_id: str | None = None
        self._deadline: float | None = None
        self._task: asyncio.Task[None] | None = None

    async def start(
        self,
        executor: WorkflowExecutor,
        repository: WorkflowRepository | None,
    ) -> None:
        if self._task is not None:
            return
        self._executor = executor
        self._repository = repository
        # A first pass establishes the baseline; starting up is not a change.
        await self.poll_once(now=time.monotonic())
        self._task = asyncio.create_task(
            self._run(), name=f"watch-{self._source.source_id}"
        )

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def poll_once(self, *, now: float) -> None:
        snapshot = await self._source.acquire()
        current = _snapshot_state(snapshot)
        if self._state is None:
            self._state = current
            return

        # An incomplete snapshot proves nothing about what is gone, so it may
        # only add to what is known, never replace it.
        comparable = current if snapshot.complete else {**self._state, **current}
        changed = {
            key
            for key in self._state.keys() | comparable.keys()
            if self._state.get(key) != comparable.get(key)
        }
        if changed:
            self._pending_paths.update(changed)
            self._deadline = now + self._settings.debounce_seconds
            self._state = comparable

        if self._pending_paths and self._deadline is not None and now >= self._deadline:
            paths = tuple(sorted(self._pending_paths))
            # Held, not consumed: if the run fails, the next tick retries the
            # same change under the same identity rather than inventing a
            # second event for it.
            self._pending_event_id = self._pending_event_id or self._event_id_factory()
            await self._run_workflow(self._pending_event_id, paths)
            self._pending_paths.clear()
            self._pending_event_id = None
            self._deadline = None

    async def _run_workflow(self, event_id: str, paths: tuple[str, ...]) -> None:
        if self._executor is None or self._repository is None:
            raise RuntimeError("watcher was not started with an executor and repository")

        workflow = await self._repository.find_by_name(self._settings.workflow)
        if workflow is None or not workflow.enabled:
            raise WorkflowNotFoundError(
                f"Watcher on source '{self._source.source_id}' found no enabled "
                f"workflow named '{self._settings.workflow}'."
            )
        version = await self._repository.find_active_version(workflow.workflow_id)
        if version is None:
            raise WorkflowNotFoundError(
                f"Workflow '{workflow.name}' has no active version."
            )

        now = datetime.now(UTC)
        trigger = Trigger(
            kind=TriggerKind.INBOUND_EVENT,
            protocol=ProtocolKind.GENERIC_JSON,
            operation=workflow.name,
            payload={
                "source_id": self._source.source_id,
                "event_id": event_id,
                "changed_paths": list(paths),
            },
            context=RequestContext(
                correlation_id=event_id,
                request_id=event_id,
                received_at=now,
                path=f"watch://{self._source.source_id}",
                method="EVENT",
                headers={},
                query_params={},
                client_host=None,
            ),
            metadata={"trigger_source": "filesystem_watch"},
        )
        logger.info(
            "watch.fire | source=%s workflow=%s changed=%d",
            self._source.source_id, workflow.name, len(paths),
        )
        if self._dispatcher is None:
            await self._executor.execute(
                WorkflowInvocation(trigger=trigger, workflow=workflow, version=version)
            )
            return

        record = await self._dispatcher.submit(
            ExecutionSubmission(
                workflow_id=workflow.workflow_id,
                workflow_version_id=version.version_id,
                trigger=trigger,
                # The event id, so a retried change coalesces onto the run it
                # already started instead of queueing a second one for the same
                # edit. The watcher holds that id until the run is away, which
                # is what makes the retry above safe.
                idempotency_key=f"watch:{self._source.source_id}:{event_id}",
                required_capabilities=tuple(workflow.capabilities or ()),
                exclusive=workflow.is_blocking,
            )
        )
        logger.info(
            "watch.submitted | source=%s workflow=%s execution=%s",
            self._source.source_id, workflow.name, record.execution_id,
        )

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(self._settings.poll_interval_seconds)
            try:
                await self.poll_once(now=time.monotonic())
            except Exception as exc:
                # A failing poll or run must not kill the watcher; the next
                # tick tries again.
                logger.warning(
                    "watch.poll_failed | source=%s error=%s",
                    self._source.source_id, type(exc).__name__,
                )


def _snapshot_state(snapshot: SourceSnapshot) -> dict[str, str]:
    revisions = {document.external_id: document.source_revision_id for document in snapshot.documents}
    return {
        external_id: revisions.get(external_id, "observed-with-diagnostic")
        for external_id in snapshot.observed_external_ids
    }
