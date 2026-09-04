# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import asyncio
import logging
import random
import socket
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from nlght.core.errors.errors import PermanentError, WorkflowConfigurationError
from nlght.core.execution import ExecutionClaim, WorkerRegistration
from nlght.core.signals.signal import Signal
from nlght.core.workflow.workflow import WorkflowInvocation
from nlght.ports.outbound.execution_repository import ExecutionRepository
from nlght.ports.outbound.execution_stream import ExecutionStreamBroker
from nlght.ports.outbound.workflow_executor import WorkflowExecutor
from nlght.ports.outbound.workflow_repository import WorkflowRepository

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class ExecutionWorkerSettings:
    instance_name: str = ""
    capabilities: tuple[str, ...] = ()
    concurrency: int = 1
    poll_interval_seconds: float = 1.0
    lease_seconds: float = 30.0
    heartbeat_seconds: float = 10.0
    retry_base_seconds: float = 1.0
    retry_max_seconds: float = 60.0

    def __post_init__(self) -> None:
        if self.concurrency < 1:
            raise ValueError("worker concurrency must be at least 1")
        if self.poll_interval_seconds <= 0:
            raise ValueError("worker poll_interval_seconds must be positive")
        if self.lease_seconds <= 0:
            raise ValueError("worker lease_seconds must be positive")
        if self.heartbeat_seconds <= 0 or self.heartbeat_seconds >= self.lease_seconds:
            raise ValueError("worker heartbeat_seconds must be positive and shorter than lease_seconds")
        if self.retry_base_seconds <= 0 or self.retry_max_seconds < self.retry_base_seconds:
            raise ValueError("worker retry bounds are invalid")


class ExecutionWorker:
    """Capability-aware worker loop for durable at-least-once executions."""

    def __init__(
        self,
        *,
        repository: ExecutionRepository,
        workflow_repository: WorkflowRepository,
        executor: WorkflowExecutor,
        settings: ExecutionWorkerSettings,
        stream_broker: ExecutionStreamBroker | None = None,
        worker_id: uuid.UUID | None = None,
        boot_token: uuid.UUID | None = None,
    ) -> None:
        self._repository = repository
        self._workflow_repository = workflow_repository
        self._executor = executor
        self._settings = settings
        self._stream_broker = stream_broker
        self._registration = WorkerRegistration(
            worker_id=worker_id or uuid.uuid4(),
            boot_token=boot_token or uuid.uuid4(),
            instance_name=settings.instance_name or socket.gethostname(),
            capabilities=settings.capabilities,
            capacity=settings.concurrency,
        )
        self._stop_event = asyncio.Event()
        self._poll_task: asyncio.Task[None] | None = None
        self._active: set[asyncio.Task[None]] = set()

    @property
    def registration(self) -> WorkerRegistration:
        return self._registration

    @property
    def active_count(self) -> int:
        return len(self._active)

    def _now(self) -> datetime:
        return datetime.now(UTC)

    def _lease_expiry(self) -> datetime:
        return self._now() + timedelta(seconds=self._settings.lease_seconds)

    async def start(self) -> None:
        if self._poll_task is not None:
            return
        self._stop_event.clear()
        await self._repository.register_worker(
            self._registration,
            expires_at=self._lease_expiry(),
        )
        self._poll_task = asyncio.create_task(
            self._run(),
            name=f"nlght-worker-{self._registration.worker_id}",
        )
        logger.info(
            "execution.worker.started | worker_id=%s instance=%s capabilities=%s capacity=%d",
            self._registration.worker_id,
            self._registration.instance_name,
            self._registration.capabilities,
            self._registration.capacity,
        )

    async def stop(self) -> None:
        if self._poll_task is None:
            return
        self._stop_event.set()
        await self._poll_task
        self._poll_task = None
        if self._active:
            for task in self._active:
                task.cancel()
            await asyncio.gather(*self._active, return_exceptions=True)
            self._active.clear()
        await self._repository.stop_worker(
            self._registration.worker_id,
            self._registration.boot_token,
        )
        logger.info("execution.worker.stopped | worker_id=%s", self._registration.worker_id)

    async def _wait_for_poll(self) -> None:
        try:
            await asyncio.wait_for(
                self._stop_event.wait(),
                timeout=self._settings.poll_interval_seconds,
            )
        except TimeoutError:
            pass

    def _discard_done(self) -> None:
        done = {task for task in self._active if task.done()}
        self._active.difference_update(done)
        for task in done:
            try:
                task.result()
            except asyncio.CancelledError:
                pass
            except Exception:
                logger.exception("execution.worker.task_failed")

    async def _run(self) -> None:
        while not self._stop_event.is_set():
            self._discard_done()
            await self._repository.heartbeat_worker(
                self._registration.worker_id,
                self._registration.boot_token,
                expires_at=self._lease_expiry(),
            )
            while (
                not self._stop_event.is_set()
                and len(self._active) < self._settings.concurrency
            ):
                claim = await self._repository.claim(
                    self._registration,
                    now=self._now(),
                    lease_expires_at=self._lease_expiry(),
                )
                if claim is None:
                    break
                task = asyncio.create_task(
                    self._execute(claim),
                    name=f"nlght-execution-{claim.execution.execution_id}",
                )
                self._active.add(task)
            await self._wait_for_poll()
        self._discard_done()

    async def _heartbeat(self, claim: ExecutionClaim, stopped: asyncio.Event) -> None:
        while not stopped.is_set():
            try:
                await asyncio.wait_for(
                    stopped.wait(),
                    timeout=self._settings.heartbeat_seconds,
                )
                return
            except TimeoutError:
                owned = await self._repository.heartbeat_execution(
                    claim,
                    lease_expires_at=self._lease_expiry(),
                )
                if not owned:
                    logger.warning(
                        "execution.worker.ownership_lost | execution_id=%s fencing=%d",
                        claim.execution.execution_id,
                        claim.fencing_token,
                    )
                    return

    def _retry_at(self, claim: ExecutionClaim) -> datetime:
        base = min(
            self._settings.retry_max_seconds,
            self._settings.retry_base_seconds * (2 ** max(0, claim.attempt_number - 1)),
        )
        jittered = min(
            self._settings.retry_max_seconds,
            random.uniform(base * 0.8, base * 1.2),  # noqa: S311 (retry jitter, not security)
        )
        return self._now() + timedelta(seconds=jittered)

    async def _execute(self, claim: ExecutionClaim) -> None:
        if not await self._repository.mark_running(
            claim,
            lease_expires_at=self._lease_expiry(),
        ):
            return

        heartbeat_stop = asyncio.Event()
        heartbeat = asyncio.create_task(self._heartbeat(claim, heartbeat_stop))
        try:
            workflow = await self._workflow_repository.find_by_id(claim.execution.workflow_id)
            version = await self._workflow_repository.find_version(
                claim.execution.workflow_id,
                claim.execution.workflow_version_id,
            )
            if workflow is None or version is None:
                raise WorkflowConfigurationError(
                    "dispatched workflow or pinned workflow version no longer exists"
                )
            invocation = WorkflowInvocation(
                trigger=claim.execution.trigger,
                workflow=workflow,
                version=version,
                # A step that fans work out needs to know which execution it is,
                # so the children it submits point back at this run.
                execution_id=claim.execution.execution_id,
            )
            if claim.execution.trigger.stream and self._stream_broker is not None:
                result = await self._run_streaming(claim, invocation)
            else:
                result = await self._executor.execute(invocation)
            await self._repository.succeed(claim, result)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # By type, never by message. A failure that will fail identically
            # however often it is repeated gets no second attempt: retrying burns
            # them on an outcome that cannot change, and where a model is in the
            # loop a retry can succeed by accident and hide the fault instead of
            # surfacing it.
            retry_at = None if isinstance(exc, PermanentError) else self._retry_at(claim)
            diagnostics: dict[str, Any] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            await self._repository.fail(claim, diagnostics, retry_at=retry_at)
            logger.warning(
                "execution.worker.execution_failed | execution_id=%s attempt=%d retry=%s error=%s",
                claim.execution.execution_id,
                claim.attempt_number,
                retry_at is not None,
                exc,
            )
        finally:
            heartbeat_stop.set()
            await heartbeat

    async def _run_streaming(
        self, claim: ExecutionClaim, invocation: WorkflowInvocation
    ) -> dict[str, Any]:
        """Run a streaming execution, publishing each signal to the broker.

        The gateway relays those signals to the caller as they arrive; the
        worker also collects them so the durable record keeps the assembled
        terminal result once the run completes.
        """
        assert self._stream_broker is not None  # noqa: S101 (guarded by the caller)
        execution_id = claim.execution.execution_id
        collected: list[Signal] = []
        try:
            async for signal in self._executor.stream_signals(invocation):
                collected.append(signal)
                await self._stream_broker.publish(execution_id, signal)
        finally:
            await self._stream_broker.close(execution_id)
        return _assemble_streamed_result(collected)


def _assemble_streamed_result(signals: list[Signal]) -> dict[str, Any]:
    """The terminal result stored for a streamed run: its assistant content.

    Mirrors the executor's buffered assembly, kept here rather than imported
    across the layer boundary — the live stream was the delivery; this is the
    durable copy the status endpoint can return after the fact.
    """
    content = "".join(
        s.content
        for s in signals
        if s.kind in ("token", "result") and s.role == "assistant"
    )
    return {
        "choices": [
            {"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
        ]
    }
