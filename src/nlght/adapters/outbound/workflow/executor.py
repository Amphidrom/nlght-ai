# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
import threading
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from nlght.adapters.outbound.signals.buffering import BufferingSignalEmitter
from nlght.adapters.outbound.signals.streaming import QueuedSignalEmitter
from nlght.adapters.outbound.workflow.loader import StepLoader
from nlght.core.errors.errors import WorkflowConfigurationError, WorkflowExecutionError
from nlght.core.metering.context import MeteringContext
from nlght.core.signals.signal import Signal
from nlght.core.workflow.step import StepBase, StepResult, WorkflowStepContext
from nlght.core.workflow.workflow import WorkflowInvocation, WorkflowStepDef
from nlght.ports.outbound.signal_emitter import SignalEmitter
from nlght.ports.outbound.workflow_executor import WorkflowExecutor

if TYPE_CHECKING:
    from nlght.core.trigger.trigger import Trigger
    from nlght.core.workspace.workspace import WorkspaceContext
    from nlght.ports.outbound.access_policy import ModelAccessPolicy
    from nlght.ports.outbound.metering import MeteringPort
    from nlght.ports.outbound.os_runtime import OsRuntime, OsRuntimeFactory
    from nlght.ports.outbound.playbook_catalog import PlaybookCatalogBuilder
    from nlght.ports.outbound.store_coordinator import StoreCoordinator, StoreCoordinatorFactory
    from nlght.ports.outbound.tool_catalog import ToolCatalogBuilder

logger = logging.getLogger(__name__)

_MAX_HOPS = 20


class StepMachineWorkflowExecutor(WorkflowExecutor):
    """Implements WorkflowExecutor — drives the step machine loop.

    Drives the step machine loop defined by the workflow version's steps.
    Step types are resolved via the injected StepLoader.

    For non-streaming requests the executor uses a BufferingSignalEmitter and
    assembles the collected signals into a response dict after the machine stops.

    For streaming requests the executor uses a QueuedSignalEmitter and runs the
    step machine as a concurrent asyncio Task, yielding SSE-encoded signals as
    they arrive — without waiting for the machine to finish first.
    """

    def __init__(
        self,
        *,
        loader: StepLoader,
        model_clients: dict[str, Any] | None = None,
        tool_catalog_builder: ToolCatalogBuilder | None = None,
        playbook_catalog_builder: PlaybookCatalogBuilder | None = None,
        model_access_policy: ModelAccessPolicy | None = None,
        coordinator_factory: StoreCoordinatorFactory | None = None,
        os_runtime: OsRuntimeFactory | None = None,
        metering_port: MeteringPort | None = None,
    ) -> None:
        self._loader = loader
        self._model_clients: dict[str, Any] = model_clients or {}
        self._tool_catalog_builder = tool_catalog_builder
        self._playbook_catalog_builder = playbook_catalog_builder
        self._model_access_policy = model_access_policy
        self._coordinator_factory = coordinator_factory
        self._os_runtime_factory = os_runtime
        self._metering_port = metering_port
        self._active_runs: set[str] = set()
        self._active_runs_lock = threading.Lock()

    def is_running(self, cid: str) -> bool:
        with self._active_runs_lock:
            return cid in self._active_runs

    def _try_register(self, cid: str) -> bool:
        """Atomically mark cid as running. Returns False if already running."""
        with self._active_runs_lock:
            if cid in self._active_runs:
                return False
            self._active_runs.add(cid)
            return True

    def _unregister(self, cid: str) -> None:
        with self._active_runs_lock:
            self._active_runs.discard(cid)

    # ------------------------------------------------------------------
    # WorkflowExecutor port
    # ------------------------------------------------------------------

    async def execute(self, invocation: WorkflowInvocation) -> dict[str, Any]:
        cid = invocation.trigger.context.correlation_id
        if not self._try_register(cid):
            raise WorkflowExecutionError(
                f"Workflow already running for cid={cid!r}."
            )
        coordinator, is_resumed = self._provision_coordinator(invocation.trigger)
        os_runtime  = self._provision_os_runtime(cid)
        emitter = BufferingSignalEmitter()
        try:
            await self._run(invocation, emitter, None, coordinator, os_runtime, is_resumed=is_resumed)
        finally:
            self._unregister(cid)
            self._release_coordinator(invocation.trigger, coordinator)
        return _signals_to_response(emitter.collected())

    async def stream(self, invocation: WorkflowInvocation) -> AsyncIterator[bytes]:
        sent_done = False
        async for signal in self.stream_signals(invocation):
            if signal.kind == "done":
                sent_done = True
            yield _signal_to_sse(signal)
        if not sent_done:
            yield b"data: [DONE]\n\n"

    async def stream_signals(self, invocation: WorkflowInvocation) -> AsyncIterator[Signal]:
        cid = invocation.trigger.context.correlation_id
        if not self._try_register(cid):
            raise WorkflowExecutionError(
                f"Workflow already running for cid={cid!r}."
            )
        coordinator, is_resumed = self._provision_coordinator(invocation.trigger)
        os_runtime  = self._provision_os_runtime(cid)
        emitter = QueuedSignalEmitter()

        async def _run_and_close() -> None:
            try:
                await self._run(invocation, emitter, None, coordinator, os_runtime, is_resumed=is_resumed)
            finally:
                self._unregister(cid)
                await emitter.close()
                self._release_coordinator(invocation.trigger, coordinator)

        task = asyncio.create_task(_run_and_close())
        try:
            async for signal in emitter:
                yield signal
        finally:
            await task

    # ------------------------------------------------------------------
    # Step machine
    # ------------------------------------------------------------------
    def _provision_os_runtime(self, cid: str) -> OsRuntime | None:
        if self._os_runtime_factory is None:
            return None
        return self._os_runtime_factory.bind(cid)

    def _provision_coordinator(self, trigger: Trigger) -> tuple[StoreCoordinator | None, bool]:
        if self._coordinator_factory is None:
            return None, False
        session_key = trigger.session_key
        if session_key is None:
            # No session key → ephemeral coordinator: runs in-memory, not persisted.
            return self._coordinator_factory.get_or_create(f"__ephemeral__{trigger.context.correlation_id}"), False
        try:
            is_resumed = self._coordinator_factory.exists(session_key)
            coordinator = self._coordinator_factory.get_or_create(session_key)
            return coordinator, is_resumed
        except Exception as exc:
            logger.warning("coordinator.provision_failed | error=%s — continuing without coordinator", exc)
            return None, False

    def _release_coordinator(self, trigger: Trigger, coordinator: StoreCoordinator | None) -> None:
        if coordinator is None or self._coordinator_factory is None:
            return
        if trigger.session_key is None:
            # Ephemeral — discard without saving.
            logger.debug("coordinator.ephemeral.discard | cid=%s", trigger.context.correlation_id)
            return
        try:
            self._coordinator_factory.save(trigger.session_key, coordinator)
        except Exception as exc:
            logger.warning("coordinator.release_failed | error=%s", exc)

    async def _run(
        self,
        invocation: WorkflowInvocation,
        emitter: SignalEmitter,
        workspace: WorkspaceContext | None = None,
        coordinator: StoreCoordinator | None = None,
        os_runtime: OsRuntime | None = None,
        *,
        is_resumed: bool = False,
    ) -> None:
        steps = invocation.version.steps
        if not steps:
            raise WorkflowConfigurationError(
                f"Workflow '{invocation.workflow.name}' has no steps."
            )

        resume_steps = [s for s in steps if s.is_resume and s.enabled]
        if is_resumed and resume_steps:
            start_steps = resume_steps
            logger.info(
                "workflow.resume | workflow=%s cid=%s",
                invocation.workflow.name,
                invocation.trigger.context.correlation_id,
            )
        else:
            start_steps = [s for s in steps if s.is_start and s.enabled]
            if not start_steps:
                raise WorkflowConfigurationError(
                    f"Workflow '{invocation.workflow.name}' has no enabled start step."
                )

        steps_by_id: dict[str, WorkflowStepDef] = {str(s.step_id): s for s in steps}
        type_index: dict[str, str] = {
            s.type: str(s.step_id) for s in steps if s.enabled
        }

        trigger           = invocation.trigger
        payload           = trigger.payload
        auto_save_session = bool(payload.get("auto_save_session", True))

        # Open the Turn-Branch so every OODA sub-branch has a root to merge into.
        _turn_nr: int = 0
        if coordinator is not None:
            _cid     = trigger.context.correlation_id
            _turn_nr = getattr(
                getattr(coordinator, "conversation", None),
                "current_turn_nr",
                0,
            )
            coordinator.open_branch(f"turn:{_cid}:{_turn_nr}")

        request_model = str(payload.get("model", "")) or None
        caller = trigger.context

        tool_catalog = (
            await self._tool_catalog_builder.build(
                model=request_model,
                caller=caller,
                store_coordinator=coordinator,
                workspace=workspace,
            )
            if self._tool_catalog_builder is not None
            else None
        )

        playbook_catalog = (
            await self._playbook_catalog_builder.build(
                tool_catalog=tool_catalog,
                model=request_model,
                caller=caller,
            )
            if self._playbook_catalog_builder is not None
            else None
        )
        ctx = WorkflowStepContext(
            correlation_id=trigger.context.correlation_id,
            trigger=trigger,
            model=str(payload.get("model", "")),
            messages=list(payload.get("messages", [])),
            stream=trigger.stream,
            emitter=emitter,
            tools=tool_catalog,
            playbooks=playbook_catalog,
            workspace=workspace,
            store_coordinator=coordinator,
            os_runtime=os_runtime,
            metering=self._metering_port,
        )

        current_step_id = str(start_steps[0].step_id)
        hops = 0

        logger.info(
            "workflow.start | workflow=%s cid=%s",
            invocation.workflow.name,
            ctx.correlation_id,
        )

        request_start = time.monotonic()
        request_status = "success"

        try:
            while True:
                if hops >= _MAX_HOPS:
                    max_hops_id = type_index.get("max_hops")
                    if max_hops_id and current_step_id != max_hops_id:
                        logger.error(
                            "workflow.max_hops | cid=%s hops=%d",
                            ctx.correlation_id, hops,
                        )
                        current_step_id = max_hops_id
                        continue
                    logger.warning(
                        "workflow.hard_stop | cid=%s hops=%d",
                        ctx.correlation_id, hops,
                    )
                    request_status = "max_hops"
                    break

                step_def = steps_by_id.get(current_step_id)
                if step_def is None:
                    raise WorkflowConfigurationError(
                        f"Step id '{current_step_id}' not found in workflow "
                        f"'{invocation.workflow.name}'."
                    )

                logger.info(
                    "step.enter | name=%s type=%s cid=%s",
                    step_def.name, step_def.type, ctx.correlation_id,
                )

                step: StepBase = self._loader.load(step_def)
                ctx = await self._bind_llm(ctx, step_def.config, invocation.workflow.name, step_def.name)

                step_start = time.monotonic()
                step_status = "success"
                try:
                    result: StepResult = await step.run(ctx)
                except Exception:
                    step_status = "error"
                    raise
                finally:
                    if self._metering_port is not None:
                        step_ms = (time.monotonic() - step_start) * 1000
                        try:
                            await self._metering_port.record_step(
                                workflow=invocation.workflow.name,
                                step=step_def.name,
                                status=step_status,
                                duration_ms=step_ms,
                            )
                        except Exception as exc:
                            logger.warning("metering.record_step.failed | %s", exc)

                ctx = result.ctx
                verdict = result.verdict

                if step_def.config.get("dump") and ctx.store_coordinator is not None:
                    if hasattr(ctx.store_coordinator, "_dump"):
                        ctx.store_coordinator._dump(step_def.type)

                if (sm := ctx.metadata.get("_snapshot_manager")) is not None:
                    if not ctx.metadata.get("_snapshot_replay", False) and ctx.store_coordinator is not None:
                        try:
                            sm.capture(
                                ctx.store_coordinator,
                                session_id     = ctx.metadata.get("_session_id", invocation.trigger.session_key or ""),
                                correlation_id = ctx.correlation_id,
                                step_name      = step_def.type,
                                step_verdict   = verdict or "",
                            )
                        except Exception as exc:
                            logger.warning("snapshot.capture_failed | step=%s error=%s", step_def.type, exc)

                if (
                    auto_save_session
                    and self._coordinator_factory is not None
                    and coordinator is not None
                    and invocation.trigger.session_key is not None
                ):
                    try:
                        self._coordinator_factory.save(invocation.trigger.session_key, coordinator)
                    except Exception as exc:
                        logger.warning("coordinator.checkpoint_failed | step=%s error=%s", step_def.name, exc)

                if step_def.is_terminal or verdict in ("done", "failed"):
                    logger.info(
                        "step.terminal | name=%s verdict=%s cid=%s",
                        step_def.name, verdict, ctx.correlation_id,
                    )
                    if verdict == "failed":
                        raise WorkflowExecutionError(
                            f"Workflow '{invocation.workflow.name}' terminated via 'failed' "
                            f"at step '{step_def.name}'."
                        )
                    break

                transitions: dict[str, Any] = step_def.transitions or {}
                if verdict and verdict in transitions:
                    current_step_id = str(transitions[verdict])
                elif "DEFAULT" in transitions:
                    current_step_id = str(transitions["DEFAULT"])
                else:
                    raise WorkflowConfigurationError(
                        f"Step '{step_def.name}' has no transition for verdict "
                        f"'{verdict}' and no DEFAULT."
                    )

                hops += 1

        except Exception:
            request_status = "error"
            raise
        finally:
            # Cleans up the per-invocation container via the factory.
            if self._os_runtime_factory is not None and hasattr(self._os_runtime_factory, "release_by_name"):
                try:
                    await self._os_runtime_factory.release_by_name(trigger.context.correlation_id)
                except Exception as exc:
                    logger.warning("workflow.runtime.release_failed | cid=%s error=%s", trigger.context.correlation_id, exc)

            # Merge Turn-Branch: promoted atoms → SessionResultStore
            if coordinator is not None:
                try:
                    promoted = coordinator.close_branch(promote=True)
                    coordinator.store_promoted_atoms(
                        promoted,
                        turn_nr=_turn_nr,
                        entities=[],
                    )
                except Exception as exc:
                    logger.warning("workflow.turn_branch.merge_failed | %s", exc)

            if self._metering_port is not None:
                request_ms = (time.monotonic() - request_start) * 1000
                try:
                    await self._metering_port.record_request(
                        caller=invocation.trigger.context,
                        session_key=invocation.trigger.session_key,
                        workflow=invocation.workflow.name,
                        status=request_status,
                        duration_ms=request_ms,
                    )
                except Exception as exc:
                    logger.warning("metering.record_request.failed | %s", exc)

        logger.info(
            "workflow.end | workflow=%s cid=%s hops=%d",
            invocation.workflow.name, ctx.correlation_id, hops,
        )

    async def _bind_llm(
        self,
        ctx: WorkflowStepContext,
        step_config: dict[str, Any],
        workflow_name: str = "",
        step_name: str = "",
    ) -> WorkflowStepContext:
        """Resolve and bind a ModelClient for this step into ctx.llm.

        Fetches a TokenBudget via OllamaModelClient.token_budget() so the
        BoundModelClient can apply the hard-cap safety net (ADR-0012).
        Budget results are cached per-process — /api/show is called at most
        once per model.
        """
        payload_provider = ctx.trigger.payload.get("model_provider") if ctx.trigger.payload else None
        provider_name = step_config.get("model_provider") or payload_provider
        if not provider_name or provider_name not in self._model_clients:
            return ctx

        payload_model = ctx.trigger.payload.get("model") if ctx.trigger.payload else None
        model = step_config.get("model") or payload_model or ctx.model
        backend = self._model_clients[provider_name]

        if self._model_access_policy is not None:
            try:
                if not await self._model_access_policy.is_allowed(model, ctx.trigger.context):
                    raise WorkflowConfigurationError(
                        f"Model '{model}' access denied for caller {ctx.trigger.context.correlation_id}."
                    )
            except WorkflowConfigurationError:
                raise
            except Exception as exc:
                logger.warning("llm.model_policy_error | model=%s error=%s — denying", model, exc)
                raise WorkflowConfigurationError(
                    f"Model '{model}' access policy check failed: {exc}"
                ) from exc

        token_budget = None
        if hasattr(backend, "token_budget"):
            try:
                token_budget = await backend.token_budget(model)
            except Exception as exc:
                logger.warning("llm.budget_fetch_failed | model=%s error=%s", model, exc)

        metering_ctx: MeteringContext | None = None
        if self._metering_port is not None:
            metering_ctx = MeteringContext(
                port=self._metering_port,
                caller=ctx.trigger.context,
                session_key=ctx.trigger.session_key,
                workflow=workflow_name,
                step=step_name,
                provider=provider_name,
            )

        bind_kwargs: dict[str, Any] = {
            "model": model,
            "emitter": ctx.emitter,
            "stream": ctx.stream,
            "token_budget": token_budget,
        }
        if metering_ctx is not None:
            bind_kwargs["metering"] = metering_ctx
        bound = backend.bind(**bind_kwargs)
        return dataclasses.replace(ctx, llm=bound)


# ------------------------------------------------------------------
# Signal serialisation helpers
# ------------------------------------------------------------------

def _signals_to_response(signals: list[Signal]) -> dict[str, Any]:
    """Assemble buffered signals into a response dict.

    Concatenates all token/result content from assistant signals into a
    single message.  This is intentionally minimal — proper response
    assembly will evolve with the step implementations.
    """
    content = "".join(
        s.content for s in signals
        if s.kind in ("token", "result") and s.role == "assistant"
    )
    return {
        "choices": [
            {
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ]
    }


def _signal_to_sse(signal: Signal) -> bytes:
    """Serialise a single signal to an OpenAI-compatible SSE frame."""
    if signal.kind == "done":
        return b"data: [DONE]\n\n"
    payload = {
        "choices": [
            {
                "delta": {"role": signal.role, "content": signal.content},
                "finish_reason": None,
            }
        ]
    }
    return f"data: {json.dumps(payload)}\n\n".encode()
