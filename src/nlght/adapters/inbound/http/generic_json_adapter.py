# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from nlght.adapters.inbound.http.response import serialize_invocation
from nlght.core.errors.errors import AIRuntimeError, WorkflowNotFoundError
from nlght.core.execution import ExecutionStatus, ExecutionSubmission

if TYPE_CHECKING:
    from nlght.core.execution import ExecutionRecord
    from nlght.core.workflow.workflow import WorkflowInvocation
    from nlght.ports.outbound.execution_dispatcher import ExecutionDispatcher
    from nlght.ports.outbound.session_key_resolver import SessionKeyResolver

logger = logging.getLogger(__name__)

SYNC = "sync"
ASYNC = "async"
_MODES = (SYNC, ASYNC)

# The sync path, when a durable dispatcher is configured, submits like the async
# one and then waits for the worker to reach a terminal state. It polls at this
# interval and holds the request open up to this ceiling.
_SYNC_POLL_INTERVAL_SECONDS = 0.2
_SYNC_MAX_WAIT_SECONDS = 3600.0


@dataclass(frozen=True, slots=True)
class _Endpoint:
    workflow: str
    mode: str


class GenericJsonHttpProtocolAdapter:
    """Catch-all route for generic JSON requests. Register last to avoid shadowing.

    ``workflow_mapping`` maps a request path to a workflow name, the same way the
    OpenAI and Ollama adapters map a model name to one. It is the configurable
    half of the trigger surface: those two are pinned to their protocols, so a
    workflow that is not a completion is exposed here or nowhere.

    Each endpoint also chooses how it answers, because the two useful shapes
    have opposite requirements and only this adapter sees both:

    - ``sync`` runs the workflow and answers with its result. Right when the
      caller wants the answer, wrong when the workflow takes minutes — the
      request is held open for its whole duration.
    - ``async`` commits the run to the durable queue and answers ``202`` with an
      execution id. The workflow then runs on a worker, survives a gateway
      restart, and the caller is free immediately.

    Both run the same workflow through the same executor; they differ only in
    who waits for it.
    """

    def __init__(
        self,
        session_key_resolver: SessionKeyResolver | None = None,
        workflow_mapping: dict[str, Any] | None = None,
        default_mode: str = SYNC,
    ) -> None:
        self._session_key_resolver = session_key_resolver
        self._default_mode = _validate_mode(default_mode, "default_mode")
        # Normalised without a trailing slash, so "/hooks/reindex" and
        # "/hooks/reindex/" name the same endpoint.
        self._endpoints = {
            _normalise(path): _endpoint_of(path, value, self._default_mode)
            for path, value in (workflow_mapping or {}).items()
        }

    def build_router(self) -> APIRouter:
        router = APIRouter()

        @router.api_route(
            "/{full_path:path}",
            methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        )
        async def generic_ingress(full_path: str, request: Request) -> JSONResponse:
            _ = full_path
            container = request.app.state.container
            endpoint = self._endpoints.get(_normalise(request.url.path))
            try:
                raw_body = await request.body()
                session_key = await self._session_key_resolver.resolve(
                    path=request.url.path,
                    method=request.method,
                    headers={k.lower(): v for k, v in request.headers.items()},
                    query_params=dict(request.query_params),
                    raw_body=raw_body,
                ) if self._session_key_resolver else None
                invocation = await container.gateway_service.process(
                    path=request.url.path,
                    method=request.method,
                    headers=dict(request.headers),
                    query_params=dict(request.query_params),
                    raw_body=raw_body,
                    client_host=request.client.host if request.client else None,
                    workflow_name=endpoint.workflow if endpoint else None,
                    session_key=session_key,
                )

                mode = endpoint.mode if endpoint else self._default_mode
                dispatcher = getattr(container, "execution_dispatcher", None)

                # With a durable dispatcher, sync and async take the same path:
                # both submit the run to the queue so one executor on a worker
                # performs it. They differ only in who waits — async answers 202,
                # sync holds the request open for the terminal result.
                if dispatcher is not None:
                    record = await self._submit(dispatcher, request, invocation)
                    if mode == ASYNC:
                        return _accepted(record, invocation)
                    return await self._await_result(dispatcher, record)

                if mode == ASYNC:
                    raise HTTPException(
                        status_code=503,
                        detail=(
                            "This endpoint is configured as 'async' but no execution "
                            "dispatcher is available. Configure workflow persistence and "
                            "run at least one worker process."
                        ),
                    )

                # Sync without durable execution: run inline (single process, no
                # queue), which is the fallback when persistence is not configured.
                executor = getattr(container, "workflow_executor", None)
                if executor is None:
                    # Resolution without a runtime to run it: report what was
                    # resolved rather than pretending the workflow ran.
                    logger.warning(
                        "generic_json.no_executor | path=%s workflow=%s",
                        request.url.path, invocation.workflow.name,
                    )
                    return JSONResponse(serialize_invocation(invocation))

                result = await executor.execute(invocation)
                return JSONResponse(content=result)
            except WorkflowNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except AIRuntimeError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        return router

    @staticmethod
    async def _submit(
        dispatcher: ExecutionDispatcher,
        request: Request,
        invocation: WorkflowInvocation,
    ) -> ExecutionRecord:
        # A caller that retries a submission must not start the run twice. Their
        # own Idempotency-Key is honoured first; otherwise the request id stands
        # in, which makes a retried *connection* idempotent even when the caller
        # sends no key of its own.
        supplied = request.headers.get("Idempotency-Key", "").strip()
        record = await dispatcher.submit(
            ExecutionSubmission(
                workflow_id=invocation.workflow.workflow_id,
                workflow_version_id=invocation.version.version_id,
                trigger=invocation.trigger,
                idempotency_key=supplied or invocation.trigger.context.request_id,
                # The workflow states what it needs; workers advertise what they
                # have. Nothing here decides which machine runs it.
                required_capabilities=tuple(invocation.workflow.capabilities or ()),
                # A blocking workflow runs one execution at a time — enforced at
                # claim. An identical trigger coalesces onto the running one via
                # the idempotency key above; a different one queues behind it.
                exclusive=invocation.workflow.is_blocking,
            )
        )
        logger.info(
            "generic_json.submitted | path=%s workflow=%s execution=%s",
            request.url.path, invocation.workflow.name, record.execution_id,
        )
        return record

    @staticmethod
    async def _await_result(
        dispatcher: ExecutionDispatcher,
        record: ExecutionRecord,
    ) -> JSONResponse:
        """Hold the request open until the dispatched execution is terminal.

        The workflow runs on a worker — the local in-process one on a
        gateway+worker — while this polls the durable record for its outcome, so
        a sync call is the same durable, claimable, concurrency-governed run as
        an async one; it just waits for the answer.
        """
        deadline = time.monotonic() + _SYNC_MAX_WAIT_SECONDS
        while True:
            current = await dispatcher.get_status(record.execution_id)
            if current is None:
                raise HTTPException(
                    status_code=500,
                    detail="Submitted execution vanished before completing.",
                )
            if current.status is ExecutionStatus.SUCCEEDED:
                return JSONResponse(content=current.result or {})
            if current.status is ExecutionStatus.FAILED:
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": "workflow execution failed",
                        "diagnostics": current.diagnostics,
                    },
                )
            if current.status is ExecutionStatus.CANCELLED:
                raise HTTPException(status_code=409, detail="Execution was cancelled.")
            if time.monotonic() >= deadline:
                raise HTTPException(
                    status_code=504,
                    detail=(
                        f"Execution {current.execution_id} did not complete within "
                        f"{int(_SYNC_MAX_WAIT_SECONDS)}s."
                    ),
                )
            await asyncio.sleep(_SYNC_POLL_INTERVAL_SECONDS)


def _accepted(record: ExecutionRecord, invocation: WorkflowInvocation) -> JSONResponse:
    """The async answer: the run is committed, here is its id."""
    return JSONResponse(
        status_code=202,
        content={
            "execution_id": str(record.execution_id),
            "status": record.status.value,
            "workflow": invocation.workflow.name,
        },
    )


def _validate_mode(mode: str, where: str) -> str:
    normalised = str(mode).strip().lower()
    if normalised not in _MODES:
        raise ValueError(
            f"generic_json {where} must be one of {list(_MODES)}, got {mode!r}"
        )
    return normalised


def _endpoint_of(path: str, value: Any, default_mode: str) -> _Endpoint:  # noqa: ANN401 - raw YAML
    """One entry of the `workflows` map.

    A bare string is the common case and stays the short form; a mapping adds
    the mode for the endpoints that need it.
    """
    if isinstance(value, str):
        return _Endpoint(workflow=value, mode=default_mode)
    if isinstance(value, dict):
        workflow = str(value.get("workflow", "")).strip()
        if not workflow:
            raise ValueError(f"generic_json endpoint '{path}' names no workflow")
        return _Endpoint(
            workflow=workflow,
            mode=_validate_mode(value.get("mode", default_mode), f"mode for '{path}'"),
        )
    raise ValueError(
        f"generic_json endpoint '{path}' must map to a workflow name or to "
        f"{{workflow: ..., mode: ...}}, got {type(value).__name__}"
    )


def _normalise(path: str) -> str:
    stripped = path.rstrip("/")
    return stripped or "/"
