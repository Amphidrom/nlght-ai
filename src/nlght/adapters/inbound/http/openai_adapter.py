# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import ValidationError

from nlght.adapters.inbound.http.openai_schemas import ChatCompletionRequest
from nlght.adapters.outbound.signals.buffering import BufferingSignalEmitter
from nlght.adapters.outbound.signals.streaming import QueuedSignalEmitter

if TYPE_CHECKING:
    from nlght.ports.outbound.model_provider_backend import ModelProviderBackend
    from nlght.ports.outbound.session_key_resolver import SessionKeyResolver

_NO_BACKEND = "No model provider configured for this adapter."


class OpenAIHttpProtocolAdapter:
    """Registers OpenAI-compatible HTTP routes on the gateway.

    ``GET /v1/models`` is served via the configured ``ModelProviderBackend``.
    The response is always in OpenAI JSON format regardless of the underlying
    provider kind (Ollama, Anthropic, OpenAI, Google).

    ``POST /v1/chat/completions`` is routed via WorkflowExecutor when both an
    executor and a workflow mapping are present in the container.  Falls back
    to a direct ``ModelClient`` call via the configured backend.
    """

    def __init__(
        self,
        base_path: str = "/v1",
        model_backend: ModelProviderBackend | None = None,
        workflow_mapping: dict[str, str] | None = None,
        session_key_resolver: SessionKeyResolver | None = None,
        model_provider_name: str = "",
    ) -> None:
        self._base_path = base_path.rstrip("/")
        self._model_backend = model_backend
        self._workflow_mapping: dict[str, str] = workflow_mapping or {}
        self._session_key_resolver = session_key_resolver
        self._model_provider_name = model_provider_name

    # ------------------------------------------------------------------
    # Router
    # ------------------------------------------------------------------

    def build_router(self) -> APIRouter:
        router = APIRouter(prefix=self._base_path)
        adapter = self

        @router.get("/models")
        async def list_models(request: Request) -> Response:
            return await adapter._list_models(request)

        @router.post("/chat/completions")
        async def chat_completions(request: Request) -> Response:
            return await adapter._handle_chat_completions(request)

        return router

    # ------------------------------------------------------------------
    # GET /models
    # ------------------------------------------------------------------

    async def _list_models(self, request: Request) -> JSONResponse:
        if self._model_backend is None:
            raise HTTPException(503, _NO_BACKEND)
        models = await self._model_backend.list_models()
        data = [
            {
                "id": m.name,
                "object": "model",
                "created": 0,
                "owned_by": "nlght",
            }
            for m in models
        ]
        return JSONResponse(content={"object": "list", "data": data})

    # ------------------------------------------------------------------
    # POST /chat/completions
    # ------------------------------------------------------------------

    async def _handle_chat_completions(self, request: Request) -> Response:
        container = getattr(request.app.state, "container", None)
        executor = getattr(container, "workflow_executor", None)
        workflow_name = self._workflow_mapping.get("chat_completions")

        if executor is not None and workflow_name and container is not None:
            raw_body = await request.body()
            _parse_chat_request(raw_body)  # validate before handing off
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
                workflow_name=workflow_name,
                session_key=session_key,
            )
            if self._model_provider_name and "model_provider" not in invocation.trigger.payload:
                enriched = replace(
                    invocation.trigger,
                    payload={**invocation.trigger.payload, "model_provider": self._model_provider_name},
                )
                invocation = replace(invocation, trigger=enriched)
            if invocation.trigger.stream:
                return StreamingResponse(
                    executor.stream(invocation),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no",
                        "Connection": "keep-alive",
                    },
                )
            result = await executor.execute(invocation)
            return JSONResponse(content=result)

        return await self._direct_inference(request)

    async def _direct_inference(self, request: Request) -> Response:
        """Direct model inference — replaces the legacy httpx proxy fallback."""
        raw_body = await request.body()
        chat_req = _parse_chat_request(raw_body)  # validate first — 422 before 503

        if self._model_backend is None:
            raise HTTPException(503, _NO_BACKEND)
        messages = [{"role": m.role, "content": m.content} for m in chat_req.messages]
        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        if chat_req.stream:
            emitter = QueuedSignalEmitter()
            bound = self._model_backend.bind(
                model=chat_req.model, emitter=emitter, stream=True
            )

            async def _run_and_close() -> None:
                try:
                    await bound.call(messages)
                finally:
                    await emitter.close()

            asyncio.create_task(_run_and_close())
            return StreamingResponse(
                _signals_to_sse(emitter, model=chat_req.model, chunk_id=chunk_id),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "Connection": "keep-alive",
                },
            )

        buffering_emitter = BufferingSignalEmitter()
        bound = self._model_backend.bind(
            model=chat_req.model, emitter=buffering_emitter, stream=False
        )
        await bound.call(messages)
        content = "".join(
            s.content for s in buffering_emitter.collected() if s.kind == "result"
        )
        return JSONResponse(
            content=_build_openai_response(content, chat_req.model, chunk_id)
        )


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _parse_chat_request(raw_body: bytes) -> ChatCompletionRequest:
    try:
        return ChatCompletionRequest.model_validate_json(raw_body)
    except ValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=json.loads(exc.json()),
        ) from exc


def _build_openai_response(content: str, model: str, response_id: str) -> dict[str, Any]:
    return {
        "id": response_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


async def _signals_to_sse(
    emitter: QueuedSignalEmitter,
    model: str,
    chunk_id: str,
) -> AsyncIterator[bytes]:
    created = int(time.time())
    async for signal in emitter:
        if signal.kind == "done":
            frame = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            }
        else:
            frame = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": signal.content},
                        "finish_reason": None,
                    }
                ],
            }
        yield f"data: {json.dumps(frame, ensure_ascii=False)}\n\n".encode()
    yield b"data: [DONE]\n\n"
