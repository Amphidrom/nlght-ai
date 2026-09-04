# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import httpx2
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from nlght.adapters.inbound.http.streaming import stream_execution_signals
from nlght.adapters.outbound.signals.buffering import BufferingSignalEmitter
from nlght.adapters.outbound.signals.streaming import QueuedSignalEmitter
from nlght.core.signals.signal import Signal
from nlght.ports.outbound.model_provider_backend import (
    NativeProxyCapable,
    RunningModelsCapable,
)

if TYPE_CHECKING:
    from nlght.ports.outbound.model_provider_backend import ModelProviderBackend
    from nlght.ports.outbound.session_key_resolver import SessionKeyResolver

logger = logging.getLogger(__name__)

_NO_BACKEND = "No model provider configured for this adapter."
_NOT_SUPPORTED = "This endpoint is not supported by the configured model provider."


class OllamaHttpProtocolAdapter:
    """Registers Ollama-native API routes on the gateway.

    Exposes the Ollama ``/api/`` surface:
    - ``GET  /api/tags``      — list models (via ModelProviderBackend.list_models)
    - ``GET  /api/ps``        — running models (RunningModelsCapable or empty list)
    - ``GET  /api/version``   — version (NativeProxyCapable or 501)
    - ``POST /api/show``      — model info (NativeProxyCapable or 501)
    - ``POST /api/generate``  — generate (NativeProxyCapable or 501)
    - ``POST /api/chat``      — chat (workflow executor or direct ModelClient inference)

    Model listing and running-model queries go through the configured
    ModelProviderBackend instead of an HTTP proxy, so any provider kind
    (Ollama, Anthropic, OpenAI, Google) can back an Ollama-protocol adapter.

    Ollama-native-only endpoints (generate, show, version) are proxied via
    ``NativeProxyCapable.native_base_url`` when available; otherwise 501.
    """

    def __init__(
        self,
        model_backend: ModelProviderBackend | None = None,
        workflow_mapping: dict[str, str] | None = None,
        session_key_resolver: SessionKeyResolver | None = None,
        model_provider_name: str = "",
    ) -> None:
        self._model_backend = model_backend
        self._session_key_resolver = session_key_resolver
        self._workflow_mapping: dict[str, str] = workflow_mapping or {}
        self._model_provider_name = model_provider_name

    # ------------------------------------------------------------------
    # Router
    # ------------------------------------------------------------------

    def build_router(self) -> APIRouter:
        router = APIRouter(prefix="/api")
        adapter = self

        @router.get("/tags")
        async def tags(request: Request) -> Response:
            return await adapter._list_models(request)

        @router.get("/ps")
        async def ps(request: Request) -> Response:
            return await adapter._list_running(request)

        @router.get("/version")
        async def version(request: Request) -> Response:
            return await adapter._native_proxy_get(request, "/api/version")

        @router.post("/show")
        async def show(request: Request) -> Response:
            return await adapter._native_proxy_post(request, "/api/show")

        @router.post("/generate")
        async def generate(request: Request) -> Response:
            return await adapter._native_proxy_post_stream(request, "/api/generate")

        @router.post("/chat")
        async def chat(request: Request) -> Response:
            return await adapter._handle_chat(request)

        return router

    # ------------------------------------------------------------------
    # GET /api/tags
    # ------------------------------------------------------------------

    async def _list_models(self, request: Request) -> JSONResponse:
        if self._model_backend is None:
            raise HTTPException(503, _NO_BACKEND)
        models = await self._model_backend.list_models()
        entries: list[dict[str, Any]] = []
        for m in models:
            entries.append(
                {
                    "name": m.name,
                    "model": m.name,
                    "modified_at": m.modified_at.isoformat() if m.modified_at else None,
                    "size": m.size or 0,
                    "digest": m.digest or "",
                    "details": {},
                }
            )
        return JSONResponse(content={"models": entries})

    # ------------------------------------------------------------------
    # GET /api/ps
    # ------------------------------------------------------------------

    async def _list_running(self, request: Request) -> JSONResponse:
        if self._model_backend is not None and isinstance(
            self._model_backend, RunningModelsCapable
        ):
            running = await self._model_backend.list_running_models()
            entries: list[dict[str, Any]] = [
                {
                    "name": r.name,
                    "model": r.name,
                    "size": r.size or 0,
                    "digest": r.digest or "",
                    "details": {},
                    "expires_at": r.expires_at.isoformat() if r.expires_at else None,
                    "size_vram": r.size_vram,
                }
                for r in running
            ]
            return JSONResponse(content={"models": entries})
        return JSONResponse(content={"models": []})

    # ------------------------------------------------------------------
    # POST /api/chat  — workflow executor or direct inference
    # ------------------------------------------------------------------

    async def _handle_chat(self, request: Request) -> Response:
        container = getattr(request.app.state, "container", None)
        executor = getattr(container, "workflow_executor", None)
        workflow_name = self._workflow_mapping.get("chat")

        if executor is not None and workflow_name and container is not None:
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
                workflow_name=workflow_name,
                session_key=session_key,
            )
            if self._model_provider_name and "model_provider" not in invocation.trigger.payload:
                enriched = replace(
                    invocation.trigger,
                    payload={**invocation.trigger.payload, "model_provider": self._model_provider_name},
                )
                invocation = replace(invocation, trigger=enriched)
            model = invocation.trigger.payload.get("model", "")
            if invocation.trigger.stream:
                return StreamingResponse(
                    _stream_as_ndjson(
                        stream_execution_signals(container, invocation), model=model
                    ),
                    media_type="application/x-ndjson",
                    headers={
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no",
                        "Connection": "keep-alive",
                    },
                )
            result = await executor.execute(invocation)
            return JSONResponse(content=_openai_result_to_ollama(result, model=model))

        return await self._direct_inference(request)

    async def _direct_inference(self, request: Request) -> Response:
        """Direct model inference — replaces the legacy httpx proxy fallback."""
        if self._model_backend is None:
            raise HTTPException(503, _NO_BACKEND)

        raw_body = await request.body()
        try:
            payload = json.loads(raw_body.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise HTTPException(400, "Invalid JSON body.") from exc

        messages = payload.get("messages", [])
        model: str | None = payload.get("model") or None
        stream = bool(payload.get("stream", True))

        if stream:
            emitter = QueuedSignalEmitter()
            bound = self._model_backend.bind(
                model=model, emitter=emitter, stream=True
            )

            async def _run_and_close() -> None:
                try:
                    await bound.call(messages)
                finally:
                    await emitter.close()

            asyncio.create_task(_run_and_close())
            return StreamingResponse(
                _stream_as_ndjson(emitter, model=model or ""),
                media_type="application/x-ndjson",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "Connection": "keep-alive",
                },
            )

        buffering_emitter = BufferingSignalEmitter()
        bound = self._model_backend.bind(model=model, emitter=buffering_emitter, stream=False)
        await bound.call(messages)
        content = "".join(
            s.content for s in buffering_emitter.collected() if s.kind == "result"
        )
        return JSONResponse(
            content=_build_ollama_response(content, model=model or "")
        )

    # ------------------------------------------------------------------
    # Native-proxy helpers (Ollama-only endpoints)
    # ------------------------------------------------------------------

    async def _native_proxy_get(self, request: Request, path: str) -> Response:
        base = self._native_url()
        target = base + path
        async with httpx2.AsyncClient() as client:
            try:
                upstream = await client.get(
                    target,
                    headers=_forward_headers(request),
                    timeout=10.0,
                )
            except httpx2.RequestError as exc:
                raise HTTPException(502, f"Model provider unreachable: {exc}") from exc
        return JSONResponse(content=upstream.json(), status_code=upstream.status_code)

    async def _native_proxy_post(self, request: Request, path: str) -> Response:
        base = self._native_url()
        raw_body = await request.body()
        target = base + path
        async with httpx2.AsyncClient() as client:
            try:
                upstream = await client.post(
                    target,
                    headers=_forward_headers(request),
                    content=raw_body,
                    timeout=30.0,
                )
            except httpx2.RequestError as exc:
                raise HTTPException(502, f"Model provider unreachable: {exc}") from exc
        return JSONResponse(content=upstream.json(), status_code=upstream.status_code)

    async def _native_proxy_post_stream(self, request: Request, path: str) -> Response:
        base = self._native_url()
        raw_body = await request.body()
        target = base + path
        return StreamingResponse(
            _stream_upstream("POST", target, _forward_headers(request), raw_body),
            media_type="application/x-ndjson",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    def _native_url(self) -> str:
        if self._model_backend is not None and isinstance(
            self._model_backend, NativeProxyCapable
        ):
            return self._model_backend.native_base_url.rstrip("/")
        raise HTTPException(501, _NOT_SUPPORTED)


# ------------------------------------------------------------------
# Signal → Ollama NDJSON serialisation
# ------------------------------------------------------------------

async def _stream_as_ndjson(
    signals: AsyncIterable[Signal],
    model: str,
) -> AsyncIterator[bytes]:
    created_at = datetime.now(UTC).isoformat()
    async for signal in signals:
        if signal.kind == "done":
            frame = {
                "model": model,
                "created_at": created_at,
                "message": {"role": "assistant", "content": ""},
                "done": True,
            }
        else:
            frame = {
                "model": model,
                "created_at": created_at,
                "message": {"role": signal.role or "assistant", "content": signal.content},
                "done": False,
            }
        yield (json.dumps(frame, ensure_ascii=False) + "\n").encode()


def _openai_result_to_ollama(result: dict[str, Any], model: str) -> dict[str, Any]:
    """Convert execute() OpenAI-format dict to Ollama response format."""
    choices = result.get("choices", [{}])
    message = choices[0].get("message", {}) if choices else {}
    return {
        "model": model,
        "created_at": datetime.now(UTC).isoformat(),
        "message": {
            "role": message.get("role", "assistant"),
            "content": message.get("content", ""),
        },
        "done": True,
        "done_reason": choices[0].get("finish_reason", "stop") if choices else "stop",
    }


def _build_ollama_response(content: str, model: str) -> dict[str, Any]:
    return {
        "model": model,
        "created_at": datetime.now(UTC).isoformat(),
        "message": {"role": "assistant", "content": content},
        "done": True,
        "done_reason": "stop",
    }


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _forward_headers(request: Request) -> dict[str, str]:
    return {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length", "transfer-encoding")
    }


async def _stream_upstream(
    method: str,
    url: str,
    headers: dict[str, str],
    body: bytes,
) -> AsyncIterator[bytes]:
    async with httpx2.AsyncClient() as client:
        try:
            async with client.stream(
                method, url, headers=headers, content=body, timeout=None
            ) as upstream:
                async for chunk in upstream.aiter_bytes():
                    yield chunk
        except httpx2.RequestError as exc:
            yield (json.dumps({"error": str(exc)}) + "\n").encode()
