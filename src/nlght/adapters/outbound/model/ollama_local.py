# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Sequence
from typing import TYPE_CHECKING, Any

import httpx2

from nlght.adapters.outbound.model._context_chunking import (
    apply_budget_to_messages,
    needs_chunking,
    prepare_chunked_session,
)
from nlght.adapters.outbound.model._ollama_http import _OllamaHttp
from nlght.adapters.outbound.model._ollama_messages import to_ollama_messages
from nlght.adapters.outbound.model._tool_helpers import (
    append_ollama_native_tool_turn,
    append_openai_tool_turn,
    contract_to_openai_tool,
    terminal_tool_names,
)
from nlght.core.model.messages import CanonicalMessage, MessageLike, append_canonical_tool_turn
from nlght.core.model.model_info import ModelInfo, RunningModelInfo
from nlght.core.signals.signal import Signal
from nlght.ports.outbound.model_client import ModelClient, ModelStreamEvent
from nlght.ports.outbound.model_provider_backend import ModelProviderBackend, NativeProxyCapable, RunningModelsCapable
from nlght.ports.outbound.signal_emitter import SignalEmitter

if TYPE_CHECKING:
    from nlght.core.metering.context import MeteringContext
    from nlght.core.model.budget import TokenBudget
    from nlght.ports.outbound.tool_catalog import ToolCatalog

logger = logging.getLogger(__name__)


def _has_tool_protocol_messages(messages: list[dict[str, Any]]) -> bool:
    for msg in messages:
        if msg.get("role") == "tool":
            return True
        if msg.get("tool_calls"):
            return True
    return False


class OllamaClient(ModelProviderBackend, RunningModelsCapable, NativeProxyCapable):
    """Local Ollama provider.

    Tool calls use the native ``/api/chat`` NDJSON wire format.
    Plain inference falls back to ``/v1/chat/completions`` SSE.
    """

    def __init__(
        self,
        *,
        http_client: httpx2.AsyncClient,
        base_url: str,
        default_model: str,
        api_key: str = "",
        extra_headers: dict[str, str] | None = None,
        request_timeout_s: float | None = None,
        stream_connect_timeout_s: float | None = None,
        stream_read_timeout_s: float | None = None,
    ) -> None:
        self._http = _OllamaHttp(
            http_client=http_client,
            base_url=base_url,
            api_key=api_key,
            extra_headers=extra_headers,
            request_timeout_s=request_timeout_s,
            stream_connect_timeout_s=stream_connect_timeout_s,
            stream_read_timeout_s=stream_read_timeout_s,
        )
        self._default_model = default_model

    @property
    def native_base_url(self) -> str:
        return self._http.base_url

    async def list_models(self) -> list[ModelInfo]:
        return await self._http.list_models()

    async def list_running_models(self) -> list[RunningModelInfo]:
        return await self._http.list_running_models()

    async def token_budget(
        self,
        model: str | None = None,
        *,
        client_max_tokens: int | None = None,
    ) -> TokenBudget:
        return await self._http.token_budget(
            model or self._default_model,
            client_max_tokens=client_max_tokens,
        )

    def bind(
        self,
        *,
        model: str | None,
        emitter: SignalEmitter,
        stream: bool,
        token_budget: TokenBudget | None = None,
        metering: MeteringContext | None = None,
        tool_catalog: ToolCatalog | None = None,
    ) -> BoundOllamaClient:
        return BoundOllamaClient(
            backend=self,
            http=self._http,
            model=model or self._default_model,
            emitter=emitter,
            stream=stream,
            token_budget=token_budget,
            metering=metering,
            tool_catalog=tool_catalog,
        )


class BoundOllamaClient(ModelClient):
    """Request-scoped client for OllamaClient."""

    def __init__(
        self,
        *,
        backend: OllamaClient,
        http: _OllamaHttp,
        model: str,
        emitter: SignalEmitter,
        stream: bool,
        token_budget: TokenBudget | None = None,
        metering: MeteringContext | None = None,
        tool_catalog: ToolCatalog | None = None,
    ) -> None:
        self._backend = backend
        self._http = http
        self._model = model
        self._emitter = emitter
        self._stream = stream
        self._token_budget = token_budget
        self._metering = metering
        self._tool_catalog = tool_catalog

    @property
    def token_budget(self) -> TokenBudget | None:
        """The budget this client will enforce, so a prompt is built to it."""
        return self._token_budget

    async def call(self, messages: Sequence[MessageLike], *, temperature: float | None = None) -> None:
        messages = to_ollama_messages(messages)
        if self._token_budget is not None and needs_chunking(messages, self._token_budget):
            base, chunks, task = prepare_chunked_session(messages, self._token_budget)
            if chunks:
                await self._http.execute_chunked_call(
                    model=self._model,
                    base_messages=base,
                    content_chunks=chunks,
                    task_message=task,
                    token_budget=self._token_budget,
                    emitter=self._emitter,
                    stream=self._stream,
                    metering=self._metering,
                )
                return
        effective = messages
        if self._token_budget is not None:
            effective, _ = apply_budget_to_messages(messages, self._token_budget)
        target = f"{self._http.base_url}/v1/chat/completions"
        if self._stream:
            body: dict[str, Any] = {"model": self._model, "messages": list(effective), "stream": True}
            if temperature is not None:
                body["temperature"] = temperature
            await self._http.stream_via_emitter(target, body, self._emitter, self._metering)
        elif self._tool_catalog is not None:
            current = list(effective)
            input_t = output_t = 0
            terminal_names = terminal_tool_names(self._tool_catalog)
            # Derive OpenAI-format tool schemas from the catalog and send them on
            # every request — the non-streaming /v1/chat/completions loop used to
            # omit them, so the model was never told the tools existed and never
            # called them (proven by the tools_in_request=None evidence log).
            tool_defs = [contract_to_openai_tool(c) for c in self._tool_catalog.all()] or None
            for _round in range(10):
                body = {"model": self._model, "messages": current, "stream": False}
                if tool_defs:
                    body["tools"] = tool_defs
                if temperature is not None:
                    body["temperature"] = temperature
                msg, usage = await self._http.complete_message(target, body)
                input_t += usage.get("prompt_tokens", 0) or 0
                output_t += usage.get("completion_tokens", 0) or 0
                tool_calls_native = msg.get("tool_calls") or []
                content = msg.get("content") or ""
                if not tool_calls_native:
                    await self._emitter.emit(Signal(role="assistant", content=content, kind="result"))
                    break
                tool_calls_raw = [
                    {
                        "id": tc.get("id", ""),
                        "name": (tc.get("function") or {}).get("name", ""),
                        "input": (
                            args if isinstance(args := (tc.get("function") or {}).get("arguments") or {}, dict)
                            else json.loads(args or "{}")
                        ),
                    }
                    for tc in tool_calls_native
                ]
                results = [await self._tool_catalog.execute(tc) for tc in tool_calls_raw]
                current = append_openai_tool_turn(current, tool_calls_raw, results, content)
                if terminal_names and any(tc.get("name") in terminal_names for tc in tool_calls_raw):
                    break
            if self._metering is not None:
                await self._http._record_tokens(
                    self._metering, self._model, {"prompt_tokens": input_t, "completion_tokens": output_t}
                )
        else:
            body = {"model": self._model, "messages": list(effective), "stream": False}
            if temperature is not None:
                body["temperature"] = temperature
            await self._http.complete(target, body, self._emitter, self._metering)

    async def stream(
        self,
        messages: Sequence[MessageLike],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | str | None = None,
        *,
        temperature: float | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        messages = to_ollama_messages(messages)
        if self._token_budget is not None and needs_chunking(messages, self._token_budget):
            base, chunks, task = prepare_chunked_session(messages, self._token_budget)
            if chunks:
                input_t = output_t = 0
                async for event in self._http.execute_chunked_stream_events(
                    model=self._model,
                    base_messages=base,
                    content_chunks=chunks,
                    task_message=task,
                    token_budget=self._token_budget,
                ):
                    if event.kind == "usage":
                        if event.raw:
                            input_t = event.raw.get("input_tokens", 0) or 0
                            output_t = event.raw.get("output_tokens", 0) or 0
                    else:
                        yield event
                if self._metering is not None:
                    await self._http._record_tokens(self._metering, self._model, {"prompt_tokens": input_t, "completion_tokens": output_t})
                return

        effective = messages
        if self._token_budget is not None:
            effective, _ = apply_budget_to_messages(messages, self._token_budget)

        # Derive tools from catalog if none explicitly passed
        effective_tools = tools
        terminal_names: set[str] = set()
        if self._tool_catalog is not None:
            contracts = self._tool_catalog.all()
            if effective_tools is None:
                effective_tools = [contract_to_openai_tool(c) for c in contracts] or None
            terminal_names = {c.name for c in contracts if getattr(c, "terminal", False)}

        input_t = output_t = 0

        def _make_events(msgs: list[dict[str, Any]]) -> AsyncIterator[ModelStreamEvent]:
            use_native = bool(effective_tools) or _has_tool_protocol_messages(msgs)
            if use_native:
                target = f"{self._http.base_url}/api/chat"
                body: dict[str, Any] = {"model": self._model, "messages": msgs, "stream": True}
                if effective_tools:
                    body["tools"] = effective_tools
                if temperature is not None:
                    body["options"] = {"temperature": temperature}
                return self._http.stream_events_ndjson(
                    target, body, tool_choice, terminal_tools=terminal_names or None,
                )
            target = f"{self._http.base_url}/v1/chat/completions"
            body = {"model": self._model, "messages": msgs, "stream": True}
            if temperature is not None:
                body["temperature"] = temperature
            return self._http.stream_events_sse(target, body)

        # Without a catalog the caller handles tool calls — pass all events through.
        if self._tool_catalog is None:
            async for event in _make_events(list(effective)):
                if event.kind == "usage":
                    if event.raw:
                        input_t += event.raw.get("input_tokens", 0) or 0
                        output_t += event.raw.get("output_tokens", 0) or 0
                else:
                    yield event
            if self._metering is not None:
                await self._http._record_tokens(self._metering, self._model, {"prompt_tokens": input_t, "completion_tokens": output_t})
            return

        # With catalog: internal tool loop — only token/done reach the caller.
        current_messages = list(effective)
        for _round in range(10):
            text_parts: list[str] = []
            tool_calls_raw: list[dict[str, Any]] = []

            async for event in _make_events(current_messages):
                if event.kind == "token" and event.content:
                    text_parts.append(event.content)
                    yield event
                elif event.kind == "tool_call" and event.raw:
                    tool_calls_raw.append(event.raw)
                elif event.kind == "usage":
                    if event.raw:
                        input_t += event.raw.get("input_tokens", 0) or 0
                        output_t += event.raw.get("output_tokens", 0) or 0
                elif event.kind == "done":
                    break

            if not tool_calls_raw:
                break

            assistant_text = "".join(text_parts).strip()
            results = [await self._tool_catalog.execute(tc) for tc in tool_calls_raw]
            current_messages = append_ollama_native_tool_turn(current_messages, tool_calls_raw, results, assistant_text)

            # A terminal tool call ends the turn — do not start another round.
            if terminal_names and any(tc.get("name") in terminal_names for tc in tool_calls_raw):
                break

        yield ModelStreamEvent(kind="done")
        if self._metering is not None:
            await self._http._record_tokens(self._metering, self._model, {"prompt_tokens": input_t, "completion_tokens": output_t})

    def append_tool_turn(
        self,
        messages: Sequence[MessageLike],
        tool_calls_raw: list[dict[str, Any]],
        results: list[str],
        assistant_text: str = "",
    ) -> list[CanonicalMessage]:
        return append_canonical_tool_turn(messages, tool_calls_raw, results, assistant_text)
