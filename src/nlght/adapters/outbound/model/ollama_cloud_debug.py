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
from nlght.adapters.outbound.model._ollama_http_debug import _OllamaHttp
from nlght.adapters.outbound.model._tool_helpers import (
    append_openai_tool_turn,
    contract_to_openai_tool,
    terminal_tool_names,
)
from nlght.adapters.outbound.model.openai_cloud import (
    _to_openai_messages as _to_ollama_cloud_messages,
)
from nlght.core.model.messages import CanonicalMessage, MessageLike, append_canonical_tool_turn
from nlght.core.model.model_info import ModelInfo, RunningModelInfo
from nlght.core.signals.signal import Signal
from nlght.ports.outbound.model_client import ModelClient, ModelStreamEvent
from nlght.ports.outbound.model_provider_backend import ModelProviderBackend
from nlght.ports.outbound.signal_emitter import SignalEmitter

if TYPE_CHECKING:
    from nlght.core.metering.context import MeteringContext
    from nlght.core.model.budget import TokenBudget
    from nlght.ports.outbound.tool_catalog import ToolCatalog

logger = logging.getLogger(__name__)

# Hardcoded context windows for known Ollama Cloud models. Cloud inference uses
# the OpenAI-compatible /v1 endpoint and does NOT expose the native /api/show
# metadata used by the local daemon, so the window is resolved from this
# provider-specific table (prefix-matched) instead. Last resort: the default.
_CONTEXT_WINDOWS: dict[str, int] = {
    "qwen3-coder:480b": 262_144,
    "qwen3-coder": 262_144,
    "qwen3": 40_960,
    "deepseek-v3.1": 163_840,
    "deepseek-v3": 163_840,
    "gpt-oss:120b": 131_072,
    "gpt-oss:20b": 131_072,
    "gpt-oss": 131_072,
    "kimi-k2": 131_072,
    "glm-4.6": 200_000,
    "glm-4": 131_072,
}
_DEFAULT_CONTEXT_WINDOW = 131_072

# Ollama Cloud's public endpoint. Used when no base_url is configured, so a
# cloud provider needs only an API key — override for a self-hosted remote.
DEFAULT_CLOUD_BASE_URL = "https://ollama.com"


class OllamaCloudClient(ModelProviderBackend):
    """Cloud/remote Ollama provider.

    Inference uses the OpenAI-compatible ``/v1/chat/completions`` wire format;
    model discovery still uses Ollama's ``/api/tags`` and ``/api/ps`` endpoints.
    Defaults to ``https://ollama.com``; pass ``base_url`` for a remote endpoint.
    """

    def __init__(
        self,
        *,
        http_client: httpx2.AsyncClient,
        base_url: str = DEFAULT_CLOUD_BASE_URL,
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
        from nlght.core.model.budget import TokenBudget as _Budget  # noqa: PLC0415 (lazy import: optional extra or deliberate startup-cost/cycle avoidance)

        resolved = model or self._default_model
        # Prefix-match so versioned/suffixed names (…:480b, …-cloud) resolve.
        window = _DEFAULT_CONTEXT_WINDOW
        for key, w in _CONTEXT_WINDOWS.items():
            if resolved.startswith(key):
                window = w
                break
        logger.info("token_budget.window | model=%s window=%d source=static", resolved, window)
        return _Budget(window, client_max_tokens=client_max_tokens)

    def bind(
        self,
        *,
        model: str | None,
        emitter: SignalEmitter,
        stream: bool,
        token_budget: TokenBudget | None = None,
        metering: MeteringContext | None = None,
        tool_catalog: ToolCatalog | None = None,
    ) -> BoundOllamaCloudClient:
        return BoundOllamaCloudClient(
            backend=self,
            http=self._http,
            model=model or self._default_model,
            emitter=emitter,
            stream=stream,
            token_budget=token_budget,
            metering=metering,
            tool_catalog=tool_catalog,
        )


class BoundOllamaCloudClient(ModelClient):
    """Request-scoped client for OllamaCloudClient."""

    def __init__(
        self,
        *,
        backend: OllamaCloudClient,
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
        messages = _to_ollama_cloud_messages(messages)
        if self._token_budget is not None and needs_chunking(messages, self._token_budget):
            base, chunks, task = prepare_chunked_session(messages, self._token_budget)
            if chunks:
                logger.info("llm.auto_chunk | model=%s chunks=%d stream=%s", self._model, len(chunks), self._stream)
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
        logger.debug("llm.call | model=%s stream=%s msgs=%d", self._model, self._stream, len(effective))
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
                # Evidence log: exactly which tool definitions this request carries.
                _req_tools = body.get("tools")
                logger.info(
                    "ollama_cloud.call.request | round=%d model=%s tools_in_request=%s",
                    _round, self._model,
                    [(t.get("function") or {}).get("name") for t in _req_tools] if _req_tools else None,
                )
                msg, usage = await self._http.complete_message(target, body)
                input_t += usage.get("prompt_tokens", 0) or 0
                output_t += usage.get("completion_tokens", 0) or 0
                tool_calls_native = msg.get("tool_calls") or []
                content = msg.get("content") or ""
                # Evidence log: the tool-call names the provider actually returned,
                # verbatim, before any of our processing — proves whether a prefix
                # like "tool." originates upstream or in our code.
                logger.info(
                    "ollama_cloud.call.response_tool_calls | round=%d names=%s",
                    _round, [(tc.get("function") or {}).get("name") for tc in tool_calls_native],
                )
                logger.debug("ollama_cloud.call.response_tool_calls_raw | round=%d raw=%s", _round, tool_calls_native)
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
                logger.info(
                    "ollama_cloud.call.tool_loop | round=%d calls=%d dispatch_names=%s",
                    _round, len(tool_calls_raw), [tc["name"] for tc in tool_calls_raw],
                )
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
        messages = _to_ollama_cloud_messages(messages)
        if self._token_budget is not None and needs_chunking(messages, self._token_budget):
            base, chunks, task = prepare_chunked_session(messages, self._token_budget)
            if chunks:
                logger.info("llm.auto_chunk_stream | model=%s chunks=%d", self._model, len(chunks))
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
        if effective_tools is None and self._tool_catalog is not None:
            contracts = self._tool_catalog.all()
            effective_tools = [contract_to_openai_tool(c) for c in contracts] or None

        target = f"{self._http.base_url}/v1/chat/completions"
        input_t = output_t = 0

        def _make_body(msgs: list[dict[str, Any]]) -> dict[str, Any]:
            b: dict[str, Any] = {"model": self._model, "messages": msgs, "stream": True}
            if effective_tools:
                b["tools"] = effective_tools
            if tool_choice is not None:
                b["tool_choice"] = tool_choice
            if temperature is not None:
                b["temperature"] = temperature
            return b

        # Without a catalog the caller handles tool calls — pass all events through.
        if self._tool_catalog is None:
            async for event in self._http.stream_events_sse(target, _make_body(list(effective))):
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
        terminal_names = terminal_tool_names(self._tool_catalog)
        for _round in range(10):
            text_parts: list[str] = []
            tool_calls_raw: list[dict[str, Any]] = []

            logger.info(
                "ollama_cloud.stream.request | round=%d model=%s tools=%s",
                _round, self._model,
                [(t.get("function") or {}).get("name") for t in effective_tools] if effective_tools else None,
            )
            async for event in self._http.stream_events_sse(target, _make_body(current_messages)):
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
            logger.info(
                "ollama_cloud.stream.tool_loop | round=%d calls=%d names=%s",
                _round, len(tool_calls_raw), [tc.get("name") for tc in tool_calls_raw],
            )
            results = [await self._tool_catalog.execute(tc) for tc in tool_calls_raw]
            for tc, result_text in zip(tool_calls_raw, results, strict=False):
                logger.info("ollama_cloud.tool_call | name=%s chars=%d", tc["name"], len(result_text))
            current_messages = append_openai_tool_turn(current_messages, tool_calls_raw, results, assistant_text)
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
