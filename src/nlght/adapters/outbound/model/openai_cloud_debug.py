# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Sequence
from typing import TYPE_CHECKING, Any, cast

from nlght.adapters.outbound.model._tool_helpers import (
    append_openai_tool_turn,
    contract_to_openai_tool,
    terminal_tool_names,
)
from nlght.adapters.outbound.model.openai_cloud import _to_openai_messages
from nlght.core.model.messages import CanonicalMessage, MessageLike, append_canonical_tool_turn
from nlght.core.model.model_info import ModelInfo
from nlght.core.signals.signal import Signal
from nlght.ports.outbound.model_client import ModelClient, ModelStreamEvent
from nlght.ports.outbound.model_provider_backend import ModelProviderBackend
from nlght.ports.outbound.signal_emitter import SignalEmitter

if TYPE_CHECKING:
    from nlght.core.metering.context import MeteringContext
    from nlght.core.model.budget import TokenBudget
    from nlght.ports.outbound.tool_catalog import ToolCatalog

logger = logging.getLogger(__name__)

# Hardcoded context windows for common OpenAI models.
_CONTEXT_WINDOWS: dict[str, int] = {
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    "gpt-3.5-turbo": 16_385,
    "o1": 200_000,
    "o1-mini": 128_000,
    "o3": 200_000,
    "o3-mini": 200_000,
    "o4-mini": 200_000,
}
_DEFAULT_CONTEXT_WINDOW = 128_000
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


class OpenAICloudModelClient(ModelProviderBackend):
    """OpenAI cloud model provider — one instance per configured provider.

    Requires ``pip install 'nlght-ai[openai]'``.

    If ``api_key`` is empty the OpenAI SDK reads ``OPENAI_API_KEY`` from
    the environment automatically.  Set ``base_url`` to use Azure OpenAI
    or any OpenAI-compatible endpoint.
    """

    def __init__(
        self,
        *,
        api_key: str,
        default_model: str,
        base_url: str = DEFAULT_OPENAI_BASE_URL,
    ) -> None:
        try:
            from openai import AsyncOpenAI  # noqa: PLC0415 (lazy import: optional extra or deliberate startup-cost/cycle avoidance)

            kwargs: dict[str, Any] = {}
            if api_key:
                kwargs["api_key"] = api_key
            # Always pass an absolute URL. The OpenAI SDK otherwise reads an
            # explicitly empty OPENAI_BASE_URL from the environment and builds
            # relative request URLs such as /chat/completions.
            kwargs["base_url"] = base_url or DEFAULT_OPENAI_BASE_URL
            self._client = AsyncOpenAI(**kwargs)
        except ImportError as exc:
            raise ImportError(
                "openai package is required for the OpenAI cloud model provider. "
                "Install with: pip install 'nlght-ai[openai]'"
            ) from exc
        self._default_model = default_model

    async def list_models(self) -> list[ModelInfo]:
        """Fetch available models from the OpenAI /v1/models endpoint."""
        try:
            page = await self._client.models.list()
            return [ModelInfo(name=m.id) for m in page.data]
        except Exception as exc:
            logger.warning("openai_cloud.list_models.failed | %s", exc)
            return [ModelInfo(name=name) for name in _CONTEXT_WINDOWS]

    def bind(
        self,
        *,
        model: str | None,
        emitter: SignalEmitter,
        stream: bool,
        token_budget: TokenBudget | None = None,
        metering: MeteringContext | None = None,
        tool_catalog: ToolCatalog | None = None,
    ) -> BoundOpenAICloudModelClient:
        return BoundOpenAICloudModelClient(
            backend=self,
            model=model or self._default_model,
            emitter=emitter,
            stream=stream,
            token_budget=token_budget,
            metering=metering,
            tool_catalog=tool_catalog,
        )

    async def token_budget(
        self,
        model: str | None = None,
        *,
        client_max_tokens: int | None = None,
    ) -> TokenBudget:
        from nlght.core.model.budget import TokenBudget as _Budget  # noqa: PLC0415 (lazy import: optional extra or deliberate startup-cost/cycle avoidance)

        resolved = model or self._default_model
        window = _DEFAULT_CONTEXT_WINDOW
        for key, w in _CONTEXT_WINDOWS.items():
            if resolved.startswith(key):
                window = w
                break
        logger.info("token_budget.window | model=%s window=%d source=static", resolved, window)
        return _Budget(window, client_max_tokens=client_max_tokens)

    # ------------------------------------------------------------------
    # Internal transport
    # ------------------------------------------------------------------

    async def _call(
        self,
        messages: list[dict[str, Any]],
        model: str,
        emitter: SignalEmitter,
        stream: bool,
        metering: MeteringContext | None = None,
        *,
        temperature: float | None = None,
    ) -> None:
        logger.debug("openai_cloud.call | model=%s stream=%s msgs=%d", model, stream, len(messages))
        if stream:
            await self._stream(messages, model, emitter, metering=metering, temperature=temperature)
        else:
            await self._complete(messages, model, emitter, metering=metering, temperature=temperature)

    async def _complete(
        self,
        messages: list[dict[str, Any]],
        model: str,
        emitter: SignalEmitter,
        metering: MeteringContext | None = None,
        *,
        temperature: float | None = None,
    ) -> None:
        create_kwargs: dict[str, Any] = {
            "model": model,
            "messages": cast(Any, messages),
            "stream": False,
        }
        if temperature is not None:
            create_kwargs["temperature"] = temperature
        response = await self._client.chat.completions.create(**create_kwargs)
        try:
            content = response.choices[0].message.content or ""
        except (IndexError, AttributeError):
            content = ""
        await emitter.emit(Signal(role="assistant", content=content, kind="result"))
        if metering is not None:
            try:
                input_t = (response.usage.prompt_tokens if response.usage else 0) or 0
                output_t = (response.usage.completion_tokens if response.usage else 0) or 0
                await metering.port.record_tokens(
                    caller=metering.caller,
                    session_key=metering.session_key,
                    workflow=metering.workflow,
                    step=metering.step,
                    provider=metering.provider,
                    model=model,
                    input_tokens=input_t,
                    output_tokens=output_t,
                )
            except Exception as exc:
                logger.warning("metering.record_tokens.failed | %s", exc)

    async def _stream(
        self,
        messages: list[dict[str, Any]],
        model: str,
        emitter: SignalEmitter,
        metering: MeteringContext | None = None,
        *,
        temperature: float | None = None,
    ) -> None:
        input_t = output_t = 0
        async for event in self._stream_events(messages, model, temperature=temperature):
            if event.kind == "usage":
                if event.raw:
                    input_t = event.raw.get("input_tokens", 0) or 0
                    output_t = event.raw.get("output_tokens", 0) or 0
            else:
                await emitter.emit(Signal(role="assistant", content=event.content, kind=event.kind))
        if metering is not None:
            try:
                await metering.port.record_tokens(
                    caller=metering.caller,
                    session_key=metering.session_key,
                    workflow=metering.workflow,
                    step=metering.step,
                    provider=metering.provider,
                    model=model,
                    input_tokens=input_t,
                    output_tokens=output_t,
                )
            except Exception as exc:
                logger.warning("metering.record_tokens.failed | %s", exc)

    async def _stream_events(
        self,
        messages: list[dict[str, Any]],
        model: str,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | str | None = None,
        *,
        temperature: float | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        input_t = output_t = 0
        create_kwargs: dict[str, Any] = dict(
            model=model,
            messages=cast(Any, messages),
            stream=True,
            stream_options={"include_usage": True},
        )
        if tools:
            create_kwargs["tools"] = cast(Any, tools)
        if tool_choice is not None:
            create_kwargs["tool_choice"] = tool_choice
        if temperature is not None:
            create_kwargs["temperature"] = temperature
        stream = await self._client.chat.completions.create(**create_kwargs)
        tc_builder: dict[int, dict[str, Any]] = {}
        async for chunk in stream:
            logger.debug("openai.stream_chunk | model=%s chunk=%.500s", model, str(chunk))
            usage = getattr(chunk, "usage", None)
            if usage:
                input_t = usage.prompt_tokens or 0
                output_t = usage.completion_tokens or 0
            try:
                delta = chunk.choices[0].delta
                token = delta.content or ""
                if token:
                    yield ModelStreamEvent(kind="token", content=token)
                for tc_delta in (getattr(delta, "tool_calls", None) or []):
                    idx = tc_delta.index
                    if idx not in tc_builder:
                        tc_builder[idx] = {"id": "", "name": "", "args": ""}
                    if tc_delta.id:
                        tc_builder[idx]["id"] = tc_delta.id
                    fn = tc_delta.function
                    if fn:
                        if fn.name:
                            tc_builder[idx]["name"] = fn.name
                        if fn.arguments:
                            tc_builder[idx]["args"] += fn.arguments
                finish = chunk.choices[0].finish_reason
                if finish == "tool_calls":
                    for tc in tc_builder.values():
                        try:
                            inp = json.loads(tc["args"]) if tc["args"] else {}
                        except json.JSONDecodeError:
                            inp = {}
                        yield ModelStreamEvent(
                            kind="tool_call",
                            raw={"id": tc["id"], "name": tc["name"], "input": inp},
                        )
                    tc_builder.clear()
            except (IndexError, AttributeError):
                pass
        yield ModelStreamEvent(kind="usage", raw={"input_tokens": input_t, "output_tokens": output_t})
        yield ModelStreamEvent(kind="done")


# ---------------------------------------------------------------------------
# BoundOpenAICloudModelClient
# ---------------------------------------------------------------------------


class BoundOpenAICloudModelClient(ModelClient):
    """Request-scoped ModelClient for OpenAI — implements the ModelClient port."""

    def __init__(
        self,
        *,
        backend: OpenAICloudModelClient,
        model: str,
        emitter: SignalEmitter,
        stream: bool,
        token_budget: TokenBudget | None = None,
        metering: MeteringContext | None = None,
        tool_catalog: ToolCatalog | None = None,
    ) -> None:
        self._backend = backend
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
        messages = _to_openai_messages(messages)
        if self._stream or self._tool_catalog is None:
            await self._backend._call(
                messages=messages,
                model=self._model,
                emitter=self._emitter,
                stream=self._stream,
                metering=self._metering,
                temperature=temperature,
            )
            return

        # Non-streaming tool loop
        contracts = self._tool_catalog.all()
        tools_openai = [contract_to_openai_tool(c) for c in contracts] or None
        terminal_names = terminal_tool_names(self._tool_catalog)
        current = list(messages)
        input_t = output_t = 0

        for _round in range(10):
            create_kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": cast(Any, current),
                "stream": False,
            }
            if tools_openai:
                create_kwargs["tools"] = cast(Any, tools_openai)
            if temperature is not None:
                create_kwargs["temperature"] = temperature

            logger.info(
                "openai.call.request | round=%d model=%s tools=%s",
                _round, self._model,
                [t["function"]["name"] for t in tools_openai] if tools_openai else None,
            )
            response = await self._backend._client.chat.completions.create(**create_kwargs)
            if response.usage:
                input_t += response.usage.prompt_tokens or 0
                output_t += response.usage.completion_tokens or 0

            msg = response.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None) or []
            content = msg.content or ""
            logger.info(
                "openai.call.response_tool_calls | round=%d names=%s",
                _round, [tc.function.name for tc in tool_calls],
            )

            if not tool_calls:
                await self._emitter.emit(Signal(role="assistant", content=content, kind="result"))
                break

            tool_calls_raw = [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "input": json.loads(tc.function.arguments or "{}"),
                }
                for tc in tool_calls
            ]
            logger.info(
                "openai.call.tool_loop | round=%d calls=%d dispatch_names=%s",
                _round, len(tool_calls_raw), [tc["name"] for tc in tool_calls_raw],
            )
            results = [await self._tool_catalog.execute(tc) for tc in tool_calls_raw]
            current = append_openai_tool_turn(current, tool_calls_raw, results, content)
            if terminal_names and any(tc.get("name") in terminal_names for tc in tool_calls_raw):
                break

        if self._metering is not None:
            try:
                await self._metering.port.record_tokens(
                    caller=self._metering.caller,
                    session_key=self._metering.session_key,
                    workflow=self._metering.workflow,
                    step=self._metering.step,
                    provider=self._metering.provider,
                    model=self._model,
                    input_tokens=input_t,
                    output_tokens=output_t,
                )
            except Exception as exc:
                logger.warning("metering.record_tokens.failed | %s", exc)

    async def stream(
        self,
        messages: Sequence[MessageLike],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | str | None = None,
        *,
        temperature: float | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        messages = _to_openai_messages(messages)
        # Derive tools from catalog if none explicitly passed
        effective_tools = tools
        if effective_tools is None and self._tool_catalog is not None:
            contracts = self._tool_catalog.all()
            effective_tools = [contract_to_openai_tool(c) for c in contracts] or None

        input_t = output_t = 0

        # Without a catalog the caller handles tool calls — pass all events through.
        if self._tool_catalog is None:
            async for event in self._backend._stream_events(
                messages, self._model, tools=effective_tools, tool_choice=tool_choice, temperature=temperature,
            ):
                if event.kind == "usage":
                    if event.raw:
                        input_t += event.raw.get("input_tokens", 0) or 0
                        output_t += event.raw.get("output_tokens", 0) or 0
                else:
                    yield event
            if self._metering is not None:
                try:
                    await self._metering.port.record_tokens(
                        caller=self._metering.caller,
                        session_key=self._metering.session_key,
                        workflow=self._metering.workflow,
                        step=self._metering.step,
                        provider=self._metering.provider,
                        model=self._model,
                        input_tokens=input_t,
                        output_tokens=output_t,
                    )
                except Exception as exc:
                    logger.warning("metering.record_tokens.failed | %s", exc)
            return

        # With catalog: internal tool loop — only token/done reach the caller.
        current_messages = list(messages)
        terminal_names = terminal_tool_names(self._tool_catalog)
        for _round in range(10):
            text_parts: list[str] = []
            tool_calls_raw: list[dict[str, Any]] = []

            logger.info(
                "openai.stream.request | round=%d model=%s tools=%s",
                _round, self._model,
                [t["function"]["name"] for t in effective_tools] if effective_tools else None,
            )
            async for event in self._backend._stream_events(
                current_messages, self._model,
                tools=effective_tools, tool_choice=tool_choice, temperature=temperature,
            ):
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
                "openai.stream.tool_loop | round=%d calls=%d names=%s",
                _round, len(tool_calls_raw), [tc["name"] for tc in tool_calls_raw],
            )
            results = [await self._tool_catalog.execute(tc) for tc in tool_calls_raw]
            for tc, result_text in zip(tool_calls_raw, results, strict=False):
                logger.info("openai.tool_call | name=%s chars=%d", tc["name"], len(result_text))
            current_messages = append_openai_tool_turn(current_messages, tool_calls_raw, results, assistant_text)
            if terminal_names and any(tc.get("name") in terminal_names for tc in tool_calls_raw):
                break

        yield ModelStreamEvent(kind="done")

        if self._metering is not None:
            try:
                await self._metering.port.record_tokens(
                    caller=self._metering.caller,
                    session_key=self._metering.session_key,
                    workflow=self._metering.workflow,
                    step=self._metering.step,
                    provider=self._metering.provider,
                    model=self._model,
                    input_tokens=input_t,
                    output_tokens=output_t,
                )
            except Exception as exc:
                logger.warning("metering.record_tokens.failed | %s", exc)

    def append_tool_turn(
        self,
        messages: Sequence[MessageLike],
        tool_calls_raw: list[dict[str, Any]],
        results: list[str],
        assistant_text: str = "",
    ) -> list[CanonicalMessage]:
        return append_canonical_tool_turn(messages, tool_calls_raw, results, assistant_text)
