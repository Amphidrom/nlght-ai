# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import base64
import logging
from collections.abc import AsyncIterator
from types import ModuleType
from typing import TYPE_CHECKING, Any, cast

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

# Hardcoded context windows for common Google Gemini models — used only as a
# token-budget fallback (list_models() itself queries the live API; see
# below). Newest first.
_CONTEXT_WINDOWS: dict[str, int] = {
    "gemini-3.5-flash": 1_048_576,
    "gemini-2.5-pro": 1_048_576,
    "gemini-2.5-flash": 1_048_576,
    "gemini-2.0-flash": 1_048_576,
    "gemini-2.0-flash-lite": 1_048_576,
    "gemini-1.5-pro": 2_097_152,
    "gemini-1.5-flash": 1_048_576,
    "gemini-1.0-pro": 32_768,
}
_DEFAULT_CONTEXT_WINDOW = 1_048_576


def _to_google_contents(
    messages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Convert OpenAI-format messages to Google Generative AI format.

    Returns ``(system_instruction, contents)`` where:
    - ``system_instruction`` is a concatenation of all system messages (empty string if none).
    - ``contents`` uses ``role: "user" | "model"`` with ``parts`` lists.

    Consecutive same-role messages are merged to satisfy the Gemini API's
    requirement that turns must alternate between user and model.

    Also handles canonical (Ollama-style) tool-call turns — an ``assistant``
    message with ``tool_calls`` followed by ``role: "tool"`` result messages
    — lowering them to Gemini's ``function_call``/``function_response`` parts.
    This is what a step gets back from ``ModelClient.append_tool_turn()`` and
    feeds into the next ``call()``/``stream()``.
    """
    system_parts = [m.get("content", "") for m in messages if m.get("role") == "system"]
    system_text = "\n\n".join(filter(None, system_parts))

    raw_contents: list[dict[str, Any]] = []
    i = 0
    while i < len(messages):
        m = messages[i]
        role = m.get("role", "user")

        if role == "system":
            i += 1
            continue

        if role == "assistant" and m.get("tool_calls"):
            parts: list[dict[str, Any]] = []
            text = m.get("content") or ""
            if text:
                parts.append({"text": text})
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                part: dict[str, Any] = {
                    "function_call": {
                        "name": fn.get("name", ""),
                        "args": fn.get("arguments", {}) or {},
                    },
                }
                # Gemini 3.x requires the model's thought_signature to be echoed
                # back on the function_call part; it rides through the canonical
                # tool-turn as base64 (see append_ollama_native_tool_turn) and is
                # decoded back to the bytes the SDK originally handed us.
                ts_b64 = tc.get("thought_signature")
                if ts_b64:
                    part["thought_signature"] = base64.b64decode(ts_b64)
                parts.append(part)
            raw_contents.append({"role": "model", "parts": parts})
            i += 1

            # Consume consecutive tool result messages, paired positionally
            # with the tool_calls above (canonical tool messages carry no name).
            names = [tc.get("function", {}).get("name", "") for tc in m["tool_calls"]]
            response_parts: list[dict[str, Any]] = []
            j = 0
            while i < len(messages) and messages[i].get("role") == "tool":
                name = names[j] if j < len(names) else ""
                response_parts.append({
                    "function_response": {
                        "name": name,
                        "response": {"output": messages[i].get("content", "")},
                    },
                })
                i += 1
                j += 1
            if response_parts:
                raw_contents.append({"role": "user", "parts": response_parts})
            continue

        google_role = "model" if role == "assistant" else "user"
        raw_contents.append({"role": google_role, "parts": [{"text": m.get("content", "")}]})
        i += 1

    # Merge consecutive same-role turns (Gemini requires alternating turns).
    merged: list[dict[str, Any]] = []
    for turn in raw_contents:
        if merged and merged[-1]["role"] == turn["role"]:
            merged[-1]["parts"].extend(turn["parts"])
        else:
            merged.append({"role": turn["role"], "parts": list(turn["parts"])})

    return system_text, merged


def _json_schema_to_google(schema: dict[str, Any], _types: ModuleType) -> object:
    """Recursively convert a JSON-Schema object to a google.genai ``Schema``.

    Carries ``enum``, ``items`` (array element schema) and nested ``properties``
    through, so a tool's full JSON Schema survives the trip to Gemini instead of
    being flattened to type+description.
    """
    kwargs: dict[str, Any] = {"type": str(schema.get("type") or "string").upper()}
    if schema.get("description"):
        kwargs["description"] = schema["description"]
    if schema.get("enum"):
        kwargs["enum"] = [str(e) for e in schema["enum"]]
    items = schema.get("items")
    if isinstance(items, dict):
        kwargs["items"] = _json_schema_to_google(items, _types)
    props = schema.get("properties")
    if isinstance(props, dict):
        kwargs["properties"] = {
            name: _json_schema_to_google(sub, _types) for name, sub in props.items()
        }
        if schema.get("required"):
            kwargs["required"] = list(schema["required"])
    return _types.Schema(**kwargs)


def _to_google_tools(openai_tools: list[dict[str, Any]]) -> object | None:
    """Convert OpenAI-format tool definitions to a Google ``Tool`` object."""
    try:
        from google.genai import types as _types  # noqa: PLC0415 (lazy import: optional extra or deliberate startup-cost/cycle avoidance)
    except ImportError:
        return None
    declarations = []
    for t in openai_tools:
        fn = t.get("function", {})
        params = fn.get("parameters") or {"type": "object", "properties": {}}
        declarations.append(_types.FunctionDeclaration(
            name=fn["name"],
            description=fn.get("description", ""),
            parameters=_json_schema_to_google(params, _types),
        ))
    tool: object = _types.Tool(function_declarations=declarations)
    return tool


class GoogleModelClient(ModelProviderBackend):
    """Google Gemini cloud model provider — one instance per configured provider.

    Requires ``pip install 'nlght-ai[google]'``.

    If ``api_key`` is empty the SDK reads ``GOOGLE_API_KEY`` or
    ``GEMINI_API_KEY`` from the environment automatically.
    """

    def __init__(self, *, api_key: str, default_model: str) -> None:
        try:
            from google import genai as _genai  # noqa: PLC0415 (lazy import: optional extra or deliberate startup-cost/cycle avoidance)

            self._genai = _genai
            self._client = _genai.Client(api_key=api_key or None)
        except ImportError as exc:
            raise ImportError(
                "google-genai package is required for the Google model provider. "
                "Install with: pip install 'nlght-ai[google]'"
            ) from exc
        self._default_model = default_model

    async def list_models(self) -> list[ModelInfo]:
        """Fetch available models from the Google Generative AI API.

        Falls back to the static table on any error (network, auth, an SDK
        surface change) so a live-API hiccup doesn't take down model
        discovery — mirrors the OpenAI cloud client's list_models().
        """
        try:
            pager = await self._client.aio.models.list()
            names = [(m.name or "").rsplit("/", 1)[-1] async for m in pager]
            models = [ModelInfo(name=name) for name in names if name]
            return models or [ModelInfo(name=name) for name in _CONTEXT_WINDOWS]
        except Exception as exc:
            logger.warning("google.list_models.failed | %s", exc)
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
    ) -> BoundGoogleModelClient:
        return BoundGoogleModelClient(
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
        system_text, contents = _to_google_contents(messages)
        logger.debug(
            "google.call | model=%s stream=%s turns=%d",
            model, stream, len(contents),
        )
        if stream:
            await self._stream(contents, model, system_text, emitter, metering=metering, temperature=temperature)
        else:
            await self._complete(contents, model, system_text, emitter, metering=metering, temperature=temperature)

    async def _complete(
        self,
        contents: list[dict[str, Any]],
        model: str,
        system_text: str,
        emitter: SignalEmitter,
        metering: MeteringContext | None = None,
        *,
        temperature: float | None = None,
    ) -> None:
        from google.genai import types as _types  # noqa: PLC0415 (lazy import: optional extra or deliberate startup-cost/cycle avoidance)

        config = _types.GenerateContentConfig(
            system_instruction=system_text or None,
            temperature=temperature,
        )
        response = await self._client.aio.models.generate_content(
            model=model,
            contents=cast(Any, contents),
            config=config,
        )
        try:
            content = response.text or ""
        except (AttributeError, ValueError):
            content = ""
        await emitter.emit(Signal(role="assistant", content=content, kind="result"))
        if metering is not None:
            try:
                meta = getattr(response, "usage_metadata", None)
                input_t = (meta.prompt_token_count if meta else 0) or 0
                output_t = (meta.candidates_token_count if meta else 0) or 0
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
        contents: list[dict[str, Any]],
        model: str,
        system_text: str,
        emitter: SignalEmitter,
        metering: MeteringContext | None = None,
        *,
        temperature: float | None = None,
    ) -> None:
        input_t = output_t = 0
        async for event in self._stream_events(contents, model, system_text, temperature=temperature):
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
        contents: list[dict[str, Any]],
        model: str,
        system_text: str,
        *,
        temperature: float | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        from google.genai import types as _types  # noqa: PLC0415 (lazy import: optional extra or deliberate startup-cost/cycle avoidance)

        config = _types.GenerateContentConfig(
            system_instruction=system_text or None,
            temperature=temperature,
        )
        last_usage = None
        async for chunk in await self._client.aio.models.generate_content_stream(
            model=model,
            contents=cast(Any, contents),
            config=config,
        ):
            logger.debug("google.stream_chunk | model=%s chunk=%.500s", model, str(chunk))
            meta = getattr(chunk, "usage_metadata", None)
            if meta:
                last_usage = meta
            try:
                token = chunk.text or ""
            except (AttributeError, ValueError):
                token = ""
            if token:
                yield ModelStreamEvent(kind="token", content=token)
        input_t = (last_usage.prompt_token_count if last_usage else 0) or 0
        output_t = (last_usage.candidates_token_count if last_usage else 0) or 0
        yield ModelStreamEvent(kind="usage", raw={"input_tokens": input_t, "output_tokens": output_t})
        yield ModelStreamEvent(kind="done")


# ---------------------------------------------------------------------------
# BoundGoogleModelClient
# ---------------------------------------------------------------------------


class BoundGoogleModelClient(ModelClient):
    """Request-scoped ModelClient for Google Gemini — implements the ModelClient port."""

    def __init__(
        self,
        *,
        backend: GoogleModelClient,
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

    async def call(self, messages: list[dict[str, Any]], *, temperature: float | None = None) -> None:
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

        # Non-streaming tool loop (Google Gemini)
        from google.genai import types as _types  # noqa: PLC0415 (lazy import: optional extra or deliberate startup-cost/cycle avoidance)

        from nlght.adapters.outbound.model._tool_helpers import (  # noqa: PLC0415 (lazy import: optional extra or deliberate startup-cost/cycle avoidance)
            contract_to_openai_tool,
            terminal_tool_names,
        )
        from nlght.core.signals.signal import Signal  # noqa: PLC0415 (lazy import: optional extra or deliberate startup-cost/cycle avoidance)

        contracts = self._tool_catalog.all()
        effective_openai_tools = [contract_to_openai_tool(c) for c in contracts] or None
        google_tool = _to_google_tools(effective_openai_tools) if effective_openai_tools else None
        terminal_names = terminal_tool_names(self._tool_catalog)

        system_text, current_contents = _to_google_contents(messages)
        input_t = output_t = 0

        for _round in range(10):
            config = _types.GenerateContentConfig(
                system_instruction=system_text or None,
                tools=[google_tool] if google_tool else None,
                temperature=temperature,
            )
            logger.info(
                "google.call.request | round=%d model=%s tools=%s",
                _round, self._model,
                [t["function"]["name"] for t in effective_openai_tools] if effective_openai_tools else None,
            )
            response = await self._backend._client.aio.models.generate_content(
                model=self._model,
                contents=cast(Any, current_contents),
                config=config,
            )
            meta = getattr(response, "usage_metadata", None)
            if meta:
                input_t += meta.prompt_token_count or 0
                output_t += meta.candidates_token_count or 0

            function_calls: list[Any] = []
            text_parts: list[str] = []
            try:
                for part in (response.candidates[0].content.parts or []):
                    fc = getattr(part, "function_call", None)
                    if fc is not None:
                        function_calls.append(fc)
                    elif getattr(part, "text", None):
                        text_parts.append(part.text)
            except (IndexError, AttributeError):
                pass
            logger.info(
                "google.call.response_tool_calls | round=%d names=%s",
                _round, [fc.name for fc in function_calls],
            )

            if not function_calls:
                try:
                    content = response.text or ""
                except (AttributeError, ValueError):
                    content = "".join(text_parts)
                await self._emitter.emit(Signal(role="assistant", content=content, kind="result"))
                break

            # Re-append the model's own turn verbatim rather than rebuilding the
            # Part — a rebuilt function_call drops its thought_signature, and
            # Gemini 3.x then rejects the continuation (400 INVALID_ARGUMENT,
            # "Function call is missing a thought_signature").
            current_contents = list(current_contents) + [response.candidates[0].content]

            logger.info(
                "google.call.tool_loop | round=%d calls=%d dispatch_names=%s",
                _round, len(function_calls), [fc.name for fc in function_calls],
            )
            response_parts: list[Any] = []
            for fc in function_calls:
                tc = {"name": fc.name, "input": dict(fc.args or {})}
                result_text = await self._tool_catalog.execute(tc)
                logger.info("google.call.tool_call | name=%s chars=%d", fc.name, len(result_text))
                response_parts.append(
                    _types.Part.from_function_response(name=fc.name, response={"output": result_text})
                )
            current_contents = current_contents + [_types.Content(role="user", parts=response_parts)]

            if terminal_names and any(fc.name in terminal_names for fc in function_calls):
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
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | str | None = None,
        *,
        temperature: float | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        from google.genai import types as _types  # noqa: PLC0415 (lazy import: optional extra or deliberate startup-cost/cycle avoidance)

        system_text, initial_contents = _to_google_contents(messages)

        # Build Google tool object from explicit tools or catalog
        effective_openai_tools: list[dict[str, Any]] | None = tools
        if effective_openai_tools is None and self._tool_catalog is not None:
            from nlght.adapters.outbound.model._tool_helpers import (  # noqa: PLC0415 (lazy import: optional extra or deliberate startup-cost/cycle avoidance)
                contract_to_openai_tool,  # noqa: PLC0415 (lazy import: optional extra or deliberate startup-cost/cycle avoidance)
            )
            contracts = self._tool_catalog.all()
            effective_openai_tools = [contract_to_openai_tool(c) for c in contracts] or None
        google_tool = _to_google_tools(effective_openai_tools) if effective_openai_tools else None

        input_t = output_t = 0

        # Without a catalog the caller handles tool calls — pass all events through.
        if self._tool_catalog is None:
            config = _types.GenerateContentConfig(
                system_instruction=system_text or None,
                tools=[google_tool] if google_tool else None,
                temperature=temperature,
            )
            last_usage = None
            async for chunk in await self._backend._client.aio.models.generate_content_stream(
                model=self._model,
                contents=cast(Any, initial_contents),
                config=config,
            ):
                logger.debug("google.stream_chunk | model=%s chunk=%.500s", self._model, str(chunk))
                meta = getattr(chunk, "usage_metadata", None)
                if meta:
                    last_usage = meta
                try:
                    token = chunk.text or ""
                except (AttributeError, ValueError):
                    token = ""
                if token:
                    yield ModelStreamEvent(kind="token", content=token)
                try:
                    for part in (chunk.candidates[0].content.parts or []):
                        fc = getattr(part, "function_call", None)
                        if fc is not None:
                            raw: dict[str, Any] = {"id": f"call_{fc.name}", "name": fc.name, "input": dict(fc.args or {})}
                            # Carry Gemini 3.x's thought_signature (bytes) through
                            # the step-driven loop as base64 so append_tool_turn can
                            # echo it back on the continuation (else 400).
                            ts = getattr(part, "thought_signature", None)
                            if ts:
                                raw["thought_signature"] = base64.b64encode(ts).decode("ascii")
                            yield ModelStreamEvent(kind="tool_call", raw=raw)
                except (IndexError, AttributeError):
                    pass
            if last_usage:
                input_t = last_usage.prompt_token_count or 0
                output_t = last_usage.candidates_token_count or 0
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
            return

        # With catalog: internal tool loop — only token/done reach the caller.
        from nlght.adapters.outbound.model._tool_helpers import (  # noqa: PLC0415 (lazy import: optional extra or deliberate startup-cost/cycle avoidance)
            terminal_tool_names,  # noqa: PLC0415 (lazy import: optional extra or deliberate startup-cost/cycle avoidance)
        )
        terminal_names = terminal_tool_names(self._tool_catalog)
        current_contents = initial_contents
        for _round in range(10):
            text_parts: list[str] = []
            function_calls: list[Any] = []  # raw FunctionCall objects
            fc_parts: list[Any] = []  # original Parts (carry thought_signature)

            config = _types.GenerateContentConfig(
                system_instruction=system_text or None,
                tools=[google_tool] if google_tool else None,
                temperature=temperature,
            )
            last_usage = None
            logger.info(
                "google.stream.request | round=%d model=%s tools=%s",
                _round, self._model,
                [t["function"]["name"] for t in effective_openai_tools] if effective_openai_tools else None,
            )
            async for chunk in await self._backend._client.aio.models.generate_content_stream(
                model=self._model,
                contents=cast(Any, current_contents),
                config=config,
            ):
                logger.debug("google.stream_chunk | model=%s chunk=%.500s", self._model, str(chunk))
                meta = getattr(chunk, "usage_metadata", None)
                if meta:
                    last_usage = meta
                try:
                    token = chunk.text or ""
                except (AttributeError, ValueError):
                    token = ""
                if token:
                    text_parts.append(token)
                    yield ModelStreamEvent(kind="token", content=token)
                # Collect function_call parts from the final chunk
                try:
                    for part in (chunk.candidates[0].content.parts or []):
                        fc = getattr(part, "function_call", None)
                        if fc is not None:
                            function_calls.append(fc)
                            fc_parts.append(part)
                except (IndexError, AttributeError):
                    pass

            logger.info(
                "google.stream.response_tool_calls | round=%d names=%s",
                _round, [fc.name for fc in function_calls],
            )
            if last_usage:
                input_t += (last_usage.prompt_token_count or 0)
                output_t += (last_usage.candidates_token_count or 0)

            if not function_calls or self._tool_catalog is None:
                break

            # Append the model turn so Gemini keeps context. Re-use the original
            # function_call Parts rather than rebuilding them — a rebuilt Part
            # drops its thought_signature, which Gemini 3.x then rejects on the
            # continuation (400 "Function call is missing a thought_signature").
            model_parts: list[Any] = []
            if text_parts:
                model_parts.append(_types.Part.from_text("".join(text_parts)))
            model_parts.extend(fc_parts)
            current_contents = list(current_contents) + [
                _types.Content(role="model", parts=model_parts)
            ]

            # Execute tools and build function_response turn
            logger.info(
                "google.stream.tool_loop | round=%d calls=%d dispatch_names=%s",
                _round, len(function_calls), [fc.name for fc in function_calls],
            )
            response_parts: list[Any] = []
            for fc in function_calls:
                tc = {"name": fc.name, "input": dict(fc.args or {})}
                result_text = await self._tool_catalog.execute(tc)
                logger.info("google.tool_call | name=%s chars=%d", fc.name, len(result_text))
                response_parts.append(
                    _types.Part.from_function_response(name=fc.name, response={"output": result_text})
                )
            current_contents = current_contents + [
                _types.Content(role="user", parts=response_parts)
            ]

            if terminal_names and any(fc.name in terminal_names for fc in function_calls):
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
        messages: list[dict[str, Any]],
        tool_calls_raw: list[dict[str, Any]],
        results: list[str],
        assistant_text: str = "",
    ) -> list[dict[str, Any]]:
        """Append in canonical (Ollama-style) shape — ``_to_google_contents``
        lowers it to Gemini's function_call/function_response parts on the next call.
        """
        from nlght.adapters.outbound.model._tool_helpers import (  # noqa: PLC0415 (lazy import: optional extra or deliberate startup-cost/cycle avoidance)
            append_ollama_native_tool_turn,  # noqa: PLC0415 (lazy import: optional extra or deliberate startup-cost/cycle avoidance)
        )

        return append_ollama_native_tool_turn(messages, tool_calls_raw, results, assistant_text)
