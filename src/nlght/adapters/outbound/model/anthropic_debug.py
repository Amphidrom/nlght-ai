# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Sequence
from typing import TYPE_CHECKING, Any

from nlght.adapters.outbound.model._tool_helpers import (
    contract_to_openai_tool,
    terminal_tool_names,
)
from nlght.adapters.outbound.model.anthropic import _to_anthropic_wire_messages
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

_DEFAULT_MAX_TOKENS = 4096

# Hardcoded context windows — Anthropic does not expose a /api/show equivalent.
_CONTEXT_WINDOWS: dict[str, int] = {
    "claude-opus-4-6": 200_000,
    "claude-sonnet-4-6": 200_000,
    "claude-haiku-4-5": 200_000,
    "claude-3-5-sonnet-20241022": 200_000,
    "claude-3-5-haiku-20241022": 200_000,
    "claude-3-opus-20240229": 200_000,
    "claude-3-haiku-20240307": 200_000,
    "claude-3-sonnet-20240229": 200_000,
}
_DEFAULT_CONTEXT_WINDOW = 200_000


def _split_messages(
    messages: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Extract system messages into a single string; return remaining messages.

    Anthropic requires ``system`` as a top-level string, not inside ``messages``.
    """
    system_parts = [m.get("content", "") for m in messages if m.get("role") == "system"]
    others = [m for m in messages if m.get("role") != "system"]
    return "\n\n".join(filter(None, system_parts)), others



def _to_anthropic_tool(t: dict[str, Any]) -> dict[str, Any]:
    """Convert canonical OpenAI-format tool definition to Anthropic format."""
    fn = t.get("function", {})
    return {
        "name": fn["name"],
        "description": fn.get("description", ""),
        "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
    }


def _to_anthropic_tool_choice(tool_choice: dict[str, Any] | str) -> dict[str, Any]:
    """Convert canonical tool_choice to Anthropic format."""
    if isinstance(tool_choice, str):
        mapping = {"auto": "auto", "none": "none", "required": "any"}
        return {"type": mapping.get(tool_choice, "auto")}
    if isinstance(tool_choice, dict) and tool_choice.get("type") == "function":
        fn = tool_choice.get("function", {})
        return {"type": "tool", "name": fn.get("name", "")}
    return {"type": "auto"}


def _convert_messages_for_anthropic(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert canonical (Ollama-style) tool messages to Anthropic block format.

    Pairs assistant tool_calls with the following tool result messages,
    generating tool_use IDs positionally since Ollama /api/chat returns no IDs.
    """
    result: list[dict[str, Any]] = []
    _id_counter = 0

    def gen_id() -> str:
        nonlocal _id_counter
        _id_counter += 1
        return f"toolu_{_id_counter:04d}"

    i = 0
    while i < len(messages):
        msg = messages[i]
        role = msg.get("role", "")

        if role == "system":
            i += 1
            continue

        if role == "assistant" and msg.get("tool_calls"):
            content_list: list[dict[str, Any]] = []
            text = msg.get("content") or ""
            if text:
                content_list.append({"type": "text", "text": text})
            tool_ids: list[str] = []
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                tid = gen_id()
                tool_ids.append(tid)
                content_list.append({
                    "type":  "tool_use",
                    "id":    tid,
                    "name":  fn.get("name", ""),
                    "input": fn.get("arguments", {}),
                })
            result.append({"role": "assistant", "content": content_list})
            i += 1
            # Consume consecutive tool result messages
            tool_results: list[dict[str, Any]] = []
            while i < len(messages) and messages[i].get("role") == "tool":
                tool_results.append(messages[i])
                i += 1
            if tool_results:
                blocks: list[dict[str, Any]] = []
                for j, tr in enumerate(tool_results):
                    tid = tool_ids[j] if j < len(tool_ids) else gen_id()
                    blocks.append({
                        "type":        "tool_result",
                        "tool_use_id": tid,
                        "content":     tr.get("content", ""),
                    })
                result.append({"role": "user", "content": blocks})
        else:
            result.append(msg)
            i += 1

    return result


class AnthropicModelClient(ModelProviderBackend):
    """Anthropic cloud model provider — one instance per configured provider.

    Requires ``pip install 'nlght-ai[anthropic]'``.

    If ``api_key`` is empty the Anthropic SDK reads ``ANTHROPIC_API_KEY``
    from the environment automatically.
    """

    def __init__(self, *, api_key: str, default_model: str) -> None:
        try:
            import anthropic as _anthropic  # noqa: PLC0415 (lazy import: optional extra or deliberate startup-cost/cycle avoidance)

            self._client = _anthropic.AsyncAnthropic(api_key=api_key or None)
        except ImportError as exc:
            raise ImportError(
                "anthropic package is required for the Anthropic model provider. "
                "Install with: pip install 'nlght-ai[anthropic]'"
            ) from exc
        self._default_model = default_model

    async def list_models(self) -> list[ModelInfo]:
        """Return the statically known Anthropic Claude models."""
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
    ) -> BoundAnthropicModelClient:
        return BoundAnthropicModelClient(
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
        # Prefix-match for versioned model names like claude-sonnet-4-6-20250101
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
        system, chat_messages = _split_messages(messages)
        logger.debug(
            "anthropic.call | model=%s stream=%s msgs=%d",
            model, stream, len(chat_messages),
        )
        if stream:
            await self._stream(chat_messages, model, system, emitter, metering=metering, temperature=temperature)
        else:
            await self._complete(chat_messages, model, system, emitter, metering=metering, temperature=temperature)

    async def _complete(
        self,
        messages: list[dict[str, Any]],
        model: str,
        system: str,
        emitter: SignalEmitter,
        metering: MeteringContext | None = None,
        *,
        temperature: float | None = None,
    ) -> None:
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": _DEFAULT_MAX_TOKENS,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if temperature is not None:
            kwargs["temperature"] = temperature
        response = await self._client.messages.create(**kwargs)
        try:
            content = response.content[0].text or ""
        except (IndexError, AttributeError):
            content = ""
        await emitter.emit(Signal(role="assistant", content=content, kind="result"))
        if metering is not None:
            try:
                input_t = (response.usage.input_tokens if response.usage else 0) or 0
                output_t = (response.usage.output_tokens if response.usage else 0) or 0
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
        system: str,
        emitter: SignalEmitter,
        metering: MeteringContext | None = None,
        *,
        temperature: float | None = None,
    ) -> None:
        input_t = output_t = 0
        async for event in self._stream_events(messages, model, system, temperature=temperature):
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
        system: str,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: dict[str, Any] | None = None,
        *,
        temperature: float | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": _DEFAULT_MAX_TOKENS,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        if temperature is not None:
            kwargs["temperature"] = temperature
        # ── Log input ─────────────────────────────────────────────────────────
        _last_msg = messages[-1] if messages else {}
        logger.info(
            "anthropic.stream_input | model=%s messages=%d last_role=%s last_len=%d",
            model,
            len(messages),
            _last_msg.get("role", "?"),
            len(str(_last_msg.get("content") or "")),
        )
        if logger.isEnabledFor(logging.DEBUG):
            for _i, _m in enumerate(messages):
                logger.debug(
                    "anthropic.stream_input.msg[%d] | role=%s content=%s",
                    _i, _m.get("role", "?"), str(_m.get("content") or "")[:2000],
                )
        logger.info(
            "anthropic.stream_start | model=%s tools=%s",
            model, [t.get("name") for t in tools] if tools else None,
        )

        input_t = output_t = 0
        _content_parts: list[str] = []
        _tool_calls_seen: list[str] = []
        _first_token_logged = False
        async with self._client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                logger.debug("anthropic.stream_token | model=%s text=%.200s", model, text)
                if text:
                    if not _first_token_logged:
                        logger.info("anthropic.stream_first_token | model=%s", model)
                        _first_token_logged = True
                    _content_parts.append(text)
                    yield ModelStreamEvent(kind="token", content=text)
            try:
                # get_final_message() is async on AsyncMessageStream — it MUST be
                # awaited. Without the await, `final` is a coroutine, `final.usage`
                # raises AttributeError, the except swallows it, and tool_use blocks
                # are never surfaced as tool_call events (streaming tool loop breaks).
                final = await stream.get_final_message()
                logger.debug("anthropic.stream_final | model=%s msg=%.500s", model, str(final))
                if final:
                    if final.usage:
                        input_t = final.usage.input_tokens or 0
                        output_t = final.usage.output_tokens or 0
                    for block in (final.content or []):
                        if getattr(block, "type", None) == "tool_use":
                            _tool_calls_seen.append(block.name)
                            yield ModelStreamEvent(
                                kind="tool_call",
                                raw={"id": block.id, "name": block.name, "input": block.input},
                            )
            except Exception as exc:
                logger.warning("anthropic.stream.final_extract_failed | %s", exc)
        logger.info(
            "anthropic.stream_response | model=%s content_len=%d tool_calls=%d content=%s tool_call_names=%s",
            model,
            len("".join(_content_parts)),
            len(_tool_calls_seen),
            "".join(_content_parts),
            _tool_calls_seen,
        )
        yield ModelStreamEvent(kind="usage", raw={"input_tokens": input_t, "output_tokens": output_t})
        yield ModelStreamEvent(kind="done")


# ---------------------------------------------------------------------------
# BoundAnthropicModelClient
# ---------------------------------------------------------------------------


class BoundAnthropicModelClient(ModelClient):
    """Request-scoped ModelClient for Anthropic — implements the ModelClient port."""

    def __init__(
        self,
        *,
        backend: AnthropicModelClient,
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
        messages = _to_anthropic_wire_messages(messages)
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

        # Non-streaming tool loop (Anthropic)
        from nlght.core.signals.signal import Signal  # noqa: PLC0415 (lazy import: optional extra or deliberate startup-cost/cycle avoidance)

        contracts = self._tool_catalog.all()
        openai_tools = [contract_to_openai_tool(c) for c in contracts] or None
        anthropic_tools = [_to_anthropic_tool(t) for t in openai_tools] if openai_tools else None
        terminal_names = terminal_tool_names(self._tool_catalog)

        system, unprivileged_messages = _split_messages(messages)
        current_messages = _convert_messages_for_anthropic(unprivileged_messages)
        input_t = output_t = 0

        for _round in range(10):
            kwargs: dict[str, Any] = {
                "model": self._model,
                "max_tokens": _DEFAULT_MAX_TOKENS,
                "messages": current_messages,
            }
            if system:
                kwargs["system"] = system
            if anthropic_tools:
                kwargs["tools"] = anthropic_tools
            if temperature is not None:
                kwargs["temperature"] = temperature

            logger.info(
                "anthropic.call.request | round=%d model=%s tools=%s",
                _round, self._model,
                [t["name"] for t in anthropic_tools] if anthropic_tools else None,
            )
            response = await self._backend._client.messages.create(**kwargs)
            if response.usage:
                input_t += response.usage.input_tokens or 0
                output_t += response.usage.output_tokens or 0

            tool_blocks = [b for b in (response.content or []) if getattr(b, "type", None) == "tool_use"]
            text_blocks = [b for b in (response.content or []) if getattr(b, "type", None) == "text"]
            content = text_blocks[0].text if text_blocks else ""
            logger.info(
                "anthropic.call.response_tool_calls | round=%d names=%s",
                _round, [b.name for b in tool_blocks],
            )

            if not tool_blocks:
                await self._emitter.emit(Signal(role="assistant", content=content, kind="result"))
                break

            assistant_content: list[dict[str, Any]] = []
            if content:
                assistant_content.append({"type": "text", "text": content})
            for b in tool_blocks:
                assistant_content.append({"type": "tool_use", "id": b.id, "name": b.name, "input": b.input})
            current_messages = list(current_messages) + [{"role": "assistant", "content": assistant_content}]

            logger.info(
                "anthropic.call.tool_loop | round=%d calls=%d dispatch_names=%s",
                _round, len(tool_blocks), [b.name for b in tool_blocks],
            )
            result_blocks: list[dict[str, Any]] = []
            for b in tool_blocks:
                tc = {"name": b.name, "input": b.input if isinstance(b.input, dict) else {}}
                result_text = await self._tool_catalog.execute(tc)
                logger.info("anthropic.call.tool_call | name=%s chars=%d", b.name, len(result_text))
                result_blocks.append({"type": "tool_result", "tool_use_id": b.id, "content": result_text})
            current_messages = current_messages + [{"role": "user", "content": result_blocks}]

            if terminal_names and any(b.name in terminal_names for b in tool_blocks):
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
        messages = _to_anthropic_wire_messages(messages)
        # Derive tool definitions: explicit > catalog > none
        effective_tools = tools
        if effective_tools is None and self._tool_catalog is not None:
            contracts = self._tool_catalog.all()
            effective_tools = [contract_to_openai_tool(c) for c in contracts] or None

        if effective_tools:
            anthropic_tools = [_to_anthropic_tool(t) for t in effective_tools]
            system, unprivileged_messages = _split_messages(messages)
            current_messages = _convert_messages_for_anthropic(unprivileged_messages)
            anthropic_tool_choice = _to_anthropic_tool_choice(tool_choice) if tool_choice is not None else None
        else:
            anthropic_tools = None
            anthropic_tool_choice = None
            system, current_messages = _split_messages(messages)

        input_t = output_t = 0

        # Without a catalog the caller (e.g. act.py) handles tool calls itself —
        # pass all events through unchanged.
        if self._tool_catalog is None:
            async for event in self._backend._stream_events(
                current_messages, self._model, system,
                tools=anthropic_tools, tool_choice=anthropic_tool_choice, temperature=temperature,
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

        # With catalog: run the tool loop internally — only token/done reach the caller.
        terminal_names = terminal_tool_names(self._tool_catalog)
        for _round in range(10):
            text_parts: list[str] = []
            tool_calls_raw: list[dict[str, Any]] = []

            async for event in self._backend._stream_events(
                current_messages, self._model, system,
                tools=anthropic_tools, tool_choice=anthropic_tool_choice, temperature=temperature,
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

            # Append assistant turn (Anthropic format — content blocks)
            assistant_text = "".join(text_parts).strip()
            assistant_content: list[dict[str, Any]] = []
            if assistant_text:
                assistant_content.append({"type": "text", "text": assistant_text})
            for tc in tool_calls_raw:
                assistant_content.append({
                    "type":  "tool_use",
                    "id":    tc["id"],
                    "name":  tc["name"],
                    "input": tc.get("input", {}),
                })
            current_messages = list(current_messages) + [{"role": "assistant", "content": assistant_content}]

            # Execute tools and append results turn (Anthropic tool_result blocks)
            result_blocks: list[dict[str, Any]] = []
            for tc in tool_calls_raw:
                result_text = await self._tool_catalog.execute(tc)
                logger.info("anthropic.tool_call | name=%s chars=%d", tc["name"], len(result_text))
                result_blocks.append({
                    "type":        "tool_result",
                    "tool_use_id": tc["id"],
                    "content":     result_text,
                })
            current_messages = current_messages + [{"role": "user", "content": result_blocks}]

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
