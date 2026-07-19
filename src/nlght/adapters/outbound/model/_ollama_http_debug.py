# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from datetime import datetime
from typing import TYPE_CHECKING, Any

import httpx2

from nlght.adapters.outbound.model._context_chunking import (
    CHUNK_ACK_MAX_TOKENS,
    FINALIZE_PREFIX,
    OUTPUT_HEADROOM,
    estimate_tokens,
    total_message_tokens,
)
from nlght.core.model.model_info import ModelInfo, RunningModelInfo
from nlght.core.signals.signal import Signal
from nlght.ports.outbound.model_client import ModelStreamEvent
from nlght.ports.outbound.signal_emitter import SignalEmitter

if TYPE_CHECKING:
    from nlght.core.metering.context import MeteringContext
    from nlght.core.model.budget import TokenBudget

logger = logging.getLogger(__name__)


def _merge_streamed_fragment(existing: str, fragment: str) -> str:
    """Merge streamed text fragments without duplicating repeated overlap."""
    if not fragment:
        return existing
    if not existing:
        return fragment
    if existing == fragment or existing.endswith(fragment):
        return existing
    if fragment.startswith(existing):
        return fragment
    max_overlap = min(len(existing), len(fragment))
    for overlap in range(max_overlap, 0, -1):
        if existing.endswith(fragment[:overlap]):
            return existing + fragment[overlap:]
    return existing + fragment


class _OllamaHttp:
    """HTTP backend shared by OllamaClient and OllamaCloudClient.

    Not part of the public API — composed, not subclassed.
    """

    def __init__(
        self,
        *,
        http_client: httpx2.AsyncClient,
        base_url: str,
        api_key: str = "",
        extra_headers: dict[str, str] | None = None,
        request_timeout_s: float | None = None,
        stream_connect_timeout_s: float | None = None,
        stream_read_timeout_s: float | None = None,
    ) -> None:
        self._http = http_client
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key.strip()
        self._extra_headers = dict(extra_headers or {})
        self._request_timeout_s = None if request_timeout_s is None else float(request_timeout_s)
        self._stream_connect_timeout_s = None if stream_connect_timeout_s is None else float(stream_connect_timeout_s)
        self._stream_read_timeout_s = None if stream_read_timeout_s is None else float(stream_read_timeout_s)

    # ------------------------------------------------------------------
    # Management API
    # ------------------------------------------------------------------

    async def list_models(self) -> list[ModelInfo]:
        target = f"{self.base_url}/api/tags"
        try:
            response = await self._http.get(target, timeout=10.0, headers=self.request_headers())
            response.raise_for_status()
            data = response.json()
        except (httpx2.RequestError, httpx2.HTTPStatusError) as exc:
            logger.warning("ollama.list_models.failed | %s", exc)
            return []
        models: list[ModelInfo] = []
        for entry in data.get("models", []):
            modified_at = None
            raw_ts = entry.get("modified_at")
            if raw_ts:
                try:
                    modified_at = datetime.fromisoformat(raw_ts)
                except ValueError:
                    pass
            models.append(ModelInfo(
                name=entry.get("name", ""),
                size=entry.get("size"),
                modified_at=modified_at,
                digest=entry.get("digest"),
            ))
        return models

    async def list_running_models(self) -> list[RunningModelInfo]:
        target = f"{self.base_url}/api/ps"
        try:
            response = await self._http.get(target, timeout=10.0, headers=self.request_headers())
            response.raise_for_status()
            data = response.json()
        except (httpx2.RequestError, httpx2.HTTPStatusError) as exc:
            logger.warning("ollama.list_running_models.failed | %s", exc)
            return []
        running: list[RunningModelInfo] = []
        for entry in data.get("models", []):
            expires_at = None
            raw_ts = entry.get("expires_at")
            if raw_ts:
                try:
                    expires_at = datetime.fromisoformat(raw_ts)
                except ValueError:
                    pass
            running.append(RunningModelInfo(
                name=entry.get("name", ""),
                size_vram=entry.get("size_vram", 0),
                expires_at=expires_at,
                digest=entry.get("digest"),
                size=entry.get("size"),
            ))
        return running

    async def token_budget(
        self,
        model: str,
        *,
        client_max_tokens: int | None = None,
    ) -> TokenBudget:
        from nlght.core.model.budget import TokenBudget as _Budget  # noqa: PLC0415 (lazy import: optional extra or deliberate startup-cost/cycle avoidance)

        return await _Budget.for_model(
            model,
            base_url=self.base_url,
            http_client=self._http,
            client_max_tokens=client_max_tokens,
            headers=self.request_headers(),
        )

    # ------------------------------------------------------------------
    # Non-streaming
    # ------------------------------------------------------------------

    async def complete(
        self,
        target: str,
        body: dict[str, Any],
        emitter: SignalEmitter,
        metering: MeteringContext | None = None,
    ) -> None:
        response = await self._http.post(
            target,
            json=body,
            timeout=self._request_timeout_s,
            headers=self.request_headers(),
        )
        response.raise_for_status()
        data = response.json()
        try:
            content = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError):
            content = ""
        await emitter.emit(Signal(role="assistant", content=content, kind="result"))
        if metering is not None:
            await self._record_tokens(metering, body.get("model", ""), data.get("usage") or {})

    async def call_raw(
        self,
        messages: list[dict[str, Any]],
        model: str,
        *,
        max_tokens: int | None = None,
    ) -> str:
        target = f"{self.base_url}/v1/chat/completions"
        body: dict[str, Any] = {"model": model, "messages": messages, "stream": False}
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        response = await self._http.post(
            target,
            json=body,
            timeout=self._request_timeout_s,
            headers=self.request_headers(),
        )
        response.raise_for_status()
        try:
            return response.json()["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError):
            return ""

    async def complete_message(
        self,
        target: str,
        body: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return (message_dict, usage_dict) from a non-streaming completion.

        Unlike complete(), this returns the full message object so callers can
        inspect tool_calls and implement their own tool-execution loop.
        """
        response = await self._http.post(
            target,
            json=body,
            timeout=self._request_timeout_s,
            headers=self.request_headers(),
        )
        response.raise_for_status()
        data = response.json()
        try:
            msg: dict[str, Any] = data["choices"][0].get("message") or {}
        except (KeyError, IndexError):
            msg = {}
        return msg, data.get("usage") or {}

    # ------------------------------------------------------------------
    # Streaming — OpenAI SSE
    # ------------------------------------------------------------------

    async def stream_events_sse(
        self,
        target: str,
        body: dict[str, Any],
    ) -> AsyncIterator[ModelStreamEvent]:
        body = {**body, "stream_options": {"include_usage": True}}
        input_t = output_t = 0
        tc_builder: dict[int, dict[str, Any]] = {}
        _content_parts: list[str] = []
        _tool_calls_seen: list[str] = []
        _finish_reason: str = "unknown"
        _first_token_logged = False
        try:
            # ── Log input ─────────────────────────────────────────────────────
            _messages: list[dict[str, Any]] = body.get("messages") or []
            _last = _messages[-1] if _messages else {}
            logger.info(
                "ollama.stream_input | target=%s model=%s messages=%d last_role=%s last_len=%d",
                target,
                body.get("model", "?"),
                len(_messages),
                _last.get("role", "?"),
                len(str(_last.get("content") or "")),
            )
            if logger.isEnabledFor(logging.DEBUG):
                for _i, _m in enumerate(_messages):
                    logger.debug(
                        "ollama.stream_input.msg[%d] | role=%s content=%s",
                        _i, _m.get("role", "?"), str(_m.get("content") or "")[:2000],
                    )
            logger.info(
                "ollama.stream_start | target=%s read_timeout=%s",
                target,
                "none" if self._stream_read_timeout_s is None else f"{self._stream_read_timeout_s:.1f}s",
            )
            async with self._http.stream(
                "POST",
                target,
                json=body,
                timeout=self._stream_timeout(),
                headers=self.request_headers(),
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    logger.debug("ollama.sse_line | target=%s line=%s", target, line)
                    if not line.startswith("data:"):
                        continue
                    raw = line[len("data:"):].strip()
                    if raw == "[DONE]":
                        break
                    try:
                        chunk = json.loads(raw)
                        if chunk.get("usage"):
                            input_t = chunk["usage"].get("prompt_tokens", 0) or 0
                            output_t = chunk["usage"].get("completion_tokens", 0) or 0
                        choice = chunk["choices"][0]
                        delta = choice.get("delta", {})
                        token = delta.get("content") or ""
                        if token:
                            if not _first_token_logged:
                                logger.info("ollama.stream_first_token | target=%s model=%s", target, body.get("model", "?"))
                                _first_token_logged = True
                            _content_parts.append(token)
                            yield ModelStreamEvent(kind="token", content=token, raw=chunk)
                        for tc_delta in (delta.get("tool_calls") or []):
                            idx = tc_delta.get("index", 0)
                            if idx not in tc_builder:
                                tc_builder[idx] = {"id": "", "name": "", "args": ""}
                            if tc_delta.get("id"):
                                tc_builder[idx]["id"] = tc_delta["id"]
                            fn = tc_delta.get("function", {})
                            if fn.get("name"):
                                tc_builder[idx]["name"] = _merge_streamed_fragment(
                                    tc_builder[idx]["name"],
                                    str(fn["name"]),
                                )
                            if fn.get("arguments"):
                                tc_builder[idx]["args"] = _merge_streamed_fragment(
                                    tc_builder[idx]["args"],
                                    str(fn["arguments"]),
                                )
                        fr = choice.get("finish_reason")
                        if fr:
                            _finish_reason = fr
                        if fr == "tool_calls":
                            for tc in tc_builder.values():
                                try:
                                    inp = json.loads(tc["args"]) if tc["args"] else {}
                                except json.JSONDecodeError:
                                    inp = {}
                                _tool_calls_seen.append(tc["name"])
                                yield ModelStreamEvent(
                                    kind="tool_call",
                                    raw={"id": tc["id"], "name": tc["name"], "input": inp},
                                )
                            tc_builder.clear()
                    except (KeyError, IndexError, json.JSONDecodeError):
                        continue
        except httpx2.TimeoutException:
            logger.warning("ollama.stream_timeout | target=%s", target)
            raise
        finally:
            _full_content = "".join(_content_parts)
            logger.info(
                "ollama.stream_response | target=%s finish_reason=%s content_len=%d tool_calls=%d content=%s tool_call_names=%s",
                target,
                _finish_reason,
                len(_full_content),
                len(_tool_calls_seen),
                _full_content,
                _tool_calls_seen,
            )
            logger.info("ollama.stream_closed | target=%s", target)
        yield ModelStreamEvent(kind="usage", raw={"input_tokens": input_t, "output_tokens": output_t})
        yield ModelStreamEvent(kind="done")

    # ------------------------------------------------------------------
    # Streaming — native Ollama NDJSON
    # ------------------------------------------------------------------

    async def stream_events_ndjson(
        self,
        target: str,
        body: dict[str, Any],
        tool_choice: dict[str, Any] | str | None = None,
        terminal_tools: set[str] | None = None,
    ) -> AsyncIterator[ModelStreamEvent]:
        if tool_choice is not None:
            body = {**body, "tool_choice": tool_choice}
        tc_counter = 0
        _content_parts: list[str] = []
        _thinking_parts: list[str] = []
        _tool_calls_seen: list[str] = []
        _finish_reason: str = "unknown"
        _first_token_logged = False
        try:
            # ── Log input ─────────────────────────────────────────────────────
            _messages: list[dict[str, Any]] = body.get("messages") or []
            _last = _messages[-1] if _messages else {}
            logger.info(
                "ollama.stream_input | target=%s model=%s messages=%d last_role=%s last_len=%d",
                target,
                body.get("model", "?"),
                len(_messages),
                _last.get("role", "?"),
                len(str(_last.get("content") or "")),
            )
            if logger.isEnabledFor(logging.DEBUG):
                for _i, _m in enumerate(_messages):
                    logger.debug(
                        "ollama.stream_input.msg[%d] | role=%s content=%s",
                        _i, _m.get("role", "?"), str(_m.get("content") or "")[:2000],
                    )
            logger.info(
                "ollama.stream_start | target=%s read_timeout=%s",
                target,
                "none" if self._stream_read_timeout_s is None else f"{self._stream_read_timeout_s:.1f}s",
            )
            async with self._http.stream(
                "POST",
                target,
                json=body,
                timeout=self._stream_timeout(),
                headers=self.request_headers(),
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    logger.debug("ollama.ndjson_line | target=%s line=%s", target, line)
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                        msg = chunk.get("message", {})
                        token = msg.get("content") or ""
                        if token:
                            if not _first_token_logged:
                                logger.debug("ollama.stream_first_token | target=%s model=%s", target, body.get("model", "?"))
                                _first_token_logged = True
                            _content_parts.append(token)
                            yield ModelStreamEvent(kind="token", content=token)
                        think = msg.get("thinking") or ""
                        if think:
                            _thinking_parts.append(think)
                            yield ModelStreamEvent(kind="thinking", content=think)
                        _terminal_hit = False
                        for tc in (msg.get("tool_calls") or []):
                            fn = tc.get("function", {})
                            tc_name = fn.get("name", "")
                            _tool_calls_seen.append(tc_name)
                            yield ModelStreamEvent(
                                kind="tool_call",
                                raw={
                                    "id": f"call_{tc_counter}",
                                    "name": tc_name,
                                    "input": fn.get("arguments", {}),
                                },
                            )
                            tc_counter += 1
                            if terminal_tools and tc_name in terminal_tools:
                                _terminal_hit = True
                        # A terminal tool call ends the turn: stop reading and let the
                        # `async with` close the connection — this cancels any trailing
                        # generation server-side instead of waiting for the model's own
                        # `done` (the chain-analyzer hang).
                        if _terminal_hit:
                            _finish_reason = "terminal_tool"
                            logger.info(
                                "ollama.stream_terminal_toolcall | target=%s — closing stream early", target,
                            )
                            break
                        if chunk.get("done"):
                            _finish_reason = chunk.get("done_reason", "stop")
                            break
                    except json.JSONDecodeError:
                        continue
        except httpx2.TimeoutException:
            logger.warning("ollama.stream_timeout | target=%s", target)
            raise
        finally:
            _full_content = "".join(_content_parts)
            _full_thinking = "".join(_thinking_parts)
            logger.info(
                "ollama.stream_response | target=%s finish_reason=%s content_len=%d thinking_len=%d tool_calls=%d content=%s tool_call_names=%s",
                target,
                _finish_reason,
                len(_full_content),
                len(_full_thinking),
                len(_tool_calls_seen),
                _full_content,
                _tool_calls_seen,
            )
            if _full_thinking:
                logger.info("ollama.stream_thinking | target=%s thinking=%s", target, _full_thinking)
            logger.info("ollama.stream_closed | target=%s", target)
        yield ModelStreamEvent(kind="usage", raw={"input_tokens": 0, "output_tokens": 0})
        yield ModelStreamEvent(kind="done")

    # ------------------------------------------------------------------
    # Streaming via emitter (for call() streaming path)
    # ------------------------------------------------------------------

    async def stream_via_emitter(
        self,
        target: str,
        body: dict[str, Any],
        emitter: SignalEmitter,
        metering: MeteringContext | None = None,
    ) -> None:
        input_t = output_t = 0
        async for event in self.stream_events_sse(target, body):
            if event.kind == "usage":
                if event.raw:
                    input_t = event.raw.get("input_tokens", 0) or 0
                    output_t = event.raw.get("output_tokens", 0) or 0
            else:
                await emitter.emit(Signal(role="assistant", content=event.content, kind=event.kind))
        if metering is not None:
            await self._record_tokens(metering, body.get("model", ""), {"prompt_tokens": input_t, "completion_tokens": output_t})

    # ------------------------------------------------------------------
    # Chunked session execution
    # ------------------------------------------------------------------

    async def execute_chunked_call(
        self,
        *,
        model: str,
        base_messages: list[dict[str, Any]],
        content_chunks: list[str],
        task_message: dict[str, Any],
        token_budget: TokenBudget,
        emitter: SignalEmitter,
        stream: bool,
        metering: MeteringContext | None = None,
    ) -> None:
        session = await self._feed_chunks_async(model, base_messages, content_chunks, token_budget)
        finalize_content = FINALIZE_PREFIX + (task_message.get("content") or "")
        session.append({"role": "user", "content": finalize_content})
        logger.info(
            "llm.auto_chunk.finalize | model=%s chunks=%d session_msgs=%d stream=%s",
            model, len(content_chunks), len(session), stream,
        )
        target = f"{self.base_url}/v1/chat/completions"
        if stream:
            await self.stream_via_emitter(
                target, {"model": model, "messages": session, "stream": True}, emitter, metering,
            )
        else:
            await self.complete(
                target, {"model": model, "messages": session, "stream": False}, emitter, metering,
            )

    async def execute_chunked_stream_events(
        self,
        *,
        model: str,
        base_messages: list[dict[str, Any]],
        content_chunks: list[str],
        task_message: dict[str, Any],
        token_budget: TokenBudget,
    ) -> AsyncIterator[ModelStreamEvent]:
        session = await self._feed_chunks_async(model, base_messages, content_chunks, token_budget)
        finalize_content = FINALIZE_PREFIX + (task_message.get("content") or "")
        session.append({"role": "user", "content": finalize_content})
        logger.info(
            "llm.auto_chunk_stream.finalize | model=%s chunks=%d session_msgs=%d",
            model, len(content_chunks), len(session),
        )
        target = f"{self.base_url}/v1/chat/completions"
        async for event in self.stream_events_sse(target, {"model": model, "messages": session, "stream": True}):
            yield event

    async def _feed_chunks_async(
        self,
        model: str,
        base_messages: list[dict[str, Any]],
        content_chunks: list[str],
        token_budget: TokenBudget,
    ) -> list[dict[str, Any]]:
        context_limit = token_budget.total
        session: list[dict[str, Any]] = list(base_messages)
        for i, chunk in enumerate(content_chunks):
            chunk_msg = {"role": "user", "content": f"[Part {i + 1}/{len(content_chunks)}]\n\n{chunk}"}
            projected = total_message_tokens(session) + estimate_tokens(chunk_msg["content"])
            if projected + OUTPUT_HEADROOM > context_limit:
                logger.warning(
                    "llm.auto_chunk.session_trim | projected=%d limit=%d chunk=%d/%d — resetting ack history",
                    projected, context_limit, i + 1, len(content_chunks),
                )
                session = list(base_messages)
            session.append(chunk_msg)
            reply = await self.call_raw(session, model, max_tokens=CHUNK_ACK_MAX_TOKENS)
            session.append({"role": "assistant", "content": reply})
            logger.debug(
                "llm.auto_chunk.fed | chunk=%d/%d reply_len=%d session_msgs=%d",
                i + 1, len(content_chunks), len(reply), len(session),
            )
        return session

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def request_headers(self) -> dict[str, str]:
        headers = dict(self._extra_headers)
        if self._api_key and "Authorization" not in headers:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _stream_timeout(self) -> httpx2.Timeout:
        return httpx2.Timeout(
            connect=self._stream_connect_timeout_s,
            read=self._stream_read_timeout_s,
            write=self._stream_connect_timeout_s,
            pool=self._stream_connect_timeout_s,
        )

    async def _record_tokens(
        self,
        metering: MeteringContext,
        model: str,
        usage: dict[str, Any],
    ) -> None:
        try:
            input_t = usage.get("prompt_tokens", 0) or 0
            output_t = usage.get("completion_tokens", 0) or 0
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
