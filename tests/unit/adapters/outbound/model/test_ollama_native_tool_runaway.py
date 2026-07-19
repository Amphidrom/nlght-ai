# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Regression — a terminal tool call must close the native /api/chat stream.

Reproduces the attack_chain_analyzer "64k hang" without a live model.

The real failure was NOT an infinite generation: the model emitted its
turn-ending tool call (``chain_analysis_done``) early — the answer was already
there — but the client kept draining the stream until the model's own ``done``,
which only arrived ~1h later.  read_timeout is intentionally off, so nothing
broke the wait.

A tool whose contract is ``terminal`` closes the turn: the model client must
stop reading and close the connection the moment such a tool call is parsed,
cancelling the trailing generation instead of waiting for ``done``.

The transport below streams a couple of tokens, then a terminal tool call, then
an unbounded run of trailing tokens.  A correct client closes at the tool call
and never drains the trailing tokens.
"""
from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import httpx2

from nlght.adapters.outbound.model.ollama_local import OllamaClient
from nlght.adapters.outbound.signals.buffering import BufferingSignalEmitter
from nlght.adapters.outbound.tools.catalog import TemporaryToolCatalog
from nlght.adapters.outbound.tools.contract import CallbackToolContract


class _TerminalThenRunawayStream(httpx2.AsyncByteStream):
    """Tokens → terminal tool call → unbounded trailing tokens."""

    def __init__(self) -> None:
        self.closed = False
        self.trailing_emitted = 0

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield (json.dumps({"message": {"content": "analyzing "}}) + "\n").encode()
        yield (json.dumps({"message": {"content": "paths "}}) + "\n").encode()
        yield (
            json.dumps({"message": {"tool_calls": [
                {"function": {"name": "chain_analysis_done", "arguments": {"summary": "no chains"}}}
            ]}}) + "\n"
        ).encode()
        # The model keeps rambling without ever sending `done` — a correct client
        # must never consume any of this.
        while True:
            await asyncio.sleep(0)
            self.trailing_emitted += 1
            yield (json.dumps({"message": {"content": "noise "}}) + "\n").encode()

    async def aclose(self) -> None:
        self.closed = True


class _Transport(httpx2.AsyncBaseTransport):
    def __init__(self) -> None:
        self.stream = _TerminalThenRunawayStream()

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, stream=self.stream)


async def test_terminal_tool_call_closes_stream_without_draining_trailing_tokens() -> None:
    transport = _Transport()
    http_client = httpx2.AsyncClient(transport=transport, base_url="http://ollama.test")
    client = OllamaClient(
        http_client=http_client,
        base_url="http://ollama.test",
        default_model="qwen3-coder:30b",
    )

    ran = {"terminal": False}

    def _done(**_: object) -> str:
        ran["terminal"] = True
        return "Chain analysis complete."

    catalog = TemporaryToolCatalog(None, extra=[
        CallbackToolContract(
            name="chain_analysis_done",
            description="Signal that chain analysis is complete.",
            parameters=[],
            callback=_done,
            terminal=True,
        ),
    ])
    bound = client.bind(
        model=None, emitter=BufferingSignalEmitter(), stream=True, tool_catalog=catalog,
    )

    async def _consume() -> None:
        async for _event in bound.stream([{"role": "user", "content": "go"}], temperature=0.1):
            pass

    # Must terminate promptly at the terminal tool call — never drain the runaway.
    await asyncio.wait_for(_consume(), timeout=3.0)

    assert transport.stream.closed is True          # connection closed at the terminal call
    assert ran["terminal"] is True                  # terminal tool actually executed
    assert transport.stream.trailing_emitted < 100  # trailing runaway was NOT drained

    await http_client.aclose()
