# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Shared model-client e2e scenario suite.

One suite, every ModelClient. The five scenarios below are provider-agnostic —
they drive whatever backend a `backend` fixture supplies, so the cloud clients
(``tests/e2e``) and the local Ollama client (``tests/local-e2e``) run byte-for-byte
the same behavioural contract:

  1. non-streaming call() → a result signal with content
  2. streaming stream()   → token events and a final done event
  3. native tool loop, non-streaming — call() with a ToolCatalog: the tool
     is really executed and its result reaches the final answer
  4. native tool loop, streaming — stream() with a ToolCatalog
  5. step-driven tool loop — stream(tools=...) without a catalog, executing
     the tool_call events manually and continuing via append_tool_turn()

This module is imported, never collected directly (its filename does not match
the ``test_*`` pattern). The importing module supplies a ``backend`` fixture and
sets the appropriate markers (``e2e`` for cloud, ``e2e`` + ``local_e2e`` locally).
"""
from __future__ import annotations

from typing import Any

from nlght.adapters.outbound.tools.catalog import AdapterToolCatalog
from nlght.adapters.outbound.tools.contract import BoundToolContract
from nlght.core.signals.signal import Signal
from nlght.core.tools.tool import ToolBase, ToolSignature

MAGIC_NUMBER = "473529"


# ---------------------------------------------------------------------------
# Test doubles: signal collector + a real tool the model must call
# ---------------------------------------------------------------------------


class _CollectingEmitter:
    def __init__(self) -> None:
        self.signals: list[Signal] = []

    async def emit(self, signal: Signal) -> None:
        self.signals.append(signal)

    def text(self) -> str:
        return "".join(s.content for s in self.signals if s.content)


class _MagicNumberTool(ToolBase):
    KIND = "magic_number"

    executions: list[str] = []  # class-level: rebound per catalog build below

    @classmethod
    def signatures(cls) -> list[ToolSignature]:
        return [
            ToolSignature(
                name="get_magic_number",
                description=(
                    "Returns the current secret magic number. The number changes "
                    "constantly — it can only be obtained through this tool."
                ),
                method_name="get_magic_number",
            )
        ]

    async def get_magic_number(self) -> str:
        self.executions.append("get_magic_number")
        return MAGIC_NUMBER


def _make_catalog() -> tuple[AdapterToolCatalog, _MagicNumberTool]:
    tool = _MagicNumberTool(name="magic-number", config={})
    tool.executions = []
    sig = _MagicNumberTool.signatures()[0]
    contract = BoundToolContract(
        name=sig.name,
        description=sig.description,
        parameters=sig.parameters,
        instance=tool,
        method_name=sig.method_name,
    )
    return AdapterToolCatalog({sig.name: contract}), tool


TOOL_PROMPT = (
    "Call the get_magic_number tool, then answer with exactly the number it "
    "returned and nothing else."
)


# ---------------------------------------------------------------------------
# 1. Plain non-streaming call
# ---------------------------------------------------------------------------


async def test_call_returns_assistant_result(backend) -> None:
    emitter = _CollectingEmitter()
    bound = backend.bind(model=None, emitter=emitter, stream=False)

    await bound.call([{"role": "user", "content": "Reply with the single word: pong"}])

    results = [s for s in emitter.signals if s.kind == "result"]
    assert results, f"no result signal, got kinds={[s.kind for s in emitter.signals]}"
    assert results[-1].role == "assistant"
    assert results[-1].content.strip()


# ---------------------------------------------------------------------------
# 2. Streaming tokens
# ---------------------------------------------------------------------------


async def test_stream_yields_tokens_and_done(backend) -> None:
    emitter = _CollectingEmitter()
    bound = backend.bind(model=None, emitter=emitter, stream=True)

    kinds: list[str] = []
    parts: list[str] = []
    async for event in bound.stream(
        [{"role": "user", "content": "Count from 1 to 5, digits separated by spaces."}]
    ):
        kinds.append(event.kind)
        if event.kind == "token":
            parts.append(event.content)

    assert "token" in kinds, f"no token events, kinds={kinds}"
    assert kinds[-1] == "done"
    assert "".join(parts).strip()


# ---------------------------------------------------------------------------
# 3. Native tool loop — non-streaming call() with a catalog
# ---------------------------------------------------------------------------


async def test_native_tool_loop_call_executes_tool(backend) -> None:
    catalog, tool = _make_catalog()
    emitter = _CollectingEmitter()
    bound = backend.bind(model=None, emitter=emitter, stream=False, tool_catalog=catalog)

    await bound.call([{"role": "user", "content": TOOL_PROMPT}])

    assert tool.executions, "the model never invoked get_magic_number"
    final = emitter.text()
    assert MAGIC_NUMBER in final, f"tool result did not reach the answer: {final!r}"


# ---------------------------------------------------------------------------
# 4. Native tool loop — streaming stream() with a catalog
# ---------------------------------------------------------------------------


async def test_native_tool_loop_stream_executes_tool(backend) -> None:
    catalog, tool = _make_catalog()
    emitter = _CollectingEmitter()
    bound = backend.bind(model=None, emitter=emitter, stream=True, tool_catalog=catalog)

    kinds: list[str] = []
    parts: list[str] = []
    async for event in bound.stream([{"role": "user", "content": TOOL_PROMPT}]):
        kinds.append(event.kind)
        if event.kind == "token":
            parts.append(event.content)

    assert tool.executions, "the model never invoked get_magic_number"
    assert kinds[-1] == "done"
    assert MAGIC_NUMBER in "".join(parts), f"tool result missing from stream: {''.join(parts)!r}"


# ---------------------------------------------------------------------------
# 5. Step-driven tool loop — manual execution + append_tool_turn
# ---------------------------------------------------------------------------


async def test_step_driven_tool_loop_round_trip(backend) -> None:
    catalog, tool = _make_catalog()
    emitter = _CollectingEmitter()
    bound = backend.bind(model=None, emitter=emitter, stream=True)

    sig = _MagicNumberTool.signatures()[0]
    tool_defs: list[dict[str, Any]] = [{
        "type": "function",
        "function": {
            "name": sig.name,
            "description": sig.description,
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }]

    messages: list[dict[str, Any]] = [{"role": "user", "content": TOOL_PROMPT}]

    # Round 1: expect at least one tool_call event.
    tool_calls: list[dict[str, Any]] = []
    async for event in bound.stream(messages, tools=tool_defs):
        if event.kind == "tool_call" and event.raw:
            tool_calls.append(event.raw)

    assert tool_calls, "the model produced no tool_call event"

    # Execute manually and append the turn in the client's wire format.
    results = [await catalog.execute(tc) for tc in tool_calls]
    assert tool.executions
    assert results and results[0] == MAGIC_NUMBER
    messages = bound.append_tool_turn(messages, tool_calls, results)

    # Round 2: the final answer must contain the tool result.
    parts: list[str] = []
    async for event in bound.stream(messages, tools=tool_defs):
        if event.kind == "token":
            parts.append(event.content)

    assert MAGIC_NUMBER in "".join(parts), f"final answer missing tool result: {''.join(parts)!r}"
