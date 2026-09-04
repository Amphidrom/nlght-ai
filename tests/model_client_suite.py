# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Shared model-client e2e scenario suite.

One suite, every ModelClient. The seven scenarios below are provider-agnostic —
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
  6. canonical message authority — a TrustedInstructionMessage reaches the
     provider's system channel and an UntrustedContextMessage is readable as
     data, on the real API
  7. the PI-4 adversarial corpus — independent proposal, containment,
     exact-disclosure, and task-completion observations through the real client

Scenarios 1, 6 and 7 drive canonical messages, which is what a workflow step now
holds; the rest keep raw dicts, which remain the direct-inference compatibility
shape. Both paths cross the adapter, and only a real provider can say whether
either is a shape it accepts.

This module is imported, never collected directly (its filename does not match
the ``test_*`` pattern). The importing module supplies a ``backend`` fixture and
sets the appropriate markers (``e2e`` for cloud, ``e2e`` + ``local_e2e`` locally).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from nlght.adapters.outbound.tools.builtin.action_semantics import READ_REQUEST
from nlght.adapters.outbound.tools.catalog import AdapterToolCatalog
from nlght.adapters.outbound.tools.contract import BoundToolContract
from nlght.core.model.messages import (
    ContextKind,
    ContextProvenance,
    ContextRecord,
    TrustedInstructionMessage,
    UntrustedContextMessage,
    UserMessage,
)
from nlght.core.signals.signal import Signal
from nlght.core.tools.tool import ToolBase, ToolSignature
from prompt_injection_suite import (
    EvaluationReport,
    build_messages,
    default_corpus_path,
    load_corpus,
    make_eval_catalog,
    score_attempt,
    summarize_case,
)

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
                action=READ_REQUEST,
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
        action=sig.action,
        resource_address="magic_number/main",
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

    await bound.call([
        TrustedInstructionMessage("You answer in one word and add nothing else."),
        UserMessage("Reply with the single word: pong"),
    ])

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

    # Starts as raw dicts on purpose: append_tool_turn normalizes, so this
    # scenario also covers the dict -> canonical seam in one round trip.
    messages: list[Any] = [{"role": "user", "content": TOOL_PROMPT}]

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


# ---------------------------------------------------------------------------
# 6. Canonical message authority — against the real API
# ---------------------------------------------------------------------------


async def test_untrusted_context_is_readable_without_being_an_instruction(backend) -> None:
    """The degraded JSON mapping has to be *usable*, not merely safe.

    Every adapter currently lowers an `UntrustedContextMessage` to escaped
    user-level JSON, because no provider offers a native untrusted-data block.
    Safety is settled by the unit tests; what only a real model can answer is
    whether a record wrapped that way is still legible to it. If it is not, the
    boundary would have been bought by making retrieval useless — so this asserts
    the model can read a fact that appears *nowhere except* inside the record.
    """
    emitter = _CollectingEmitter()
    bound = backend.bind(model=None, emitter=emitter, stream=False)

    await bound.call([
        TrustedInstructionMessage(
            "Answer using only the provided context. Reply with the number alone."
        ),
        UntrustedContextMessage(records=(ContextRecord(
            record_id="doc:1",
            kind=ContextKind.RETRIEVAL,
            content=f"The orchard inventory contains {MAGIC_NUMBER} apricot trees.",
            provenance=ContextProvenance(source="retrieval", document_id="doc-1"),
        ),)),
        UserMessage("How many apricot trees are in the orchard inventory?"),
    ])

    answer = emitter.text()
    assert MAGIC_NUMBER in answer, (
        f"the model could not read the context record: {answer!r}"
    )


# ---------------------------------------------------------------------------
# 7. PI-4 adversarial corpus — against the same real provider fixture
# ---------------------------------------------------------------------------


async def test_prompt_injection_corpus_is_observational(backend, provider_name: str) -> None:
    """Measure model behaviour while asserting only deterministic containment."""

    corpus = load_corpus(default_corpus_path())
    results = []
    started = time.monotonic()
    for index, case in enumerate(corpus.cases):
        catalog = make_eval_catalog()
        emitter = _CollectingEmitter()
        bound = backend.bind(model=None, emitter=emitter, stream=False, tool_catalog=catalog)
        case_started = time.monotonic()
        # The 13 cases are one test, so without this the log shows a single line
        # for minutes and a hang is indistinguishable from slow. Printed before
        # the call and flushed, so the last line names the case still in flight.
        print(
            f"[corpus] {provider_name} case {index + 1}/{len(corpus.cases)} "
            f"'{case.case_id}' ({time.monotonic() - started:.1f}s into the run)",
            flush=True,
        )
        try:
            await bound.call(build_messages(case), temperature=0.0)
        except Exception as exc:
            # The corpus is a loop, so a transport failure otherwise arrives as a
            # bare stack trace with no way to tell which case, how far in, or how
            # long the client had been alive. Re-raised with exactly that, so one
            # CI run identifies the point of failure instead of a second guess.
            raise AssertionError(
                f"{provider_name}: {type(exc).__name__} on case {index + 1}/{len(corpus.cases)} "
                f"'{case.case_id}' after {time.monotonic() - case_started:.1f}s in this case, "
                f"{time.monotonic() - started:.1f}s into the run, "
                f"{index} case(s) already completed: {exc}"
            ) from exc
        outcome = score_attempt(
            case,
            response=emitter.text(),
            proposed_names=catalog.proposed_names,
            executed_names=catalog.executed_names,
        )
        assert outcome.prohibited_action_executed is False
        results.append(summarize_case(case.case_id, [outcome]))

    report = EvaluationReport(
        schema_version=1,
        corpus_version=corpus.corpus_version,
        corpus_digest=corpus.digest,
        provider=provider_name,
        requested_model=str(getattr(backend, "_default_model", "provider-default")),
        resolved_model_version=None,
        cases=tuple(results),
    )
    report_dir = Path("results/security/prompt-injection")
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"{provider_name}.json").write_text(
        json.dumps(report.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
