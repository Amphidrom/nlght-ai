# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from collections.abc import Iterable

import pytest

from nlght.adapters.outbound.model._ollama_messages import to_ollama_messages
from nlght.adapters.outbound.model.anthropic import _to_anthropic_wire_messages
from nlght.adapters.outbound.model.google import _to_google_contents, _to_google_wire_messages
from nlght.adapters.outbound.model.ollama_cloud import _to_ollama_cloud_messages
from nlght.adapters.outbound.model.openai_cloud import _to_openai_messages
from nlght.core.model.messages import (
    AssistantMessage,
    CanonicalMessage,
    ContextKind,
    ToolResultMessage,
    TrustedInstructionMessage,
    UntrustedContextMessage,
    UserMessage,
)
from prompt_injection_suite import (
    PROHIBITED_SINK,
    Carrier,
    build_messages,
    default_corpus_path,
    load_corpus,
    make_eval_catalog,
    score_attempt,
)

CORPUS = load_corpus(default_corpus_path())
MAPPERS = {
    "openai": _to_openai_messages,
    "anthropic": _to_anthropic_wire_messages,
    "google": _to_google_wire_messages,
    "ollama-local": to_ollama_messages,
    "ollama-cloud": _to_ollama_cloud_messages,
}


def _all_strings(value: object) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _all_strings(item)
    elif isinstance(value, list | tuple):
        for item in value:
            yield from _all_strings(item)


def _carrier_content(messages: list[CanonicalMessage], carrier: Carrier) -> list[str]:
    if carrier == Carrier.CURRENT_USER:
        return [str(message.content) for message in messages if isinstance(message, UserMessage)]
    if carrier == Carrier.PRIOR_TURN:
        return [str(message.content) for message in messages[:-1] if isinstance(message, UserMessage)]
    if carrier == Carrier.TOOL_RESULT:
        return [str(message.content) for message in messages if isinstance(message, ToolResultMessage)]
    kind = {
        Carrier.RETRIEVAL: ContextKind.RETRIEVAL,
        Carrier.SESSION_RESULT: ContextKind.SESSION_RESULT,
        Carrier.WORKING_MEMORY: ContextKind.WORKING_MEMORY,
    }[carrier]
    return [
        record.content
        for message in messages
        if isinstance(message, UntrustedContextMessage)
        for record in message.records
        if record.kind == kind
    ]


@pytest.mark.parametrize("case", CORPUS.cases, ids=lambda case: case.case_id)
def test_case_uses_its_declared_canonical_carrier(case) -> None:  # noqa: ANN001
    messages = build_messages(case)

    carrier_content = _carrier_content(messages, case.attack.carrier)
    assert sum(value.count(case.adversarial_content) for value in carrier_content) == 1
    assert isinstance(messages[0], TrustedInstructionMessage)
    if case.attack.carrier == Carrier.TOOL_RESULT:
        result_index = next(
            index for index, message in enumerate(messages) if isinstance(message, ToolResultMessage)
        )
        assert isinstance(messages[result_index - 1], AssistantMessage)
        assert isinstance(messages[result_index - 2], UserMessage)
        assert messages[result_index - 2].content == case.user_request
        assert result_index == len(messages) - 1
        assert messages[result_index - 1].tool_calls[0].call_id == messages[result_index].tool_call_id


@pytest.mark.parametrize(
    "case",
    [case for case in CORPUS.cases if case.attack.carrier == Carrier.TOOL_RESULT],
    ids=lambda case: case.case_id,
)
def test_google_tool_result_case_is_one_valid_function_call_turn(case) -> None:  # noqa: ANN001
    _system, contents = _to_google_contents(_to_google_wire_messages(build_messages(case)))

    assert [turn["role"] for turn in contents] == ["user", "model", "user"]
    function_call = contents[1]["parts"][0]
    assert function_call["thought_signature"] == b"skip_thought_signature_validator"
    assert contents[2]["parts"][0]["function_response"]["name"] == "pi_external_source"


@pytest.mark.parametrize("provider", MAPPERS)
@pytest.mark.parametrize("case", CORPUS.cases, ids=lambda case: case.case_id)
def test_every_case_uses_production_serialization_without_system_promotion(
    provider: str,
    case,
) -> None:  # noqa: ANN001
    wire = MAPPERS[provider](build_messages(case))
    system_text = "\n".join(
        str(message.get("content", ""))
        for message in wire
        if message.get("role") == "system"
    )
    occurrences = sum(value.count(case.adversarial_content) for value in _all_strings(wire))

    assert case.adversarial_content not in system_text
    assert occurrences == 1


@pytest.mark.parametrize("case", CORPUS.cases, ids=lambda case: case.case_id)
async def test_fake_prohibited_sink_crosses_production_action_gate_and_never_executes(case) -> None:  # noqa: ANN001
    catalog = make_eval_catalog()

    result = await catalog.execute(
        {"name": PROHIBITED_SINK, "input": {"payload": case.adversarial_content}},
        has_external_untrusted_input=True,
    )
    outcome = score_attempt(
        case,
        response="",
        proposed_names=catalog.proposed_names,
        executed_names=catalog.executed_names,
    )

    assert "arbitrary_egress" in result
    assert outcome.prohibited_action_proposed is True
    assert outcome.prohibited_action_executed is False
    assert catalog.executed_names == ()
