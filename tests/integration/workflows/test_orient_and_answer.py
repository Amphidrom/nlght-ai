# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""The whole path, asserted where it actually matters: at the model call.

    retrieval.hits → prompt.compose → passthrough → ModelClient.call(messages)

Nothing in the middle is mocked. The only fake is the model itself, which records
what it was sent — because what a model receives is the only place the pipeline's
claims can be checked rather than inferred.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from nlght.adapters.outbound.workflow.steps.passthrough import PassthroughStep
from nlght.adapters.outbound.workflow.steps.prompt import PromptComposeStep
from nlght.core.entry.context import RequestContext
from nlght.core.hive_mind.models import (
    TurnSummary,
    estimate_tokens,
)
from nlght.core.model.budget import TokenBudget
from nlght.core.model.messages import (
    CanonicalMessage,
    TrustedInstructionMessage,
    UntrustedContextMessage,
    UserMessage,
)
from nlght.core.protocol.protocol import ProtocolKind
from nlght.core.retrieval import (
    DOCUMENT,
    LEXICAL,
    Provenance,
    RetrievalHit,
    fuse,
    rank_within_sources,
)
from nlght.core.trigger.trigger import Trigger, TriggerKind
from nlght.core.workflow.step import WorkflowStepContext

pytestmark = pytest.mark.integration


class _RecordingModel:
    """A model that answers nothing and remembers everything it was sent.

    It implements the whole `ModelClient` contract including `token_budget`,
    because a double that stands in for a port has to satisfy it — a double that
    is missing a member only proves the production code was duck-typing.
    """

    def __init__(self) -> None:
        self.calls: list[list[CanonicalMessage]] = []

    @property
    def token_budget(self) -> TokenBudget | None:
        """No budget from here, which is a definite answer and not a gap."""
        return None

    async def call(self, messages: list[CanonicalMessage], **_kwargs: object) -> None:
        self.calls.append(list(messages))

    def stream(self, *args: object, **kwargs: object):  # noqa: ANN201, ARG002
        raise AssertionError("this workflow does not stream")


class _Emitter:
    async def emit(self, signal: object) -> None:  # noqa: ARG002
        return None


class _Memory:
    def __init__(self, turns=(), directives=(), results=()) -> None:  # noqa: ANN001
        self._turns, self._directives, self._results = list(turns), list(directives), list(results)

    def get_directives_list(self):  # noqa: ANN201
        return self._directives

    def get_recent_turns(self, n: int):  # noqa: ANN201
        return self._turns[-n:]


def _ctx(*, hits=(), memory=None, question="what is the limit?") -> WorkflowStepContext:  # noqa: ANN001
    context = RequestContext(
        correlation_id="run-1", request_id="rid",
        received_at=datetime(2026, 1, 1, tzinfo=UTC),
        path="/", method="POST", headers={}, query_params={}, client_host=None,
    )
    ctx = WorkflowStepContext(
        correlation_id="run-1",
        trigger=Trigger(
            kind=TriggerKind.INBOUND_EVENT, protocol=ProtocolKind.GENERIC_JSON,
            operation="ask", payload={}, context=context,
        ),
        model="m",
        messages=[{"role": "user", "content": question}],
        stream=False,
        emitter=_Emitter(),
        store_coordinator=memory,
    )
    if hits:
        ctx.metadata["retrieval.hits"] = tuple(hits)
    return ctx


def _hits(*texts: str):  # noqa: ANN202
    return fuse(rank_within_sources([
        RetrievalHit(
            source=LEXICAL, carrier=DOCUMENT, carrier_id=f"d{index}", content=text,
            provenance=Provenance(document_id=f"d{index}", path=f"docs/{index}.adoc"),
        )
        for index, text in enumerate(texts)
    ]))


async def _run(ctx: WorkflowStepContext, *, config: dict | None = None) -> _RecordingModel:
    model = _RecordingModel()
    ctx.llm = model
    ctx = (await PromptComposeStep(config=config or {}).run(ctx)).ctx
    await PassthroughStep(config={}).run(ctx)
    return model


def _trusted(messages: list[CanonicalMessage]) -> TrustedInstructionMessage:
    return next(message for message in messages if isinstance(message, TrustedInstructionMessage))


def _context(messages: list[CanonicalMessage]) -> UntrustedContextMessage | None:
    return next(
        (message for message in messages if isinstance(message, UntrustedContextMessage)),
        None,
    )


def _context_text(messages: list[CanonicalMessage]) -> str:
    context = _context(messages)
    return "\n".join(record.content for record in context.records) if context else ""


def _estimated_request_tokens(messages: list[CanonicalMessage]) -> int:
    context = _context(messages)
    return estimate_tokens(_trusted(messages).content) + sum(
        record.estimated_tokens for record in (context.records if context else ())
    )


# The path, end to end
# ---------------------------------------------------------------------------

async def test_retrieved_material_reaches_the_model_call() -> None:
    model = await _run(_ctx(hits=_hits("The value is limited to 100 characters.")))

    assert len(model.calls) == 1
    messages = model.calls[0]
    assert sum(isinstance(message, TrustedInstructionMessage) for message in messages) == 1
    assert "The value is limited to 100 characters." not in _trusted(messages).content
    assert "The value is limited to 100 characters." in _context_text(messages)


async def test_the_user_question_survives_untouched() -> None:
    model = await _run(_ctx(hits=_hits("something")))

    users = [message for message in model.calls[0] if isinstance(message, UserMessage)]
    assert [message.content for message in users] == ["what is the limit?"]


async def test_a_passage_never_becomes_a_role_of_its_own() -> None:
    """Retrieved text is one typed data message, not a provider role per hit.

    The old prototype appended a system message per file and per claim kind. That
    makes the provider's message list carry the pipeline's structure, which no
    model contract promises to respect.
    """
    model = await _run(_ctx(hits=_hits("a", "b", "c")))

    assert [type(message) for message in model.calls[0]] == [
        TrustedInstructionMessage,
        UntrustedContextMessage,
        UserMessage,
    ]


async def test_the_same_workflow_works_with_no_retrieval_at_all() -> None:
    # The regression that matters most: retrieval is an input to the chain, never
    # its precondition.
    model = await _run(_ctx())

    assert len(model.calls) == 1
    assert any(isinstance(message, TrustedInstructionMessage) for message in model.calls[0])


async def test_memory_and_retrieval_arrive_in_one_untrusted_context_message() -> None:
    memory = _Memory(turns=[TurnSummary(
        turn_nr=1, user_input="we discussed limits", intent="ask", topic="limits",
    )])

    model = await _run(_ctx(hits=_hits("The limit is 100 characters."), memory=memory))

    assert "The limit is 100 characters." in _context_text(model.calls[0])
    assert "we discussed limits" in _context_text(model.calls[0])
    assert "The limit is 100 characters." not in _trusted(model.calls[0]).content


async def test_the_prompt_stays_inside_the_budget_it_was_given() -> None:
    """Budget pressure across both sources at once.

    The passages here cost far more than the budget allows, so the reduction has
    to give some of them up — and what it produces must actually fit, not merely
    be smaller.
    """
    long_passages = _hits(*[f"passage number {n} " + ("x" * 400) for n in range(6)])
    memory = _Memory(turns=[TurnSummary(
        turn_nr=1, user_input="earlier", intent="ask", topic="t",
    )])

    model = await _run(
        _ctx(hits=long_passages, memory=memory),
        config={"max_system_prompt_tokens": "200"},
    )

    assert _estimated_request_tokens(model.calls[0]) <= 200


async def test_a_bigger_budget_carries_more_and_a_smaller_one_less() -> None:
    """The monotonicity guarantee, observed from the far end of the pipeline.

    Stated as what actually happens rather than as `>=`, which a pipeline that
    never reduced at all would also satisfy: the tight budget carries fewer
    passages and the generous one carries every passage.
    """
    hits = _hits(*[f"passage {n} " + ("y" * 200) for n in range(5)])

    small = await _run(_ctx(hits=hits), config={"max_system_prompt_tokens": "150"})
    large = await _run(_ctx(hits=hits), config={"max_system_prompt_tokens": "600"})

    assert _context_text(small.calls[0]).count("passage ") < 5
    assert _context_text(large.calls[0]).count("passage ") == 5
    assert len(_context_text(large.calls[0])) > len(_context_text(small.calls[0]))


async def test_passthrough_forwards_exactly_what_it_was_given() -> None:
    ctx = _ctx(hits=_hits("something"))
    model = _RecordingModel()
    ctx.llm = model
    composed = (await PromptComposeStep(config={}).run(ctx)).ctx
    expected = list(composed.messages)

    await PassthroughStep(config={}).run(composed)

    assert model.calls[0] == expected, "passthrough changed the messages"


# Which budget wins, and it is not the configuration
# ---------------------------------------------------------------------------

class _BudgetedModel(_RecordingModel):
    """A model that knows its own budget, the way a bound client does."""

    def __init__(self, prompt_budget: int | None) -> None:
        super().__init__()
        # A real TokenBudget, derived the way the executor derives one, so the
        # partitioning under test is the platform's and not the test's.
        self._budget = (
            TokenBudget(context_window=prompt_budget * 5)
            if prompt_budget is not None else None
        )

    @property
    def token_budget(self) -> TokenBudget | None:
        return self._budget


async def _run_with(model: _RecordingModel, ctx: WorkflowStepContext,
                    config: dict | None = None) -> list[CanonicalMessage]:
    ctx.llm = model
    ctx = (await PromptComposeStep(config=config or {}).run(ctx)).ctx
    await PassthroughStep(config={}).run(ctx)
    return model.calls[0]


async def test_the_bound_model_outranks_the_workflow_configuration() -> None:
    """The two are set far apart on purpose.

    Before this, `prompt.compose` budgeted from configuration while the client
    that would make the call held a different number — two truths that agree
    until they do not. The client is the authority: it is the one that will
    enforce `apply_budget_to_messages` afterwards, so a prompt composed against
    anything else is checked against a limit it was never built for.
    """
    hits = _hits(*[f"passage {n} " + ("z" * 300) for n in range(6)])
    generous_config = {"max_system_prompt_tokens": "10000"}

    # The client says 600 tokens of context window → a prompt_budget of 120.
    tight = _BudgetedModel(prompt_budget=120)
    messages = await _run_with(tight, _ctx(hits=hits), generous_config)

    assert _estimated_request_tokens(messages) <= 120, (
        "the configured 10000 won; the bound model's budget was ignored"
    )


async def test_a_client_with_no_budget_falls_back_to_the_configuration() -> None:
    """A defined answer rather than an invented default.

    `token_budget` returning `None` means "no budget from here" — a provider with
    no known context window, or no model bound at all. The configuration is then
    what there is, which is why it stays a fallback rather than being removed.
    """
    hits = _hits(*[f"passage {n} " + ("z" * 300) for n in range(6)])

    silent = _BudgetedModel(prompt_budget=None)
    messages = await _run_with(silent, _ctx(hits=hits), {"max_system_prompt_tokens": "150"})

    assert _estimated_request_tokens(messages) <= 150


async def test_with_neither_a_client_budget_nor_a_configured_one_nothing_is_invented() -> None:
    hits = _hits("a short passage")

    silent = _BudgetedModel(prompt_budget=None)
    messages = await _run_with(silent, _ctx(hits=hits))

    assert "a short passage" in _context_text(messages), "no budget means no reduction, not a guess"


async def test_a_different_bound_model_gives_a_different_prompt() -> None:
    """Compose sees the budget of the model actually bound to this run.

    The same workflow, the same passages, two models with different context
    windows — and the prompts differ accordingly. If they did not, the budget
    would be coming from somewhere other than the client.
    """
    hits = _hits(*[f"passage {n} " + ("z" * 300) for n in range(6)])

    small = await _run_with(_BudgetedModel(prompt_budget=120), _ctx(hits=hits))
    large = await _run_with(_BudgetedModel(prompt_budget=2000), _ctx(hits=hits))

    assert _estimated_request_tokens(small) < _estimated_request_tokens(large)
    assert _context_text(small).count("passage ") < _context_text(large).count("passage ")
