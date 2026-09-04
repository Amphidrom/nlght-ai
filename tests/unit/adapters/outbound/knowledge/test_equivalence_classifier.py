# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Reading a model's answer about two complete propositions.

What the adapter does with a *non-answer* matters as much as what it does with an
answer, and it is the whole reason this file is long:

    ambiguous   a judge looked and could not tell
    unjudged    no judgement happened

A backend that timed out did not weigh the claims and find them hard — it did not
weigh them. Recording that as `ambiguous` would put a deliberation in the trail
that never took place, and the trail's value is that it reports what happened.

So every technical non-answer lands on `unjudged`, and only `yes`, `no` and
`uncertain` from a model that actually replied produce a verdict.
"""

from __future__ import annotations

import pytest

from nlght.adapters.outbound.knowledge.equivalence_classifier import (
    ModelPropositionEquivalence,
)
from nlght.core.knowledge import Proposition
from nlght.core.knowledge.equivalence import (
    AMBIGUOUS,
    DIFFERENT,
    SAME,
    UNJUDGED,
    AssertionObservation,
)
from nlght.ports.outbound.model_client import ModelStreamEvent


class _Llm:
    def __init__(self, response: str = "yes", fail: bool = False) -> None:
        self.response = response
        self.fail = fail
        self.prompts: list[str] = []
        self.temperature: float | None = None

    def stream(self, messages, tools=None, tool_choice=None, *, temperature=None):  # noqa: ANN001, ANN201
        self.prompts.append(str(messages[0]["content"]))
        self.temperature = temperature
        if self.fail:
            raise RuntimeError("backend unreachable")

        async def _gen():  # noqa: ANN202
            yield ModelStreamEvent(kind="token", content=self.response)
            yield ModelStreamEvent(kind="done")

        return _gen()


def _p(**fields: object) -> AssertionObservation:
    """A fact's claim: its proposition, its kind, and the sentence it came from."""
    return AssertionObservation.of(Proposition(fields))


_A = _p(predicate="transfers", sender="Alice", recipient="Bob")
_B = _p(predicate="transfers", actor="Alice", target="Bob")


async def test_a_plain_answer_is_taken_as_given() -> None:
    assert await ModelPropositionEquivalence(_Llm("yes")).classify(_A, _B) == SAME
    assert await ModelPropositionEquivalence(_Llm("no")).classify(_A, _B) == DIFFERENT


async def test_the_only_doubt_a_model_may_express_is_uncertain() -> None:
    assert await ModelPropositionEquivalence(_Llm("uncertain")).classify(_A, _B) == AMBIGUOUS


@pytest.mark.parametrize("response", ["", "maybe", "possibly the same", "YES?!?", "1"])
async def test_an_answer_that_is_not_one_of_the_three_is_no_judgement(response: str) -> None:
    """Not doubt. Nothing decided anything, so nothing may be recorded as having.

    An unreadable reply is the model failing to answer, which is a different fact
    from the model answering "I cannot tell" — and the second is worth recording
    as a deliberation while the first is not.
    """
    assert await ModelPropositionEquivalence(_Llm(response)).classify(_A, _B) == UNJUDGED


async def test_an_unreachable_backend_is_no_judgement_either() -> None:
    """The correction that matters most here.

    It would have been `ambiguous` by analogy with the semantic-change
    classifier, and that analogy is wrong: there, doubt and failure both mean
    "have a person look", so collapsing them costs nothing. Here they are
    different entries in a record somebody will read as evidence.
    """
    assert await ModelPropositionEquivalence(_Llm(fail=True)).classify(_A, _B) == UNJUDGED


async def test_both_propositions_reach_the_model_whole() -> None:
    llm = _Llm()

    await ModelPropositionEquivalence(llm).classify(_A, _B)

    prompt = llm.prompts[0]
    assert "sender" in prompt and "recipient" in prompt
    assert "actor" in prompt and "target" in prompt


async def test_one_pair_is_one_question_whichever_way_round_it_is_asked() -> None:
    """Otherwise a pair could get two answers depending on how it was held.

    The record would then have two rows for one question and no way to show why
    they differ.
    """
    forward, backward = _Llm(), _Llm()

    await ModelPropositionEquivalence(forward).classify(_A, _B)
    await ModelPropositionEquivalence(backward).classify(_B, _A)

    assert forward.prompts == backward.prompts


async def test_the_question_is_asked_at_zero_temperature() -> None:
    # An answer that varied between runs would merge and unmerge claims nobody
    # edited.
    llm = _Llm()

    await ModelPropositionEquivalence(llm).classify(_A, _B)

    assert llm.temperature == 0.0


async def test_a_model_that_thinks_aloud_is_read_by_its_answer() -> None:
    llm = _Llm(
        "The roles could be the same, or this could be a swap, so maybe no.\n"
        "Answer: **yes**"
    )

    assert await ModelPropositionEquivalence(llm).classify(_A, _B) == SAME


async def test_the_prompt_says_that_roles_are_part_of_the_claim() -> None:
    """Because a swap and a rename look identical, and end opposite.

    Nothing in the structure separates them, so the only place the
    directed-by-default decision can be enforced is in the question itself.
    """
    llm = _Llm()

    await ModelPropositionEquivalence(llm).classify(_A, _B)

    assert "exchanged" in llm.prompts[0]
    assert "Different\nnames for the same role" in llm.prompts[0]


async def test_the_judge_names_itself_completely() -> None:
    # A verdict without its judge cannot be revisited when the judge changes.
    judge = ModelPropositionEquivalence(_Llm(), model="qwen3.8:27B").judge

    assert judge.classifier == "proposition-equivalence"
    assert judge.model == "qwen3.8:27B"
    assert judge.version
    assert str(judge) == "proposition-equivalence/qwen3.8:27B/1"
