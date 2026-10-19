# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Interpret semantic-change classifications without losing review safety.

Malformed or missing model answers require review rather than silently
classifying an edit as a harmless rewrite."""

from __future__ import annotations

import pytest

from nlght.adapters.outbound.knowledge.semantic_classifier import (
    ModelPropositionClassifier,
)
from nlght.core.knowledge.semantics import AMBIGUOUS, NORMATIVE, REWRITE
from nlght.ports.outbound.model_client import ModelStreamEvent


class _Llm:
    def __init__(self, response: str = "rewrite", fail: bool = False) -> None:
        self.response = response
        self.fail = fail
        self.prompts: list[str] = []

    def stream(self, messages, tools=None, tool_choice=None, **kwargs):
        self.prompts.append(str(messages[0]["content"]))
        self.call_options = kwargs
        if self.fail:
            raise RuntimeError("backend unreachable")

        async def _gen():
            yield ModelStreamEvent(kind="token", content=self.response)
            yield ModelStreamEvent(kind="done")

        return _gen()


async def test_a_plain_answer_is_taken_as_given() -> None:
    llm = _Llm("normative")

    assert await ModelPropositionClassifier(llm).classify("a", "b") == NORMATIVE


async def test_both_sentences_reach_the_model() -> None:
    llm = _Llm()

    await ModelPropositionClassifier(llm).classify(
        "Requests must be signed.", "Requests shall be signed."
    )

    assert "Requests must be signed." in llm.prompts[0]
    assert "Requests shall be signed." in llm.prompts[0]


async def test_the_question_carries_no_sampling_setting() -> None:
    # Determinism is the model provider's configuration (`temperature: 0`), so
    # the classifier must not pass one of its own that a provider would ignore
    # or reject.
    llm = _Llm()

    await ModelPropositionClassifier(llm).classify("a", "b")

    assert llm.call_options == {}


async def test_a_model_that_thinks_aloud_is_read_by_its_answer() -> None:
    # Local reasoning models narrate before answering, and the narration usually
    # contains every one of the three words.
    llm = _Llm(
        "The word 'not' could make this normative, but it might be a rewrite.\n"
        "Answer: **normative**"
    )

    assert await ModelPropositionClassifier(llm).classify("a", "b") == NORMATIVE


@pytest.mark.parametrize("response", ["", "maybe", "yes", "REWRITE?!"])
async def test_an_answer_that_is_not_one_of_the_three_is_doubt(response: str) -> None:
    assert await ModelPropositionClassifier(_Llm(response)).classify("a", "b") == AMBIGUOUS


async def test_an_unreachable_backend_is_doubt_not_agreement() -> None:
    """The failure that would otherwise be silent and expensive.

    A classifier that cannot answer must not decide that nothing changed. Falling
    towards a review costs a minute; falling towards `rewrite` carries an
    approval across an edit nobody saw.
    """
    assert await ModelPropositionClassifier(_Llm(fail=True)).classify("a", "b") == AMBIGUOUS


async def test_a_recognised_answer_is_still_recognised(caplog) -> None:
    assert await ModelPropositionClassifier(_Llm("rewrite")).classify("a", "b") == REWRITE
