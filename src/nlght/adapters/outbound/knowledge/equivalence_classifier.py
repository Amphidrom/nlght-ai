# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Asking a model whether two complete propositions say the same thing.

One question, three answers, and a fourth state the model never gives.

    yes | no | uncertain     what the model may answer
    unjudged                 what happens when no answer was reached

The distinction is the reason this adapter is careful rather than convenient.
`ambiguous` means a judge looked and could not tell. `unjudged` means no judgement
happened at all. A backend that timed out never weighed the claims. Recording
that as `ambiguous` would record a deliberation that never took place. The trail
must report what actually happened.

So every technical non-answer lands on `unjudged`: an unreachable backend, a
timeout, a reply that is not one of the three words. Only a model that answered
produces a verdict.

The pair is put to the model in a canonical order, so the same two propositions
are always asked in the same way. Otherwise one pair could get two answers
depending on which side the caller happened to hold first, and the record would
have no way to show it.

Run it on a model provider configured with `temperature: 0`, for the reason
extraction is: an answer that varied between runs would merge and unmerge claims
nobody edited. Sampling is the provider's configuration, not this call's.
"""

from __future__ import annotations

import json
import logging

from nlght.core.knowledge.equivalence import (
    AMBIGUOUS,
    DIFFERENT,
    SAME,
    UNJUDGED,
    AssertionObservation,
    Judge,
)
from nlght.ports.outbound.model_client import ModelClient

logger = logging.getLogger(__name__)

CLASSIFIER = "proposition-equivalence"
PROMPT_VERSION = "1"

_PROMPT = """Two claims of kind "{kind}", extracted from technical or policy
documents. The field names were chosen by whatever read each sentence and may
differ even where the claim is the same.

A: {left}
   read from: {left_source}
B: {right}
   read from: {right_source}

Do A and B assert the same thing about the world?

"read from" is what the source actually says, word for word. The claim beside it
says which assertion inside that sentence is being asked about. One sentence
often carries several. So A and B may quote the same sentence and still assert
different things, and quoting the same sentence is no evidence at all that they
are the same claim.

Take the roles seriously: who does what to whom is part of the claim, so a claim
with its participants exchanged is a different claim, not a rephrasing. Different
names for the same role are a rephrasing.

Answer with exactly one word:
  yes        the same claim, described differently
  no         different claims
  uncertain  you cannot tell

Answer:"""

#: What the model may say, and what each means to us. `uncertain` is the only
#: doubt a model is allowed to express. Everything unrecognised is a
#: non-decision rather than a doubt.
_ANSWERS = {"yes": SAME, "no": DIFFERENT, "uncertain": AMBIGUOUS}


class ModelPropositionEquivalence:
    """The equivalence judgement, answered by a model."""

    def __init__(self, llm: ModelClient, model: str = "") -> None:
        self._llm = llm
        self._model = model

    @property
    def judge(self) -> Judge:
        return Judge(classifier=CLASSIFIER, model=self._model, version=PROMPT_VERSION)

    async def classify(self, a: AssertionObservation, b: AssertionObservation) -> str:
        """Classify whether two observations express the same assertion.

        Args:
            a (AssertionObservation): First observation to compare.
            b (AssertionObservation): Second observation to compare.

        Returns:
            str: The last recognized model verdict, or ``UNJUDGED`` when the
            model fails or produces no recognized verdict.

        Raises:
            ValueError: The observations have different assertion kinds.

        Side Effects:
            Streams one canonical comparison prompt through the model and logs
            model failures. Canonical ordering makes both input orders equivalent.
        """
        if a.kind != b.kind:
            raise ValueError(
                f"a pair of different kinds is settled without a model, so it "
                f"should never reach the judge: '{a.kind}' and '{b.kind}'"
            )

        def ordering(claim: AssertionObservation) -> str:
            return json.dumps(claim.as_dict(), sort_keys=True, ensure_ascii=False)

        left, right = sorted((a, b), key=ordering)
        prompt = _PROMPT.format(
            kind=left.kind,
            left=ordering(left),
            right=ordering(right),
            left_source=left.observed_text or "(not recorded)",
            right_source=right.observed_text or "(not recorded)",
        )

        collected: list[str] = []
        try:
            async for event in self._llm.stream([{"role": "user", "content": prompt}]):
                if event.kind == "token" and event.content:
                    collected.append(event.content)
                elif event.kind == "done":
                    break
        except Exception:
            # A model failure cannot establish an equivalence judgement.
            logger.warning("knowledge.equivalence.unavailable", exc_info=True)
            return UNJUDGED

        # The last recognized verdict takes precedence over earlier reasoning.
        answer = UNJUDGED
        for word in "".join(collected).lower().replace("*", " ").split():
            candidate = word.strip(".,:;`'\"()[]")
            if candidate in _ANSWERS:
                answer = _ANSWERS[candidate]
        return answer
