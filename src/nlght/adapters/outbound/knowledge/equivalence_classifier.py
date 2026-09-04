# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Asking a model whether two complete propositions say the same thing.

One question, three answers, and a fourth state the model never gives.

    yes | no | uncertain     what the model may answer
    unjudged                 what happens when no answer was reached

The distinction is the reason this adapter is careful rather than convenient.
`ambiguous` means a judge looked and could not tell; `unjudged` means no judgement
happened at all. A backend that timed out did not weigh the claims and find them
hard — it did not weigh them. Recording that as `ambiguous` would put a
deliberation in the trail that never took place, and the trail's whole value is
that it reports what actually happened.

So every technical non-answer lands on `unjudged`: an unreachable backend, a
timeout, a reply that is not one of the three words. Only a model that answered
produces a verdict.

The pair is put to the model in a canonical order, so the same two propositions
are always asked in the same way. Otherwise one pair could get two answers
depending on which side the caller happened to hold first, and the record would
have no way to show it.

Pinned at `temperature=0` for the reason extraction is: an answer that varied
between runs would merge and unmerge claims nobody edited.
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
says which assertion inside that sentence is being asked about — one sentence
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
#: doubt a model is allowed to express; everything unrecognised is a
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
        """Whether these two observations are one assertion.

        One question and one prompt, for every kind. Four prompts would grow
        four slightly different notions of truth, and "is this the same
        assertion" has to mean one thing.

        A pair of different kinds never arrives here: `settled` answers it `no`
        without a model, because `kind` is part of the identity namespace and the
        answer is already known. The guard below is a programming check, not a
        classification.

        The pair is put in a canonical order so one pair is one question however
        the caller held it. Both sides carry their `observed_text` — what the
        source actually says — and their representation, which says *which*
        assertion inside that sentence is being judged.
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
            async for event in self._llm.stream(
                [{"role": "user", "content": prompt}], temperature=0.0
            ):
                if event.kind == "token" and event.content:
                    collected.append(event.content)
                elif event.kind == "done":
                    break
        except Exception:  # noqa: BLE001 (any backend failure is a non-decision)
            # Not doubt. Nothing weighed these claims, so nothing may be recorded
            # as having weighed them.
            logger.warning("knowledge.equivalence.unavailable", exc_info=True)
            return UNJUDGED

        # The last recognised word, so a model that thinks aloud before answering
        # is read by its answer rather than by its reasoning.
        answer = UNJUDGED
        for word in "".join(collected).lower().replace("*", " ").split():
            candidate = word.strip(".,:;`'\"()[]")
            if candidate in _ANSWERS:
                answer = _ANSWERS[candidate]
        return answer
