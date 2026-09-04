# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Asking a model whether two wordings say the same thing.

The prompt is deliberately small. The model is not asked to extract, summarise or
judge quality — only to answer one question with one word, about two sentences it
is handed. A narrow question is one a small local model can answer reliably, and
one whose answer is cheap to validate: anything that is not one of the three
words is doubt, and doubt is a review.

Pinned at `temperature=0` for the same reason extraction is: an answer that
varies between runs would make a corpus nobody edited drift, which is the defect
this whole design removes.
"""

from __future__ import annotations

import logging

from nlght.core.knowledge.semantics import AMBIGUOUS, NORMATIVE, REWRITE
from nlght.ports.outbound.model_client import ModelClient

logger = logging.getLogger(__name__)

_PROMPT = """Two versions of one sentence from a technical or policy document.

OLD: {old}
NEW: {new}

Did the NEW version change what the sentence requires, permits or asserts?

Answer with exactly one word:
  rewrite    - same meaning, different words
  normative  - the meaning changed: an obligation, permission, default or
               condition is different
  ambiguous  - you cannot tell

Answer:"""

_ANSWERS = {REWRITE, NORMATIVE, AMBIGUOUS}


class ModelPropositionClassifier:
    """The `PropositionClassifier` port, answered by a model."""

    def __init__(self, llm: ModelClient) -> None:
        self._llm = llm

    async def classify(self, old: str, new: str) -> str:
        collected: list[str] = []
        try:
            async for event in self._llm.stream(
                [{"role": "user", "content": _PROMPT.format(old=old, new=new)}],
                temperature=0.0,
            ):
                if event.kind == "token" and event.content:
                    collected.append(event.content)
                elif event.kind == "done":
                    break
        except Exception:  # noqa: BLE001 (any backend failure is doubt, not a guess)
            # A classifier that cannot reach its backend must not decide that
            # nothing changed. Saying so sends the revision to review, which is
            # the direction a failure should fall in.
            logger.warning("knowledge.semantic_change.unavailable", exc_info=True)
            return AMBIGUOUS

        # The last recognised word, so a model that thinks aloud before
        # answering is read by its answer rather than by its reasoning.
        answer = AMBIGUOUS
        for word in "".join(collected).lower().replace("*", " ").split():
            candidate = word.strip(".,:;`'\"()[]")
            if candidate in _ANSWERS:
                answer = candidate
        return answer
