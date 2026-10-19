# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Adapt retrieved passages into budgetable mental elements."""

from __future__ import annotations

from collections.abc import Sequence

from nlght.core.context.passage import ContextPassage
from nlght.core.hive_mind.models import (
    Level,
    MentalElement,
    Presentation,
    Representation,
    Retention,
    estimate_tokens,
)

#: Where retrieved material goes in the prompt. After the session's own memory,
#: because what the corpus says about the question reads as support for it rather
#: than as the state of the conversation.
SECTION_RETRIEVED = Presentation("Retrieved from the corpus:", order=60)


def passage_elements(
    passages: Sequence[ContextPassage],
    *,
    section: Presentation = SECTION_RETRIEVED,
    retention: Retention = Retention.USEFUL,
) -> list[MentalElement]:
    """Adapt ranked passages for shared relevance and budget reduction.

    Args:
        passages (Sequence[ContextPassage]): Ranked context passages.
        section (Presentation): Presentation section for each element.
        retention (Retention): Retention policy during budget reduction.

    Returns:
        list[MentalElement]: Full-or-omit elements with positional relevance and provenance.
    """
    total = len(passages)
    elements: list[MentalElement] = []
    for passage in passages:
        text = f"  - {passage.text}"
        elements.append(MentalElement(
            element_id=f"passage:{passage.carrier}:{passage.carrier_id}",
            kind="passage",
            representations=(
                Representation(level=Level.FULL, text=text, cost=estimate_tokens(text)),
                Representation(level=Level.OMIT, text="", cost=0),
            ),
            presentation=section,
            retention=retention,
            # 1 for the first of n, falling to just above 0 for the last. A
            # position turned into an ordering, not a score turned into one.
            relevance=(total - passage.rank + 1) / total if total else 0.0,
            provenance=passage.provenance,
            payload=passage,
        ))
    return elements
