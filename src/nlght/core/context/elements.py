# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Retrieved passages as elements a budget can weigh.

The adapter is where retrieval-specific knowledge is allowed to live, and this is
the last place it appears. After here a passage is a `MentalElement` like any
other and is weighed against a remembered fact by the same rules — which is the
point of the whole arrangement (ADR-0054).

Two things it deliberately does not do.

**The fusion rank becomes relevance, never retention.** They answer different
questions and collapsing them is the mistake this design keeps avoiding:

    rank       how well did this match the question, among the things found
    retention  what would it cost to lose this

A passage that ranked first is a good match, not an important one. Retention for
retrieved material is `USEFUL`: it is what a source says about the question, and
losing it costs a less well-founded answer rather than a broken one.

**No compact form is invented.** There is no honest short version of a passage —
truncating produces text that ends mid-claim, and pulling out a sentence is a
judgement about which sentence carries it. A passage is offered whole or not at
all, which the reduction handles because representation levels are optional.
"""

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
    """Passages as elements, in the order retrieval ranked them.

    `relevance` is derived from the fusion position rather than carried as a
    score, because a position is what fusion actually decided (ADR-0050) and the
    numbers behind it were never comparable across stores. First place scores
    highest, and the scale is only ever used to order passages against each other
    and against whatever else is competing for the same budget.

    Provenance travels untouched, so a passage that survives can still say which
    documents currently carry it (ADR-0053) — and a later citation layer has
    everything it needs without this one rendering anything.
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
