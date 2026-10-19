# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Convert ranked findings into whole passages within a character budget."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace

from nlght.core.retrieval.fusion import FusedHit
from nlght.core.retrieval.hit import Provenance, RetrievalHit


@dataclass(slots=True, frozen=True)
class ContextPassage:
    """One store finding as text with its unchanged retrieval provenance.

    Keep citations outside this data type. A citation renderer can use the
    current support recorded for a knowledge assertion without inventing a
    path or line reference from the passage text.
    """

    text: str
    carrier: str
    carrier_id: str
    #: Preserve every contributing source because agreement between stores is
    #: evidence for one carrier, not a duplicate finding.
    found_by: tuple[str, ...]
    #: Keep rank as a position because store score scales are incomparable.
    rank: int
    provenance: Provenance = field(default_factory=Provenance)

    @property
    def cost(self) -> int:
        """Measure a passage against a character budget.

        Returns:
            int: Character count of the text.
        """
        return len(self.text)


@dataclass(slots=True, frozen=True)
class ContextBudget:
    """Character limit for selected passage text, independent of model tokens."""

    max_chars: int

    def __post_init__(self) -> None:
        if self.max_chars < 0:
            raise ValueError("context max_chars must not be negative")


@dataclass(slots=True, frozen=True)
class ContextSelection:
    """What fitted, what did not, and what it cost."""

    passages: tuple[ContextPassage, ...] = ()
    #: Retain omitted passage identities in rank order so callers can explain
    #: what the budget excluded.
    omitted: tuple[ContextPassage, ...] = ()
    used_chars: int = 0


def passages_from(fused: Sequence[FusedHit]) -> list[ContextPassage]:
    """Convert fused findings into revision-specific passages.

    Args:
        fused (Sequence[FusedHit]): Ranked fused retrieval hits.

    Returns:
        list[ContextPassage]: Ordered passages, split when constituent revisions differ.
    """
    passages: list[ContextPassage] = []
    for found in fused:
        by_revision: dict[str, list[RetrievalHit]] = {}
        for hit in found.hits:
            by_revision.setdefault(hit.provenance.state_revision, []).append(hit)
        for revision in sorted(by_revision):
            # Reuse FusedHit selection so revision splitting cannot introduce
            # a different best-hit rule.
            group = replace(found, hits=tuple(by_revision[revision]))
            best = group.best
            passages.append(
                ContextPassage(
                    text=best.content,
                    carrier=found.carrier,
                    carrier_id=found.key[1],
                    found_by=group.sources,
                    # Assign ranks after revision splitting establishes final order.
                    rank=0,
                    provenance=best.provenance,
                )
            )
    return [replace(passage, rank=position) for position, passage in enumerate(passages, 1)]


def select(
    passages: Sequence[ContextPassage], budget: ContextBudget
) -> ContextSelection:
    """Keep the longest prefix of whole ranked passages within a budget.

    Args:
        passages (Sequence[ContextPassage]): Passages in ranking order.
        budget (ContextBudget): Character budget for selected text.

    Returns:
        ContextSelection: Selected and omitted passages with the number of characters used.
    """
    taken: list[ContextPassage] = []
    used = 0
    for position, passage in enumerate(passages):
        if used + passage.cost > budget.max_chars:
            return ContextSelection(
                passages=tuple(taken),
                omitted=tuple(passages[position:]),
                used_chars=used,
            )
        taken.append(passage)
        used += passage.cost
    return ContextSelection(passages=tuple(taken), omitted=(), used_chars=used)


def build_context(
    fused: Sequence[FusedHit], budget: ContextBudget
) -> ContextSelection:
    """Build budgeted context passages from fused retrieval hits.

    Args:
        fused (Sequence[FusedHit]): Fused hits in ranking order.
        budget (ContextBudget): Character budget for selected passages.

    Returns:
        ContextSelection: Selected passage prefix and omitted remainder.
    """
    return select(passages_from(fused), budget)
