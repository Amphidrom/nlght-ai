# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Findings become budgeted passages, and nothing more.

Retrieval answers "these systems found these things, in this order". This turns
that into the list of texts a caller may actually spend, and it is deliberately
the dullest step in the pipeline: it adds no text, removes no text from within a
passage, and computes no relevance of its own.

**No expansion.** `passage.text` is exactly `hit.content`. Growing a chunk into
its neighbours is not possible from what the pipeline indexes — the Qdrant
payload carries no position and no offsets, though `DocumentChunk` computes both
— and growing it into its whole document would inflate a finding the retrieval
deliberately narrowed to one chunk. Neither is half-implemented here; the gap is
recorded instead.

**No interpretation of overlap.** A document and a chunk of it are two findings.
That their text overlaps does not make either redundant, and deciding which is
"better" is a judgement this layer has no ground for. Only the same carrier, the
same id and the same revision are one passage.

**Characters, not tokens.** The budget measures `len(text)`. The platform has a
`TokenBudget` (`core/model/budget.py`) that partitions a model's context window,
and two places that estimate tokens from characters at 3.5 — this adds a third of
neither. The translation belongs where the model is known:

    model context window  →  TokenBudget  →  reserve prompt and answer overhead
                          →  translate the remaining share into max_chars
                          →  ContextBudget

So this layer never claims a precision it does not have. `max_chars` is the
length of the passage texts, not estimated model tokens and not the overhead of
whatever prompt they later land in.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace

from nlght.core.retrieval.fusion import FusedHit
from nlght.core.retrieval.hit import Provenance, RetrievalHit


@dataclass(slots=True, frozen=True)
class ContextPassage:
    """One finding, as text a caller may spend, with its provenance intact.

    The provenance is the retrieval `Provenance` unchanged rather than a flat
    copy of its fields: a second copy is a second thing to keep in sync, and
    what a citation needs is exactly what the store recorded.

    It is data and never a rendered citation. `[1] foo.adoc:23` is a decision
    about a prompt — which labels, which order, which format — and it is made
    where the prompt is written, with everything else that goes in it.

    Only what a store actually recorded is carried. A knowledge hit has an
    assertion id and nothing else today, so a passage from one has an empty
    path: showing a line reference the corpus cannot support would be a citation
    to something nobody can check.
    """

    text: str
    carrier: str
    carrier_id: str
    #: Which systems found it. A finding two stores made is one passage that
    #: names both, because the second finding is evidence and not a duplicate.
    found_by: tuple[str, ...]
    #: Position in the retrieval ranking, 1-based. A position, not a score:
    #: nothing here recomputes relevance, and this layer would have nothing to
    #: recompute it from.
    rank: int
    provenance: Provenance = field(default_factory=Provenance)

    @property
    def cost(self) -> int:
        """What spending this passage costs against a budget."""
        return len(self.text)


@dataclass(slots=True, frozen=True)
class ContextBudget:
    """How much passage text a caller can afford.

    In characters, and the name says so. See the module docstring for why this
    is not a token count and where the translation from one belongs.
    """

    max_chars: int

    def __post_init__(self) -> None:
        if self.max_chars < 0:
            raise ValueError("context max_chars must not be negative")


@dataclass(slots=True, frozen=True)
class ContextSelection:
    """What fitted, what did not, and what it cost."""

    passages: tuple[ContextPassage, ...] = ()
    #: The passages the budget did not reach, in the order they were ranked, so
    #: a caller can say what it left out rather than only how much.
    omitted: tuple[ContextPassage, ...] = ()
    used_chars: int = 0


def passages_from(fused: Sequence[FusedHit]) -> list[ContextPassage]:
    """The ranked findings as passages, in the order retrieval put them.

    Deduplication is by identity and by identity alone: the same carrier, the
    same id, the same revision. Where two stores found it, they are united into
    one passage naming both — that is what `FusedHit` already grouped, and
    nothing here regroups it.

    A fused finding whose stores disagree about the *revision* is split rather
    than merged. Two revisions of one carrier are two states, and picking one
    would be this layer deciding which state the corpus is in — a question the
    knowledge model answers elsewhere and retrieval must not answer twice.
    """
    passages: list[ContextPassage] = []
    for found in fused:
        by_revision: dict[str, list[RetrievalHit]] = {}
        for hit in found.hits:
            by_revision.setdefault(hit.provenance.state_revision, []).append(hit)
        for revision in sorted(by_revision):
            # Rebuilt as a `FusedHit` so `best` and `sources` decide this the
            # one way they are already defined, rather than a second rule that
            # agrees with them until it does not.
            group = replace(found, hits=tuple(by_revision[revision]))
            best = group.best
            passages.append(
                ContextPassage(
                    text=best.content,
                    carrier=found.carrier,
                    carrier_id=found.key[1],
                    found_by=group.sources,
                    # Filled in below, once the whole list is in its final order.
                    rank=0,
                    provenance=best.provenance,
                )
            )
    return [replace(passage, rank=position) for position, passage in enumerate(passages, 1)]


def select(
    passages: Sequence[ContextPassage], budget: ContextBudget
) -> ContextSelection:
    """The longest prefix of the ranking that fits, whole passages only.

    Two rules, and the second follows from the first being taken seriously.

    **A passage that does not fit is not cut down.** Half a passage is text that
    never existed as a finding, ending wherever the budget happened to run out —
    possibly mid-sentence, possibly mid-negation. A citation would then point at
    a source that does not say what was quoted.

    **And the selection stops there rather than skipping ahead.** Taking a later,
    smaller passage instead would make the result depend on the sizes of things
    rather than on the ranking: a smaller budget could then drop a passage from
    the *middle* of the list, and two budgets would no longer be comparable. A
    prefix is what makes "less budget means less context, from the bottom" true.

    The cost is stated rather than hidden: a large passage in second place can
    leave the budget unspent. That is visible in `omitted` and is the price of a
    selection anybody can explain.
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
    """Ranked findings in, budgeted passages out."""
    return select(passages_from(fused), budget)
