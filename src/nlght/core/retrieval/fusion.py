# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Combining rankings from systems whose scores mean different things.

The contract, and everything here follows from it:

    a store decides local relevance
    retrieval fuses rankings
    retrieval never reads store scores as a shared unit

BM25 lands near 7, cosine below 1, a confidence is a probability. Weighting
those raw numbers lets one store shout over another for no reason but its scale
— which a first attempt here did, and a document carrying none of the question
outranked an assertion that carried it, purely on BM25 magnitude.

Normalising per store does not fix it either, and that is the subtler trap. Over

    store A   7.1, 7.0, 6.9      three near-identical results
    store B   0.81, 0.42, 0.10   one good result and two poor ones

min-max gives both a 1.0 at the top and a 0.0 at the bottom, inventing a spread
where there was none and flattening one where there was. The shape of a store's
score distribution is not information about the question.

So only **position** is fused, by reciprocal rank:

    fused(hit) = Σ  weight(source) / (k + rank_in_that_source)

`k` damps the top: without it the first place is worth twice the second and
nothing below fourth matters. It is the one tuning knob, and it is configuration.

What this buys is stated as invariants and tested as such: scaling or shifting a
store's scores changes nothing; a store returning a long tail of poor results
cannot outvote a store returning three good ones; a hit found by two systems is
strengthened by the second; and the order is deterministic.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace

from nlght.core.retrieval.hit import SOURCES, RetrievalHit

#: What each store's opinion is worth. Equal by default: the platform has no
#: ground for preferring one index over another, and a preference is exactly the
#: kind of thing an operator should state rather than inherit.
DEFAULT_SOURCE_WEIGHTS: Mapping[str, float] = dict.fromkeys(SOURCES, 1.0)

#: How hard the top of each store's list is damped. 60 is the value reciprocal
#: rank fusion was published with and it behaves sanely for lists of tens: first
#: place is worth about 1.6% more than second, so a store is not decided by its
#: single best guess.
DEFAULT_DAMPING = 60.0


def _float(value: object) -> float:
    if not isinstance(value, (str, bytes, bytearray, int, float)):
        raise TypeError
    return float(value)


@dataclass(slots=True, frozen=True)
class FusionWeights:
    """What each store contributes, and how far down its list still counts."""

    sources: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_SOURCE_WEIGHTS)
    )
    damping: float = DEFAULT_DAMPING

    def of(self, source: str) -> float:
        return float(self.sources.get(source, 1.0))

    @classmethod
    def from_config(cls, config: Mapping[str, object] | None) -> FusionWeights:
        """Read weights from step configuration, per store and falling back.

        Naming one store replaces that store's weight and leaves the others, so
        raising `knowledge` does not silently mute `lexical`. A weight of zero
        means the store contributes nothing to the ranking, which is a thing an
        operator may legitimately want and is not the same as not searching it —
        its hits still arrive, with their provenance, and can still be shown.
        """
        if not config:
            return cls()
        weights = dict(DEFAULT_SOURCE_WEIGHTS)
        raw = config.get("source_weights")
        if isinstance(raw, Mapping):
            for name, value in raw.items():
                key = str(name).strip().lower()
                if key not in SOURCES:
                    raise ValueError(
                        f"weight given for unknown retrieval source '{name}'; "
                        f"expected any of {', '.join(SOURCES)}"
                    )
                try:
                    weights[key] = _float(value)
                except (TypeError, ValueError) as error:
                    raise ValueError(f"weight for '{name}' is not a number") from error
        damping = config.get("damping")
        try:
            damped = _float(damping) if damping is not None else DEFAULT_DAMPING
        except (TypeError, ValueError):
            damped = DEFAULT_DAMPING
        if damped <= 0:
            raise ValueError("damping must be positive; it is added to a rank")
        return cls(sources=weights, damping=damped)


@dataclass(slots=True, frozen=True)
class FusedHit:
    """One finding, and every store that made it.

    The hits are kept whole rather than summarised. A finding that two systems
    made is two hits with two provenances, and context building later needs both
    — the chunk's text and the document's fields are not interchangeable.
    """

    key: tuple[str, str]
    score: float
    hits: tuple[RetrievalHit, ...]

    @property
    def best(self) -> RetrievalHit:
        """The hit its own store ranked highest, which is the one to show."""
        return min(self.hits, key=lambda hit: (hit.source_rank, hit.source))

    @property
    def sources(self) -> tuple[str, ...]:
        return tuple(sorted({hit.source for hit in self.hits}))

    @property
    def carrier(self) -> str:
        return self.key[0]

    @property
    def contributions(self) -> Mapping[str, float]:
        """What each store contributed, so a ranking can be read rather than trusted."""
        return {hit.source: hit.source_rank for hit in self.hits}


def rank_within_sources(hits: Sequence[RetrievalHit]) -> list[RetrievalHit]:
    """Fill in `source_rank` from the order each store answered in.

    A store's own ordering *is* its ranking, so a caller that preserved the
    order does not have to count. Where a store returned one carrier twice, the
    better-ranked appearance is the one kept: counting it twice would let a
    store vote for the same thing repeatedly, which is the one way a single
    store could outweigh the others without saying anything more.
    """
    ranked: list[RetrievalHit] = []
    positions: dict[str, int] = {}
    seen: dict[tuple[str, tuple[str, str]], int] = {}
    for hit in hits:
        marker = (hit.source, hit.key)
        if marker in seen:
            continue
        positions[hit.source] = positions.get(hit.source, 0) + 1
        seen[marker] = positions[hit.source]
        ranked.append(replace(hit, source_rank=positions[hit.source]))
    return ranked


def fuse(
    hits: Sequence[RetrievalHit],
    *,
    weights: FusionWeights | None = None,
) -> list[FusedHit]:
    """Rank findings across stores by position, never by score.

    Ordering is total and deterministic: the fused score, then the best rank any
    store gave it, then the carrier and its id. Two runs over the same input
    produce the same list, which is what makes a golden ranking test meaningful
    at all.
    """
    scale = weights or FusionWeights()
    grouped: dict[tuple[str, str], list[RetrievalHit]] = {}
    for hit in hits:
        rank = hit.source_rank
        if rank <= 0:
            raise ValueError(
                f"hit {hit.key} from '{hit.source}' has no source rank; "
                f"call rank_within_sources() before fusing"
            )
        grouped.setdefault(hit.key, []).append(hit)

    fused: list[FusedHit] = []
    for key, found in grouped.items():
        # One store contributing the same carrier twice would be one store
        # voting twice. `rank_within_sources` drops the duplicate; this keeps
        # the guarantee even for a caller that ranked its hits itself.
        by_source: dict[str, RetrievalHit] = {}
        for hit in found:
            held = by_source.get(hit.source)
            if held is None or hit.source_rank < held.source_rank:
                by_source[hit.source] = hit
        score = sum(
            scale.of(source) / (scale.damping + hit.source_rank)
            for source, hit in by_source.items()
        )
        fused.append(
            FusedHit(key=key, score=score, hits=tuple(by_source.values()))
        )

    fused.sort(
        key=lambda item: (
            -item.score,
            min(hit.source_rank for hit in item.hits),
            item.key,
        )
    )
    return fused
