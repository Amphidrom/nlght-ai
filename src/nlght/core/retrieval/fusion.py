# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Fuse store rankings without comparing incompatible score scales."""

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
        """Resolve per-store weights and rank damping from configuration.

        Args:
            config (Mapping[str, object] | None): Optional configuration mapping.

        Returns:
            FusionWeights: Fusion weights with defaults for unspecified stores.

        Side Effects:
            None. Invalid stores or damping raise ValueError.
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
    because the chunk's text and the document's fields are not interchangeable.
    """

    key: tuple[str, str]
    score: float
    hits: tuple[RetrievalHit, ...]

    @property
    def best(self) -> RetrievalHit:
        """Select the highest-ranked source hit for display.

        Returns:
            RetrievalHit: Hit with the lowest source rank and source-name tie-breaker.
        """
        return min(self.hits, key=lambda hit: (hit.source_rank, hit.source))

    @property
    def sources(self) -> tuple[str, ...]:
        return tuple(sorted({hit.source for hit in self.hits}))

    @property
    def carrier(self) -> str:
        return self.key[0]

    @property
    def contributions(self) -> Mapping[str, int]:
        """Expose source ranks behind a fused finding.

        Returns:
            Mapping[str, int]: Source names mapped to their source ranks.
        """
        return {hit.source: hit.source_rank for hit in self.hits}


def rank_within_sources(hits: Sequence[RetrievalHit]) -> list[RetrievalHit]:
    """Assign ranks within each source and discard duplicate carriers.

    Args:
        hits (Sequence[RetrievalHit]): Source hits in each store's returned order.

    Returns:
        list[RetrievalHit]: Hits with one-based source ranks and first duplicate retained.
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
    """Combine ranked findings through reciprocal rank fusion.

    Args:
        hits (Sequence[RetrievalHit]): Ranked findings from the retrieval sources.
        weights (FusionWeights | None): Optional source weights and damping settings.

    Returns:
        list[FusedHit]: Deterministically ordered fused hits with one vote per source.

    Raises:
        ValueError: A finding has no source rank.
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
        # voting twice. `rank_within_sources` drops the duplicate. This keeps
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
