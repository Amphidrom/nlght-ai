# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Which stores get asked, and how far down each list is fetched.

Two separate questions, and conflating them is what made a first attempt refuse
to run:

    requested   what this workflow wants searched
    available   what this platform actually has

A search runs over the intersection. A platform with a lexical index and a
knowledge store but no vectors is a perfectly ordinary platform, and telling its
operator that they have misconfigured something is inventing an error they did
not make. A source that is genuinely indispensable to a workflow says so —
`required` — and then its absence is a real failure.

The three cut-offs are also separate, and were not:

    fetch_k    how deep each store is asked
    fusion_k   how many of each store's hits take part in the ranking
    final_k    how many findings leave the step

Setting `fetch_k` to `final_k` looks tidy and quietly destroys evidence. A hit
the lexical index put 20th and the knowledge store put 1st is exactly the case
fusion exists to catch, and fetching ten from each throws away the half of it
that would have made the point.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from nlght.core.retrieval.hit import KNOWLEDGE, LEXICAL, SOURCES, VECTOR

#: Asked in this order. Lexical first because it is free and its result decides
#: whether the vector store is worth a model round trip.
_ORDER = (LEXICAL, VECTOR, KNOWLEDGE)


class RetrievalPlanError(ValueError):
    """A plan that cannot be honoured as written.

    Raised for a source that does not exist and for a *required* source that is
    not available. Never for a source that is merely absent: a search silently
    missing a store looks exactly like a search finding little, and a search
    refusing to run because a platform lacks an optional index is worse.
    """


@dataclass(slots=True, frozen=True)
class RetrievalPlan:
    """What to search, how deep, and what has to be there."""

    requested: tuple[str, ...] = SOURCES
    available: tuple[str, ...] = SOURCES
    required: tuple[str, ...] = ()
    #: How deep each store is asked. Deliberately larger than what leaves the
    #: step: evidence a second store agrees with is often well down the first
    #: store's list.
    fetch_k: Mapping[str, int] = field(
        default_factory=lambda: {LEXICAL: 50, VECTOR: 50, KNOWLEDGE: 20}
    )
    #: How many of each store's hits take part in the ranking. `0` means all of
    #: what was fetched, which is the honest default — cutting before fusing
    #: discards evidence for no reason but tidiness.
    fusion_k: int = 0
    #: How many findings leave the step.
    final_k: int = 20
    kinds: tuple[str, ...] = ("fact", "rule", "pattern", "decision")
    #: Below the floor or above the ceiling of lexical hits, the vector store is
    #: worth its model round trip.
    lexical_floor: int = 1
    lexical_ceiling: int = 100

    def __post_init__(self) -> None:
        missing = [source for source in self.required if source not in self.available]
        if missing:
            raise RetrievalPlanError(
                f"required retrieval source(s) unavailable: {', '.join(sorted(missing))}"
            )

    @property
    def sources(self) -> tuple[str, ...]:
        """What will actually be searched: what was asked for, and exists."""
        return tuple(
            source for source in _ORDER
            if source in self.requested and source in self.available
        )

    @property
    def skipped(self) -> tuple[str, ...]:
        """Asked for and not present. Reported, never an error on its own."""
        return tuple(
            source for source in _ORDER
            if source in self.requested and source not in self.available
        )

    def fetch_for(self, source: str) -> int:
        return max(1, int(self.fetch_k.get(source, 20)))

    def wants_vector(self, lexical_hits: int) -> bool:
        """Whether the vector store is worth embedding the query for.

        Only where it is planned *and* present. Where lexical was not searched
        there is nothing to judge the call against, so it runs — a plan naming
        only the vector store means exactly that.
        """
        if VECTOR not in self.sources:
            return False
        if LEXICAL not in self.sources:
            return True
        return lexical_hits < self.lexical_floor or lexical_hits > self.lexical_ceiling


def plan_from_config(
    config: Mapping[str, object] | None,
    *,
    available: Sequence[str] = SOURCES,
) -> RetrievalPlan:
    """Read a plan, refusing only what genuinely cannot be honoured."""
    values = dict(config or {})
    at_hand = tuple(str(item).strip().lower() for item in available)
    # Naming no sources means "search what this platform has", not "search all
    # three". Defaulting to the full set made every absent store show up as
    # requested-and-missing, which reads as a misconfiguration the operator
    # never made.
    requested = _sources(values.get("sources"), "sources") or at_hand
    required = _sources(values.get("required_sources"), "required_sources")

    fetch = {LEXICAL: 50, VECTOR: 50, KNOWLEDGE: 20}
    raw_fetch = values.get("fetch_k")
    if isinstance(raw_fetch, Mapping):
        for name, value in raw_fetch.items():
            key = str(name).strip().lower()
            if key not in SOURCES:
                raise RetrievalPlanError(f"fetch_k given for unknown source '{name}'")
            fetch[key] = _int(value, fetch[key])
    elif raw_fetch is not None:
        fetch = dict.fromkeys(SOURCES, _int(raw_fetch, 50))

    kinds = values.get("kinds")
    if isinstance(kinds, str):
        kinds = [kinds]
    resolved = (
        tuple(str(item).strip().lower() for item in kinds if str(item).strip())
        if isinstance(kinds, Sequence)
        else ()
    )

    return RetrievalPlan(
        requested=requested,
        available=at_hand,
        required=required,
        fetch_k=fetch,
        fusion_k=_int(values.get("fusion_k"), 0),
        final_k=_int(values.get("final_k", values.get("limit")), 20),
        kinds=resolved or ("fact", "rule", "pattern", "decision"),
        lexical_floor=_int(values.get("lexical_floor"), 1),
        lexical_ceiling=_int(values.get("lexical_ceiling"), 100),
    )


def _sources(raw: object, name: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        raw = [item for item in raw.replace(",", " ").split() if item]
    if not isinstance(raw, Sequence):
        raise RetrievalPlanError(f"{name} must be a list, got {type(raw).__name__}")
    named = tuple(str(item).strip().lower() for item in raw if str(item).strip())
    unknown = [item for item in named if item not in SOURCES]
    if unknown:
        raise RetrievalPlanError(
            f"unknown retrieval source(s) in {name}: {', '.join(sorted(unknown))}; "
            f"expected any of {', '.join(SOURCES)}"
        )
    return named


def _int(value: object, fallback: int) -> int:
    if not isinstance(value, (str, bytes, bytearray, int, float)):
        return fallback
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback
