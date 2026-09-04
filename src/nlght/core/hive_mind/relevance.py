# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""RelevanceEngine — scores knowledge objects on four dimensions.

Dimensions:
    Recency     — how recently was the object updated?
    Proximity   — how close are the object's entities to the current signal?
    Volatility  — how often has the object been mutated?
    Dependency  — is the object in the dependency chain of the current intent?

Usage::

    engine = RelevanceEngine(weights=RelevanceWeights(), strategy=ScoringStrategy.WEIGHTED_SIGMOID)
    score  = engine.score_atom(atom, signal_entities, intent, now)
    score  = engine.score_result(result, signal_entities, intent, now)
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Any

from nlght.core.hive_mind.models import (
    LEVELS,
    Level,
    MentalElement,
    RelevanceInputs,
    RelevanceScore,
    RelevanceWeights,
    Representation,
    Retention,
    ScoredAtom,
    ScoredResult,
    ScoringStrategy,
)

logger = logging.getLogger(__name__)

# Half-life in seconds — after this time, recency score is 0.5 (5 minutes)
_RECENCY_HALF_LIFE_SECONDS = 300.0


def _recency_score(updated_at: datetime | None, now: datetime) -> float:
    if updated_at is None:
        return 1.0
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    age_seconds = max(0.0, (now - updated_at).total_seconds())
    return math.pow(0.5, age_seconds / _RECENCY_HALF_LIFE_SECONDS)


def _proximity_score(obj_entities: list[str], signal_entities: list[str], intent: str) -> float:
    if not obj_entities:
        return 0.1
    signal_set = {e.lower() for e in signal_entities}
    obj_set    = {e.lower() for e in obj_entities}
    overlap = len(obj_set & signal_set)
    if overlap == 0:
        intent_lower = intent.lower()
        if any(intent_lower in e or e in intent_lower for e in obj_set):
            return 0.35
        return 0.05
    recall    = overlap / len(obj_set)
    precision = overlap / len(signal_set) if signal_set else 0.0
    return min(1.0, 0.6 * recall + 0.4 * precision + 0.1 * overlap)


def _volatility_score(mutation_count: int) -> float:
    if mutation_count <= 0:
        return 0.0
    return min(1.0, math.log1p(mutation_count) / math.log1p(10))


def _dependency_score(obj_id: str, dependency_ids: set[str]) -> float:
    if not dependency_ids:
        return 0.0
    return 1.0 if obj_id in dependency_ids else 0.0


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class RelevanceEngine:
    """Scores KnowledgeAtoms and SessionResults for relevance to the current signal.

    Parameters:
        weights:   Dimension weights (must sum to 1.0).
        strategy:  Scoring formula (linear / weighted_sigmoid / multiplicative).

    There is no threshold. It existed to discard anything scoring below 0.15, and
    that is a decision this class is in the wrong place to make: a low score says
    "this does not match the moment", which is not the same as "this may be
    forgotten". What may be forgotten depends on retention and on what else wants
    the room, and is settled once, later, by the reduction.
    """

    def __init__(
        self,
        weights:   RelevanceWeights | None = None,
        strategy:  ScoringStrategy         = ScoringStrategy.WEIGHTED_SIGMOID,
    ) -> None:
        self.weights   = weights or RelevanceWeights()
        self.weights.validate()
        self.strategy  = strategy
        logger.debug(
            "RelevanceEngine init | strategy=%s "
            "weights=(r=%.2f p=%.2f v=%.2f d=%.2f)",
            strategy,
            self.weights.recency, self.weights.proximity,
            self.weights.volatility, self.weights.dependency,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(
        self,
        signals:         RelevanceInputs,
        signal_entities: list[str],
        intent:          str,
        now:             datetime,
        dependency_ids:  set[str] | None = None,
    ) -> RelevanceScore:
        """How well this matches the moment — one definition, for everything.

        There used to be two, `score_atom` and `score_result`, differing in which
        field each read and in one hardcoded `volatility = 0.0`. That difference
        is real and it is an *input*: a session result does not mutate. It was
        never a second meaning of relevance, and having it look like one is what
        let two storage types drift apart.
        """
        recency    = _recency_score(signals.updated_at, now)
        proximity  = _proximity_score(list(signals.entities), signal_entities, intent)
        volatility = _volatility_score(signals.mutation_count)
        dependency = _dependency_score(signals.identity, dependency_ids or set())
        total = self._compute_total(recency, proximity, volatility, dependency)
        return RelevanceScore(
            recency=round(recency, 4),
            proximity=round(proximity, 4),
            volatility=round(volatility, 4),
            dependency=round(dependency, 4),
            total=round(total, 4),
        )

    def rank(
        self,
        elements:        Sequence[MentalElement],
        signal_entities: list[str],
        intent:          str,
        now:             datetime,
        dependency_ids:  set[str] | None = None,
    ) -> list[MentalElement]:
        """Every element, scored and ordered — and **none of them discarded**.

        This is the half of the architecture that was missing. Reduction was made
        blind to what an element is made of; deciding what may *reach* the
        reduction was not, and it was doing so with a score threshold and a
        per-type count. A durable fact about the user could therefore be dropped
        for being the eleventh session result, before anything had a chance to
        notice it was the sort of thing you must not lose:

            13 results stored, max_results = 10
            → the one marked IMPORTANT was gone
            → the budget engine never saw it

        So nothing is forgotten before it becomes an element. What survives is a
        question about retention, cost and relevance together, and it is asked in
        exactly one place (ADR-0054).

        Ordered for readability and for the reduction's tie-breaks; the order is
        not a selection.
        """
        scored = [
            replace(
                element,
                relevance=self.score(
                    element.signals, signal_entities, intent, now, dependency_ids
                ).total,
            )
            for element in elements
        ]
        scored.sort(key=lambda item: (-item.relevance, item.element_id))
        return scored

    def score_atom(
        self,
        atom:            object,
        signal_entities: list[str],
        intent:          str,
        now:             datetime,
        dependency_ids:  set[str] | None = None,
    ) -> RelevanceScore:
        """A working atom's signals, mapped onto the one scoring function.

        Two of the four are read from fields that exist, and that is the whole
        of the fix: `created_at` rather than an `updated_at` no atom has ever
        had, and the entities the caller supplies.

        The other two are neutral, and honestly so. `mutation_count` is zero
        because nothing in the platform ever modifies an atom after it is
        written — branches move them, never rewrite them — so it is the same
        statement `score_result` makes about session results. `dependency` needs
        `dependency_ids`, which no production path populates for any sort; that
        is a dormant dimension platform-wide and not an atom's gap to fill.
        """
        return self.score(
            RelevanceInputs(
                updated_at=getattr(atom, "created_at", None),
                entities=tuple(getattr(atom, "entities", []) or ()),
                mutation_count=0,
                identity=str(getattr(atom, "id", "")),
            ),
            signal_entities, intent, now, dependency_ids,
        )

    def score_result(
        self,
        result:          object,
        signal_entities: list[str],
        intent:          str,
        now:             datetime,
        dependency_ids:  set[str] | None = None,
    ) -> RelevanceScore:
        """A session result's signals, mapped onto the one scoring function.

        `mutation_count` is zero because a session result does not mutate — an
        input, stated once, rather than a scoring rule of its own.
        """
        return self.score(
            RelevanceInputs(
                updated_at=getattr(result, "created_at", None),
                entities=tuple(
                    [*(getattr(result, "tags", []) or []), getattr(result, "topic", "")]
                ),
                mutation_count=0,
                identity=str(getattr(result, "id", "")),
            ),
            signal_entities, intent, now, dependency_ids,
        )

    def rank_atoms(
        self,
        atoms:           list[Any],
        signal_entities: list[str],
        intent:          str,
        now:             datetime,
        dependency_ids:  set[str] | None = None,
    ) -> list[ScoredAtom]:
        """Score every atom and order them. Discards nothing — see `rank`."""
        scored = [
            ScoredAtom(
                atom=atom,
                score=self.score_atom(atom, signal_entities, intent, now, dependency_ids),
            )
            for atom in atoms
        ]
        scored.sort(key=lambda item: item.score.total, reverse=True)
        return scored

    def rank_results(
        self,
        results:         list[Any],
        signal_entities: list[str],
        intent:          str,
        now:             datetime,
        dependency_ids:  set[str] | None = None,
    ) -> list[ScoredResult]:
        """Score every result and order them. Discards nothing — see `rank`."""
        scored = [
            ScoredResult(
                result=result,
                score=self.score_result(result, signal_entities, intent, now, dependency_ids),
            )
            for result in results
        ]
        scored.sort(key=lambda item: item.score.total, reverse=True)
        return scored

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_total(
        self,
        recency:    float,
        proximity:  float,
        volatility: float,
        dependency: float,
    ) -> float:
        w = self.weights

        if self.strategy == ScoringStrategy.LINEAR:
            return (
                w.recency    * recency    +
                w.proximity  * proximity  +
                w.volatility * volatility +
                w.dependency * dependency
            )

        if self.strategy == ScoringStrategy.WEIGHTED_SIGMOID:
            boosted_proximity = min(1.0, proximity * w.proximity_boost)
            linear = (
                w.recency    * recency           +
                w.proximity  * boosted_proximity +
                w.volatility * volatility        +
                w.dependency * dependency
            )
            raw = _sigmoid((linear - 0.5) * 6)
            return max(0.0, min(1.0, raw))

        if self.strategy == ScoringStrategy.MULTIPLICATIVE:
            r = max(0.01, recency)
            p = max(0.01, proximity)
            v = max(0.01, volatility) if volatility > 0 else 1.0
            d = max(0.01, dependency) if dependency > 0 else 1.0
            # float.__pow__ types as Any in typeshed for the general case (a
            # fractional exponent on a negative base could yield complex) --
            # r/p/v/d are all guaranteed positive above, so this is always a
            # real float; the float() cast just makes that explicit to mypy.
            raw = float((r ** w.recency) * (p ** w.proximity) * (v ** w.volatility) * (d ** w.dependency))
            return max(0.0, min(1.0, raw))

        return 0.0

# ---------------------------------------------------------------------------
# Reduction: what survives a budget, and in which form
# ---------------------------------------------------------------------------
#
# `filter_atoms` and `filter_results` above answer a different question — what is
# relevant to this turn at all — and they answer it per storage type, by a
# threshold and a count. What follows answers "what fits", over elements, and it
# is deliberately blind to what those elements are made of.
#
# The one that was replaced: `_compress_session_results` in the prompt builder,
# which met budget pressure by dropping session results and nothing else. That is
# not a reduction policy, it is the one storage type somebody happened to make
# droppable.


@dataclass(frozen=True, slots=True)
class ReducedView:
    """What survived, at what level, and what it cost."""

    #: Every element that is being said, with the level it is said at. Elements
    #: reduced to `OMIT` are not here — they are in `forgotten`.
    kept: tuple[tuple[MentalElement, Representation], ...] = ()
    #: Dropped entirely, worst first, so a report can say what was lost.
    forgotten: tuple[MentalElement, ...] = ()
    #: Said more briefly than it could have been.
    compacted: tuple[MentalElement, ...] = ()
    cost: int = 0
    #: True when even the mandatory elements do not fit. Reported rather than
    #: hidden: nothing here may drop a mandatory element to make a number work,
    #: so the caller is told it is over budget instead of being lied to.
    over_budget: bool = False


#: The order in which information is given up. A fixed total order, and every
#: reduction is a **prefix** of it — which is what makes the guarantees hold by
#: construction rather than by testing:
#:
#:     more budget → fewer steps applied → every element at the same level or fuller
#:     less budget → more steps applied  → information only ever decreases
#:
#: Read it as a sentence: spend the dispensable first, then say the merely useful
#: briefly, then stop saying it, then say the important briefly, and only then
#: stop saying the important. The mandatory is never given up, only shortened.
_GIVING_UP: tuple[tuple[Retention, Level], ...] = (
    (Retention.DISPENSABLE, Level.COMPACT),
    (Retention.DISPENSABLE, Level.OMIT),
    (Retention.USEFUL, Level.COMPACT),
    (Retention.USEFUL, Level.OMIT),
    (Retention.IMPORTANT, Level.COMPACT),
    (Retention.IMPORTANT, Level.OMIT),
    (Retention.MANDATORY, Level.COMPACT),
)


def _steps(elements: Sequence[MentalElement]) -> list[tuple[str, Level]]:
    """Every reduction available, in the order they are taken.

    Within one band the least relevant element gives way first, and ties break on
    the element id so two runs over one input reduce identically.
    """
    ordered: list[tuple[str, Level]] = []
    for retention, level in _GIVING_UP:
        band = sorted(
            (item for item in elements if item.retention is retention),
            key=lambda item: (item.relevance, item.element_id),
        )
        ordered.extend(
            (item.element_id, level) for item in band if item.at(level) is not None
        )
    return ordered


def reduce_to_budget(
    elements: Sequence[MentalElement], *, budget: int
) -> ReducedView:
    """The fullest view of these elements that fits, and what it cost to get there.

    **Type-blind on purpose.** Nothing here reads `element.kind`, and nothing may:
    the moment a reduction branches on where information came from, a corpus has
    one policy per store again and no way to weigh a memory against a passage.
    What decides is what the element *is worth* (`retention`), how well it
    matches the moment (`relevance`), and what it costs to say.

    **A prefix of a fixed order.** The reduction applies steps from `_GIVING_UP`
    until the total fits, and stops. Two consequences follow without being
    separately arranged: a larger budget applies a prefix of the steps a smaller
    one applies, so no element is ever *less* complete when there is more room;
    and information only ever decreases as the budget shrinks.

    **Nothing is invented.** An element is only ever moved to a level it already
    offers, and the text of that level was written by whatever knows what the
    element is. Provenance rides along untouched, so a compacted element can
    still say where it came from — and that it is no longer a verbatim quote of
    it.
    """
    if budget < 0:
        raise ValueError("a reduction budget cannot be negative")

    chosen: dict[str, Level] = {}
    for item in elements:
        fullest = item.levels[0]
        chosen[item.element_id] = fullest
    by_id = {item.element_id: item for item in elements}

    def _total() -> int:
        return sum(
            representation.cost
            for element_id, level in chosen.items()
            if (representation := by_id[element_id].at(level)) is not None
        )

    for element_id, level in _steps(elements):
        if _total() <= budget:
            break
        current = chosen[element_id]
        # A step never moves an element back up. Levels are ordered, and the
        # order is the whole guarantee.
        if LEVELS.index(level) > LEVELS.index(current):
            chosen[element_id] = level

    cost = _total()
    kept: list[tuple[MentalElement, Representation]] = []
    forgotten: list[MentalElement] = []
    compacted: list[MentalElement] = []
    for item in elements:
        level = chosen[item.element_id]
        if level is Level.OMIT:
            forgotten.append(item)
            continue
        representation = item.at(level)
        if representation is None:  # pragma: no cover - `chosen` only holds offered levels
            continue
        if level is not item.levels[0]:
            compacted.append(item)
        kept.append((item, representation))

    return ReducedView(
        kept=tuple(kept),
        forgotten=tuple(forgotten),
        compacted=tuple(compacted),
        cost=cost,
        over_budget=cost > budget,
    )
