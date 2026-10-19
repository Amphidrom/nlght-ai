# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Score memory elements and reduce them to a token budget."""

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

# Set the half-life to five minutes, when recency falls to 0.5.
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

    Dimension weights sum to 1.0. The scoring strategy selects a linear,
    weighted sigmoid, or multiplicative formula.

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


    def score(
        self,
        signals:         RelevanceInputs,
        signal_entities: list[str],
        intent:          str,
        now:             datetime,
        dependency_ids:  set[str] | None = None,
    ) -> RelevanceScore:
        """Score one element on recency, proximity, volatility, and dependency.

        Args:
            signals (RelevanceInputs): Relevance signals for one element.
            signal_entities (list[str]): Entities in the current signal.
            intent (str): Current user intent.
            now (datetime): Time used for recency scoring.
            dependency_ids (set[str] | None): Optional IDs of prerequisite elements.

        Returns:
            RelevanceScore: Weighted relevance score and its components.
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
        """Score and order all elements without discarding any.

        Args:
            elements (Sequence[MentalElement]): Elements to score and order.
            signal_entities (list[str]): Entities in the current signal.
            intent (str): Current user intent.
            now (datetime): Time used for recency scoring.
            dependency_ids (set[str] | None): Optional IDs of prerequisite elements.

        Returns:
            list[MentalElement]: Every element with an updated score, highest first.
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
        """Map a working atom to the shared relevance score.

        Args:
            atom (object): Working atom to score.
            signal_entities (list[str]): Entities in the current signal.
            intent (str): Current user intent.
            now (datetime): Time used for recency scoring.
            dependency_ids (set[str] | None): Optional IDs of prerequisite elements.

        Returns:
            RelevanceScore: Relevance score for the atom.
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
        """Map a session result to the shared relevance score.

        Args:
            result (object): Session result to score.
            signal_entities (list[str]): Entities in the current signal.
            intent (str): Current user intent.
            now (datetime): Time used for recency scoring.
            dependency_ids (set[str] | None): Optional IDs of prerequisite elements.

        Returns:
            RelevanceScore: Relevance score for the result.
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
        """Score and order every working atom.

        Args:
            atoms (list[Any]): Working atoms to score and order.
            signal_entities (list[str]): Entities in the current signal.
            intent (str): Current user intent.
            now (datetime): Time used for recency scoring.
            dependency_ids (set[str] | None): Optional IDs of prerequisite elements.

        Returns:
            list[ScoredAtom]: Every atom with its score, highest first.
        """
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
        """Score and order every session result.

        Args:
            results (list[Any]): Session results to score and order.
            signal_entities (list[str]): Entities in the current signal.
            intent (str): Current user intent.
            now (datetime): Time used for recency scoring.
            dependency_ids (set[str] | None): Optional IDs of prerequisite elements.

        Returns:
            list[ScoredResult]: Every result with its score, highest first.
        """
        scored = [
            ScoredResult(
                result=result,
                score=self.score_result(result, signal_entities, intent, now, dependency_ids),
            )
            for result in results
        ]
        scored.sort(key=lambda item: item.score.total, reverse=True)
        return scored


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
            # real float. The float() cast just makes that explicit to mypy.
            raw = float((r ** w.recency) * (p ** w.proximity) * (v ** w.volatility) * (d ** w.dependency))
            return max(0.0, min(1.0, raw))

        return 0.0

#
# `filter_atoms` and `filter_results` above determine what is
# relevant to this turn at all, and they answer it per storage type, by a
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
    #: reduced to `OMIT` are in `forgotten` instead.
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
#: reduction is a **prefix** of it, which is what makes the guarantees hold by
#: construction rather than by testing:
#:
#:     More budget applies fewer steps, preserving each element's level or increasing it.
#:     Less budget applies more steps, so information only decreases.
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
    """Order available reduction steps by retention and relevance.

    Args:
        elements (Sequence[MentalElement]): Mental elements and their offered representation levels.

    Returns:
        list[tuple[str, Level]]: Ordered element IDs and levels to give up.
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
    """Select the fullest offered representations within a budget.

    Args:
        elements (Sequence[MentalElement]): Mental elements in preference order.
        budget (int): Nonnegative token budget.

    Returns:
        ReducedView: Selected representations and forgotten elements.

    Raises:
        ValueError: The budget is negative.
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
