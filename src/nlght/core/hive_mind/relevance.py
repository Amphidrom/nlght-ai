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
from datetime import UTC, datetime
from typing import Any

from nlght.core.hive_mind.models import (
    RelevanceScore,
    RelevanceWeights,
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
        threshold: Objects below this score are filtered out (default: 0.15).
    """

    def __init__(
        self,
        weights:   RelevanceWeights | None = None,
        strategy:  ScoringStrategy         = ScoringStrategy.WEIGHTED_SIGMOID,
        threshold: float                   = 0.15,
    ) -> None:
        self.weights   = weights or RelevanceWeights()
        self.weights.validate()
        self.strategy  = strategy
        self.threshold = threshold
        logger.debug(
            "RelevanceEngine init | strategy=%s threshold=%.2f "
            "weights=(r=%.2f p=%.2f v=%.2f d=%.2f)",
            strategy, threshold,
            self.weights.recency, self.weights.proximity,
            self.weights.volatility, self.weights.dependency,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score_atom(
        self,
        atom:            object,
        signal_entities: list[str],
        intent:          str,
        now:             datetime,
        dependency_ids:  set[str] | None = None,
    ) -> RelevanceScore:
        recency    = _recency_score(getattr(atom, "updated_at", None), now)
        proximity  = _proximity_score(getattr(atom, "entities", []), signal_entities, intent)
        volatility = _volatility_score(getattr(atom, "mutation_count", 0))
        dependency = _dependency_score(getattr(atom, "id", ""), dependency_ids or set())
        total = self._compute_total(recency, proximity, volatility, dependency)
        return RelevanceScore(
            recency=round(recency, 4),
            proximity=round(proximity, 4),
            volatility=round(volatility, 4),
            dependency=round(dependency, 4),
            total=round(total, 4),
        )

    def score_result(
        self,
        result:          object,
        signal_entities: list[str],
        intent:          str,
        now:             datetime,
        dependency_ids:  set[str] | None = None,
    ) -> RelevanceScore:
        recency    = _recency_score(getattr(result, "created_at", None), now)
        proximity  = _proximity_score(
            getattr(result, "tags", []) + [getattr(result, "topic", "")],
            signal_entities,
            intent,
        )
        volatility = 0.0  # SessionResults do not mutate
        dependency = _dependency_score(getattr(result, "id", ""), dependency_ids or set())
        total = self._compute_total(recency, proximity, volatility, dependency)
        return RelevanceScore(
            recency=round(recency, 4),
            proximity=round(proximity, 4),
            volatility=round(volatility, 4),
            dependency=round(dependency, 4),
            total=round(total, 4),
        )

    def filter_atoms(
        self,
        atoms:           list[Any],
        signal_entities: list[str],
        intent:          str,
        now:             datetime,
        dependency_ids:  set[str] | None = None,
        limit:           int             = 20,
    ) -> list[ScoredAtom]:
        dep_ids = dependency_ids or set()
        scored: list[ScoredAtom] = []
        for atom in atoms:
            score = self.score_atom(atom, signal_entities, intent, now, dep_ids)
            if score.total >= self.threshold:
                depth = self._depth_from_score(score.total)
                scored.append(ScoredAtom(atom=atom, score=score, depth=depth))
        scored.sort(key=lambda s: s.score.total, reverse=True)
        return scored[:limit]

    def filter_results(
        self,
        results:         list[Any],
        signal_entities: list[str],
        intent:          str,
        now:             datetime,
        dependency_ids:  set[str] | None = None,
        limit:           int             = 20,
    ) -> list[ScoredResult]:
        dep_ids = dependency_ids or set()
        scored: list[ScoredResult] = []
        for result in results:
            score = self.score_result(result, signal_entities, intent, now, dep_ids)
            if score.total >= self.threshold:
                scored.append(ScoredResult(result=result, score=score))
        scored.sort(key=lambda s: s.score.total, reverse=True)
        return scored[:limit]

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

    @staticmethod
    def _depth_from_score(total: float) -> int:
        if total >= 0.75:
            return 3
        if total >= 0.40:
            return 2
        return 1
