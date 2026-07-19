# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""MentalModelBuilder — assembles a MentalModel from store snapshots.

Pure logic — no LLM call.  The optional ``summary`` string comes from
an LLM call in OrientStep and is passed in from outside.

Usage::

    snapshot = ContextSnapshot(
        directives    = coordinator.get_directives_list(),
        recent_turns  = coordinator.conversation.last_n(5),
        known_results = coordinator.results.find_by_entities(entities),
        active_atoms  = coordinator.working.read_active(),
    )
    builder = MentalModelBuilder(relevance_engine=engine)
    model   = builder.build(
        turn_id        = cid,
        context        = snapshot,
        signal_entities = entities,
        intents        = ["informational"],
        summary        = "...",   # optional, from a prior LLM call
    )
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from nlght.core.hive_mind.models import (
    MentalModel,
    RelevanceWeights,
    ScoredAtom,
    ScoredResult,
    ScoringStrategy,
)
from nlght.core.hive_mind.relevance import RelevanceEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Context Snapshot
# ---------------------------------------------------------------------------

class ContextSnapshot:
    """Compact snapshot of all four stores for a single turn.

    MentalModelBuilder works exclusively on this object so steps
    never need to pass store references into the builder directly.
    """

    def __init__(
        self,
        directives:     list[Any] | None = None,
        recent_turns:   list[Any] | None = None,
        known_results:  list[Any] | None = None,
        active_atoms:   list[Any] | None = None,
        dependency_ids: set[str]  | None = None,
    ) -> None:
        self.directives     = directives    or []
        self.recent_turns   = recent_turns  or []
        self.known_results  = known_results or []
        self.active_atoms   = active_atoms  or []
        self.dependency_ids = dependency_ids or set()


# ---------------------------------------------------------------------------
# MentalModelBuilder
# ---------------------------------------------------------------------------

class MentalModelBuilder:
    """Assembles a MentalModel from a store snapshot and the current signal.

    Parameters:
        relevance_engine:    Pre-configured RelevanceEngine.
        relevance_threshold: Objects below this score are dropped.
        max_atoms:           Maximum atoms in the assembled model.
        max_results:         Maximum session results in the assembled model.
        max_turns:           Maximum recent turns included.
    """

    def __init__(
        self,
        relevance_engine:    RelevanceEngine,
        relevance_threshold: float = 0.20,
        max_atoms:           int   = 15,
        max_results:         int   = 10,
        max_turns:           int   = 8,
    ) -> None:
        self.engine      = relevance_engine
        self.threshold   = relevance_threshold
        self.max_atoms   = max_atoms
        self.max_results = max_results
        self.max_turns   = max_turns

    @classmethod
    def from_step_config(cls, step_config: Mapping[str, object]) -> MentalModelBuilder:
        """Build from the public ``session_memory`` step-config block.

        Missing blocks and values retain the same defaults as the direct
        constructors. Invalid values fail at step construction time instead of
        silently changing context selection.
        """
        memory = _mapping_value(step_config, "session_memory")
        relevance = _mapping_value(memory, "relevance")
        weights_config = _mapping_value(relevance, "weights")
        mental_model = _mapping_value(memory, "mental_model")

        weights = RelevanceWeights(
            recency=_float_value(weights_config, "recency", 0.25),
            proximity=_float_value(weights_config, "proximity", 0.40),
            volatility=_float_value(weights_config, "volatility", 0.15),
            dependency=_float_value(weights_config, "dependency", 0.20),
            proximity_boost=_float_value(weights_config, "proximity_boost", 1.8),
        )
        raw_strategy = relevance.get("strategy", ScoringStrategy.WEIGHTED_SIGMOID.value)
        try:
            strategy = ScoringStrategy(str(raw_strategy))
        except ValueError as exc:
            choices = ", ".join(item.value for item in ScoringStrategy)
            raise ValueError(
                f"session_memory.relevance.strategy must be one of: {choices}"
            ) from exc

        engine = RelevanceEngine(
            weights=weights,
            strategy=strategy,
            threshold=_float_value(relevance, "threshold", 0.15),
        )
        return cls(
            relevance_engine=engine,
            max_atoms=_positive_int_value(mental_model, "max_atoms", 15),
            max_results=_positive_int_value(mental_model, "max_results", 10),
            max_turns=_positive_int_value(mental_model, "max_turns", 8),
        )

    def build(
        self,
        turn_id:         str,
        context:         ContextSnapshot,
        signal_entities: list[str],
        intents:         list[str],
        delta:           object | None = None,
        summary:         str | None = None,
    ) -> MentalModel:
        """Build a complete MentalModel.  Synchronous, no LLM call.

        Parameters:
            turn_id:         Correlation ID for the current turn.
            context:         ContextSnapshot from all four stores.
            signal_entities: Entities from the current signal (Observe output).
            intents:         All intents for this turn (one per sub-signal).
            delta:           Optional DeltaReport from a state reconciler.
            summary:         Optional summary string from a prior LLM call.
        """
        now = datetime.now(UTC)
        intent_str = ",".join(intents) if intents else "informational"

        logger.debug(
            "[%s] MentalModelBuilder.build | intents=%s entities=%d atoms=%d results=%d",
            turn_id, intents, len(signal_entities),
            len(context.active_atoms), len(context.known_results),
        )

        scored_atoms = self.engine.filter_atoms(
            atoms           = context.active_atoms,
            signal_entities = signal_entities,
            intent          = intent_str,
            now             = now,
            dependency_ids  = context.dependency_ids,
            limit           = self.max_atoms,
        )

        scored_results = self.engine.filter_results(
            results         = context.known_results,
            signal_entities = signal_entities,
            intent          = intent_str,
            now             = now,
            dependency_ids  = context.dependency_ids,
            limit           = self.max_results,
        )

        recent_turns = context.recent_turns[-self.max_turns:]

        all_entities = _deduplicate_entities(signal_entities, scored_atoms, scored_results)

        logger.debug(
            "[%s] MentalModelBuilder: scored | atoms=%d/%d results=%d/%d entities=%d",
            turn_id,
            len(scored_atoms),   len(context.active_atoms),
            len(scored_results), len(context.known_results),
            len(all_entities),
        )

        model = MentalModel(
            turn_id       = turn_id,
            built_at      = now,
            is_valid      = True,
            directives    = context.directives,
            recent_turns  = recent_turns,
            known_results = scored_results,
            active_atoms  = scored_atoms,
            delta         = delta,
            summary       = summary,
            intents       = intents,
            entities      = all_entities,
        )

        logger.info(
            "[%s] MentalModelBuilder: built | atoms=%d results=%d "
            "directives=%d turns=%d intents=%s summary=%s",
            turn_id,
            len(scored_atoms), len(scored_results),
            len(context.directives), len(recent_turns),
            intents,
            "yes" if summary else "no",
        )

        return model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deduplicate_entities(
    signal_entities: list[str],
    scored_atoms:    list[ScoredAtom],
    scored_results:  list[ScoredResult],
) -> list[str]:
    seen:   set[str]  = set()
    result: list[str] = []

    for e in signal_entities:
        key = e.lower()
        if key not in seen:
            seen.add(key)
            result.append(e)

    for sr in scored_results:
        for e in sr.result.entities:
            key = e.lower()
            if key not in seen:
                seen.add(key)
                result.append(e)

    return result


def _mapping_value(config: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = config.get(key, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be an object")
    return value


def _float_value(config: Mapping[str, object], key: str, default: float) -> float:
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be a number")
    return float(value)


def _positive_int_value(config: Mapping[str, object], key: str, default: int) -> int:
    value = config.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{key} must be a positive integer")
    return value
