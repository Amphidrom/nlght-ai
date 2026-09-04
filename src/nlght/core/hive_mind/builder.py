# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""MentalModelBuilder — assembles a MentalModel from store snapshots.

Pure logic — no LLM call.  The optional ``summary`` string comes from
an LLM call in OrientStep and is passed in from outside.

Usage::

    snapshot = ContextSnapshot(
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
        recent_turns:   list[Any] | None = None,
        known_results:  list[Any] | None = None,
        active_atoms:   list[Any] | None = None,
        dependency_ids: set[str]  | None = None,
    ) -> None:
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
        relevance_engine: Pre-configured RelevanceEngine.

    **It selects nothing.** It assembles and scores; what the model is actually
    told is decided once, later, by the reduction, which is the only place that
    can see retention, cost and everything competing for the same room at the
    same time.

    The caps it used to apply — `max_atoms`, `max_results`, `max_turns` — and the
    `relevance_threshold` are gone. Three storage types had three different
    selection rules, none of which knew what it would cost to lose what they were
    dropping. If a store ever grows large enough to need a *loading* limit, that
    is acquisition and has to be named and reported as such: information lost to
    a cap is degradation, not the relevance engine finding it unimportant.

    (`relevance_threshold` was additionally never read: it was assigned to
    `self.threshold` and nothing consulted it.)
    """

    def __init__(self, relevance_engine: RelevanceEngine) -> None:
        self.engine = relevance_engine

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

        # Settings that no longer do anything are refused rather than accepted
        # in silence. `relevance.threshold` and the `mental_model` caps used to
        # remove candidates before anything could weigh them (ADR-0056); a
        # deployment that still sets one is asking for a behaviour that is gone,
        # and letting it start would be the worst of both — configured, ignored,
        # and no way to tell.
        for gone, where in (("threshold", relevance), ("max_atoms", mental_model),
                            ("max_results", mental_model), ("max_turns", mental_model)):
            if gone in where:
                raise ValueError(
                    f"session_memory sets '{gone}', which no longer exists: nothing "
                    f"is discarded before it becomes an element, and what a model "
                    f"is told is decided once by the prompt budget (ADR-0056)."
                )
        return cls(relevance_engine=RelevanceEngine(weights=weights, strategy=strategy))

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

        # Scored and ordered, and **nothing discarded**. What may be forgotten
        # is decided once, later, with retention and cost in view — a score
        # threshold and a per-type count answered it here, before either
        # existed, and a durable fact about the user could be dropped for being
        # the eleventh session result.
        scored_atoms = self.engine.rank_atoms(
            atoms           = context.active_atoms,
            signal_entities = signal_entities,
            intent          = intent_str,
            now             = now,
            dependency_ids  = context.dependency_ids,
        )

        scored_results = self.engine.rank_results(
            results         = context.known_results,
            signal_entities = signal_entities,
            intent          = intent_str,
            now             = now,
            dependency_ids  = context.dependency_ids,
        )

        # Every turn the snapshot holds. `[-max_turns:]` was the third selection
        # rule for the third storage type, and the only one that never even
        # consulted relevance.
        recent_turns = context.recent_turns

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
            "turns=%d intents=%s summary=%s",
            turn_id,
            len(scored_atoms), len(scored_results),
            len(recent_turns),
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
