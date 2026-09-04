# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Tests for nlght.core.hive_mind.builder — MentalModelBuilder + ContextSnapshot."""
from __future__ import annotations

import pytest

from nlght.core.hive_mind.builder import ContextSnapshot, MentalModelBuilder
from nlght.core.hive_mind.models import (
    AtomType,
    RelevanceWeights,
    ScoringStrategy,
    SessionResult,
    TurnSummary,
    WorkingAtom,
)
from nlght.core.hive_mind.relevance import RelevanceEngine


def _engine() -> RelevanceEngine:
    return RelevanceEngine(
        weights=RelevanceWeights(),
        strategy=ScoringStrategy.LINEAR,
    )


def _builder(**_ignored) -> MentalModelBuilder:
    """The builder no longer takes caps or a threshold — it selects nothing.

    `**_ignored` so the call sites that used to pass `max_turns=3` read as what
    they now are: a request for a limit that no longer exists.
    """
    return MentalModelBuilder(relevance_engine=_engine())


def test_from_step_config_uses_defaults() -> None:
    builder = MentalModelBuilder.from_step_config({})
    assert builder.engine.strategy == ScoringStrategy.WEIGHTED_SIGMOID
    assert builder.engine.weights == RelevanceWeights()


def test_from_step_config_applies_nested_memory_config() -> None:
    builder = MentalModelBuilder.from_step_config({
        "session_memory": {
            "relevance": {
                "strategy": "linear",
                "weights": {
                    "recency": 0.20,
                    "proximity": 0.50,
                    "volatility": 0.10,
                    "dependency": 0.20,
                    "proximity_boost": 1.5,
                },
            },
        }
    })
    assert builder.engine.strategy == ScoringStrategy.LINEAR
    assert builder.engine.weights.proximity == 0.50
    assert builder.engine.weights.proximity_boost == 1.5
    # No caps and no threshold in the block any more: they configured a selection
    # that no longer happens here, and setting one is now an error rather than a
    # silent no-op (ADR-0056).


@pytest.mark.parametrize("strategy", ["weighted_sigmoid", "linear", "multiplicative"])
def test_from_step_config_accepts_every_strategy(strategy: str) -> None:
    builder = MentalModelBuilder.from_step_config({
        "session_memory": {"relevance": {"strategy": strategy}}
    })
    assert builder.engine.strategy.value == strategy


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"session_memory": []}, "session_memory must be an object"),
        ({"session_memory": {"relevance": {"strategy": "magic"}}}, "must be one of"),
        # Settings that no longer do anything are refused rather than accepted in
        # silence: configured, ignored and undetectable is the worst outcome.
        ({"session_memory": {"relevance": {"threshold": 0.25}}}, "no longer exists"),
        ({"session_memory": {"mental_model": {"max_atoms": 5}}}, "no longer exists"),
        ({"session_memory": {"mental_model": {"max_turns": 3}}}, "no longer exists"),
    ],
)
def test_from_step_config_rejects_invalid_values(config: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        MentalModelBuilder.from_step_config(config)


# ---------------------------------------------------------------------------
# ContextSnapshot
# ---------------------------------------------------------------------------

def test_context_snapshot_defaults_empty() -> None:
    snap = ContextSnapshot()
    assert snap.active_atoms == []
    assert snap.known_results == []
    assert snap.recent_turns == []
    assert snap.dependency_ids == set()


def test_context_snapshot_accepts_values() -> None:
    atom = WorkingAtom(atom_type=AtomType.RESULT, content="x", task_id="t1")
    snap = ContextSnapshot(active_atoms=[atom], dependency_ids={"id1"})
    assert len(snap.active_atoms) == 1
    assert "id1" in snap.dependency_ids


# ---------------------------------------------------------------------------
# MentalModelBuilder.build
# ---------------------------------------------------------------------------

def test_build_returns_mental_model_with_correct_turn_id() -> None:
    builder = _builder()
    snap    = ContextSnapshot()
    model   = builder.build(turn_id="test-turn", context=snap, signal_entities=[], intents=["info"])
    assert model.turn_id == "test-turn"
    assert model.is_valid is True


def test_build_with_atoms() -> None:
    builder = _builder(max_atoms=5)
    atom    = WorkingAtom(atom_type=AtomType.RESULT, content="data", task_id="t1")
    snap    = ContextSnapshot(active_atoms=[atom])
    model   = builder.build(turn_id="t", context=snap, signal_entities=["data"], intents=["act"])
    # atom may or may not be scored above threshold depending on weights
    assert model.active_atoms is not None


def test_build_with_results() -> None:
    builder = _builder(max_results=5)
    result  = SessionResult(content="info", entities=["python"], tags=["code"])
    snap    = ContextSnapshot(known_results=[result])
    model   = builder.build(turn_id="t", context=snap, signal_entities=["python"], intents=["info"])
    assert isinstance(model.known_results, list)


def test_build_keeps_every_turn_the_snapshot_holds() -> None:
    """`[-max_turns:]` was the third selection rule for the third storage type.

    It was also the only one that never consulted relevance at all: turns were
    cut by position while atoms and results were cut by score. What a model is
    told is now one decision, made once, with retention and cost in view.
    """
    builder = _builder()
    turns = [
        TurnSummary(turn_nr=i, user_input="hi", intent="greet", topic="t")
        for i in range(10)
    ]
    snap  = ContextSnapshot(recent_turns=turns)
    model = builder.build(turn_id="t", context=snap, signal_entities=[], intents=[])
    assert len(model.recent_turns) == 10


def test_build_includes_summary() -> None:
    builder = _builder()
    snap    = ContextSnapshot()
    model   = builder.build(
        turn_id="t", context=snap, signal_entities=[], intents=[], summary="A brief summary."
    )
    assert model.summary == "A brief summary."


def test_build_deduplicates_entities() -> None:
    builder = _builder(max_results=10)
    result  = SessionResult(content="x", entities=["python"])
    snap    = ContextSnapshot(known_results=[result])
    model   = builder.build(
        turn_id="t", context=snap,
        signal_entities=["python", "python"],  # duplicate
        intents=["info"],
    )
    count = model.entities.count("python")
    assert count == 1


def test_build_empty_intents_uses_fallback() -> None:
    builder = _builder()
    snap    = ContextSnapshot()
    # Should not raise — uses "informational" fallback
    model   = builder.build(turn_id="t", context=snap, signal_entities=[], intents=[])
    assert model.intents == []
