# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Tests for nlght.core.hive_mind.models — all data types."""
from __future__ import annotations

import pytest

from nlght.core.hive_mind.models import (
    AtomType,
    ControlSignalDecision,
    DiagnosticItem,
    Diagnostics,
    MentalModel,
    MentalModelCache,
    PromotionStatus,
    RelevanceScore,
    RelevanceWeights,
    ScoredAtom,
    ScoredResult,
    ScoringStrategy,
    SessionResult,
    Severity,
    WorkingAtom,
)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

def test_atom_type_values() -> None:
    assert AtomType.RESULT == "RESULT"
    assert AtomType.SPEC == "SPEC"

def test_promotion_status_values() -> None:
    assert PromotionStatus.FINAL == "FINAL"
    assert PromotionStatus.SUPERSEDED == "SUPERSEDED"

def test_scoring_strategy_values() -> None:
    assert ScoringStrategy.LINEAR == "linear"
    assert ScoringStrategy.WEIGHTED_SIGMOID == "weighted_sigmoid"
    assert ScoringStrategy.MULTIPLICATIVE == "multiplicative"

def test_severity_values() -> None:
    assert Severity.error == "error"
    assert Severity.warning == "warning"


# ---------------------------------------------------------------------------
# RelevanceWeights
# ---------------------------------------------------------------------------

def test_relevance_weights_default_valid() -> None:
    w = RelevanceWeights()
    w.validate()  # should not raise

def test_relevance_weights_validation_fails_if_sum_wrong() -> None:
    w = RelevanceWeights(recency=0.5, proximity=0.5, volatility=0.5, dependency=0.5)
    with pytest.raises(ValueError, match="sum to 1.0"):
        w.validate()

def test_relevance_weights_validation_fails_if_value_out_of_range() -> None:
    w = RelevanceWeights(recency=-0.1, proximity=0.5, volatility=0.3, dependency=0.3)
    with pytest.raises(ValueError, match="between 0 and 1"):
        w.validate()

def test_relevance_weights_validation_rejects_negative_proximity_boost() -> None:
    with pytest.raises(ValueError, match="proximity_boost"):
        RelevanceWeights(proximity_boost=-0.1).validate()


# ---------------------------------------------------------------------------
# RelevanceScore
# ---------------------------------------------------------------------------

def test_relevance_score_repr() -> None:
    score = RelevanceScore(recency=0.8, proximity=0.6, volatility=0.2, dependency=0.4, total=0.65)
    r = repr(score)
    assert "r=0.80" in r
    assert "0.65" in r


# ---------------------------------------------------------------------------
# SessionResult
# ---------------------------------------------------------------------------

def test_session_result_is_file_backed_false_by_default() -> None:
    r = SessionResult(content="hello")
    assert r.is_file_backed() is False

def test_session_result_is_file_backed_true_when_path_set() -> None:
    r = SessionResult(content="", workspace_path="/tmp/file.txt")
    assert r.is_file_backed() is True

def test_session_result_peek_content() -> None:
    r = SessionResult(content="hello world")
    assert r.peek(5) == "hello"

def test_session_result_peek_file_backed() -> None:
    r = SessionResult(content="", workspace_path="/tmp/x.txt")
    assert "[file:" in r.peek()


# ---------------------------------------------------------------------------
# MentalModel
# ---------------------------------------------------------------------------

def _make_mental_model(**kwargs) -> MentalModel:
    from datetime import UTC, datetime
    defaults = dict(turn_id="abc123", built_at=datetime.now(UTC))
    defaults.update(kwargs)
    return MentalModel(**defaults)

def test_mental_model_top_atoms_sorted() -> None:
    atom = WorkingAtom(atom_type=AtomType.RESULT, content="x", task_id="t1")
    low  = ScoredAtom(atom=atom, score=RelevanceScore(total=0.2))
    high = ScoredAtom(atom=atom, score=RelevanceScore(total=0.9))
    model = _make_mental_model(active_atoms=[low, high])
    top = model.top_atoms(1)
    assert top[0] is high

def test_mental_model_top_results_sorted() -> None:
    result = SessionResult(content="data")
    low    = ScoredResult(result=result, score=RelevanceScore(total=0.1))
    high   = ScoredResult(result=result, score=RelevanceScore(total=0.8))
    model  = _make_mental_model(known_results=[low, high])
    top = model.top_results(1)
    assert top[0] is high

def test_mental_model_has_delta_false() -> None:
    model = _make_mental_model()
    assert model.has_delta() is False

def test_mental_model_has_delta_true() -> None:
    model = _make_mental_model(delta=object())
    assert model.has_delta() is True

def test_mental_model_repr() -> None:
    model = _make_mental_model()
    r = repr(model)
    assert "MentalModel" in r
    assert "valid=True" in r


# ---------------------------------------------------------------------------
# MentalModelCache
# ---------------------------------------------------------------------------

def test_mental_model_cache_empty_by_default() -> None:
    cache = MentalModelCache()
    assert cache.model is None
    assert cache.valid is False

def test_mental_model_cache_set_and_read() -> None:
    cache = MentalModelCache()
    model = _make_mental_model()
    cache.set(model)
    assert cache.valid is True
    assert cache.model is model

def test_mental_model_cache_invalidate() -> None:
    cache = MentalModelCache()
    model = _make_mental_model()
    cache.set(model)
    cache.invalidate()
    assert cache.valid is False
    assert cache.model is None
    assert model.is_valid is False

def test_mental_model_cache_clear() -> None:
    cache = MentalModelCache()
    cache.set(_make_mental_model())
    cache.clear()
    assert cache.model is None

def test_mental_model_cache_repr_empty() -> None:
    assert "empty" in repr(MentalModelCache())

def test_mental_model_cache_repr_valid() -> None:
    cache = MentalModelCache()
    cache.set(_make_mental_model())
    assert "valid=True" in repr(cache)


# ---------------------------------------------------------------------------
# ControlSignalDecision
# ---------------------------------------------------------------------------

def test_control_signal_pass_through() -> None:
    d = ControlSignalDecision(pass_through=True)
    assert "pass_through" in repr(d)

def test_control_signal_stop_disables_pass_through() -> None:
    d = ControlSignalDecision(should_stop=True, stop_reason="cancel", pass_through=True)
    assert d.pass_through is False

def test_control_signal_repr_with_stop() -> None:
    d = ControlSignalDecision(should_stop=True, stop_reason="abort", pass_through=False)
    r = repr(d)
    assert "stop=" in r


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def test_diagnostics_summary_auto_populated() -> None:
    item = DiagnosticItem(message="err", severity=Severity.error)
    diag = Diagnostics(tool_name="lint", tool_call_id="call-1", errors=[item])
    assert diag.summary is not None
    assert diag.summary.error_count == 1
    assert diag.summary.warning_count == 0
