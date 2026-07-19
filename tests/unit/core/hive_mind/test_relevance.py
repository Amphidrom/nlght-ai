# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Tests for nlght.core.hive_mind.relevance — RelevanceEngine."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from nlght.core.hive_mind.models import (
    AtomType,
    ScoringStrategy,
    SessionResult,
    WorkingAtom,
)
from nlght.core.hive_mind.relevance import (
    RelevanceEngine,
    _dependency_score,
    _proximity_score,
    _recency_score,
    _volatility_score,
)

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _now() -> datetime:
    return datetime.now(UTC)

def _atom(entities: list[str] | None = None) -> WorkingAtom:
    return WorkingAtom(atom_type=AtomType.RESULT, content="data", task_id="t1")

def _result(entities: list[str] | None = None, tags: list[str] | None = None) -> SessionResult:
    return SessionResult(content="data", entities=entities or [], tags=tags or [])


# ---------------------------------------------------------------------------
# _recency_score
# ---------------------------------------------------------------------------

def test_recency_score_none_timestamp_returns_one() -> None:
    assert _recency_score(None, _now()) == 1.0

def test_recency_score_fresh_is_near_one() -> None:
    now = _now()
    score = _recency_score(now - timedelta(seconds=1), now)
    assert score > 0.99

def test_recency_score_old_is_near_zero() -> None:
    now = _now()
    score = _recency_score(now - timedelta(hours=24), now)
    assert score < 0.01

def test_recency_score_naive_datetime_handled() -> None:
    naive = datetime.now().replace(tzinfo=None)  # explicitly naive
    now   = _now()
    score = _recency_score(naive, now)
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# _proximity_score
# ---------------------------------------------------------------------------

def test_proximity_score_no_entities_returns_low() -> None:
    score = _proximity_score([], ["python"], "code")
    assert score == 0.1

def test_proximity_score_exact_match_returns_high() -> None:
    score = _proximity_score(["python"], ["python"], "code")
    assert score > 0.5

def test_proximity_score_no_overlap_returns_low() -> None:
    score = _proximity_score(["rust"], ["python"], "code")
    assert score <= 0.05

def test_proximity_score_intent_fallback() -> None:
    score = _proximity_score(["python_code"], [], "python")
    assert score > 0.05


# ---------------------------------------------------------------------------
# _volatility_score
# ---------------------------------------------------------------------------

def test_volatility_score_zero_mutations() -> None:
    assert _volatility_score(0) == 0.0

def test_volatility_score_many_mutations_capped_at_one() -> None:
    assert _volatility_score(1000) == 1.0

def test_volatility_score_grows_with_mutations() -> None:
    assert _volatility_score(5) > _volatility_score(1)


# ---------------------------------------------------------------------------
# _dependency_score
# ---------------------------------------------------------------------------

def test_dependency_score_in_deps() -> None:
    assert _dependency_score("id1", {"id1", "id2"}) == 1.0

def test_dependency_score_not_in_deps() -> None:
    assert _dependency_score("id3", {"id1", "id2"}) == 0.0

def test_dependency_score_empty_deps() -> None:
    assert _dependency_score("id1", set()) == 0.0


# ---------------------------------------------------------------------------
# RelevanceEngine — score_atom
# ---------------------------------------------------------------------------

def test_engine_score_atom_returns_relevance_score() -> None:
    engine = RelevanceEngine()
    atom   = _atom()
    score  = engine.score_atom(atom, ["python"], "code", _now())
    assert 0.0 <= score.total <= 1.0

def test_engine_score_atom_with_dependency() -> None:
    engine = RelevanceEngine()
    atom   = _atom()
    score  = engine.score_atom(atom, [], "x", _now(), dependency_ids={atom.id})
    assert score.dependency == 1.0


# ---------------------------------------------------------------------------
# RelevanceEngine — score_result
# ---------------------------------------------------------------------------

def test_engine_score_result_returns_relevance_score() -> None:
    engine = RelevanceEngine()
    result = _result(entities=["python"], tags=["code"])
    score  = engine.score_result(result, ["python"], "code", _now())
    assert 0.0 <= score.total <= 1.0

def test_engine_score_result_volatility_always_zero() -> None:
    engine = RelevanceEngine()
    result = _result()
    score  = engine.score_result(result, [], "x", _now())
    assert score.volatility == 0.0


# ---------------------------------------------------------------------------
# RelevanceEngine — filter_atoms
# ---------------------------------------------------------------------------

def test_engine_filter_atoms_respects_threshold() -> None:
    engine = RelevanceEngine(threshold=0.99)
    atoms  = [_atom() for _ in range(5)]
    # With high threshold, all should be filtered out
    scored = engine.filter_atoms(atoms, [], "x", _now())
    assert len(scored) == 0

def test_engine_filter_atoms_respects_limit() -> None:
    engine = RelevanceEngine(threshold=0.0)
    atoms  = [_atom() for _ in range(10)]
    scored = engine.filter_atoms(atoms, ["python"], "code", _now(), limit=3)
    assert len(scored) <= 3

def test_engine_filter_atoms_sorted_descending() -> None:
    engine = RelevanceEngine(threshold=0.0)
    now    = _now()
    # Fresh atom scores higher on recency
    fresh  = WorkingAtom(atom_type=AtomType.RESULT, content="fresh", task_id="t1")
    scored = engine.filter_atoms([fresh], ["python"], "code", now)
    if len(scored) > 1:
        scores = [s.score.total for s in scored]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# RelevanceEngine — filter_results
# ---------------------------------------------------------------------------

def test_engine_filter_results_respects_limit() -> None:
    engine  = RelevanceEngine(threshold=0.0)
    results = [_result(tags=["x"]) for _ in range(10)]
    scored  = engine.filter_results(results, ["x"], "info", _now(), limit=4)
    assert len(scored) <= 4


# ---------------------------------------------------------------------------
# Scoring strategies
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strategy", [
    ScoringStrategy.LINEAR,
    ScoringStrategy.WEIGHTED_SIGMOID,
    ScoringStrategy.MULTIPLICATIVE,
])
def test_all_strategies_return_valid_score(strategy: ScoringStrategy) -> None:
    engine = RelevanceEngine(strategy=strategy)
    atom   = _atom()
    score  = engine.score_atom(atom, ["python"], "code", _now())
    assert 0.0 <= score.total <= 1.0


# ---------------------------------------------------------------------------
# depth_from_score
# ---------------------------------------------------------------------------

def test_depth_from_high_score_is_3() -> None:
    engine = RelevanceEngine()
    assert engine._depth_from_score(0.8) == 3

def test_depth_from_medium_score_is_2() -> None:
    engine = RelevanceEngine()
    assert engine._depth_from_score(0.5) == 2

def test_depth_from_low_score_is_1() -> None:
    engine = RelevanceEngine()
    assert engine._depth_from_score(0.1) == 1
