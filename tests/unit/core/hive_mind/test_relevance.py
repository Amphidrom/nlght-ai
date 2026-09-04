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
# RelevanceEngine — ranking, which discards nothing
# ---------------------------------------------------------------------------
#
# These replace four tests that asserted the opposite: that a score threshold and
# a per-type limit removed candidates here. They did, and that was the defect —
# a durable fact about the user could be dropped for being the eleventh session
# result, before anything knew what it would cost to lose it (ADR-0056).

def test_ranking_atoms_keeps_every_one_of_them() -> None:
    engine = RelevanceEngine()
    atoms = [_atom() for _ in range(10)]

    assert len(engine.rank_atoms(atoms, ["python"], "code", _now())) == 10


def test_ranking_results_keeps_every_one_of_them() -> None:
    engine = RelevanceEngine()
    results = [_result(tags=["x"]) for _ in range(10)]

    assert len(engine.rank_results(results, ["x"], "info", _now())) == 10


def test_a_poorly_matching_candidate_is_ranked_last_and_not_removed() -> None:
    """Low relevance orders it; it does not delete it.

    "This does not match the moment" and "this may be forgotten" are different
    statements, and only the second can be made with retention and cost in view.
    """
    engine = RelevanceEngine()
    now = _now()
    matching = _result(tags=["python"])
    unrelated = _result(tags=["knitting"])

    ranked = engine.rank_results([unrelated, matching], ["python"], "code", now)

    assert len(ranked) == 2
    assert ranked[0].result is matching


def test_an_atom_is_scored_on_the_entities_it_carries() -> None:
    """The signal the caller always supplied and every writer dropped.

    `WriteIntent.entities` has always existed and `memory_ingestion` fills it
    with real extracted entities; all three `write()` implementations mapped five
    fields and let it fall. `WorkingAtom` had no field for it, so proximity
    scored every atom against an empty list and returned the same floor — every
    atom in the platform scored exactly 0.2558, whatever it said (ADR-0057).

    This test replaces the one that pinned that defect. That one documented
    something known to be wrong, not a contract, and it goes with the defect.
    """
    engine = RelevanceEngine()
    now = _now()
    about_python = WorkingAtom(atom_type=AtomType.RESULT, content="c", task_id="t",
                               entities=["python"])
    about_knitting = WorkingAtom(atom_type=AtomType.RESULT, content="c", task_id="t",
                                 entities=["knitting"])

    matching = engine.score_atom(about_python, ["python"], "code", now)
    unrelated = engine.score_atom(about_knitting, ["python"], "code", now)

    assert matching.proximity > unrelated.proximity
    assert matching.total > unrelated.total


def test_only_proximity_moves_when_only_the_entities_differ() -> None:
    engine = RelevanceEngine()
    now = _now()
    made = datetime.now(UTC)
    one = WorkingAtom(atom_type=AtomType.RESULT, content="c", task_id="t",
                      entities=["python"], created_at=made)
    other = WorkingAtom(atom_type=AtomType.RESULT, content="c", task_id="t",
                        entities=["knitting"], created_at=made)

    a, b = (engine.score_atom(x, ["python"], "code", now) for x in (one, other))

    assert a.proximity != b.proximity
    assert (a.recency, a.volatility, a.dependency) == (b.recency, b.volatility, b.dependency)


def test_only_recency_moves_when_only_the_age_differs() -> None:
    """`created_at` is what an atom has; `updated_at` is what it never had.

    The scorer read the second through a `getattr` default, so recency was 1.0
    for every atom regardless of age.
    """
    engine = RelevanceEngine()
    now = _now()
    fresh = WorkingAtom(atom_type=AtomType.RESULT, content="c", task_id="t",
                        entities=["python"], created_at=now)
    old = WorkingAtom(atom_type=AtomType.RESULT, content="c", task_id="t",
                      entities=["python"], created_at=now - timedelta(hours=1))

    a, b = (engine.score_atom(x, ["python"], "code", now) for x in (fresh, old))

    assert a.recency > b.recency
    assert (a.proximity, a.volatility, a.dependency) == (b.proximity, b.volatility, b.dependency)


def test_volatility_stays_zero_because_an_atom_is_never_rewritten() -> None:
    # Honest rather than synthesised: nothing in the platform modifies an atom
    # after it is written — branches move them, never rewrite them — so this is
    # the same statement `score_result` makes about session results.
    engine = RelevanceEngine()
    atom = WorkingAtom(atom_type=AtomType.RESULT, content="c", task_id="t",
                       entities=["python"])

    assert engine.score_atom(atom, ["python"], "code", _now()).volatility == 0.0


def test_dependency_stays_neutral_while_nothing_populates_it() -> None:
    """A dormant dimension, and platform-wide rather than an atom's gap.

    `ContextSnapshot.dependency_ids` is a constructor parameter no production
    path passes, so dependency is 0 for every sort. Left neutral rather than
    activated to make a number look richer.
    """
    engine = RelevanceEngine()
    now = _now()
    atom = WorkingAtom(atom_type=AtomType.RESULT, content="c", task_id="t",
                       entities=["python"])

    assert engine.score_atom(atom, ["python"], "code", now).dependency == 0.0
    assert engine.score_atom(atom, ["python"], "code", now, set()).dependency == 0.0


def test_an_atom_and_a_result_with_the_same_signals_score_the_same() -> None:
    """One definition of relevance, two field mappings onto it.

    `score_result` hardcodes `volatility = 0` because a session result does not
    mutate. That is an input, and with an atom that has not mutated either the
    two must agree exactly — if they do not, there are two meanings of relevance
    again.
    """
    from nlght.core.hive_mind.models import RelevanceInputs

    engine = RelevanceEngine()
    now = _now()
    signals = RelevanceInputs(
        updated_at=now, entities=("alpha",), mutation_count=0, identity="x",
    )

    # Whatever produced them, identical signals give an identical score. That is
    # the whole of "one definition of relevance".
    assert engine.score(signals, ["alpha"], "x", now) == engine.score(
        signals, ["alpha"], "x", now
    )
    assert engine.score(signals, ["alpha"], "x", now).proximity > engine.score(
        RelevanceInputs(updated_at=now, entities=("beta",)), ["alpha"], "x", now
    ).proximity


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


def test_the_signals_survive_a_working_memory_round_trip() -> None:
    """Both serialisers, because the loss would otherwise return one level later.

    `WorkingMemory.snapshot` uses `asdict` and picks a new field up for free;
    `pipeline_snapshot` writes an explicit field list and would have dropped it
    silently. Adding a model field and fixing the production writer is not
    enough — the same information has to survive every path it is written to.
    """
    from nlght.adapters.outbound.hive_mind.pipeline_snapshot import _de_atom, _ser_atom
    from nlght.core.hive_mind.stores import WorkingMemory

    made = datetime.now(UTC) - timedelta(minutes=7)
    atom = WorkingAtom(atom_type=AtomType.RESULT, content="c", task_id="t",
                       entities=["python", "spring"], created_at=made)

    memory = WorkingMemory()
    memory.branch("b")
    memory.write(atom)
    restored = WorkingMemory.from_snapshot(memory.snapshot()).read_all()[0]

    assert restored.entities == ["python", "spring"]
    assert restored.created_at == made

    through_pipeline = _de_atom(_ser_atom(atom))
    assert through_pipeline.entities == ["python", "spring"]
    assert through_pipeline.created_at == made


def test_the_entities_a_caller_supplies_reach_the_atom() -> None:
    """Every writer, because all three dropped it the same way."""
    from nlght.adapters.outbound.hive_mind.simple import SimpleStoreCoordinator
    from nlght.core.hive_mind.models import WriteIntent

    store = SimpleStoreCoordinator()
    store.write(WriteIntent(
        atom_type=AtomType.RESULT, content="c", task_id="t",
        entities=["python", "spring"],
    ))

    written = next(iter(store._working.values()))[0]  # noqa: SLF001
    assert written.entities == ["python", "spring"]


def test_the_key_and_kind_survive_both_serialisers() -> None:
    """The same boundary as `entities`, and the same two paths out.

    `WorkingMemory.snapshot` picks a new field up through `asdict`;
    `pipeline_snapshot` writes an explicit list and would drop it silently. A
    compact representation is built from these two fields, so losing them one
    level down would quietly remove the ability to say anything briefly.
    """
    from nlght.adapters.outbound.hive_mind.pipeline_snapshot import _de_atom, _ser_atom
    from nlght.core.hive_mind.stores import WorkingMemory

    atom = WorkingAtom(atom_type=AtomType.RESULT, content="c", task_id="t",
                       key="ssl-keystore-path", kind="fact")

    memory = WorkingMemory()
    memory.branch("b")
    memory.write(atom)
    restored = WorkingMemory.from_snapshot(memory.snapshot()).read_all()[0]
    assert (restored.key, restored.kind) == ("ssl-keystore-path", "fact")

    through = _de_atom(_ser_atom(atom))
    assert (through.key, through.kind) == ("ssl-keystore-path", "fact")


def test_an_older_snapshot_without_key_or_kind_still_reads() -> None:
    from nlght.adapters.outbound.hive_mind.pipeline_snapshot import _de_atom

    older = {
        "id": "a1", "atom_type": "RESULT", "content": "c", "task_id": "t",
        "entities": [], "tags": [], "promote_to_parent": False,
        "created_at": datetime.now(UTC).isoformat(),
    }

    atom = _de_atom(older)

    assert (atom.key, atom.kind) == ("", "")


def test_the_key_a_caller_supplies_reaches_the_atom() -> None:
    from nlght.adapters.outbound.hive_mind.simple import SimpleStoreCoordinator
    from nlght.core.hive_mind.models import WriteIntent

    store = SimpleStoreCoordinator()
    store.write(WriteIntent(
        atom_type=AtomType.RESULT, content="c", task_id="t",
        key="ssl-keystore-path", kind="fact",
    ))

    written = next(iter(store._working.values()))[0]  # noqa: SLF001
    assert (written.key, written.kind) == ("ssl-keystore-path", "fact")
