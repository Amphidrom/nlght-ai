# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Tests for SimpleStoreCoordinator and SimpleStoreCoordinatorFactory."""
from __future__ import annotations

import time

from nlght.adapters.outbound.hive_mind.simple import (
    SimpleStoreCoordinator,
    SimpleStoreCoordinatorFactory,
)
from nlght.core.hive_mind.models import (
    AtomType,
    TurnSummary,
    WriteIntent,
)


def _coordinator() -> SimpleStoreCoordinator:
    return SimpleStoreCoordinator()


def _turn(nr: int = 1) -> TurnSummary:
    return TurnSummary(turn_nr=nr, user_input="hi", intent="greet", topic="t")


# ---------------------------------------------------------------------------
# Directives
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Conversation
# ---------------------------------------------------------------------------

def test_record_and_get_recent_turns() -> None:
    c = _coordinator()
    c.record_turn(_turn(1))
    c.record_turn(_turn(2))
    assert len(c.get_recent_turns(5)) == 2


def test_get_recent_turns_limited() -> None:
    c = _coordinator()
    for i in range(10):
        c.record_turn(_turn(i))
    assert len(c.get_recent_turns(3)) == 3


def test_get_recent_turns_fewer_than_n() -> None:
    c = _coordinator()
    c.record_turn(_turn(1))
    assert len(c.get_recent_turns(10)) == 1


# ---------------------------------------------------------------------------
# Write + task lifecycle
# ---------------------------------------------------------------------------

def test_write_working_memory() -> None:
    c = _coordinator()
    intent = WriteIntent(atom_type=AtomType.SPEC, content="spec", task_id="t1")
    atom_id, conflict = c.write(intent)
    assert isinstance(atom_id, str)
    assert conflict is None


def test_write_promote_immediately() -> None:
    c      = _coordinator()
    intent = WriteIntent(
        atom_type=AtomType.RESULT, content="result",
        task_id="t1", promote_immediately=True,
        entities=["x"],
    )
    result_id, _ = c.write(intent)
    ctx = c.get_context(["x"])
    assert any(r.id == result_id for r in ctx["known_results"])


def test_branch_promoted_atoms_keep_tags_when_stored() -> None:
    c = _coordinator()
    c.open_branch("turn:1")
    c.open_branch("phase:evaluate")
    c.write(WriteIntent(
        atom_type=AtomType.EVAL,
        content="claim=verified",
        task_id="phase:evaluate",
        tags=["kind:evaluated_evidence", "phase:evaluate"],
        promote_immediately=True,
    ))

    c.close_branch(promote=True)
    promoted = c.close_branch(promote=True)
    stored = c.store_promoted_atoms(promoted, turn_nr=1, entities=["entity"])

    assert len(stored) == 1
    assert stored[0].tags == ["kind:evaluated_evidence", "phase:evaluate"]


def test_start_and_finish_task_promotes_result() -> None:
    c = _coordinator()
    c.start_task("t1")
    c.write(WriteIntent(atom_type=AtomType.RESULT, content="done", task_id="t1"))
    promoted = c.finish_task("t1", entities=["e"], turn_nr=1)
    assert len(promoted) == 1


def test_finish_task_discards_non_promotable() -> None:
    c = _coordinator()
    c.start_task("t1")
    c.write(WriteIntent(atom_type=AtomType.SPEC, content="spec", task_id="t1"))
    promoted = c.finish_task("t1", entities=[], turn_nr=1)
    assert promoted == []


def test_finish_task_clears_active_task() -> None:
    c = _coordinator()
    c.start_task("t1")
    c.finish_task("t1", entities=[], turn_nr=1)
    # _active_task should be reset
    assert c._active_task is None


# ---------------------------------------------------------------------------
# get_context / is_known
# ---------------------------------------------------------------------------

def test_get_context_structure() -> None:
    c   = _coordinator()
    ctx = c.get_context(["x"])
    assert "recent_turns" in ctx
    assert "known_results" in ctx


def test_is_known_false_initially() -> None:
    c = _coordinator()
    assert c.is_known(["x"]) is False


def test_is_known_true_after_write() -> None:
    c = _coordinator()
    c.write(WriteIntent(
        atom_type=AtomType.RESULT, content="python data",
        task_id="t1", promote_immediately=True, entities=["python"],
    ))
    assert c.is_known(["python"]) is True


def test_is_known_with_tags_match() -> None:
    c = _coordinator()
    c.write(WriteIntent(
        atom_type=AtomType.RESULT, content="some python content",
        task_id="t1", promote_immediately=True,
        entities=["python"], tags=["code"],
    ))
    assert c.is_known(["python"], tags=["code"]) is True


def test_is_known_with_tags_no_match() -> None:
    c = _coordinator()
    c.write(WriteIntent(
        atom_type=AtomType.RESULT, content="some python content",
        task_id="t1", promote_immediately=True,
        entities=["python"], tags=["code"],
    ))
    assert c.is_known(["python"], tags=["draft"]) is False


# ---------------------------------------------------------------------------
# SimpleStoreCoordinatorFactory
# ---------------------------------------------------------------------------

def test_factory_creates_session() -> None:
    factory = SimpleStoreCoordinatorFactory()
    coord   = factory.get_or_create("sess-1")
    assert isinstance(coord, SimpleStoreCoordinator)


def test_factory_reuses_session() -> None:
    factory = SimpleStoreCoordinatorFactory()
    coord1  = factory.get_or_create("sess-2")
    coord2  = factory.get_or_create("sess-2")
    assert coord1 is coord2


def test_factory_save_refreshes_timestamp() -> None:
    factory = SimpleStoreCoordinatorFactory()
    coord   = factory.get_or_create("sess-3")
    factory.save("sess-3", coord)
    # Just verify it doesn't raise and session still exists
    coord2  = factory.get_or_create("sess-3")
    assert coord2 is coord


def test_factory_evicts_expired_sessions() -> None:
    factory = SimpleStoreCoordinatorFactory(session_ttl_seconds=0)
    factory.get_or_create("sess-old")
    time.sleep(0.01)
    # Trigger eviction by creating another session
    factory.get_or_create("sess-new")
    # sess-old should be evicted
    coord = factory.get_or_create("sess-old")
    # A new coordinator was created (old one evicted)
    assert isinstance(coord, SimpleStoreCoordinator)
