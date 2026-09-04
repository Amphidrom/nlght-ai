# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Tests for nlght.core.hive_mind.stores — the four store classes."""
from __future__ import annotations

import pytest

from nlght.core.hive_mind.models import (
    AtomType,
    PromotionStatus,
    SessionResult,
    TurnSummary,
    WorkingAtom,
)
from nlght.core.hive_mind.stores import (
    ConversationStore,
    SessionResultStore,
    WorkingMemory,
)

# ---------------------------------------------------------------------------
# ConversationStore
# ---------------------------------------------------------------------------

def _turn(nr: int, entities: list[str] | None = None) -> TurnSummary:
    return TurnSummary(
        turn_nr=nr, user_input="hi", intent="greet",
        topic="test", entities=entities or [],
    )

def test_conversation_store_append_and_get() -> None:
    store = ConversationStore()
    t = _turn(1)
    store.append(t)
    assert store.get_turn(1) is t

def test_conversation_store_last_n() -> None:
    store = ConversationStore()
    for i in range(5):
        store.append(_turn(i))
    assert len(store.last_n(3)) == 3
    assert len(store.last_n(10)) == 5

def test_conversation_store_resolve_reference_with_entity() -> None:
    store = ConversationStore()
    store.append(_turn(1, entities=["python"]))
    store.append(_turn(2, entities=["rust"]))
    result = store.resolve_reference("python")
    assert result is not None
    assert result.turn_nr == 1

def test_conversation_store_resolve_reference_fallback() -> None:
    store = ConversationStore()
    store.append(_turn(1))
    result = store.resolve_reference("unknown_entity")
    assert result is not None
    assert result.turn_nr == 1

def test_conversation_store_resolve_reference_empty() -> None:
    store = ConversationStore()
    assert store.resolve_reference("x") is None

def test_conversation_store_find_by_entities() -> None:
    store = ConversationStore()
    store.append(_turn(1, entities=["python"]))
    store.append(_turn(2, entities=["rust"]))
    store.append(_turn(3, entities=["python", "rust"]))
    results = store.find_by_entities(["python"])
    assert all(t.turn_nr in {1, 3} for t in results)

def test_conversation_store_current_turn_nr() -> None:
    store = ConversationStore()
    assert store.current_turn_nr == 0
    store.append(_turn(1))
    assert store.current_turn_nr == 1


# ---------------------------------------------------------------------------
# SessionResultStore
# ---------------------------------------------------------------------------

def _result(content: str, entities: list[str] | None = None, tags: list[str] | None = None) -> SessionResult:
    return SessionResult(content=content, entities=entities or [], tags=tags or [])

def test_session_result_store_store_and_get() -> None:
    store = SessionResultStore()
    r = _result("hello", entities=["world"])
    store.store(r)
    assert store.get(r.id) is r

def test_session_result_store_find_by_entities() -> None:
    store = SessionResultStore()
    r1 = _result("a", entities=["python"])
    r2 = _result("b", entities=["rust"])
    store.store(r1)
    store.store(r2)
    found = store.find_by_entities(["python"])
    assert r1 in found
    assert r2 not in found

def test_session_result_store_find_by_tags() -> None:
    store = SessionResultStore()
    r1 = _result("a", tags=["code", "python"])
    r2 = _result("b", tags=["code"])
    store.store(r1)
    store.store(r2)
    found = store.find_by_tags(["code", "python"])
    assert r1 in found
    assert r2 not in found

def test_session_result_store_find_by_tags_empty() -> None:
    store = SessionResultStore()
    assert store.find_by_tags([]) == []

def test_session_result_store_exists_true() -> None:
    store = SessionResultStore()
    store.store(_result("data", entities=["x"]))
    assert store.exists(["x"]) is True

def test_session_result_store_exists_false() -> None:
    store = SessionResultStore()
    assert store.exists(["y"]) is False

def test_session_result_store_exists_with_tags() -> None:
    store = SessionResultStore()
    store.store(_result("data", entities=["x"], tags=["final"]))
    assert store.exists(["x"], tags=["final"]) is True
    assert store.exists(["x"], tags=["draft"]) is False

def test_session_result_store_update_status() -> None:
    store = SessionResultStore()
    r = _result("data")
    store.store(r)
    assert store.update_status(r.id, PromotionStatus.SUPERSEDED) is True
    assert store.get(r.id).status == PromotionStatus.SUPERSEDED

def test_session_result_store_update_status_missing() -> None:
    store = SessionResultStore()
    assert store.update_status("nonexistent", PromotionStatus.FINAL) is False

def test_session_result_store_all_final() -> None:
    store = SessionResultStore()
    r1 = _result("a")
    r2 = SessionResult(content="b", status=PromotionStatus.SUPERSEDED)
    store.store(r1)
    store.store(r2)
    finals = store.all_final()
    assert r1 in finals
    assert r2 not in finals


# ---------------------------------------------------------------------------
# WorkingMemory
# ---------------------------------------------------------------------------

def _atom(task_id: str, atom_type: AtomType = AtomType.RESULT, promoted: bool = False) -> WorkingAtom:
    return WorkingAtom(atom_type=atom_type, content="data", task_id=task_id, promote_to_parent=promoted)


# branch / write / read_active

def test_working_memory_branch_and_write() -> None:
    mem = WorkingMemory()
    mem.branch("t1")
    atom = _atom("t1")
    mem.write(atom)
    assert mem.read_active() == [atom]

def test_working_memory_write_without_branch_fails_explicitly() -> None:
    mem = WorkingMemory()

    with pytest.raises(RuntimeError, match="active WorkingMemory branch"):
        mem.write(_atom("orphan"))

def test_working_memory_read_active_empty_when_no_branch() -> None:
    mem = WorkingMemory()
    assert mem.read_active() == []

def test_working_memory_read_active_by_type() -> None:
    mem = WorkingMemory()
    mem.branch("t1")
    a_result = _atom("t1", AtomType.RESULT)
    a_spec   = _atom("t1", AtomType.SPEC)
    mem.write(a_result)
    mem.write(a_spec)
    assert mem.read_active(AtomType.RESULT) == [a_result]

def test_working_memory_active_task_id() -> None:
    mem = WorkingMemory()
    assert mem.active_task_id is None
    mem.branch("t1")
    assert mem.active_task_id == "t1"

def test_working_memory_depth() -> None:
    mem = WorkingMemory()
    assert mem.depth == 0
    mem.branch("turn")
    assert mem.depth == 1
    mem.branch("observe")
    assert mem.depth == 2


# read_all across stack

def test_working_memory_read_all_across_branches() -> None:
    mem = WorkingMemory()
    mem.branch("turn")
    mem.write(_atom("turn", AtomType.RESULT))
    mem.branch("observe")
    mem.write(_atom("observe", AtomType.SPEC))
    assert len(mem.read_all()) == 2

def test_working_memory_read_all_by_type() -> None:
    mem = WorkingMemory()
    mem.branch("t1")
    mem.write(_atom("t1", AtomType.RESULT))
    mem.write(_atom("t1", AtomType.SPEC))
    assert len(mem.read_all(AtomType.RESULT)) == 1

def test_working_memory_read_active_vs_read_all_across_nested_stack() -> None:
    mem = WorkingMemory()
    mem.branch("turn")
    root_atom = WorkingAtom(atom_type=AtomType.CONTEXT, content="root", task_id="turn")
    mem.write(root_atom)
    mem.branch("phase")
    phase_atom = WorkingAtom(atom_type=AtomType.EVAL, content="phase", task_id="phase", promote_to_parent=True)
    mem.write(phase_atom)
    mem.branch("tool")
    tool_atom = WorkingAtom(atom_type=AtomType.RESULT, content="tool", task_id="tool", promote_to_parent=True)
    mem.write(tool_atom)

    assert mem.read_active() == [tool_atom]
    assert mem.read_all() == [root_atom, phase_atom, tool_atom]
    assert mem.read_all(AtomType.EVAL) == [phase_atom]


# merge — promote / discard

def test_working_memory_merge_promotes_marked_atoms() -> None:
    mem = WorkingMemory()
    mem.branch("turn")
    mem.branch("observe")
    kept    = _atom("observe", AtomType.RESULT, promoted=True)
    dropped = _atom("observe", AtomType.CONTEXT)
    mem.write(kept)
    mem.write(dropped)
    promoted = mem.merge()
    assert promoted == [kept]
    # promoted atom now lives in parent (turn) branch
    assert kept in mem.read_active()
    assert dropped not in mem.read_active()

def test_working_memory_merge_returns_promoted_when_no_parent() -> None:
    mem = WorkingMemory()
    mem.branch("turn")
    atom = _atom("turn", AtomType.RESULT, promoted=True)
    mem.write(atom)
    returned = mem.merge()
    assert returned == [atom]
    assert mem.depth == 0

def test_working_memory_merge_empty_returns_empty() -> None:
    mem = WorkingMemory()
    assert mem.merge() == []

def test_working_memory_merge_discards_non_promoted() -> None:
    mem = WorkingMemory()
    mem.branch("turn")
    mem.branch("observe")
    mem.write(_atom("observe", AtomType.CONTEXT))
    promoted = mem.merge()
    assert promoted == []
    assert mem.read_active() == []

def test_working_memory_merge_promote_false_discards_even_promoted_atoms() -> None:
    mem = WorkingMemory()
    mem.branch("turn")
    mem.branch("phase")
    promoted_atom = _atom("phase", AtomType.RESULT, promoted=True)
    mem.write(promoted_atom)

    returned = mem.merge(promote=False)

    assert returned == []
    assert mem.depth == 1
    assert mem.active_task_id == "turn"
    assert mem.read_active() == []

def test_working_memory_sibling_branches_are_isolated_after_merge() -> None:
    mem = WorkingMemory()
    mem.branch("turn")
    turn_atom = WorkingAtom(atom_type=AtomType.CONTEXT, content="turn", task_id="turn")
    mem.write(turn_atom)

    mem.branch("phase:a")
    kept_a = WorkingAtom(atom_type=AtomType.RESULT, content="kept-a", task_id="phase:a", promote_to_parent=True)
    dropped_a = WorkingAtom(atom_type=AtomType.CONTEXT, content="dropped-a", task_id="phase:a")
    mem.write(kept_a)
    mem.write(dropped_a)
    assert mem.merge(promote=True) == [kept_a]

    mem.branch("phase:b")
    kept_b = WorkingAtom(atom_type=AtomType.RESULT, content="kept-b", task_id="phase:b", promote_to_parent=True)
    mem.write(kept_b)
    assert mem.read_active() == [kept_b]
    assert mem.read_all() == [turn_atom, kept_a, kept_b]
    assert mem.merge(promote=False) == []

    assert mem.active_task_id == "turn"
    assert mem.read_active() == [turn_atom, kept_a]
    assert dropped_a not in mem.read_active()
    assert kept_b not in mem.read_active()


# nested branch cascade

def test_working_memory_nested_promotion_cascade() -> None:
    mem = WorkingMemory()
    mem.branch("turn")
    mem.branch("observe")
    mem.branch("inner")
    payload = _atom("inner", AtomType.PAYLOAD, promoted=True)
    mem.write(payload)
    mem.merge()                # inner → observe
    assert payload in mem.read_active()
    mem.merge()                # observe → turn (not promoted further — atom.promote_to_parent stays True)
    assert payload in mem.read_active()

def test_working_memory_full_turn_branch_returns_only_cascaded_promoted_atoms() -> None:
    mem = WorkingMemory()
    mem.branch("turn")
    root_noise = WorkingAtom(atom_type=AtomType.CONTEXT, content="root-noise", task_id="turn")
    mem.write(root_noise)
    mem.branch("phase")
    phase_result = WorkingAtom(
        atom_type=AtomType.RESULT,
        content="phase-result",
        task_id="phase",
        promote_to_parent=True,
    )
    phase_noise = WorkingAtom(atom_type=AtomType.CONTEXT, content="phase-noise", task_id="phase")
    mem.write(phase_result)
    mem.write(phase_noise)

    assert mem.merge(promote=True) == [phase_result]
    assert mem.read_active() == [root_noise, phase_result]
    assert mem.merge(promote=True) == [phase_result]
    assert mem.depth == 0


# snapshot roundtrip

def test_working_memory_snapshot_roundtrip() -> None:
    mem = WorkingMemory()
    mem.branch("turn:abc:1")
    mem.write(_atom("turn:abc:1", AtomType.RESULT, promoted=True))
    mem.branch("observe:abc:1")
    mem.write(_atom("observe:abc:1", AtomType.CONTEXT))

    snap = mem.snapshot()
    restored = WorkingMemory.from_snapshot(snap)

    assert restored.depth == 2
    assert restored.active_task_id == "observe:abc:1"
    restored_all = restored.read_all()
    assert len(restored_all) == 2
    promoted_atoms = [a for a in restored_all if a.promote_to_parent]
    assert len(promoted_atoms) == 1

def test_working_memory_snapshot_restored_stack_continues_promotion_cascade() -> None:
    mem = WorkingMemory()
    mem.branch("turn")
    mem.branch("phase")
    mem.branch("tool")
    result = WorkingAtom(
        atom_type=AtomType.RESULT,
        content="restored-result",
        task_id="tool",
        tags=["kind:test_result"],
        promote_to_parent=True,
    )
    mem.write(result)

    restored = WorkingMemory.from_snapshot(mem.snapshot())

    assert restored.depth == 3
    assert restored.active_task_id == "tool"
    tool_promoted = restored.merge(promote=True)
    assert [atom.content for atom in tool_promoted] == ["restored-result"]
    assert restored.active_task_id == "phase"
    assert [atom.content for atom in restored.read_active()] == ["restored-result"]
    phase_promoted = restored.merge(promote=True)
    assert [atom.content for atom in phase_promoted] == ["restored-result"]
    assert restored.active_task_id == "turn"
    turn_promoted = restored.merge(promote=True)
    assert [atom.content for atom in turn_promoted] == ["restored-result"]
    assert turn_promoted[0].tags == ["kind:test_result"]
    assert restored.depth == 0

def test_working_memory_snapshot_empty() -> None:
    mem = WorkingMemory()
    snap = mem.snapshot()
    assert snap == []
    restored = WorkingMemory.from_snapshot(snap)
    assert restored.depth == 0
