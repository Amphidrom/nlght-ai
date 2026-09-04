# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Tests for HiveMindStoreCoordinator and HiveMindStoreCoordinatorFactory."""
from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from nlght.adapters.outbound.hive_mind.coordinator import (
    HiveMindStoreCoordinator,
    HiveMindStoreCoordinatorFactory,
)
from nlght.core.hive_mind.models import (
    AtomType,
    PromotionStatus,
    SessionSnapshot,
    TurnSummary,
    WriteIntent,
)


def _coordinator() -> HiveMindStoreCoordinator:
    return HiveMindStoreCoordinator()


def _session_root() -> Path:
    root = Path.cwd() / "test-output" / f"hive-mind-{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    return root


# ---------------------------------------------------------------------------
# Directives
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Conversation
# ---------------------------------------------------------------------------

def _turn(nr: int = 1) -> TurnSummary:
    return TurnSummary(turn_nr=nr, user_input="hi", intent="greet", topic="t")


class _MemoryBackend:
    def __init__(self) -> None:
        self.snapshots: dict[str, SessionSnapshot] = {}

    def exists(self, session_id: str) -> bool:
        return session_id in self.snapshots

    def load(self, session_id: str) -> SessionSnapshot | None:
        return self.snapshots.get(session_id)

    def save(self, snapshot: SessionSnapshot) -> None:
        self.snapshots[snapshot.session_id] = snapshot


def test_record_turn_and_get_recent() -> None:
    c = _coordinator()
    c.record_turn(_turn(1))
    c.record_turn(_turn(2))
    turns = c.get_recent_turns(5)
    assert len(turns) == 2


def test_get_recent_turns_limited() -> None:
    c = _coordinator()
    for i in range(10):
        c.record_turn(_turn(i))
    assert len(c.get_recent_turns(3)) == 3


# ---------------------------------------------------------------------------
# Working Memory + write + task lifecycle
# ---------------------------------------------------------------------------

def test_write_to_working_memory() -> None:
    c = _coordinator()
    c.open_branch("t1")
    intent = WriteIntent(atom_type=AtomType.SPEC, content="spec", task_id="t1")
    atom_id, conflict = c.write(intent)
    assert isinstance(atom_id, str)
    assert conflict is None
    assert len(c.working.read_active()) == 1


def test_write_promote_immediately_marks_atom() -> None:
    c = _coordinator()
    c.open_branch("turn")
    c.open_branch("t1")
    intent = WriteIntent(
        atom_type=AtomType.RESULT, content="result",
        task_id="t1", promote_immediately=True,
        entities=["x"],
    )
    c.write(intent)
    # close sub-branch → promoted atom moves to turn branch
    c.close_branch(promote=True)
    # close turn branch → promoted atoms returned (would go to SessionResultStore in executor)
    promoted = c.close_branch(promote=True)
    assert len(promoted) == 1
    assert promoted[0].content == "result"


def test_multiturn_branch_promotion_preserves_context_across_all_store_layers() -> None:
    c = _coordinator()

    c.record_turn(TurnSummary(
        turn_nr=1,
        user_input="Was ist zwischen USA und Iran passiert?",
        intent="informational",
        topic="usa-iran",
        entities=["USA", "Iran"],
        result_summary="Recherche gestartet",
    ))
    c.open_branch("turn:1")
    c.write(WriteIntent(
        atom_type=AtomType.CONTEXT,
        content="transient search candidate",
        task_id="turn:1",
        tags=["kind:source_candidate", "phase:discover"],
    ))
    c.open_branch("turn:1:evaluate")
    c.write(WriteIntent(
        atom_type=AtomType.EVAL,
        content="claim=Talks remain fragile | document_id=doc_usa_iran | status=verified",
        task_id="turn:1:evaluate",
        tags=["kind:evaluated_evidence", "phase:evaluate"],
        promote_immediately=True,
    ))

    phase_promoted = c.close_branch(promote=True)
    assert [atom.content for atom in phase_promoted] == [
        "claim=Talks remain fragile | document_id=doc_usa_iran | status=verified"
    ]
    active_after_phase = c.get_context([])["active_atoms"]
    assert [atom.content for atom in active_after_phase] == [
        "transient search candidate",
        "claim=Talks remain fragile | document_id=doc_usa_iran | status=verified",
    ]

    turn_promoted = c.close_branch(promote=True)
    stored_turn_1 = c.store_promoted_atoms(turn_promoted, turn_nr=1, entities=["USA", "Iran"])
    assert len(stored_turn_1) == 1
    assert stored_turn_1[0].tags == ["kind:evaluated_evidence", "phase:evaluate"]
    assert c.get_context([])["active_atoms"] == []

    c.record_turn(TurnSummary(
        turn_nr=2,
        user_input="Und was bedeutet das diplomatisch?",
        intent="follow_up",
        topic="usa-iran",
        entities=["Iran"],
        result_summary="Folgefrage zu Diplomatie",
    ))
    turn_2_context = c.get_context(["Iran"])

    assert [turn.turn_nr for turn in turn_2_context["recent_turns"]] == [1, 2]
    assert [result.content for result in turn_2_context["known_results"]] == [
        "claim=Talks remain fragile | document_id=doc_usa_iran | status=verified"
    ]
    assert turn_2_context["known_results"][0].tags == ["kind:evaluated_evidence", "phase:evaluate"]
    assert c.resolve_reference("iran").turn_nr == 2

    c.open_branch("turn:2")
    c.write(WriteIntent(
        atom_type=AtomType.RESULT,
        content="Diplomatic risk remains elevated.",
        task_id="turn:2",
        tags=["kind:final_synthesis", "phase:synthesise"],
        promote_immediately=True,
    ))
    turn_2_promoted = c.close_branch(promote=True)
    stored_turn_2 = c.store_promoted_atoms(turn_2_promoted, turn_nr=2, entities=["Iran"])

    assert len(stored_turn_2) == 1
    known_after_turn_2 = c.get_context(["Iran"])["known_results"]
    known_after_turn_2 = sorted(known_after_turn_2, key=lambda result: result.turn_nr)
    assert [result.turn_nr for result in known_after_turn_2] == [1, 2]
    assert [result.tags[0] for result in known_after_turn_2] == [
        "kind:evaluated_evidence",
        "kind:final_synthesis",
    ]


def test_start_and_finish_task_promotes_result() -> None:
    c = _coordinator()
    c.start_task("t1")
    c.write(WriteIntent(atom_type=AtomType.RESULT, content="done", task_id="t1"))
    promoted = c.finish_task("t1", entities=["e1"], turn_nr=1)
    assert len(promoted) == 1
    assert promoted[0].content == "done"


def test_finish_task_discards_spec() -> None:
    c = _coordinator()
    c.start_task("t1")
    c.write(WriteIntent(atom_type=AtomType.SPEC, content="spec", task_id="t1"))
    promoted = c.finish_task("t1", entities=[], turn_nr=1)
    assert promoted == []


def test_discard_task_clears_working_memory() -> None:
    c = _coordinator()
    c.start_task("t1")
    c.write(WriteIntent(atom_type=AtomType.RESULT, content="data", task_id="t1"))
    c.discard_task("t1")
    assert c.working.read_all() == []


def test_fail_task_clears_working_memory() -> None:
    c = _coordinator()
    c.start_task("t1")
    c.write(WriteIntent(atom_type=AtomType.RESULT, content="data", task_id="t1"))
    c.fail_task("t1")
    assert c.working.read_all() == []


# ---------------------------------------------------------------------------
# MentalModelCache
# ---------------------------------------------------------------------------

def test_mental_model_cache_initially_empty() -> None:
    c = _coordinator()
    assert c.mental_model_cache.valid is False


# ---------------------------------------------------------------------------
# Payload promotion
# ---------------------------------------------------------------------------

def test_promote_payload_stores_result() -> None:
    c    = _coordinator()
    rid, stored = c.promote_payload("content", entities=["e"], tags=["tag"], turn_nr=1)
    assert stored is True
    r = c.get_payload(["e"])
    assert r is not None
    assert r.content == "content"


def test_write_keeps_working_memory_tags() -> None:
    c = _coordinator()
    c.open_branch("t1")
    c.write(WriteIntent(
        atom_type=AtomType.CONTEXT,
        content="candidate",
        task_id="t1",
        tags=["kind:source_candidate", "phase:discover"],
    ))

    atoms = c.working.read_active()

    assert len(atoms) == 1
    assert atoms[0].tags == ["kind:source_candidate", "phase:discover"]


def test_promote_payload_deduplicates_identical() -> None:
    c = _coordinator()
    c.promote_payload("content", entities=["e"], tags=[], turn_nr=1)
    rid2, stored2 = c.promote_payload("content", entities=["e"], tags=[], turn_nr=2)
    assert stored2 is False


def test_promote_payload_supersedes_on_change() -> None:
    c = _coordinator()
    rid1, _ = c.promote_payload("v1", entities=["e"], tags=[], turn_nr=1)
    rid2, stored2 = c.promote_payload("v2", entities=["e"], tags=[], turn_nr=2)
    assert stored2 is True
    old = c.results.get(rid1)
    assert old.status == PromotionStatus.SUPERSEDED


def test_get_payload_none_when_empty() -> None:
    c = _coordinator()
    assert c.get_payload(["missing"]) is None


# ---------------------------------------------------------------------------
# resolve_reference
# ---------------------------------------------------------------------------

def test_resolve_reference_returns_turn() -> None:
    c = _coordinator()
    t = TurnSummary(turn_nr=1, user_input="about python", intent="info",
                    topic="t", entities=["python"])
    c.record_turn(t)
    result = c.resolve_reference("python")
    assert result is not None
    assert result.turn_nr == 1


# ---------------------------------------------------------------------------
# get_context / is_known
# ---------------------------------------------------------------------------

def test_get_context_returns_dict() -> None:
    c   = _coordinator()
    ctx = c.get_context(["e"])
    assert "recent_turns" in ctx
    assert "known_results" in ctx


def test_is_known_false_when_empty() -> None:
    c = _coordinator()
    assert c.is_known(["e"]) is False


def test_is_known_true_after_promotion() -> None:
    c = _coordinator()
    c.start_task("t1")
    c.write(WriteIntent(atom_type=AtomType.RESULT, content="data", task_id="t1"))
    c.finish_task("t1", entities=["e1"], turn_nr=1)
    assert c.is_known(["e1"]) is True


# ---------------------------------------------------------------------------
# write conflict detection
# ---------------------------------------------------------------------------

def test_write_promote_immediately_no_conflict_returned() -> None:
    # write() no longer detects conflicts — conflict detection only in promote_payload().
    # Two writes with promote_immediately=True both return None conflict.
    c = _coordinator()
    c.open_branch("t1")
    _id1, conflict1 = c.write(WriteIntent(
        atom_type=AtomType.RESULT, content="v1",
        task_id="t1", promote_immediately=True, entities=["e"],
    ))
    _id2, conflict2 = c.write(WriteIntent(
        atom_type=AtomType.RESULT, content="v2",
        task_id="t1", promote_immediately=True, entities=["e"],
    ))
    assert conflict1 is None
    assert conflict2 is None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def test_factory_creates_new_session() -> None:
    root = _session_root()
    try:
        factory = HiveMindStoreCoordinatorFactory(base_dir=str(root))
        coord = factory.get_or_create("sess-1")
        assert isinstance(coord, HiveMindStoreCoordinator)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_factory_saves_and_restores_turns() -> None:
    root = _session_root()
    try:
        factory = HiveMindStoreCoordinatorFactory(base_dir=str(root))
        coord = factory.get_or_create("sess-3")
        coord.record_turn(_turn(1))
        factory.save("sess-3", coord)

        coord2 = factory.get_or_create("sess-3")
        assert len(coord2.get_recent_turns(10)) == 1
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_factory_restores_multiturn_hive_mind_context_without_transient_working_memory() -> None:
    factory = HiveMindStoreCoordinatorFactory(backend=_MemoryBackend())
    coord = factory.get_or_create("sess-complex")
    coord.record_turn(TurnSummary(
        turn_nr=1,
        user_input="Merke: NLght nutzt HiveMind.",
        intent="remember",
        topic="nlght",
        entities=["NLght", "HiveMind"],
        result_summary="Plattform-Fakt gespeichert",
    ))
    coord.open_branch("turn:1")
    coord.write(WriteIntent(
        atom_type=AtomType.RESULT,
        content="NLght uses HiveMind as its session memory coordinator.",
        task_id="turn:1",
        tags=["user_fact", "architecture"],
        promote_immediately=True,
    ))
    promoted = coord.close_branch(promote=True)
    coord.store_promoted_atoms(promoted, turn_nr=1, entities=["NLght", "HiveMind"])
    coord.record_turn(TurnSummary(
        turn_nr=2,
        user_input="Was hast du dir gemerkt?",
        intent="recall",
        topic="nlght",
        entities=["NLght"],
        result_summary="HiveMind-Fakt abgerufen",
    ))

    factory.save("sess-complex", coord)
    restored = factory.get_or_create("sess-complex")
    context = restored.get_context(["NLght"])

    assert [turn.turn_nr for turn in context["recent_turns"]] == [1, 2]
    assert [result.content for result in context["known_results"]] == [
        "NLght uses HiveMind as its session memory coordinator."
    ]
    assert context["known_results"][0].tags == ["user_fact", "architecture"]
    assert context["active_atoms"] == []
    assert restored.is_known(["NLght"], tags=["user_fact"]) is True
    assert restored.resolve_reference("hivemind").turn_nr == 1
