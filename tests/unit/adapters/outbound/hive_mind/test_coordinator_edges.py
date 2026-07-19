# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock

from nlght.adapters.outbound.hive_mind.coordinator import (
    HiveMindStoreCoordinator,
    HiveMindStoreCoordinatorFactory,
)
from nlght.adapters.outbound.hive_mind.simple import (
    SimpleStoreCoordinator,
    SimpleStoreCoordinatorFactory,
)
from nlght.core.hive_mind.models import (
    AtomType,
    Directive,
    DirectivePriority,
    PromotionStatus,
    SessionResult,
    SessionSnapshot,
    TurnSummary,
    WriteIntent,
)


def _turn(n: int = 1) -> TurnSummary:
    return TurnSummary(turn_nr=n, user_input="input", intent="task", topic="topic", entities=["entity"], result_summary="result")


def test_simple_coordinator_task_branches_context_payload_slots_and_dump(tmp_path) -> None:
    store = SimpleStoreCoordinator()
    store.set_directive("tone", "brief", source="user", priority=DirectivePriority.HIGH)
    turn = _turn()
    store.record_turn(turn)
    assert store.get_directives() == {"tone": "brief"}
    assert store.next_turn_nr() == 2
    assert store.get_recent_turns(5) == [turn]

    store.start_task("task")
    store.write(WriteIntent(atom_type=AtomType.RESULT, content="Berlin fact", task_id="task", entities=["Berlin"], tags=["geo"], promote_immediately=True))
    store.write(WriteIntent(atom_type=AtomType.PLAN, content="plan", task_id="task", promote_immediately=True))
    assert len(store.get_context([])["active_atoms"]) == 2
    promoted = store.finish_task("task", ["Berlin"], turn_nr=1)
    assert [result.content for result in promoted] == ["Berlin fact"]
    assert store.is_known(["Berlin"], ["geo"]) is False
    assert store.is_known(["Berlin"])

    payload_id, stored = store.promote_payload("payload", ["file"], ["tag"], 2)
    assert stored and payload_id
    assert store.get_payload(["unused"]).content == "payload"
    assert store.resolve_reference("previous") is None

    store.open_branch("outer")
    store.open_branch("inner")
    store.write(WriteIntent(atom_type=AtomType.EVAL, content="nested", task_id="inner", promote_immediately=True))
    assert len(store.close_branch(promote=True)) == 1
    assert len(store.close_branch(promote=True)) == 1
    assert store.close_branch() == []
    assert len(store.store_promoted_atoms([], turn_nr=2)) == 0

    store.set_slot("workspace", SimpleNamespace(root_path=tmp_path))
    assert store.load_artifact("../unsafe") is None
    assert store.load_artifact("missing") is None
    artifacts = tmp_path / "memory_artifacts"
    artifacts.mkdir()
    (artifacts / "abc.py").write_text("print(1)")
    assert store.load_artifact("abc").extension == ".py"
    assert store.get_slot("created", lambda: []) == []
    store._dump("edge-test")

    store.start_task("discard")
    store.discard_task("discard")
    store._working["orphan"] = []
    store.fail_task("orphan")


def test_simple_factory_reuses_saves_and_evicts_expired() -> None:
    factory = SimpleStoreCoordinatorFactory(session_ttl_seconds=1)
    first = factory.get_or_create("session")
    assert factory.exists("session")
    assert factory.get_or_create("session") is first
    factory.save("session", first)
    factory._sessions["session"] = (first, datetime.now(UTC) - timedelta(seconds=2))
    assert not factory.exists("session")


def test_hive_coordinator_directives_tasks_payload_conflicts_artifacts_and_dump(tmp_path) -> None:
    invalidated: list[list[str]] = []
    store = HiveMindStoreCoordinator(on_invalidate=invalidated.append)
    assert store.set_directive("tone", "brief", source="user") is None
    assert store.set_directive("tone", "brief", source="inferred") is None
    assert store.set_directive("tone", "verbose", source="inferred") is None
    store.set_directive("language", "de")
    assert len(store.get_directives_list()) == 2

    turn = _turn()
    store.record_turn(turn)
    assert store.next_turn_nr() == 2
    assert store.get_recent_turns(1) == [turn]

    store.start_task("task")
    store.write(WriteIntent(atom_type=AtomType.RESULT, content="fact", task_id="task", promote_immediately=True))
    store.write(WriteIntent(atom_type=AtomType.PLAN, content="plan", task_id="task"))
    assert len(store.finish_task("task", ["entity"], 1)) == 1
    assert invalidated[-1] == ["entity"]
    assert store.is_known(["entity"])
    assert store.get_context([])["known_results"]
    assert store.get_context(["entity"])["known_results"]

    first_id, first_stored = store.promote_payload("v1", ["file"], ["source"], 1)
    same_id, same_stored = store.promote_payload("v1", ["file"], ["source"], 2)
    second_id, second_stored = store.promote_payload("v2", ["file"], ["source"], 3)
    assert (first_id, True) == (same_id, first_stored)
    assert second_stored and second_id != first_id
    assert store.get_payload(["file"]).content == "v2"
    assert store.get_payload(["missing"]) is None
    assert store._check_conflict([], "x") is None
    assert store._check_conflict(["file"], "different") is not None

    store.open_branch("discard")
    store.discard_task("discard")
    store.open_branch("failure")
    store.fail_task("failure")
    store.set_slot("workspace", SimpleNamespace(root_path=tmp_path))
    assert store.load_artifact("bad/path") is None
    assert store.load_artifact("missing") is None
    artifacts = tmp_path / "memory_artifacts"
    artifacts.mkdir()
    (artifacts / "id.json").write_text("{}")
    assert store.load_artifact("id").content == "{}"
    store._dump("edge-test")


def test_hive_factory_creates_restores_and_preserves_snapshot_creation_time() -> None:
    backend = MagicMock()
    backend.exists.side_effect = [False, True]
    backend.load.side_effect = [None]
    factory = HiveMindStoreCoordinatorFactory(backend=backend)
    created = factory.get_or_create("new")
    assert isinstance(created, HiveMindStoreCoordinator)

    now = datetime.now(UTC)
    snapshot = SessionSnapshot(
        session_id="saved",
        directives=[Directive("tone", "brief")],
        turns=[_turn(2), _turn(1)],
        results=[SessionResult(content="known", status=PromotionStatus.FINAL)],
        interrupted_task_ids=["task"],
        slots={"slot": "value"},
        created_at=now,
        updated_at=now,
    )
    backend.load.side_effect = [snapshot]
    restored = factory.get_or_create("saved")
    assert restored.get_directives() == {"tone": "brief"}
    assert [turn.turn_nr for turn in restored.get_recent_turns(2)] == [1, 2]
    assert restored.get_slot("slot", lambda: None) == "value"

    restored.open_branch("running")
    backend.load.side_effect = [snapshot]
    factory.save("saved", restored)
    saved = backend.save.call_args.args[0]
    assert saved.created_at == now
    assert saved.interrupted_task_ids == ["running"]
