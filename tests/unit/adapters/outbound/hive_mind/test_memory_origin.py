# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""What session memory keeps inherits the origin of what the run had read (ADR-0074)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from nlght.adapters.outbound.hive_mind.coordinator import HiveMindStoreCoordinator
from nlght.adapters.outbound.hive_mind.pipeline_snapshot import _de_atom, _ser_atom
from nlght.adapters.outbound.hive_mind.simple import SimpleStoreCoordinator
from nlght.core.hive_mind.models import (
    AtomType,
    SessionResult,
    SessionSnapshot,
    TurnSummary,
    WorkingAtom,
    WriteIntent,
)
from nlght.core.hive_mind.stores import WorkingMemory
from nlght.core.model.trust import (
    PLATFORM_TEXT,
    ContentLedger,
    ContentOrigin,
    ContentTrust,
    SourceRef,
)

WEB = SourceRef("web", "fetch-1", "", ContentTrust.UNTRUSTED)
HANDBOOK = SourceRef("data_source", "store-a", "handbook", ContentTrust.TRUSTED)


def _ledger(*refs: SourceRef) -> ContentLedger:
    ledger = ContentLedger()
    ledger.record(ContentOrigin.of(*refs))
    return ledger


def _turn() -> TurnSummary:
    return TurnSummary(turn_nr=1, user_input="q", intent="ask", topic="t")


@pytest.fixture(params=[SimpleStoreCoordinator, HiveMindStoreCoordinator])
def coordinator(request: pytest.FixtureRequest):
    return request.param()


def test_a_turn_recorded_in_a_run_carries_the_runs_origin(coordinator) -> None:
    coordinator.bind_ledger(_ledger(HANDBOOK, WEB))

    coordinator.record_turn(_turn())

    (turn,) = coordinator.get_recent_turns(1)
    assert set(turn.origin.sources) == {HANDBOOK, WEB}


def test_an_atom_and_the_result_promoted_from_it_carry_the_origin(coordinator) -> None:
    coordinator.bind_ledger(_ledger(WEB))
    coordinator.open_branch("task")
    coordinator.write(WriteIntent(
        atom_type=AtomType.RESULT, content="found it", task_id="task", promote_immediately=True,
    ))
    atoms = coordinator.close_branch(promote=True)

    (result,) = coordinator.store_promoted_atoms(atoms, turn_nr=1)

    assert atoms[0].origin.untrusted_sources == (WEB,)
    assert result.origin.untrusted_sources == (WEB,)


def test_a_promoted_payload_carries_the_origin(coordinator) -> None:
    coordinator.bind_ledger(_ledger(WEB))

    coordinator.promote_payload("payload", ["entity"], [], 1)

    result = coordinator.get_payload(["entity"])
    assert result is not None
    assert result.origin.untrusted_sources == (WEB,)


def test_a_run_that_read_nothing_writes_platform_text(coordinator) -> None:
    coordinator.bind_ledger(ContentLedger())

    coordinator.record_turn(_turn())

    assert coordinator.get_recent_turns(1)[0].origin == PLATFORM_TEXT


def test_without_a_bound_ledger_nothing_is_stamped(coordinator) -> None:
    coordinator.record_turn(_turn())

    assert coordinator.get_recent_turns(1)[0].origin is None


def test_the_session_snapshot_keeps_origins_and_reads_missing_ones_as_unannotated() -> None:
    now = datetime(2026, 10, 13, tzinfo=UTC)
    turn = TurnSummary(turn_nr=1, user_input="q", intent="i", topic="t",
                       origin=ContentOrigin.of(HANDBOOK, WEB))
    result = SessionResult(content="r", origin=ContentOrigin.of(WEB))
    snapshot = SessionSnapshot("s", [turn], [result], [], now, now)

    restored = SessionSnapshot.from_dict(snapshot.to_dict())

    assert restored.turns[0].origin == turn.origin
    assert restored.results[0].origin == result.origin
    legacy = snapshot.to_dict()
    del legacy["turns"][0]["origin"]
    assert SessionSnapshot.from_dict(legacy).turns[0].origin is None


def test_working_memory_and_pipeline_snapshots_keep_the_atom_origin() -> None:
    atom = WorkingAtom(atom_type=AtomType.RESULT, content="c", task_id="t",
                       origin=ContentOrigin.of(WEB))
    memory = WorkingMemory()
    memory.branch("t")
    memory.write(atom)

    restored = WorkingMemory.from_snapshot(memory.snapshot())

    assert restored.read_all()[0].origin == atom.origin
    assert _de_atom(_ser_atom(atom)).origin == atom.origin
