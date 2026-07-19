# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from nlght.adapters.outbound.hive_mind.pipeline_snapshot import (
    PipelineSnapshot,
    SlotState,
    _de_mental_model,
    _ser_mental_model,
)
from nlght.core.hive_mind.models import (
    AtomType,
    Directive,
    MentalModel,
    RelevanceScore,
    ScoredAtom,
    ScoredResult,
    SessionResult,
    SessionSnapshot,
    TurnSummary,
    WorkingAtom,
)


@dataclass
class _Delta:
    signal_nature: str
    confidence: float


def _now() -> datetime:
    return datetime(2026, 1, 1, tzinfo=UTC)


def test_pipeline_snapshot_roundtrip_preserves_session_and_slots() -> None:
    session = SessionSnapshot(
        session_id="sess",
        directives=[Directive(key="lang", value="de")],
        turns=[TurnSummary(turn_nr=1, user_input="hi", intent="greet", topic="t")],
        results=[SessionResult(content="result", entities=["e"], tags=["tag"], turn_nr=1)],
        interrupted_task_ids=["task-1"],
        created_at=_now(),
        updated_at=_now(),
    )
    snapshot = PipelineSnapshot(
        snapshot_id="snap-1",
        session_id="sess",
        correlation_id="cid",
        step_name="act",
        step_verdict="done",
        created_at=_now(),
        session=session,
        slots=SlotState(
            mental_model={"turn_id": "turn-1"},
            active_task_id="task-1",
            signal_classifications=[{"kind": "x"}],
            interim_results=["partial"],
            working_memory_stack=[{"label": "turn", "atoms": []}],
        ),
    )

    restored = PipelineSnapshot.from_dict(snapshot.to_dict())

    assert restored.snapshot_id == "snap-1"
    assert restored.session.directives[0].value == "de"
    assert restored.session.turns[0].turn_nr == 1
    assert restored.session.results[0].content == "result"
    assert restored.slots.active_task_id == "task-1"
    assert restored.slots.interim_results == ["partial"]
    assert restored.slots.working_memory_stack == [{"label": "turn", "atoms": []}]


def test_mental_model_roundtrip_preserves_nested_scored_state_and_delta() -> None:
    atom = WorkingAtom(
        atom_type=AtomType.CONTEXT,
        content="candidate",
        task_id="task",
        tags=["kind:source_candidate"],
        promote_to_parent=True,
    )
    result = SessionResult(content="known", entities=["entity"], tags=["user_fact"], turn_nr=2)
    model = MentalModel(
        turn_id="turn-1",
        built_at=_now(),
        is_valid=False,
        directives=[Directive(key="lang", value="de")],
        recent_turns=[TurnSummary(turn_nr=2, user_input="u", intent="i", topic="t")],
        known_results=[ScoredResult(result=result, score=RelevanceScore(total=0.7))],
        active_atoms=[ScoredAtom(atom=atom, score=RelevanceScore(recency=0.1, total=0.9), depth=4)],
        delta=_Delta(signal_nature="correction", confidence=0.8),
        summary="summary",
        intents=["informational"],
        entities=["entity"],
    )

    restored = _de_mental_model(_ser_mental_model(model))

    assert restored.turn_id == "turn-1"
    assert restored.is_valid is False
    assert restored.directives[0].value == "de"
    assert restored.known_results[0].result.tags == ["user_fact"]
    assert restored.active_atoms[0].atom.tags == ["kind:source_candidate"]
    assert restored.active_atoms[0].depth == 4
    assert restored.delta.signal_nature == "correction"
    assert restored.delta.confidence == 0.8
    assert restored.summary == "summary"


def test_slot_state_defaults_and_missing_optional_fields() -> None:
    snapshot = PipelineSnapshot.from_dict({
        "snapshot_id": "s",
        "session_id": "sess",
        "correlation_id": "cid",
        "step_name": "step",
        "step_verdict": "done",
        "created_at": _now().isoformat(),
        "session": SessionSnapshot(
            session_id="sess",
            directives=[],
            turns=[],
            results=[],
            interrupted_task_ids=[],
            created_at=_now(),
            updated_at=_now(),
        ).to_dict(),
        "slots": {},
    })

    assert snapshot.slots.interim_results == []
    assert snapshot.slots.mental_model is None
    assert _de_mental_model(_ser_mental_model(MentalModel(turn_id="t", built_at=_now()))).delta is None
