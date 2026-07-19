# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""PipelineSnapshot — full pipeline state capture for step-level replay.

Captures both the persistent session state (SessionSnapshot) and the
transient in-memory slot state (MentalModel) after any workflow step.

Usage::

    from nlght.adapters.outbound.hive_mind.pipeline_snapshot import (
        PipelineSnapshot, SlotState,
        _ser_mental_model, _de_mental_model,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from nlght.core.hive_mind.models import (
    MentalModel,
    RelevanceScore,
    ScoredAtom,
    ScoredResult,
    SessionSnapshot,
    WorkingAtom,
    _de_directive,
    _de_result,
    _de_turn,
    _ser_directive,
    _ser_result,
    _ser_turn,
)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class _DeltaSnapshot:
    """Lightweight proxy that preserves delta signal fields across snapshots."""
    signal_nature: str | None = None
    confidence:    float | None = None


@dataclass
class SlotState:
    mental_model:            dict[str, Any] | None       = None
    active_task_id:          str  | None       = None
    signal_classifications:  list[Any] | None       = None
    interim_results:         list[str]         = field(default_factory=list)
    working_memory_stack:    list[dict[str, Any]] | None = None


@dataclass
class PipelineSnapshot:
    snapshot_id:    str
    session_id:     str
    correlation_id: str
    step_name:      str
    step_verdict:   str
    created_at:     datetime
    session:        SessionSnapshot
    slots:          SlotState

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id":    self.snapshot_id,
            "session_id":     self.session_id,
            "correlation_id": self.correlation_id,
            "step_name":      self.step_name,
            "step_verdict":   self.step_verdict,
            "created_at":     self.created_at.isoformat(),
            "session":        self.session.to_dict(),
            "slots": {
                "mental_model":           self.slots.mental_model,
                "active_task_id":         self.slots.active_task_id,
                "signal_classifications": self.slots.signal_classifications,
                "interim_results":        self.slots.interim_results,
                "working_memory_stack":   self.slots.working_memory_stack,
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineSnapshot:
        s = data["slots"]
        return cls(
            snapshot_id    = data["snapshot_id"],
            session_id     = data["session_id"],
            correlation_id = data["correlation_id"],
            step_name      = data["step_name"],
            step_verdict   = data["step_verdict"],
            created_at     = datetime.fromisoformat(data["created_at"]),
            session        = SessionSnapshot.from_dict(data["session"]),
            slots          = SlotState(
                mental_model           = s.get("mental_model"),
                active_task_id         = s.get("active_task_id"),
                signal_classifications = s.get("signal_classifications"),
                interim_results        = s.get("interim_results") or [],
                working_memory_stack   = s.get("working_memory_stack"),
            ),
        )


# ---------------------------------------------------------------------------
# RelevanceScore
# ---------------------------------------------------------------------------

def _ser_score(s: RelevanceScore) -> dict[str, Any]:
    return {
        "recency":    s.recency,
        "proximity":  s.proximity,
        "volatility": s.volatility,
        "dependency": s.dependency,
        "total":      s.total,
    }


def _de_score(d: dict[str, Any]) -> RelevanceScore:
    return RelevanceScore(
        recency    = d.get("recency",    0.0),
        proximity  = d.get("proximity",  0.0),
        volatility = d.get("volatility", 0.0),
        dependency = d.get("dependency", 0.0),
        total      = d.get("total",      0.0),
    )


# ---------------------------------------------------------------------------
# WorkingAtom
# ---------------------------------------------------------------------------

def _ser_atom(a: WorkingAtom) -> dict[str, Any]:
    return {
        "id":               a.id,
        "atom_type":        str(a.atom_type),
        "content":          a.content,
        "task_id":          a.task_id,
        "tags":             list(a.tags),
        "promote_to_parent": a.promote_to_parent,
        "created_at":       a.created_at.isoformat(),
    }


def _de_atom(d: dict[str, Any]) -> WorkingAtom:
    a = WorkingAtom(
        atom_type         = d["atom_type"],
        content           = d["content"],
        task_id           = d["task_id"],
        tags              = list(d.get("tags", []) or []),
        promote_to_parent = d.get("promote_to_parent", False),
    )
    a.id         = d["id"]
    a.created_at = datetime.fromisoformat(d["created_at"])
    return a


# ---------------------------------------------------------------------------
# ScoredAtom / ScoredResult
# ---------------------------------------------------------------------------

def _ser_scored_atom(sa: ScoredAtom) -> dict[str, Any]:
    return {
        "atom":  _ser_atom(sa.atom),
        "score": _ser_score(sa.score),
        "depth": sa.depth,
    }


def _de_scored_atom(d: dict[str, Any]) -> ScoredAtom:
    return ScoredAtom(
        atom  = _de_atom(d["atom"]),
        score = _de_score(d["score"]),
        depth = d.get("depth", 2),
    )


def _ser_scored_result(sr: ScoredResult) -> dict[str, Any]:
    return {
        "result": _ser_result(sr.result),
        "score":  _ser_score(sr.score),
    }


def _de_scored_result(d: dict[str, Any]) -> ScoredResult:
    return ScoredResult(
        result = _de_result(d["result"]),
        score  = _de_score(d["score"]),
    )


# ---------------------------------------------------------------------------
# MentalModel.delta (open type — preserved via duck-typed proxy)
# ---------------------------------------------------------------------------

def _ser_delta(delta: object | None) -> dict[str, Any] | None:
    if delta is None:
        return None
    return {
        "signal_nature": getattr(delta, "signal_nature", None),
        "confidence":    getattr(delta, "confidence",    None),
    }


def _de_delta(d: dict[str, Any] | None) -> _DeltaSnapshot | None:
    if d is None:
        return None
    return _DeltaSnapshot(
        signal_nature = d.get("signal_nature"),
        confidence    = d.get("confidence"),
    )


# ---------------------------------------------------------------------------
# MentalModel
# ---------------------------------------------------------------------------

def _ser_mental_model(m: MentalModel) -> dict[str, Any]:
    return {
        "turn_id":       m.turn_id,
        "built_at":      m.built_at.isoformat(),
        "is_valid":      m.is_valid,
        "directives":    [_ser_directive(d) for d in m.directives],
        "recent_turns":  [_ser_turn(t) for t in m.recent_turns],
        "known_results": [_ser_scored_result(sr) for sr in m.known_results],
        "active_atoms":  [_ser_scored_atom(sa) for sa in m.active_atoms],
        "delta":         _ser_delta(m.delta),
        "summary":       m.summary,
        "intents":       m.intents,
        "entities":      m.entities,
    }


def _de_mental_model(d: dict[str, Any]) -> MentalModel:
    return MentalModel(
        turn_id       = d["turn_id"],
        built_at      = datetime.fromisoformat(d["built_at"]),
        is_valid      = d.get("is_valid", True),
        directives    = [_de_directive(x) for x in d.get("directives", [])],
        recent_turns  = [_de_turn(x)      for x in d.get("recent_turns", [])],
        known_results = [_de_scored_result(x) for x in d.get("known_results", [])],
        active_atoms  = [_de_scored_atom(x)   for x in d.get("active_atoms", [])],
        delta         = _de_delta(d.get("delta")),
        summary       = d.get("summary"),
        intents       = d.get("intents", []),
        entities      = d.get("entities", []),
    )

