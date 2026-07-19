# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum, StrEnum
from typing import Any
from uuid import uuid4

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AtomType(StrEnum):
    """Built-in atom type vocabulary for WorkingMemory writes.

    This class is intentionally open: any plain string is a valid atom type.
    Predefined members cover the core pipeline vocabulary; custom steps may
    pass arbitrary strings directly to ``WriteIntent.atom_type``.

    Convention for custom types
    ---------------------------
    Use a ``namespace:label`` format to avoid collisions with built-ins::

        coordinator.write(WriteIntent(atom_type="ooda:approval", ...))
        coordinator.write(WriteIntent(atom_type="myapp:diff_patch", ...))

    Custom types are treated as *non-promotable* by default — they are
    discarded when ``finish_task`` runs.  Pass ``promote_immediately=True``
    on the ``WriteIntent`` if the result should land in the SessionResultStore.
    """

    SPEC    = "SPEC"
    PLAN    = "PLAN"
    RESULT  = "RESULT"
    EVAL    = "EVAL"
    ERROR   = "ERROR"
    SIGNAL  = "SIGNAL"
    PAYLOAD = "PAYLOAD"
    INTENT  = "INTENT"
    CONTEXT = "CONTEXT"

    @classmethod
    def _missing_(cls, value: object) -> AtomType:
        """Accept any string as a valid atom type (open extension point)."""
        obj = str.__new__(cls, value)
        obj._value_ = str(value)
        obj._name_ = str(value)
        return obj


class PromotionStatus(StrEnum):
    DRAFT      = "DRAFT"
    FINAL      = "FINAL"
    SUPERSEDED = "SUPERSEDED"


class DirectivePriority(int, Enum):
    LOW    = 0
    NORMAL = 1
    HIGH   = 2


# ---------------------------------------------------------------------------
# DirectiveStore models
# ---------------------------------------------------------------------------

@dataclass
class Directive:
    key:       str
    value:     str
    source:    str              = "inferred"   # "user" | "inferred"
    priority:  DirectivePriority = DirectivePriority.NORMAL
    created_at: datetime        = field(default_factory=lambda: datetime.now(UTC))
    overrides: str | None    = None
    id:        str              = field(default_factory=lambda: str(uuid4()))


# ---------------------------------------------------------------------------
# ConversationStore models
# ---------------------------------------------------------------------------

@dataclass
class TurnSummary:
    turn_nr:        int
    user_input:     str
    intent:         str
    topic:          str
    entities:       list[str]       = field(default_factory=list)
    result_summary: str             = ""
    correction_of:  int | None   = None
    directive:      str | None   = None
    timestamp:      datetime        = field(default_factory=lambda: datetime.now(UTC))


# ---------------------------------------------------------------------------
# SessionResultStore models
# ---------------------------------------------------------------------------

@dataclass
class SessionResult:
    """A finished, validated result or artefact of a session.

    ``content`` is always a plain string — the stores are string-based.
    File-backed payloads are a HiveMind-adapter concern; the core model
    only carries ``workspace_path`` and ``content_hash`` as optional
    metadata fields so the adapter can track them without core changes.
    """

    content:        str
    entities:       list[str]       = field(default_factory=list)
    tags:           list[str]       = field(default_factory=list)
    turn_nr:        int             = 0
    status:         PromotionStatus = PromotionStatus.FINAL
    created_at:     datetime        = field(default_factory=lambda: datetime.now(UTC))
    id:             str             = field(default_factory=lambda: str(uuid4()))
    workspace_path: str | None   = None
    content_hash:   str | None   = None

    def is_file_backed(self) -> bool:
        return self.workspace_path is not None

    def peek(self, chars: int = 80) -> str:
        return self.content[:chars] if self.content else f"[file: {self.workspace_path}]"


# ---------------------------------------------------------------------------
# SessionSnapshot — persistence-agnostic session state + (de)serialisation
#
# Moved here from adapters/outbound/hive_mind/persistence.py: it's a plain
# data shape built from other core types with json/datetime-only
# serialisation helpers, no I/O. The SessionBackend port (ports/outbound/
# session_backend.py) needs this type, and ports must not import adapters.
# ---------------------------------------------------------------------------

def _ser_directive(d: Directive) -> dict[str, Any]:
    return {
        "id": d.id, "key": d.key, "value": d.value,
        "source": d.source, "priority": d.priority.value,
        "created_at": d.created_at.isoformat(), "overrides": d.overrides,
    }


def _de_directive(data: dict[str, Any]) -> Directive:
    d = Directive(
        key=data["key"], value=data["value"],
        source=data.get("source", "inferred"),
        priority=DirectivePriority(data["priority"]),
        overrides=data.get("overrides"),
    )
    d.id = data["id"]
    d.created_at = datetime.fromisoformat(data["created_at"])
    return d


def _ser_turn(t: TurnSummary) -> dict[str, Any]:
    return {
        "turn_nr": t.turn_nr, "user_input": t.user_input,
        "intent": t.intent, "topic": t.topic,
        "entities": t.entities, "result_summary": t.result_summary,
        "correction_of": t.correction_of, "directive": t.directive,
        "timestamp": t.timestamp.isoformat(),
    }


def _de_turn(data: dict[str, Any]) -> TurnSummary:
    t = TurnSummary(
        turn_nr=data["turn_nr"], user_input=data["user_input"],
        intent=data["intent"], topic=data["topic"],
        entities=data.get("entities", []),
        result_summary=data.get("result_summary", ""),
        correction_of=data.get("correction_of"),
        directive=data.get("directive"),
    )
    t.timestamp = datetime.fromisoformat(data["timestamp"])
    return t


def _ser_result(r: SessionResult) -> dict[str, Any]:
    return {
        "id": r.id, "content": "" if r.is_file_backed() else r.content,
        "entities": r.entities, "tags": r.tags, "turn_nr": r.turn_nr,
        "status": r.status.value, "created_at": r.created_at.isoformat(),
        "workspace_path": r.workspace_path, "content_hash": r.content_hash,
    }


def _de_result(data: dict[str, Any]) -> SessionResult:
    r = SessionResult(
        content=data["content"], entities=data.get("entities", []),
        tags=data.get("tags", []), turn_nr=data.get("turn_nr", 0),
        status=PromotionStatus(data["status"]),
        workspace_path=data.get("workspace_path"),
        content_hash=data.get("content_hash"),
    )
    r.id = data["id"]
    r.created_at = datetime.fromisoformat(data["created_at"])
    return r


def _slots_to_json(slots: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *slots* containing only JSON-serialisable values."""
    result: dict[str, Any] = {}
    for k, v in slots.items():
        try:
            json.dumps(v)
            result[k] = v
        except (TypeError, ValueError):
            _logger.debug("hive_mind.slots.skip_non_serializable | key=%s type=%s", k, type(v).__name__)
    return result


@dataclass
class SessionSnapshot:
    session_id:           str
    directives:           list[Directive]
    turns:                list[TurnSummary]
    results:              list[SessionResult]
    interrupted_task_ids: list[str]
    created_at:           datetime
    updated_at:           datetime
    slots:                dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id":           self.session_id,
            "created_at":           self.created_at.isoformat(),
            "updated_at":           self.updated_at.isoformat(),
            "interrupted_task_ids": self.interrupted_task_ids,
            "directives":           [_ser_directive(d) for d in self.directives],
            "turns":                [_ser_turn(t) for t in self.turns],
            "results":              [_ser_result(r) for r in self.results],
            "slots":                self.slots,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionSnapshot:
        return cls(
            session_id           = data["session_id"],
            directives           = [_de_directive(d) for d in data.get("directives", [])],
            turns                = [_de_turn(t) for t in data.get("turns", [])],
            results              = [_de_result(r) for r in data.get("results", [])],
            interrupted_task_ids = data.get("interrupted_task_ids", []),
            created_at           = datetime.fromisoformat(data["created_at"]),
            updated_at           = datetime.fromisoformat(data["updated_at"]),
            slots                = data.get("slots", {}),
        )


# ---------------------------------------------------------------------------
# WorkingMemory models
# ---------------------------------------------------------------------------

@dataclass
class WorkingAtom:
    atom_type:         str   # AtomType member or any "namespace:label" string
    content:           str
    task_id:           str
    tags:              list[str] = field(default_factory=list)
    promote_to_parent: bool     = False
    created_at:        datetime = field(default_factory=lambda: datetime.now(UTC))
    id:                str      = field(default_factory=lambda: str(uuid4()))


# ---------------------------------------------------------------------------
# StoreCoordinator write interface
# ---------------------------------------------------------------------------

@dataclass
class WriteIntent:
    atom_type:         str  # AtomType member or any "namespace:label" string
    content:           str
    task_id:           str
    entities:          list[str] = field(default_factory=list)
    tags:              list[str] = field(default_factory=list)
    turn_nr:           int       = 0
    promote_immediately: bool    = False


@dataclass
class ConflictReport:
    existing_id:           str
    existing_content:      str
    new_content:           str
    conflicting_entities:  list[str]
    resolution:            str = "UNRESOLVED"


# ---------------------------------------------------------------------------
# Diagnostics (tool output)
# ---------------------------------------------------------------------------

class Severity(StrEnum):
    error   = "error"
    warning = "warning"
    info    = "info"


@dataclass
class Position:
    line: int
    col:  int


@dataclass
class Range:
    start: Position
    end:   Position


@dataclass
class Location:
    path:  str
    range: Range | None = None


@dataclass
class RelatedInfo:
    message:  str
    location: Location | None = None


@dataclass
class DiagnosticItem:
    message:  str
    severity: Severity
    code:     str | None           = None
    location: Location | None      = None
    related:  list[RelatedInfo]    = field(default_factory=list)


@dataclass
class DiagnosticsSummary:
    error_count:   int
    warning_count: int
    info_count:    int


@dataclass
class Diagnostics:
    tool_name:    str
    tool_call_id: str
    errors:       list[DiagnosticItem]      = field(default_factory=list)
    warnings:     list[DiagnosticItem]      = field(default_factory=list)
    infos:        list[DiagnosticItem]      = field(default_factory=list)
    summary:      DiagnosticsSummary | None = None

    def __post_init__(self) -> None:
        if self.summary is None:
            self.summary = DiagnosticsSummary(
                error_count   = len(self.errors),
                warning_count = len(self.warnings),
                info_count    = len(self.infos),
            )


# ---------------------------------------------------------------------------
# Orient: Relevance scoring
# ---------------------------------------------------------------------------

class ScoringStrategy(StrEnum):
    LINEAR           = "linear"
    WEIGHTED_SIGMOID = "weighted_sigmoid"
    MULTIPLICATIVE   = "multiplicative"


@dataclass
class RelevanceWeights:
    recency:         float = 0.25
    proximity:       float = 0.40
    volatility:      float = 0.15
    dependency:      float = 0.20
    proximity_boost: float = 1.8

    def validate(self) -> None:
        total = self.recency + self.proximity + self.volatility + self.dependency
        if not (0.99 <= total <= 1.01):
            raise ValueError(
                f"RelevanceWeights must sum to 1.0, got {total:.3f}"
            )
        for name, val in [
            ("recency", self.recency),
            ("proximity", self.proximity),
            ("volatility", self.volatility),
            ("dependency", self.dependency),
        ]:
            if not (0.0 <= val <= 1.0):
                raise ValueError(
                    f"RelevanceWeights.{name} must be between 0 and 1, got {val}"
                )
        if self.proximity_boost < 0.0:
            raise ValueError(
                "RelevanceWeights.proximity_boost must be greater than or equal "
                f"to 0, got {self.proximity_boost}"
            )


@dataclass
class RelevanceScore:
    recency:    float = 0.0
    proximity:  float = 0.0
    volatility: float = 0.0
    dependency: float = 0.0
    total:      float = 0.0

    def __repr__(self) -> str:
        return (
            f"RelevanceScore(r={self.recency:.2f} p={self.proximity:.2f} "
            f"v={self.volatility:.2f} d={self.dependency:.2f} → {self.total:.2f})"
        )


# ---------------------------------------------------------------------------
# Orient: Scored wrappers
# ---------------------------------------------------------------------------

@dataclass
class ScoredAtom:
    atom:  WorkingAtom
    score: RelevanceScore
    depth: int = 2


@dataclass
class ScoredResult:
    result: SessionResult
    score:  RelevanceScore


# ---------------------------------------------------------------------------
# Orient: Mental Model
# ---------------------------------------------------------------------------

@dataclass
class MentalModel:
    turn_id:       str
    built_at:      datetime
    is_valid:      bool             = True
    directives:    list[Directive]  = field(default_factory=list)
    recent_turns:  list[TurnSummary]  = field(default_factory=list)
    known_results: list[ScoredResult] = field(default_factory=list)
    active_atoms:  list[ScoredAtom]   = field(default_factory=list)
    delta:         object | None      = None
    summary:       str | None         = None
    intents:       list[str]          = field(default_factory=list)
    entities:      list[str]          = field(default_factory=list)

    def top_atoms(self, n: int = 5) -> list[ScoredAtom]:
        return sorted(self.active_atoms, key=lambda a: a.score.total, reverse=True)[:n]

    def top_results(self, n: int = 5) -> list[ScoredResult]:
        return sorted(self.known_results, key=lambda r: r.score.total, reverse=True)[:n]

    def has_delta(self) -> bool:
        return self.delta is not None

    def __repr__(self) -> str:
        return (
            f"MentalModel(turn={self.turn_id[:8]} "
            f"atoms={len(self.active_atoms)} results={len(self.known_results)} "
            f"directives={len(self.directives)} valid={self.is_valid})"
        )


class MentalModelCache:
    """Turn-scoped cache for the assembled MentalModel.

    Orient sets it; subsequent steps read it; cleared after each turn.
    """

    def __init__(self) -> None:
        self._model: MentalModel | None = None
        self._valid: bool = False

    @property
    def model(self) -> MentalModel | None:
        return self._model if self._valid else None

    @property
    def valid(self) -> bool:
        return self._valid and self._model is not None

    def set(self, model: MentalModel) -> None:
        self._model = model
        self._valid = True

    def invalidate(self) -> None:
        self._valid = False
        if self._model is not None:
            self._model.is_valid = False

    def clear(self) -> None:
        self._model = None
        self._valid = False

    def __repr__(self) -> str:
        if not self._valid or self._model is None:
            return "MentalModelCache(empty)"
        return f"MentalModelCache(valid=True model={self._model!r})"


# ---------------------------------------------------------------------------
# Orient: Control signal decision
# ---------------------------------------------------------------------------

@dataclass
class ControlSignalDecision:
    should_stop:    bool        = False
    stop_reason:    str | None  = None
    priority_boost: str | None  = None
    scope_change:   str | None  = None
    pass_through:   bool        = True

    def __post_init__(self) -> None:
        if self.should_stop and self.pass_through:
            self.pass_through = False

    def __repr__(self) -> str:
        if self.pass_through:
            return "ControlSignalDecision(pass_through)"
        parts = []
        if self.should_stop:
            parts.append(f"stop={self.stop_reason!r}")
        if self.priority_boost:
            parts.append(f"boost={self.priority_boost!r}")
        if self.scope_change:
            parts.append(f"scope={self.scope_change!r}")
        return f"ControlSignalDecision({', '.join(parts)})"

