# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
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

    Custom types can be marked for movement between WorkingMemory branches,
    but ``store_promoted_atoms`` and the legacy ``finish_task`` persist only
    ``RESULT`` and ``EVAL`` atoms in the SessionResultStore.
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

def _ser_turn(t: TurnSummary) -> dict[str, Any]:
    return {
        "turn_nr": t.turn_nr, "user_input": t.user_input,
        "intent": t.intent, "topic": t.topic,
        "entities": t.entities, "result_summary": t.result_summary,
        "correction_of": t.correction_of,
        "timestamp": t.timestamp.isoformat(),
    }


def _de_turn(data: dict[str, Any]) -> TurnSummary:
    t = TurnSummary(
        turn_nr=data["turn_nr"], user_input=data["user_input"],
        intent=data["intent"], topic=data["topic"],
        entities=data.get("entities", []),
        result_summary=data.get("result_summary", ""),
        correction_of=data.get("correction_of"),
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
    turns:                list[TurnSummary]
    results:              list[SessionResult]
    interrupted_task_ids: list[str]
    created_at:           datetime
    updated_at:           datetime
    slots:                dict[str, Any] = field(default_factory=dict)
    #: Which principal this session belongs to, recorded when it was created.
    #:
    #: Empty means **unowned**, and that is not a permission. A session written
    #: before ownership existed has no owner and cannot be given one by whoever
    #: asks for it next — "first caller wins" would be an account-takeover
    #: migration (ADR-0061).
    owner_principal_id:   str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id":           self.session_id,
            "created_at":           self.created_at.isoformat(),
            "updated_at":           self.updated_at.isoformat(),
            "interrupted_task_ids": self.interrupted_task_ids,
            "owner_principal_id":   self.owner_principal_id,
            "turns":                [_ser_turn(t) for t in self.turns],
            "results":              [_ser_result(r) for r in self.results],
            "slots":                self.slots,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SessionSnapshot:
        return cls(
            session_id           = data["session_id"],
            owner_principal_id   = data.get("owner_principal_id", ""),
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
    #: What this atom is about. The caller has always supplied them on
    #: `WriteIntent`, and every `write()` implementation dropped them here — so
    #: proximity scored every atom against an empty list and returned the same
    #: floor for all of them (ADR-0057). Carried now, and nothing else changes:
    #: the field the caller already fills reaches the scorer that already reads
    #: it.
    entities:          list[str] = field(default_factory=list)
    tags:              list[str] = field(default_factory=list)
    #: What this atom is *about*, as a stable short name — a `MemoryCandidate`'s
    #: key. It has always existed upstream and reached the atom only as a prefix
    #: inside `content`, which is a shorthand hiding in free text: readable to a
    #: person, unusable to anything that wants to say the same thing briefly.
    #:
    #: Empty for an atom written by anything that has no such name, and an
    #: adapter offers no short form for those rather than parsing one back out.
    key:               str      = ""
    #: What sort of remembered thing it is — fact, artifact, relation, request.
    #: **Not** `atom_type`: that says where in the pipeline this was
    #: written (SPEC, RESULT, EVAL), which is a different axis entirely. An
    #: artifact can be written at any stage and a SPEC can be about anything.
    kind:              str      = ""
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
    #: The stable short name of what this is about, and what sort of thing it
    #: is. Carried so an adapter can offer a compact representation without
    #: parsing one back out of the content (ADR-0059).
    key:               str       = ""
    kind:              str       = ""
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
    recent_turns:  list[TurnSummary]  = field(default_factory=list)
    known_results: list[ScoredResult] = field(default_factory=list)
    active_atoms:  list[ScoredAtom]   = field(default_factory=list)
    delta:         object | None      = None
    summary:       str | None         = None
    intents:       list[str]          = field(default_factory=list)
    entities:      list[str]          = field(default_factory=list)
    #: Knowledge that arrives already adapted, from wherever it came from.
    #:
    #: The four fields above are the session's own stores and keep their types.
    #: This is for everything else the model should currently know — a retrieved
    #: passage is the first — and it is deliberately **not** a second typed list
    #: per source: a `passages` field here would be the retrieval special case
    #: rebuilt one level up, and the next source would want its own.
    elements:      tuple[Any, ...]     = ()

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
            f"valid={self.is_valid})"
        )


# ---------------------------------------------------------------------------
# Orient: one element, whatever it is made of
# ---------------------------------------------------------------------------
#
# A turn, an atom, a session result and a retrieved passage are four different
# things and stay four different things — they mean different things to
# a reader and to the business. What they have in common is only visible at the
# moment a budget forces a choice: each is some information, worth something,
# costing something, and expressible more cheaply or not at all.
#
# That common layer is what these types are. Everything type-specific lives in
# the adapter that produces a `MentalElement`; nothing below this line knows what
# a SessionResult is, and no reduction rule may ever ask.


class Level(StrEnum):
    """How fully an element is being said.

    Ordered from fullest to cheapest, and that order is the contract: a reduction
    may move an element *down* this list and never up.

    `OMIT` is a level rather than the absence of one. "Choose exactly one level
    per element" is then a total function, which is what makes monotonicity a
    property of the algorithm instead of something tests hope for.
    """

    FULL = "full"
    COMPACT = "compact"
    OMIT = "omit"


#: Fullest first. The single place this order is written down.
LEVELS: tuple[Level, ...] = (Level.FULL, Level.COMPACT, Level.OMIT)


class Retention(StrEnum):
    """How hard this element should be held on to under pressure.

    Declared by whatever produced the element, never derived from its relevance
    score. The two are different questions: relevance says *how well this matches
    the moment*, retention says *what it costs to be wrong about dropping it*. A
    result nobody's question mentions may still be one that must not be dropped.

    It is a property of the element and not of its kind, so the reduction can
    honour it without knowing what kind of thing it is holding.
    """

    #: Never omitted. It may still be compacted, and a mandatory set too large
    #: for the budget is reported rather than silently trimmed.
    MANDATORY = "mandatory"
    #: Compacted before it is omitted — losing it entirely costs more than
    #: saying it briefly.
    IMPORTANT = "important"
    #: Kept while there is room; dropped without ceremony when there is not.
    USEFUL = "useful"
    #: The first thing to go.
    DISPENSABLE = "dispensable"


@dataclass(frozen=True, slots=True)
class Representation:
    """One way of saying an element, and what saying it costs.

    Produced by the adapter that knows what the element is. The reduction picks
    between representations; it never writes one, because writing one requires
    knowing what the thing is and that is exactly the knowledge the reduction
    must not have.
    """

    level: Level
    text: str
    #: In the same estimated tokens the prompt path already counts in. A number
    #: the producer states, so nothing here has to guess at a unit.
    cost: int = 0
    #: Whether this wording was derived rather than quoted. A compacted form of
    #: source-backed text is **not** a verbatim quote of the source, and anything
    #: that cites must be able to tell the difference (ADR-0053).
    derived: bool = False

    def __post_init__(self) -> None:
        if self.cost < 0:
            raise ValueError("a representation cannot cost less than nothing")
        if self.level is Level.OMIT and self.cost:
            raise ValueError("an omitted element costs nothing to say")


def estimate_tokens(text: str) -> int:
    """What saying this is expected to cost.

    The prompt path has counted in this estimate for as long as it has had a
    budget, and it lives here now because `Representation.cost` is the concept it
    serves. One definition: a second answer to "how big is this" is two numbers
    that drift, and the platform already carries one duplicate of this constant.
    """
    return max(1, int(len(text) / 3.5))


@dataclass(frozen=True, slots=True)
class RelevanceInputs:
    """What relevance is computed from, in the one shape it is computed from.

    Every sort of knowledge answers these four questions differently — an atom
    knows when it was last mutated, a session result never mutates at all, a
    passage was ranked by a fusion — and that difference is real. What is *not*
    real is a scoring function per sort: "session results do not mutate" is the
    input `mutation_count=0`, not a second definition of relevance.

    So the adapter maps its domain fields onto this, and the engine reads only
    this. After the adapter, nothing scoring anything knows what it is holding.
    """

    #: When this last changed. `None` means "no age known", which scores as
    #: freshly relevant rather than as infinitely stale — absence of a timestamp
    #: is not evidence of age.
    updated_at: datetime | None = None
    #: What this is about, for closeness to the current question.
    entities: tuple[str, ...] = ()
    #: How often it has been rewritten. Zero for anything immutable, which is a
    #: fact about that thing and not a gap in it.
    mutation_count: int = 0
    #: The id a dependency chain would name it by, where it has one.
    identity: str = ""


@dataclass(frozen=True, slots=True)
class Presentation:
    """Where a representation belongs in the prompt, and nothing more.

    Produced by the adapter, because deciding that a user fact belongs under
    "Known facts about the user" is an interpretation of what a session result
    *is* — and that interpretation is exactly what the renderer must not make.

    Deliberately two fields. It is not an ontology of prompt structure; it is the
    smallest thing that answers "which heading, in what order", which is all a
    generic renderer needs to reproduce the prompt the special cases used to
    produce.
    """

    #: The heading this element renders under, verbatim.
    section: str
    #: Position among sections. Two elements naming one section agree on it in
    #: practice; where they disagree the lowest wins, so ordering never depends
    #: on which element happened to be seen first.
    order: int = 0


@dataclass(frozen=True, slots=True)
class MentalElement:
    """One thing the model might be told, in whichever form survives.

    The payload stays whatever it was — a `SessionResult`, a `ContextPassage`,
    a `WorkingAtom` — and is carried rather than converted. Everything the reduction
    needs is on the wrapper, so a new sort of knowledge becomes reducible by
    being adapted, never by the reduction learning about it.
    """

    element_id: str
    #: What sort of thing this is. A **label**, for diagnosis, provenance and
    #: choosing an adapter. Neither the reduction nor the renderer reads it, and
    #: the moment something branches on it, storage types are deciding again.
    #: Where an element belongs in the prompt is `presentation`, which the
    #: adapter decides — a session result tagged `user_fact` and one without it
    #: are the same `kind` and render under different headings.
    kind: str
    representations: tuple[Representation, ...]
    presentation: Presentation = field(
        default_factory=lambda: Presentation(section="")
    )
    #: What relevance is computed from. Carried rather than computed, so the
    #: engine scores an element without ever asking what it was made of.
    signals: RelevanceInputs = field(default_factory=lambda: RelevanceInputs())
    retention: Retention = Retention.USEFUL
    #: How well this matches the moment, from the `RelevanceEngine`. Orders
    #: elements *within* a retention band and decides nothing across bands.
    relevance: float = 0.0
    #: Where the information came from, if it can say. Survives every change of
    #: representation, because how much of a thing is being said has nothing to
    #: do with where it came from.
    provenance: Any = None
    payload: Any = None

    def __post_init__(self) -> None:
        if not self.representations:
            raise ValueError(f"element '{self.element_id}' offers no representation")
        levels = [item.level for item in self.representations]
        if len(set(levels)) != len(levels):
            raise ValueError(f"element '{self.element_id}' offers a level twice")
        # Cheaper as it gets shorter, and this is load-bearing rather than
        # tidy: the reduction guarantees that more budget never yields a less
        # complete view, and it earns that by applying a prefix of one fixed
        # order of reductions. A "compact" form costing more than the full one
        # would make a reduction step *raise* the total, and the guarantee would
        # hold only by luck of the data.
        offered = sorted(self.representations, key=lambda item: LEVELS.index(item.level))
        for fuller, shorter in zip(offered, offered[1:], strict=False):
            if shorter.cost > fuller.cost:
                raise ValueError(
                    f"element '{self.element_id}' says {shorter.level} costs "
                    f"{shorter.cost} and {fuller.level} costs {fuller.cost}; a "
                    f"shorter form that costs more is not a shorter form"
                )

    def at(self, level: Level) -> Representation | None:
        """This element said at that level, where it can be."""
        return next((item for item in self.representations if item.level is level), None)

    @property
    def levels(self) -> tuple[Level, ...]:
        """The levels this element can be said at, fullest first."""
        offered = {item.level for item in self.representations}
        return tuple(level for level in LEVELS if level in offered)


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
