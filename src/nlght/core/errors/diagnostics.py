# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


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
    code:     str | None      = None
    location: Location | None = None
    related:  list[RelatedInfo]  = field(default_factory=list)


@dataclass
class DiagnosticsSummary:
    error_count:   int
    warning_count: int
    info_count:    int


@dataclass
class Diagnostics:
    """Structured diagnostic output produced by a step or tool.

    Callers serialise this to JSON and store it as a WorkingAtom
    (AtomType.ERROR) in the StoreCoordinator.  The stores themselves
    only ever hold plain strings — Diagnostics is a vocabulary type
    for building that string content consistently.
    """

    tool_name:    str
    tool_call_id: str
    errors:       list[DiagnosticItem] = field(default_factory=list)
    warnings:     list[DiagnosticItem] = field(default_factory=list)
    infos:        list[DiagnosticItem] = field(default_factory=list)
    summary:      DiagnosticsSummary | None = None

    def __post_init__(self) -> None:
        if self.summary is None:
            self.summary = DiagnosticsSummary(
                error_count   = len(self.errors),
                warning_count = len(self.warnings),
                info_count    = len(self.infos),
            )
