# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class PhaseExitPolicy:
    """Deterministic phase exit policy declared by the playbook contract."""

    type: Literal["output_exists", "io_coverage"]
    input_tag: str | None = None
    output_tags: list[str] = field(default_factory=list)
    terminal_tags: list[str] = field(default_factory=list)
    key_fields: list[str] = field(default_factory=list)
    min_outputs: int = 1


@dataclass(frozen=True)
class Phase:
    """An ordered execution phase of a playbook."""

    name: str
    goal: str
    tools: list[str]
    role: str | None = None
    guidance: str | None = None
    exit_condition: str | None = None
    input_tags: list[str] = field(default_factory=list)
    output_tags: list[str] = field(default_factory=list)
    visible_tags: list[str] = field(default_factory=list)
    output_example: str | None = None
    max_rounds: int | None = None
    exit: PhaseExitPolicy | None = None
    execution: str | None = None          # "programmatic" | None (= "llm")
    query_inputs: dict[str, Any] | None = None      # {tool_param: tag_field} — used when execution="programmatic"


@dataclass(frozen=True)
class PlaybookDefinition:
    """Static description of a playbook — loaded from YAML.

    ``requires`` is a list of OR groups (a list of lists).
    A playbook is available if at least one group is fully
    contained in the available tool names.

    Example:
        requires: [["web-search"], ["searxng", "langsearch"]]
        → available if "web-search" OR both "searxng" + "langsearch" are present.
    """

    name: str
    description_hint: str
    requires: list[list[str]]
    phases: list[Phase]
    optional: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    fallback: str | None = None
    use_when: list[str] = field(default_factory=list)
    use_none_only_if: list[str] = field(default_factory=list)

    def is_available(self, tool_names: set[str]) -> bool:
        """True if at least one requires group is fully satisfied."""
        if not self.requires:
            return True
        return any(all(t in tool_names for t in group) for group in self.requires)

    def active_tools(self, tool_names: set[str]) -> list[str]:
        """Required + optional tools that are actually available."""
        all_required = {t for group in self.requires for t in group}
        return [t for t in (all_required | set(self.optional)) if t in tool_names]


@dataclass
class ActivePlaybook:
    """A playbook that can be realized at runtime with the available tools."""

    name: str
    description: str
    tool_names: list[str]
