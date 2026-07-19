# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from nlght.core.playbooks.playbook import ActivePlaybook, PlaybookDefinition
from nlght.ports.outbound.playbook_catalog import PlaybookCatalog

if TYPE_CHECKING:
    from nlght.core.entry.context import RequestContext
    from nlght.ports.outbound.access_policy import PlaybookAccessPolicy
    from nlght.ports.outbound.tool_catalog import ToolCatalog

logger = logging.getLogger(__name__)


class AdapterPlaybookCatalog(PlaybookCatalog):
    """Request-scoped PlaybookCatalog — implements the PlaybookCatalog port.

    Immutable after construction. Steps read ``to_markdown()`` and integrate
    the result into the system prompt.
    """

    def __init__(
        self,
        playbooks: dict[str, ActivePlaybook],
        definitions: dict[str, PlaybookDefinition] | None = None,
    ) -> None:
        self._playbooks = playbooks
        self._definitions = definitions or {}

    def definitions(self) -> dict[str, PlaybookDefinition]:
        return self._definitions

    def get(self, name: str) -> ActivePlaybook | None:
        return self._playbooks.get(name)

    def all(self) -> list[ActivePlaybook]:
        return list(self._playbooks.values())

    def names(self) -> list[str]:
        return list(self._playbooks.keys())

    def to_markdown(self) -> str:
        if not self._playbooks:
            return "_No playbooks available for current toolset._\n"
        parts = ["# Available Playbooks\n"]
        for playbook in self._playbooks.values():
            parts.append(playbook.description)
            parts.append("\n---\n")
        return "\n".join(parts)

    def to_contract(self) -> str:
        """Compact contract listing — one entry per playbook with activation criteria."""
        if not self._playbooks:
            return ""
        lines = [
            "## Playbooks",
            "",
            "If the user's request matches one of these playbooks, call `activate_playbook` to select it.",
            "A request can match a playbook even when it also involves generation or writing — activate the playbook first.",
            "",
            "Available playbooks:",
        ]
        for playbook in self._playbooks.values():
            defn = self._definitions.get(playbook.name)
            hint = ""
            if defn and defn.description_hint:
                hint = defn.description_hint.split("\n")[0].strip()
            lines.append("")
            lines.append(f"**{playbook.name}**: {hint}")
            if defn and defn.use_when:
                criteria = [c.replace("_", " ") for c in defn.use_when]
                lines.append(f"  Activate when: {'; '.join(criteria)}")
            if defn and defn.use_none_only_if:
                criteria = [c.replace("_", " ") for c in defn.use_none_only_if]
                lines.append(f"  Skip only if: {'; '.join(criteria)}")
        return "\n".join(lines)

    def playbook_markdown(self, name: str) -> str:
        playbook = self._playbooks.get(name)
        return playbook.description if playbook else ""

    def __len__(self) -> int:
        return len(self._playbooks)

    def __bool__(self) -> bool:
        return bool(self._playbooks)


class PlaybookCatalogBuilder:
    """Builds a request-scoped AdapterPlaybookCatalog.

    Playbook definitions are loaded once from YAML at startup and held
    here. Per request, ``build()`` filters by:

    1. Tool availability (``PlaybookDefinition.is_available``)
    2. Access (optional ``PlaybookAccessPolicy`` — model- and caller-based)

    The markdown is generated deterministically from the definition —
    no LLM call during catalog construction.
    """

    def __init__(
        self,
        definitions: dict[str, PlaybookDefinition],
        *,
        access_policy: PlaybookAccessPolicy | None = None,
    ) -> None:
        self._definitions = definitions
        self._access_policy = access_policy

    async def build(
        self,
        tool_catalog: ToolCatalog | None,
        model: str | None = None,
        caller: RequestContext | None = None,
    ) -> AdapterPlaybookCatalog:
        tool_names: set[str] = set(tool_catalog.names()) if tool_catalog else set()
        playbooks: dict[str, ActivePlaybook] = {}

        for defn in self._definitions.values():
            if not defn.is_available(tool_names):
                logger.debug("playbook.catalog.unavailable | playbook=%s (tools missing)", defn.name)
                continue

            if self._access_policy is not None:
                try:
                    if not await self._access_policy.is_allowed(defn.name, caller, model):
                        logger.info(
                            "playbook.catalog.access_denied | playbook=%s cid=%s",
                            defn.name, caller.correlation_id if caller else None,
                        )
                        continue
                except Exception as exc:
                    logger.warning(
                        "playbook.catalog.policy_error | playbook=%s error=%s — denying",
                        defn.name, exc,
                    )
                    continue

            active_tool_names = defn.active_tools(tool_names)
            markdown = _build_playbook_markdown(defn, active_tool_names, tool_catalog)

            playbooks[defn.name] = ActivePlaybook(
                name=defn.name,
                description=markdown,
                tool_names=active_tool_names,
            )
            logger.debug(
                "playbook.catalog.activated | playbook=%s tools=%s",
                defn.name, active_tool_names,
            )

        logger.info(
            "playbook.catalog.built | total=%d active=%d",
            len(self._definitions), len(playbooks),
        )
        return AdapterPlaybookCatalog(playbooks, self._definitions)


# ---------------------------------------------------------------------------
# Deterministic markdown generation
# ---------------------------------------------------------------------------


def _build_playbook_markdown(
    defn: PlaybookDefinition,
    active_tool_names: list[str],
    tool_catalog: ToolCatalog | None,
) -> str:
    lines: list[str] = [
        f"# Playbook: {defn.name}",
        "",
        defn.description_hint,
    ]

    if defn.phases:
        lines += ["", "## Phases", ""]
        for i, phase in enumerate(defn.phases, 1):
            role_tag = f" [{phase.role}]" if phase.role else ""
            lines.append(f"{i}. **{phase.name}**{role_tag}: {phase.goal}")
            if phase.tools:
                lines.append(f"   - Tools: {', '.join(phase.tools)}")
            if phase.input_tags:
                lines.append(f"   - Input tags: {', '.join(phase.input_tags)}")
            if phase.output_tags:
                lines.append(f"   - Output tags: {', '.join(phase.output_tags)}")
            if phase.visible_tags:
                lines.append(f"   - Visible tags: {', '.join(phase.visible_tags)}")
            if phase.guidance:
                lines.append(f"   - Guidance: {phase.guidance.strip()}")
            if phase.exit_condition:
                lines.append(f"   - Exit: {phase.exit_condition}")
            if phase.exit is not None:
                lines.append(f"   - Exit policy: {phase.exit.type}")

    if defn.constraints:
        lines += ["", "## Constraints", ""]
        for c in defn.constraints:
            lines.append(f"- {c}")

    if defn.fallback:
        lines += ["", "## Fallback", "", defn.fallback]

    if active_tool_names and tool_catalog is not None:
        lines += ["", "## Available Tools", ""]
        for tool_name in active_tool_names:
            contract = tool_catalog.get_or_none(tool_name)
            if contract is None:
                continue
            lines.append(f"### {contract.name}")
            lines.append(f"> {contract.description}")
            lines.append("")
            lines.append("```tool")
            lines.append(contract.name)
            for p in contract.parameters or []:
                placeholder = str(p.default) if p.default is not None else f"<{p.type}>"
                lines.append(f"{p.name}: {placeholder}")
            lines.append("```")
            lines.append("")

    return "\n".join(lines)
