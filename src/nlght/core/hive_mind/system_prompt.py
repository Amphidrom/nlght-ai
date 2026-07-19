# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""SystemPromptBuilder — assembles a system prompt from a MentalModel.

All input selection is declared via PromptInputPolicy.
Extra sections from outside the core are injected through PromptSlot implementations.

Usage::

    builder = SystemPromptBuilder(ctx=ctx)
    prompt  = builder.build(
        mental_model = model,
        ego          = "You are a precise software assistant.",
        input_policy = PromptInputPolicy(
            include_directive_store=True,
            include_conversation_store=True,
            include_working_memory=True,
            include_session_result_store=True,
            include_current_request_interpretation=True,
            include_tools=True,
            include_playbooks=True,
        ),
    )
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Protocol

from nlght.core.hive_mind.directives import KnownDirective
from nlght.core.hive_mind.models import AtomType, MentalModel
from nlght.core.playbooks.playbook import Phase
from nlght.core.workflow import WorkflowStepContext

DEFAULT_CONSTRAINTS: list[str] = [
    "Never use placeholders like [value], [insert here], or similar if it wasn't asked explicitly.",
    "Never invent, estimate, or approximate values — if unsure, say so.",
]

_READABLE_ATOM_TYPES = {AtomType.CONTEXT, AtomType.RESULT, AtomType.EVAL, AtomType.ERROR}


class PromptSlot(Protocol):
    """Pluggable prompt section injected from outside the core.

    Implement render() to return a formatted string block, or None to skip.
    """

    def render(self) -> str | None: ...


@dataclass(frozen=True)
class PromptInputPolicy:
    """Declares which inputs and capabilities appear in the system prompt."""

    # Store inputs
    include_directive_store: bool = False
    include_conversation_store: bool = False
    include_working_memory: bool = False
    include_session_result_store: bool = False
    include_current_user_request: bool = False
    include_current_request_interpretation: bool = False
    resolve_working_memory_references: bool = False
    working_memory_tags: list[str] | None = None
    # Capabilities (require ctx)
    include_tools: bool = False
    exclude_tools: tuple[str, ...] = ()
    include_playbooks: bool = False
    active_playbook: str | None = None
    active_phase: Phase | None = None
    # Signals
    include_delta: bool = False


@dataclass
class PromptCompressionReport:
    """Reports prompt-side memory omissions caused by a token budget."""

    reason: str
    original_tokens: int
    final_tokens: int
    max_tokens: int
    omitted_session_results: int = 0


class SystemPromptBuilder:
    """Builds a structured system prompt from a MentalModel.

    Parameters:
        ctx: Optional workflow step context. When provided, available
             capabilities (tools, playbooks) can be listed in the prompt.
    """

    def __init__(self, ctx: WorkflowStepContext | None = None) -> None:
        self.ctx = ctx

    def build(
        self,
        mental_model: MentalModel,
        ego: str,
        constraints: list[str] | None = None,
        input_policy: PromptInputPolicy | None = None,
        working_memory_reference_resolver: Callable[[list[str]], list[str]] | None = None,
        slots: list[PromptSlot] | None = None,
        token_budget: object | None = None,
        max_system_prompt_tokens: int | None = None,
        compression_reports: list[PromptCompressionReport] | None = None,
    ) -> str:
        """Build a full system prompt string."""
        if constraints is None:
            constraints = DEFAULT_CONSTRAINTS

        policy = input_policy or PromptInputPolicy()

        sections = self._build_sections(
            mental_model=mental_model,
            ego=ego,
            constraints=constraints,
            policy=policy,
            working_memory_reference_resolver=working_memory_reference_resolver,
            slots=slots,
        )
        prompt = "\n\n".join(sections)

        max_tokens = _resolve_max_prompt_tokens(
            token_budget=token_budget,
            max_system_prompt_tokens=max_system_prompt_tokens,
        )
        if (
            max_tokens is not None
            and policy.include_session_result_store
            and _estimate_prompt_tokens(prompt) > max_tokens
            and mental_model.known_results
        ):
            return self._compress_session_results(
                prompt=prompt,
                mental_model=mental_model,
                ego=ego,
                constraints=constraints,
                policy=policy,
                working_memory_reference_resolver=working_memory_reference_resolver,
                slots=slots,
                max_tokens=max_tokens,
                compression_reports=compression_reports,
            )

        return prompt

    def _build_sections(
        self,
        *,
        mental_model: MentalModel,
        ego: str,
        constraints: list[str],
        policy: PromptInputPolicy,
        working_memory_reference_resolver: Callable[[list[str]], list[str]] | None,
        slots: list[PromptSlot] | None,
    ) -> list[str]:
        sections: list[str] = [ego.strip()]

        if policy.include_directive_store and mental_model.directives:
            sections.append(self._build_directives(mental_model))

        if self.ctx:
            if policy.include_playbooks:
                block = self._build_playbooks(policy.active_playbook, policy.active_phase)
                if block:
                    sections.append(block)
            if policy.include_tools:
                block = self._build_tools(exclude=policy.exclude_tools)
                if block:
                    sections.append(block)

        _uq = ""
        if policy.include_current_user_request and self.ctx and self.ctx.messages:
            _uq = next(
                (m.get("content", "") for m in reversed(self.ctx.messages) if m.get("role") == "user"),
                "",
            ).strip()

        context = self._build_context(
            mental_model=mental_model,
            user_query=_uq,
            policy=policy,
            working_memory_reference_resolver=working_memory_reference_resolver,
        )
        if context:
            sections.append(context)

        if slots:
            for slot in slots:
                content = slot.render()
                if content:
                    sections.append(content)

        if constraints:
            sections.append(self._constraints_block(constraints))

        return sections

    def _compress_session_results(
        self,
        *,
        prompt: str,
        mental_model: MentalModel,
        ego: str,
        constraints: list[str],
        policy: PromptInputPolicy,
        working_memory_reference_resolver: Callable[[list[str]], list[str]] | None,
        slots: list[PromptSlot] | None,
        max_tokens: int,
        compression_reports: list[PromptCompressionReport] | None,
    ) -> str:
        original_tokens = _estimate_prompt_tokens(prompt)
        ordered_results = _newest_results_first(mental_model.known_results)
        best_prompt = prompt
        best_count = len(ordered_results)

        for keep_count in range(len(ordered_results), -1, -1):
            trimmed_model = replace(
                mental_model,
                known_results=ordered_results[:keep_count],
            )
            candidate = "\n\n".join(self._build_sections(
                mental_model=trimmed_model,
                ego=ego,
                constraints=constraints,
                policy=policy,
                working_memory_reference_resolver=working_memory_reference_resolver,
                slots=slots,
            ))
            best_prompt = candidate
            best_count = keep_count
            if _estimate_prompt_tokens(candidate) <= max_tokens:
                break

        omitted = len(ordered_results) - best_count
        if omitted > 0 and compression_reports is not None:
            compression_reports.append(PromptCompressionReport(
                reason="system_prompt_token_budget",
                original_tokens=original_tokens,
                final_tokens=_estimate_prompt_tokens(best_prompt),
                max_tokens=max_tokens,
                omitted_session_results=omitted,
            ))

        return best_prompt

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _build_directives(self, mental_model: MentalModel) -> str:
        lines = ["Follow these behavioral rules:"]
        for d in mental_model.directives:
            known = KnownDirective.for_key(d.key)
            lines.append(f"- {known.render(d.value)}" if known else f"- {d.key}: {d.value}")
        return "\n".join(lines)

    def _build_playbooks(self, active_playbook: str | None, active_phase: Phase | None) -> str:
        if not self.ctx or not self.ctx.playbooks:
            return ""

        if active_playbook:
            definition = self.ctx.playbooks.definitions().get(active_playbook)
            hint = ""
            if definition and definition.description_hint:
                hint = definition.description_hint.split("\n")[0].strip()

            lines = [
                f"## Active Playbook: {active_playbook}",
                "",
                "You are currently inside this playbook.",
                "Follow the active playbook instructions and phases.",
                "Only use tools that are relevant to the current playbook phase.",
            ]
            if hint:
                lines += ["", hint]

            others = [n for n in self.ctx.playbooks.names() if n != active_playbook]
            if others:
                lines += ["", f"Other available playbooks: {', '.join(others)}"]

            phase_block = self._build_active_phase(active_phase)
            if phase_block:
                lines += ["", phase_block]

            return "\n".join(lines)

        return "\n".join([
            "## Playbook Routing",
            "",
            "`activate_playbook` is a control tool, not a task tool.",
            "",
            "Before calling any task tool, first decide whether an available playbook matches the user request.",
            "If a playbook matches, you MUST call `activate_playbook` with that playbook name before using any other tool.",
            "If no playbook matches, call `activate_playbook` with `none`.",
            "",
            "Important: a request that combines research with a generative task (e.g. 'research X and write Y') "
            "still requires the research playbook. Activate it even when the final output is generative.",
            "Call `activate_playbook(none)` ONLY when the answer is fully available from your own knowledge "
            "and the user has not explicitly asked to search, look up, or research anything.",
            "",
            self.ctx.playbooks.to_contract(),
        ])

    def _build_tools(self, exclude: tuple[str, ...] = ()) -> str:
        if not self.ctx or not self.ctx.tools:
            return ""
        playbook_tools: set[str] = set()
        if self.ctx.playbooks:
            for playbook in self.ctx.playbooks.all():
                playbook_tools.update(getattr(playbook, "tool_names", []))
        excluded = set(exclude)
        standalone = [
            t for t in self.ctx.tools.all()
            if t.name not in playbook_tools
            and t.name != "activate_playbook"
            and t.name not in excluded
        ]
        if not standalone:
            return ""
        lines = [
            "## Tools",
            "Tools are provided via the API. Use ONLY the API tool call mechanism.",
            "Never output tool calls as plain text, XML tags, or code blocks — the result will not be processed.",
            "",
        ]
        for tool in standalone:
            params = ", ".join(
                f"{p.name}: {p.type}" for p in (tool.parameters or [])
            ) if tool.parameters else ""
            desc = getattr(tool, "description", "") or ""
            entry = f"- {tool.name}({params})"
            if desc:
                entry += f": {desc}"
            lines.append(entry)
        return "\n".join(lines)

    def _build_active_phase(self, active_phase: Phase | None) -> str:
        if active_phase is None:
            return ""

        lines = [
            f"## Active Phase: {active_phase.name}",
            "",
            f"Role: {active_phase.role or 'task'}",
            f"Goal: {active_phase.goal}",
        ]

        if active_phase.tools:
            lines += ["", f"Tools for this phase: {', '.join(active_phase.tools)}"]
        if active_phase.input_tags:
            lines += ["", f"Visible inputs: {', '.join(active_phase.input_tags)}"]
        if active_phase.output_tags:
            lines += ["", f"Required outputs: {', '.join(active_phase.output_tags)}"]
            lines += ["", _phase_transform_instruction(active_phase)]
        if active_phase.guidance:
            lines += ["", f"Guidance: {active_phase.guidance}"]
        if active_phase.exit_condition:
            lines += ["", f"Exit condition: {active_phase.exit_condition}"]
        if active_phase.output_example:
            lines += ["", f"Example: {active_phase.output_example}"]
            lines += [
                "",
                "When this phase returns structured output, follow the example shape exactly.",
                "Return one compact tagged line per result.",
                "Do not use any other tagged shape.",
            ]

        return "\n".join(lines)

    @staticmethod
    def _build_context(
        mental_model: MentalModel,
        policy: PromptInputPolicy,
        user_query: str = "",
        working_memory_reference_resolver: Callable[[list[str]], list[str]] | None = None,
    ) -> str:
        lines: list[str] = []

        if user_query:
            lines.append(f"User question: {user_query}")

        if policy.include_current_request_interpretation and mental_model.intents:
            lines.append(f"User intent: {', '.join(mental_model.intents)}")

        if policy.include_current_request_interpretation and mental_model.entities:
            lines.append(f"Active entities: {', '.join(mental_model.entities)}")

        if policy.include_current_request_interpretation and mental_model.summary:
            lines.append(f"Current turn summary: {mental_model.summary}")

        if policy.include_conversation_store:
            if mental_model.recent_turns:
                lines.append("Recent conversation:")
                for t in mental_model.recent_turns[-3:]:
                    entry = f"  [{t.turn_nr}]"
                    if t.user_input:
                        entry += f" user: {t.user_input}"
                    if t.result_summary:
                        entry += f" → {t.result_summary}"
                    lines.append(entry)

        if policy.include_working_memory:
            atom_pool = (
                mental_model.active_atoms
                if policy.working_memory_tags
                else mental_model.top_atoms(10)
            )
            readable_entries = [
                _render_working_atom(sa.atom.atom_type, sa.atom.content, sa.atom.tags)
                for sa in atom_pool
                if (
                    sa.atom.atom_type in _READABLE_ATOM_TYPES
                    and _atom_matches_tags(sa.atom.tags, policy.working_memory_tags)
                )
            ]
            if readable_entries:
                rendered_entries = readable_entries
                if (
                    policy.resolve_working_memory_references
                    and working_memory_reference_resolver is not None
                ):
                    resolved_entries = working_memory_reference_resolver(readable_entries)
                    if resolved_entries:
                        rendered_entries = resolved_entries
                lines.append("Active task context:")
                for entry in rendered_entries:
                    _append_indented_block(lines, entry)

        if policy.include_session_result_store:
            user_facts = [sr for sr in mental_model.known_results if "user_fact" in sr.result.tags]
            if user_facts:
                lines.append("Known facts about the user:")
                for sr in user_facts:
                    lines.append(f"  - {sr.result.content}")

            task_results = [
                sr for sr in mental_model.top_results(10)
                if "user_fact" not in (sr.result.tags or [])
            ]
            if task_results:
                lines.append("Task results:")
                for sr in task_results:
                    lines.append(f"  - {sr.result.content}")

        if policy.include_delta:
            planning = [
                sa for sa in mental_model.top_atoms(10)
                if sa.atom.atom_type not in _READABLE_ATOM_TYPES
            ]
            if planning:
                lines.append("Planning context:")
                for sa in planning:
                    lines.append(f"  - {_render_working_atom(sa.atom.atom_type, sa.atom.content, sa.atom.tags)}")

            if mental_model.delta:
                nature     = getattr(mental_model.delta, "signal_nature", None)
                confidence = getattr(mental_model.delta, "confidence", None)
                if nature:
                    lines.append(f"Signal delta: {nature} (confidence={confidence})")

        return "\n".join(lines) if lines else ""

    @staticmethod
    def _constraints_block(constraints: list[str]) -> str:
        lines = ["Always follow these constraints:"]
        lines.extend(f"- {c}" for c in constraints)
        return "\n".join(lines)


def _append_indented_block(lines: list[str], block: str) -> None:
    parts = [part for part in str(block).splitlines() if part.strip()]
    if not parts:
        return
    lines.append(f"  - {parts[0]}")
    for part in parts[1:]:
        lines.append(f"    {part}")


def _phase_transform_instruction(phase: Phase) -> str:
    input_tags = [str(tag).strip() for tag in (getattr(phase, "input_tags", []) or []) if str(tag).strip()]
    output_tags = [str(tag).strip() for tag in (getattr(phase, "output_tags", []) or []) if str(tag).strip()]
    if input_tags and output_tags:
        return (
            f"Your task is to transform visible `{input_tags[0]}` entries into explicit "
            f"`[{output_tags[0]}]` outputs."
        )
    if output_tags:
        return f"Your task is to produce explicit `[{output_tags[0]}]` outputs for this phase."
    return ""


def _render_working_atom(atom_type: str, content: str, tags: list[str]) -> str:
    prefix = f"[{atom_type}]"
    kind_tag = next(
        (
            str(tag).strip()
            for tag in (tags or [])
            if str(tag).strip().startswith("kind:")
        ),
        "",
    )
    if kind_tag:
        prefix += f" [{kind_tag}]"
    clean_content = str(content)
    clean_content = re.sub(r"^\[(kind:[^\]]+)\]\s*\|?\s*", "", clean_content).strip()
    return f"{prefix} {clean_content}"


def _atom_matches_tags(atom_tags: list[str], required_tags: list[str] | None) -> bool:
    if not required_tags:
        return True
    if not atom_tags:
        return False
    atom_tag_set = {str(tag).strip() for tag in atom_tags if str(tag).strip()}
    required_set = {str(tag).strip() for tag in required_tags if str(tag).strip()}
    if not required_set:
        return True
    return bool(atom_tag_set & required_set)


def _resolve_max_prompt_tokens(
    *,
    token_budget: object | None,
    max_system_prompt_tokens: int | None,
) -> int | None:
    if max_system_prompt_tokens is not None:
        return max(0, int(max_system_prompt_tokens))
    if token_budget is None:
        return None
    prompt_budget = getattr(token_budget, "prompt_budget", None)
    if prompt_budget is None:
        return None
    return max(0, int(prompt_budget))


def _estimate_prompt_tokens(text: str) -> int:
    return max(1, int(len(text) / 3.5))


def _newest_results_first(results: list[Any]) -> list[Any]:
    return sorted(
        results,
        key=lambda scored: (
            getattr(scored.result, "turn_nr", 0),
            getattr(scored.result, "created_at", None) is not None,
            getattr(scored.result, "created_at", None),
        ),
        reverse=True,
    )
