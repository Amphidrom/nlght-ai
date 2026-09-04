# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Build structured, untrusted model context from runtime knowledge.

This module owns store selection and budget reduction. It cannot create a
trusted instruction message or provider role; its only output is a tuple of
``ContextRecord`` values.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from nlght.core.hive_mind.elements import (
    context_atom_elements,
    planning_atom_elements,
    result_elements,
    turn_elements,
)
from nlght.core.hive_mind.models import MentalElement, MentalModel, estimate_tokens
from nlght.core.hive_mind.relevance import ReducedView, reduce_to_budget
from nlght.core.model.budget import TokenBudget
from nlght.core.model.messages import (
    ContextKind,
    ContextProvenance,
    ContextRecord,
    ContextSupport,
)
from nlght.core.playbooks.playbook import Phase
from nlght.core.workflow import WorkflowStepContext

_UNBOUNDED = 1 << 62


class PromptSlot(Protocol):
    """Extension content, untrusted unless a future registry grants authority."""

    def render(self) -> str | None: ...


@dataclass(frozen=True)
class PromptInputPolicy:
    """Declares which runtime inputs are offered as untrusted context."""

    include_conversation_store: bool = False
    include_working_memory: bool = False
    include_session_result_store: bool = False
    include_current_request_interpretation: bool = False
    resolve_working_memory_references: bool = False
    working_memory_tags: list[str] | None = None
    include_tools: bool = False
    exclude_tools: tuple[str, ...] = ()
    include_playbooks: bool = False
    active_playbook: str | None = None
    active_phase: Phase | None = None
    include_delta: bool = False


@dataclass
class PromptCompressionReport:
    """Reports knowledge omitted when fitting the complete model request."""

    reason: str
    original_tokens: int
    final_tokens: int
    max_tokens: int
    omitted_elements: int = 0
    omitted_session_results: int = 0


class ModelContextBuilder:
    """Convert a ``MentalModel`` into typed, untrusted context records."""

    def __init__(self, ctx: WorkflowStepContext | None = None) -> None:
        self.ctx = ctx

    def build(
        self,
        mental_model: MentalModel,
        input_policy: PromptInputPolicy | None = None,
        working_memory_reference_resolver: Callable[[list[str]], list[str]] | None = None,
        slots: list[PromptSlot] | None = None,
        token_budget: TokenBudget | None = None,
        max_prompt_tokens: int | None = None,
        reserved_instruction_tokens: int = 0,
        compression_reports: list[PromptCompressionReport] | None = None,
    ) -> tuple[ContextRecord, ...]:
        """Build records while reducing only budgetable ``MentalElement`` data."""

        policy = input_policy or PromptInputPolicy()
        elements = _elements_for(mental_model, policy, working_memory_reference_resolver)
        maximum = _resolve_max_prompt_tokens(
            token_budget=token_budget,
            max_prompt_tokens=max_prompt_tokens,
        )

        def records(view: ReducedView) -> tuple[ContextRecord, ...]:
            return self._build_records(
                mental_model=mental_model,
                policy=policy,
                view=view,
                slots=slots,
            )

        whole = reduce_to_budget(elements, budget=_UNBOUNDED)
        if maximum is None:
            return records(whole)

        fixed_records = records(ReducedView())
        fixed_cost = sum(record.estimated_tokens for record in fixed_records)
        headings = sum(
            estimate_tokens(section)
            for section in {element.presentation.section for element in elements if element.presentation.section}
        )
        available = max(
            0,
            maximum - max(0, reserved_instruction_tokens) - fixed_cost - headings,
        )
        view = reduce_to_budget(elements, budget=available)
        reduced = records(view)

        if view.forgotten and compression_reports is not None:
            compression_reports.append(PromptCompressionReport(
                reason="model_context_token_budget",
                original_tokens=(
                    max(0, reserved_instruction_tokens)
                    + sum(record.estimated_tokens for record in records(whole))
                    + headings
                ),
                final_tokens=(
                    max(0, reserved_instruction_tokens)
                    + sum(record.estimated_tokens for record in reduced)
                    + headings
                ),
                max_tokens=maximum,
                omitted_elements=len(view.forgotten),
                omitted_session_results=sum(
                    1 for element in view.forgotten if element.kind == ContextKind.SESSION_RESULT.value
                ),
            ))
        return reduced

    def _build_records(
        self,
        *,
        mental_model: MentalModel,
        policy: PromptInputPolicy,
        view: ReducedView,
        slots: list[PromptSlot] | None,
    ) -> tuple[ContextRecord, ...]:
        records: list[ContextRecord] = []

        if self.ctx is not None:
            if policy.include_playbooks:
                content = self._build_playbooks(policy.active_playbook, policy.active_phase)
                if content:
                    records.append(_runtime_record(ContextKind.PLAYBOOK, "playbooks", content))
            if policy.include_tools:
                content = self._build_tools(exclude=policy.exclude_tools)
                if content:
                    records.append(_runtime_record(ContextKind.TOOL_GUIDANCE, "tools", content))

        if policy.include_current_request_interpretation:
            if mental_model.intents:
                records.append(_derived_record(
                    "request:intents",
                    ContextKind.REQUEST_INTERPRETATION,
                    f"User intent: {', '.join(mental_model.intents)}",
                ))
            if mental_model.entities:
                records.append(_derived_record(
                    "request:entities",
                    ContextKind.REQUEST_INTERPRETATION,
                    f"Active entities: {', '.join(mental_model.entities)}",
                ))
            if mental_model.summary:
                records.append(_derived_record(
                    "request:summary",
                    ContextKind.SUMMARY,
                    f"Current turn summary: {mental_model.summary}",
                ))

        ordered_elements = sorted(
            enumerate(view.kept),
            key=lambda item: (
                item[1][0].presentation.order,
                item[1][0].presentation.section,
                item[0],
            ),
        )
        for _, (element, representation) in ordered_elements:
            records.append(ContextRecord(
                record_id=element.element_id,
                kind=ContextKind(element.kind),
                content=representation.text,
                provenance=_context_provenance(element),
                section=element.presentation.section,
                representation_level=representation.level.value,
                retention=element.retention.value,
                relevance=element.relevance,
                estimated_tokens=representation.cost,
            ))

        if policy.include_delta and mental_model.delta:
            nature = getattr(mental_model.delta, "signal_nature", None)
            confidence = getattr(mental_model.delta, "confidence", None)
            if nature:
                records.append(_derived_record(
                    "request:delta",
                    ContextKind.DELTA,
                    f"Signal delta: {nature} (confidence={confidence})",
                ))

        for position, slot in enumerate(slots or ()):
            slot_content = slot.render()
            if slot_content:
                records.append(ContextRecord(
                    record_id=f"prompt-slot:{position}",
                    kind=ContextKind.PROMPT_SLOT,
                    content=slot_content,
                    provenance=ContextProvenance(source="prompt_slot", item_id=str(position)),
                    section="Extension context",
                    estimated_tokens=estimate_tokens(slot_content),
                ))
        return tuple(records)

    def _build_playbooks(self, active_playbook: str | None, active_phase: Phase | None) -> str:
        if self.ctx is None or not self.ctx.playbooks:
            return ""

        if active_playbook:
            definition = self.ctx.playbooks.definitions().get(active_playbook)
            hint = ""
            if definition and definition.description_hint:
                hint = str(definition.description_hint).split("\n")[0].strip()
            lines = [
                f"## Active Playbook: {active_playbook}",
                "",
                "You are currently inside this playbook.",
                "Follow the active playbook instructions and phases.",
                "Only use tools that are relevant to the current playbook phase.",
            ]
            if hint:
                lines += ["", hint]
            others = [name for name in self.ctx.playbooks.names() if name != active_playbook]
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
            self.ctx.playbooks.to_contract(),
        ])

    def _build_tools(self, exclude: tuple[str, ...] = ()) -> str:
        if self.ctx is None or not self.ctx.tools:
            return ""
        playbook_tools: set[str] = set()
        if self.ctx.playbooks:
            for playbook in self.ctx.playbooks.all():
                playbook_tools.update(getattr(playbook, "tool_names", []))
        excluded = set(exclude)
        standalone = [
            tool for tool in self.ctx.tools.all()
            if tool.name not in playbook_tools
            and tool.name != "activate_playbook"
            and tool.name not in excluded
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
                f"{parameter.name}: {parameter.type}" for parameter in (tool.parameters or [])
            ) if tool.parameters else ""
            entry = f"- {tool.name}({params})"
            description = getattr(tool, "description", "") or ""
            if description:
                entry += f": {description}"
            lines.append(entry)
        return "\n".join(lines)

    @staticmethod
    def _build_active_phase(active_phase: Phase | None) -> str:
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


def _runtime_record(kind: ContextKind, item_id: str, content: str) -> ContextRecord:
    return ContextRecord(
        record_id=f"runtime:{item_id}",
        kind=kind,
        content=content,
        provenance=ContextProvenance(source="runtime_catalog", item_id=item_id),
        estimated_tokens=estimate_tokens(content),
    )


def _derived_record(record_id: str, kind: ContextKind, content: str) -> ContextRecord:
    return ContextRecord(
        record_id=record_id,
        kind=kind,
        content=content,
        provenance=ContextProvenance(source="request_interpretation"),
        estimated_tokens=estimate_tokens(content),
    )


def _context_provenance(element: MentalElement) -> ContextProvenance:
    source = element.provenance
    payload = element.payload
    support = tuple(
        ContextSupport(
            document_id=str(getattr(item, "document_id", "")),
            observed_document_revision=str(getattr(item, "observed_document_revision", "")),
            document_path=str(getattr(item, "document_path", "")),
            slot_id=str(getattr(item, "slot_id", "")),
        )
        for item in (getattr(source, "support", ()) or ())
    )
    return ContextProvenance(
        source=element.kind,
        store=element.kind,
        item_id=str(getattr(payload, "id", element.element_id)),
        turn_nr=getattr(payload, "turn_nr", None),
        document_id=str(getattr(source, "document_id", "")),
        chunk_id=str(getattr(source, "chunk_id", "")),
        assertion_id=str(getattr(source, "assertion_id", "")),
        source_revision_id=str(getattr(source, "source_revision_id", "")),
        processing_revision_id=str(getattr(source, "processing_revision_id", "")),
        knowledge_revision_id=str(getattr(source, "knowledge_revision_id", "")),
        position=int(getattr(source, "position", 0) or 0),
        start_offset=int(getattr(source, "start_offset", 0) or 0),
        end_offset=int(getattr(source, "end_offset", 0) or 0),
        path=str(getattr(source, "path", "")),
        source_name=str(getattr(source, "source_name", "")),
        external_id=str(getattr(source, "external_id", "")),
        support=support,
    )


def _phase_transform_instruction(phase: Phase) -> str:
    input_tags = [str(tag).strip() for tag in (phase.input_tags or []) if str(tag).strip()]
    output_tags = [str(tag).strip() for tag in (phase.output_tags or []) if str(tag).strip()]
    if input_tags and output_tags:
        return (
            f"Your task is to transform visible `{input_tags[0]}` entries into explicit "
            f"`[{output_tags[0]}]` outputs."
        )
    if output_tags:
        return f"Your task is to produce explicit `[{output_tags[0]}]` outputs for this phase."
    return ""


def _elements_for(
    mental_model: MentalModel,
    policy: PromptInputPolicy,
    resolve_references: Callable[[list[str]], list[str]] | None,
) -> list[MentalElement]:
    elements: list[MentalElement] = []
    if policy.include_conversation_store:
        elements.extend(turn_elements(mental_model))
    if policy.include_working_memory:
        elements.extend(context_atom_elements(
            mental_model,
            tags=policy.working_memory_tags,
            resolve_references=(resolve_references if policy.resolve_working_memory_references else None),
        ))
    if policy.include_session_result_store:
        elements.extend(result_elements(mental_model))
    if policy.include_delta:
        elements.extend(planning_atom_elements(mental_model))
    elements.extend(mental_model.elements)
    return elements


def _resolve_max_prompt_tokens(
    *,
    token_budget: TokenBudget | None,
    max_prompt_tokens: int | None,
) -> int | None:
    if max_prompt_tokens is not None:
        return max(0, int(max_prompt_tokens))
    if token_budget is None:
        return None
    return max(0, int(token_budget.prompt_budget))
