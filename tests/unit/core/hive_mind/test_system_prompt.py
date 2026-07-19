# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Tests for nlght.core.hive_mind.system_prompt — SystemPromptBuilder."""
from __future__ import annotations

from datetime import UTC, datetime

from nlght.core.hive_mind.models import (
    AtomType,
    Directive,
    MentalModel,
    RelevanceScore,
    ScoredAtom,
    ScoredResult,
    SessionResult,
    TurnSummary,
    WorkingAtom,
)
from nlght.core.hive_mind.system_prompt import (
    DEFAULT_CONSTRAINTS,
    PromptCompressionReport,
    PromptInputPolicy,
    SystemPromptBuilder,
)
from nlght.core.playbooks.playbook import Phase


def _model(**kwargs) -> MentalModel:
    defaults = dict(turn_id="t1", built_at=datetime.now(UTC))
    defaults.update(kwargs)
    return MentalModel(**defaults)


def _builder() -> SystemPromptBuilder:
    return SystemPromptBuilder()


class _StubPlaybooks:
    def __init__(self) -> None:
        self._definitions = {
            "web_research": type(
                "Defn",
                (),
                {"description_hint": "Search the web for current information."},
            )(),
        }

    def definitions(self):
        return self._definitions

    def names(self):
        return ["web_research", "translation_patch_validation"]

    def to_contract(self):
        return "## Playbooks\n- web_research"


def _scored_result(content: str = "result data", turn_nr: int = 0, tags: list[str] | None = None) -> ScoredResult:
    return ScoredResult(
        result=SessionResult(content=content, entities=["x"], turn_nr=turn_nr, tags=tags or []),
        score=RelevanceScore(total=0.9),
    )

def _scored_atom(
    content: str = "atom data",
    atom_type: AtomType = AtomType.RESULT,
    tags: list[str] | None = None,
) -> ScoredAtom:
    return ScoredAtom(
        atom=WorkingAtom(atom_type=atom_type, content=content, task_id="t1", tags=tags or []),
        score=RelevanceScore(total=0.9),
    )


# ---------------------------------------------------------------------------
# DEFAULT_CONSTRAINTS
# ---------------------------------------------------------------------------

def test_default_constraints_not_empty() -> None:
    assert len(DEFAULT_CONSTRAINTS) > 0


# ---------------------------------------------------------------------------
# decision-equivalent: include_results=True only
# ---------------------------------------------------------------------------

def test_decision_includes_ego() -> None:
    builder = _builder()
    model   = _model(known_results=[_scored_result()])
    prompt  = builder.build(model, ego="You are a classifier.", input_policy=PromptInputPolicy(include_session_result_store=True))
    assert "You are a classifier." in prompt

def test_decision_includes_known_results() -> None:
    builder = _builder()
    model   = _model(known_results=[_scored_result("important fact")])
    prompt  = builder.build(model, ego="Ego.", input_policy=PromptInputPolicy(include_session_result_store=True))
    assert "important fact" in prompt

def test_decision_includes_constraints() -> None:
    builder = _builder()
    model   = _model()
    prompt  = builder.build(model, ego="Ego.", input_policy=PromptInputPolicy(include_session_result_store=True))
    assert "constraints" in prompt.lower()

def test_decision_no_results_no_context_section() -> None:
    builder = _builder()
    model   = _model()
    prompt  = builder.build(model, ego="Ego.", input_policy=PromptInputPolicy(include_session_result_store=True))
    assert "Known results" not in prompt

def test_decision_empty_constraints() -> None:
    builder = _builder()
    model   = _model()
    prompt  = builder.build(model, ego="Ego.", input_policy=PromptInputPolicy(include_session_result_store=True), constraints=[])
    assert "constraints" not in prompt.lower()


# ---------------------------------------------------------------------------
# orient-equivalent: include_directives + include_intents + include_entities
# ---------------------------------------------------------------------------

def test_orient_includes_intent() -> None:
    builder = _builder()
    model   = _model(intents=["informational"], entities=["python"])
    prompt  = builder.build(
        model, ego="Ego.",
        input_policy=PromptInputPolicy(include_current_request_interpretation=True),
    )
    assert "informational" in prompt
    assert "python" in prompt

def test_orient_includes_directives() -> None:
    builder = _builder()
    d       = Directive(key="conversation_language", value="de")
    model   = _model(directives=[d], intents=["info"])
    prompt  = builder.build(
        model, ego="Ego.",
        input_policy=PromptInputPolicy(include_directive_store=True, include_current_request_interpretation=True),
    )
    assert "de" in prompt

def test_orient_renders_all_known_directives_via_their_templates() -> None:
    # Proves the KnownDirective enum's templates (not just raw values) are
    # actually wired through SystemPromptBuilder for every known directive —
    # language, timezone, and tone — not just the single key exercised above.
    builder = _builder()
    directives = [
        Directive(key="conversation_language", value="de"),
        Directive(key="timezone", value="Europe/Berlin"),
        Directive(key="tone", value="formal"),
    ]
    model = _model(directives=directives, intents=["info"])
    prompt = builder.build(
        model, ego="Ego.",
        input_policy=PromptInputPolicy(include_directive_store=True, include_current_request_interpretation=True),
    )

    assert "Respond in the language specified by ISO 639-1 code: de." in prompt
    assert (
        "The user's local timezone is Europe/Berlin (IANA format). "
        "Use this when interpreting or formatting dates and times." in prompt
    )
    assert "Use a formal tone throughout your response." in prompt


def test_orient_no_recent_turns() -> None:
    builder = _builder()
    turns   = [TurnSummary(turn_nr=1, user_input="hi", intent="greet", topic="t")]
    model   = _model(recent_turns=turns)
    prompt  = builder.build(
        model, ego="Ego.",
        input_policy=PromptInputPolicy(include_directive_store=True, include_current_request_interpretation=True),
    )
    assert "Recent conversation" not in prompt


# ---------------------------------------------------------------------------
# task-equivalent: adds include_recent_turns + include_atoms + include_results
# ---------------------------------------------------------------------------

def test_task_includes_recent_turns() -> None:
    builder = _builder()
    turns   = [TurnSummary(turn_nr=1, user_input="hello", intent="greet", topic="t")]
    model   = _model(recent_turns=turns, intents=["info"])
    prompt  = builder.build(
        model, ego="Ego.",
        input_policy=PromptInputPolicy(include_current_request_interpretation=True, include_conversation_store=True),
    )
    assert "Recent conversation" in prompt

def test_task_includes_known_results() -> None:
    builder = _builder()
    model   = _model(known_results=[_scored_result("my data")])
    prompt  = builder.build(model, ego="Ego.", input_policy=PromptInputPolicy(include_session_result_store=True))
    assert "my data" in prompt

def test_task_no_active_atoms_section_without_flag() -> None:
    builder = _builder()
    model   = _model(active_atoms=[_scored_atom()])
    prompt  = builder.build(model, ego="Ego.")   # no include_atoms
    assert "working context" not in prompt.lower()


def test_policy_can_include_working_memory_without_session_results() -> None:
    builder = _builder()
    model = _model(
        active_atoms=[_scored_atom("working note")],
        known_results=[_scored_result("stored result")],
    )
    prompt = builder.build(
        model,
        ego="Ego.",
        input_policy=PromptInputPolicy(
            include_working_memory=True,
        ),
    )
    assert "working note" in prompt
    assert "stored result" not in prompt


def test_working_memory_render_keeps_primary_kind_tag_visible() -> None:
    builder = _builder()
    model = _model(
        active_atoms=[_scored_atom("url=https://example.com | title=Example", AtomType.CONTEXT, ["kind:selected_source"])],
    )
    prompt = builder.build(
        model,
        ego="Ego.",
        input_policy=PromptInputPolicy(
            include_working_memory=True,
            working_memory_tags=["kind:selected_source"],
        ),
    )

    assert "[kind:selected_source]" in prompt
    assert "url=https://example.com" in prompt


def test_working_memory_render_strips_duplicate_kind_prefix_from_content() -> None:
    builder = _builder()
    model = _model(
        active_atoms=[_scored_atom("[kind:source_candidate] candidate_id=cand_1 | title=Example", AtomType.CONTEXT, ["kind:source_candidate"])],
    )
    prompt = builder.build(
        model,
        ego="Ego.",
        input_policy=PromptInputPolicy(
            include_working_memory=True,
            working_memory_tags=["kind:source_candidate"],
        ),
    )

    assert "[kind:source_candidate] candidate_id=cand_1" in prompt
    assert "[kind:source_candidate] [kind:source_candidate]" not in prompt


def test_policy_can_filter_working_memory_by_tags() -> None:
    builder = _builder()
    model = _model(
        active_atoms=[
            ScoredAtom(
                atom=WorkingAtom(
                    atom_type=AtomType.CONTEXT,
                    content="candidate a",
                    task_id="t1",
                    tags=["kind:source_candidate"],
                ),
                score=RelevanceScore(total=0.9),
            ),
            ScoredAtom(
                atom=WorkingAtom(
                    atom_type=AtomType.EVAL,
                    content="evaluated fact",
                    task_id="t1",
                    tags=["kind:evaluated_evidence"],
                ),
                score=RelevanceScore(total=0.8),
            ),
        ],
    )
    prompt = builder.build(
        model,
        ego="Ego.",
        input_policy=PromptInputPolicy(
            include_working_memory=True,
            working_memory_tags=["kind:evaluated_evidence"],
        ),
    )

    assert "evaluated fact" in prompt
    assert "candidate a" not in prompt


def test_active_playbook_prompt_is_compact_not_full_contract() -> None:
    ctx = type("Ctx", (), {"playbooks": _StubPlaybooks(), "tools": None, "messages": []})()
    builder = SystemPromptBuilder(ctx=ctx)
    model = _model()

    prompt = builder.build(
        model,
        ego="Ego.",
        input_policy=PromptInputPolicy(include_playbooks=True, active_playbook="web_research"),
    )

    assert "## Active Playbook: web_research" in prompt
    assert "Search the web for current information." in prompt
    assert "Other available playbooks" in prompt
    assert "## Playbooks" not in prompt


def test_active_phase_details_render_via_system_prompt_builder() -> None:
    ctx = type("Ctx", (), {"playbooks": _StubPlaybooks(), "tools": None, "messages": []})()
    builder = SystemPromptBuilder(ctx=ctx)
    model = _model()
    phase = Phase(
        name="select",
        role="transformation",
        goal="Transform source candidates into a focused shortlist",
        guidance="Return only compact tagged selection lines.",
        exit_condition="At least one selected source was produced.",
        tools=[],
        input_tags=["kind:source_candidate"],
        output_tags=["kind:selected_source"],
        output_example="[kind:selected_source] url=https://example.com | title=Example | reason=primary_source",
    )

    prompt = builder.build(
        model,
        ego="Ego.",
        input_policy=PromptInputPolicy(include_playbooks=True, active_playbook="web_research", active_phase=phase),
    )

    assert "## Active Phase: select" in prompt
    assert "Visible inputs: kind:source_candidate" in prompt
    assert "Required outputs: kind:selected_source" in prompt
    assert "transform visible `kind:source_candidate` entries into explicit `[kind:selected_source]` outputs" in prompt
    assert "Example: [kind:selected_source] url=https://example.com | title=Example | reason=primary_source" in prompt
    assert "Return one compact tagged line per result." in prompt


def test_policy_can_include_session_results_without_conversation_store() -> None:
    builder = _builder()
    turn = TurnSummary(turn_nr=1, user_input="hello", intent="greet", topic="t")
    model = _model(
        recent_turns=[turn],
        known_results=[_scored_result("stored result")],
    )
    prompt = builder.build(
        model,
        ego="Ego.",
        input_policy=PromptInputPolicy(
            include_session_result_store=True,
        ),
    )
    assert "stored result" in prompt
    assert "Recent conversation" not in prompt


# ---------------------------------------------------------------------------
# plan-equivalent: adds include_delta (SPEC/PLAN atoms + delta)
# ---------------------------------------------------------------------------

def test_plan_includes_active_atoms() -> None:
    builder = _builder()
    model   = _model(active_atoms=[_scored_atom("atom content")])
    prompt  = builder.build(model, ego="Ego.", input_policy=PromptInputPolicy(include_working_memory=True))
    assert "atom content" in prompt

def test_plan_includes_spec_and_plan_atom_types() -> None:
    builder = _builder()
    model   = _model(active_atoms=[_scored_atom("spec data", AtomType.SPEC)])
    prompt  = builder.build(model, ego="Ego.", input_policy=PromptInputPolicy(include_delta=True))
    assert "spec data" in prompt
    assert "Planning context:" in prompt

def test_plan_no_directives_when_flag_off() -> None:
    builder = _builder()
    d       = Directive(key="tone", value="formal")
    model   = _model(directives=[d])
    prompt  = builder.build(model, ego="Ego.")   # include_directives=False (default)
    assert "behavioral rules" not in prompt


# ---------------------------------------------------------------------------
# Unknown directives — fallback rendering
# ---------------------------------------------------------------------------

def test_unknown_directive_fallback_rendering() -> None:
    builder = _builder()
    d       = Directive(key="custom_key", value="custom_val")
    model   = _model(directives=[d], intents=["info"])
    prompt  = builder.build(
        model, ego="Ego.",
        input_policy=PromptInputPolicy(include_directive_store=True, include_current_request_interpretation=True),
    )
    assert "custom_key" in prompt
    assert "custom_val" in prompt


# ---------------------------------------------------------------------------
# Turn result summary included
# ---------------------------------------------------------------------------

def test_task_turn_with_result_summary() -> None:
    builder = _builder()
    turn    = TurnSummary(
        turn_nr=1, user_input="what is x", intent="info",
        topic="t", result_summary="X is 42",
    )
    model   = _model(recent_turns=[turn], intents=["info"])
    prompt  = builder.build(
        model, ego="Ego.",
        input_policy=PromptInputPolicy(include_current_request_interpretation=True, include_conversation_store=True),
    )
    assert "X is 42" in prompt


def test_prompt_compression_omits_old_session_results_when_budget_is_exceeded() -> None:
    builder = _builder()
    reports: list[PromptCompressionReport] = []
    model = _model(
        known_results=[
            _scored_result("old fact " + ("x" * 300), turn_nr=1, tags=["user_fact"]),
            _scored_result("new fact stays", turn_nr=10, tags=["user_fact"]),
        ],
    )

    prompt = builder.build(
        model,
        ego="Ego.",
        input_policy=PromptInputPolicy(include_session_result_store=True),
        constraints=[],
        max_system_prompt_tokens=40,
        compression_reports=reports,
    )

    assert "new fact stays" in prompt
    assert "old fact" not in prompt
    assert reports
    assert reports[0].omitted_session_results == 1
    assert reports[0].reason == "system_prompt_token_budget"
