# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Tests that guard the direction of the architecture rather than its behaviour.

Every rule here was true when it was written and is the sort of thing that decays
without anybody deciding to break it: an import added for convenience, one branch
on a type "just this once". A behavioural test cannot see any of that — the prompt
still comes out right — which is exactly why these exist.

They are deliberately narrow. Each one names a direction that was argued for in an
ADR, and nothing else, so a failure means the architecture moved and not that
somebody renamed a variable.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

#: Types whose names in the wrong module would mean a storage type is deciding
#: policy again (ADR-0054).
_DOMAIN_TYPES = (
    "SessionResult", "WorkingAtom", "TurnSummary", "ContextPassage", "AtomType",
)


def _module_source(dotted: str) -> str:
    import importlib

    module = importlib.import_module(dotted)
    assert module.__file__ is not None
    return Path(module.__file__).read_text(encoding="utf-8")


def _imports(dotted: str) -> set[str]:
    """Every module this one imports, at any level, including lazy ones."""
    tree = ast.parse(_module_source(dotted))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def _names_in(tree: ast.AST) -> set[str]:
    """Every identifier the code uses, with prose excluded.

    Comments do not survive parsing and string constants are skipped, so a
    docstring may name the very thing the code must not touch — which is what a
    docstring explaining a removal does. Scanning raw text would make explaining
    a rule indistinguishable from breaking it.
    """
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant):
            continue
        if isinstance(node, ast.Attribute):
            found.add(node.attr)
        elif isinstance(node, ast.Name):
            found.add(node.id)
    return found


def _body_without_docstrings(function) -> str:  # noqa: ANN001
    """The code of a function, with its prose removed.

    A docstring may — and here does — name the very thing the code must not
    touch, which is the point of the docstring. Asserting against raw source
    would make explaining a rule the same as breaking it.
    """
    import textwrap

    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    stripped: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            continue
        if isinstance(node, ast.Attribute):
            stripped.append(node.attr)
        elif isinstance(node, ast.Name):
            stripped.append(node.id)
    return " ".join(stripped)


# The reduction weighs information and never asks what it is made of
# ---------------------------------------------------------------------------

def test_the_reduction_never_reads_what_an_element_is() -> None:
    from nlght.core.hive_mind.relevance import _steps, reduce_to_budget

    for function in (reduce_to_budget, _steps):
        names = _body_without_docstrings(function)
        assert "kind" not in names.split(), (
            f"{function.__name__} reads element.kind — storage types are deciding again"
        )
        for domain in _DOMAIN_TYPES:
            assert domain not in names, f"{function.__name__} knows about {domain}"


def test_the_reduction_module_does_not_reach_into_retrieval() -> None:
    # `relevance` may know the session's own model — its scoring half is written
    # against it — but a retrieved passage must reach it only as an element.
    assert not any(
        "context" in name or "retrieval" in name
        for name in _imports("nlght.core.hive_mind.relevance")
    )


# The renderer renders and does not judge
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "renderer",
    ["nlght.core.hive_mind.system_prompt", "nlght.core.hive_mind.model_context"],
)
def test_the_renderer_holds_no_domain_classifier(renderer: str) -> None:
    source = _module_source(renderer)

    for vocabulary in ("user_fact", "READABLE_ATOM_TYPES", "AtomType", "known_results"):
        assert vocabulary not in source, f"the renderer knows about '{vocabulary}' again"


def test_the_rendering_function_reads_only_presentation() -> None:
    # Rendering moved out of `SystemPromptBuilder` when trusted instructions were
    # split from untrusted knowledge (ADR-0058): the builder now renders nothing
    # but deployment text, and the function this rule was written for is
    # `ModelContextBuilder._build_records`.
    from nlght.core.hive_mind.model_context import ModelContextBuilder

    names = _body_without_docstrings(ModelContextBuilder._build_records)  # noqa: SLF001

    assert "presentation" in names
    for domain in _DOMAIN_TYPES:
        assert domain not in names


def test_the_rendering_function_carries_kind_without_branching_on_it() -> None:
    """`kind` may travel to the record; it may not decide anything on the way.

    The record is typed data now, so the renderer has to name the kind in order
    to *carry* it — the old rule, "never read `kind` at all", would forbid the
    very field that keeps provenance intact. What must not come back is the
    branch: the moment a condition asks what an element is made of, a storage
    type is choosing presentation again (ADR-0054).
    """
    import textwrap

    from nlght.core.hive_mind.model_context import ModelContextBuilder

    tree = ast.parse(textwrap.dedent(
        inspect.getsource(ModelContextBuilder._build_records),  # noqa: SLF001
    ))
    for node in ast.walk(tree):
        tests = (
            [node.test] if isinstance(node, ast.If | ast.IfExp | ast.While)
            else [node.subject] if isinstance(node, ast.Match)
            else []
        )
        for test in tests:
            assert "kind" not in _names_in(test), (
                "_build_records branches on an element's kind — the renderer is "
                "classifying storage types again"
            )


# Adapters depend on the generic vocabulary, never the other way round
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "adapter",
    ["nlght.core.hive_mind.elements", "nlght.core.context.elements"],
)
def test_an_adapter_does_not_know_the_renderer(adapter: str) -> None:
    # An adapter that imported the prompt builder could start formatting for it,
    # and the judgement about *where* something belongs would drift back into
    # how it looks.
    assert not any("system_prompt" in name for name in _imports(adapter))


def test_retrieval_does_not_depend_on_the_mental_model() -> None:
    # Retrieval is upstream. It produces passages; whether a passage becomes an
    # element is somebody else's question.
    for module in ("nlght.core.retrieval.hit", "nlght.core.retrieval.fusion",
                   "nlght.core.retrieval.plan"):
        assert not any("hive_mind" in name for name in _imports(module))


def test_the_model_call_knows_nothing_about_how_its_prompt_was_made() -> None:
    """The seam between preparing a prompt and executing one (ADR-0055).

    `passthrough` sends what it is given. If it ever imported retrieval or the
    mental model it would be able to *decide* something about them, and the two
    halves would stop being separable.
    """
    imported = _imports("nlght.adapters.outbound.workflow.steps.passthrough")

    for forbidden in ("retrieval", "hive_mind", "core.context"):
        assert not any(forbidden in name for name in imported), (
            f"passthrough imports {forbidden}"
        )


# Nothing selects before the engine
# ---------------------------------------------------------------------------

def test_the_builder_no_longer_discards_candidates() -> None:
    """The defect this guards against is a `[:limit]` reappearing "just here".

    `MentalModelBuilder` assembles and scores; what a model is told is decided
    once, later, with retention and cost in view (ADR-0056). A cap or a threshold
    back in this file would remove information before anything could weigh what
    losing it costs — and every behavioural test would still pass, because the
    prompt would simply be a little shorter.
    """
    names = _names_in(ast.parse(_module_source("nlght.core.hive_mind.builder")))

    for gone in ("filter_atoms", "filter_results", "max_atoms",
                 "max_results", "max_turns", "threshold"):
        assert gone not in names, f"the builder is selecting again, via {gone}"


def test_the_engine_offers_no_way_to_discard() -> None:
    import inspect

    from nlght.core.hive_mind.relevance import RelevanceEngine

    assert not hasattr(RelevanceEngine, "filter_atoms")
    assert not hasattr(RelevanceEngine, "filter_results")
    assert "threshold" not in inspect.signature(RelevanceEngine.__init__).parameters


def test_ranking_never_slices_its_input() -> None:
    from nlght.core.hive_mind.relevance import RelevanceEngine

    for name in ("rank", "rank_atoms", "rank_results"):
        body = _body_without_docstrings(getattr(RelevanceEngine, name))
        assert "limit" not in body.split(), f"{name} takes a limit again"


def test_there_is_only_one_notion_of_how_fully_to_say_something() -> None:
    """No second `1/2/3` beside the representations.

    `ScoredAtom.depth` was derived from the relevance total and read only by a
    debug dump, and by the end it serialised the same constant for every atom.
    The danger was never the dead field: it was that somebody would later reach
    for it as a shortcut for "how fully should this render", and go around
    retention and the budget entirely (ADR-0058).
    """
    from nlght.core.hive_mind.models import ScoredAtom
    from nlght.core.hive_mind.relevance import RelevanceEngine

    assert not hasattr(ScoredAtom, "depth")
    assert "depth" not in {field.name for field in ScoredAtom.__dataclass_fields__.values()}
    assert not hasattr(RelevanceEngine, "_depth_from_score")

    # And the one that remains is the real one.
    from nlght.core.hive_mind.models import LEVELS, Level

    assert LEVELS == (Level.FULL, Level.COMPACT, Level.OMIT)


def test_no_production_path_reaches_a_session_around_its_owner() -> None:
    """The whole boundary is worth exactly as much as the paths that go through it.

    `SessionAccess` decides who may open a session — and decides nothing at all
    for a caller that holds the factory and asks it directly. That call is one
    line, it works, and it is invisible in review because it looks like every
    other factory call in the codebase. So the rule is checked here rather than
    trusted:

        No production path may reach session-bound state by `session_key` alone,
        without a successful `SessionAccess` first.

    Outside the boundary itself, that means no production module calls
    `get_or_create`, `owner_of`, `claim_ownership` or `list_sessions` on a store
    coordinator factory or a session backend.

    `WorkspaceManager.get_or_create(session_key)` is held to the same rule, and
    for the same reason rather than a new one. A workspace is **part of its
    session** — same key, same lifecycle — so it has no owner of its own and
    needs no second ownership layer: whoever may not open the session may not
    have the session's workspace either, and the authorization is inherited.

    Neither that port nor `SessionBackend.delete`/`list_sessions` has a
    production caller today, which is exactly why they are in here. Nothing is
    being called unsafe. The day somebody wires one up — the session-management
    endpoints those backend methods exist for, or a step that wants a working
    directory — this test is what says the call has to sit behind an authorized
    session (ADR-0061).
    """
    root = Path(__file__).resolve().parents[3] / "src" / "nlght"
    assert (root / "core" / "session" / "access.py").exists(), (
        "the scan root is wrong, so this test would pass over an empty directory"
    )
    allowed = {
        root / "core" / "session" / "access.py",          # the boundary
        root / "adapters" / "outbound" / "hive_mind" / "simple.py",       # implements it
        root / "adapters" / "outbound" / "hive_mind" / "coordinator.py",  # implements it
        root / "adapters" / "outbound" / "hive_mind" / "persistence.py",  # implements it
        root / "ports" / "outbound" / "store_coordinator.py",             # declares it
        root / "ports" / "outbound" / "session_backend.py",               # declares it
    }
    # Every name here is one only a session factory or backend answers to.
    # `claim_ownership` is spelled out rather than `claim` for that reason: the
    # execution queue also claims things, and a guard that cannot tell the two
    # apart gets relaxed by the first person it inconveniences.
    guarded = {"get_or_create", "owner_of", "claim_ownership", "list_sessions"}

    offenders: list[str] = []
    for path in root.rglob("*.py"):
        if path in allowed:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr not in guarded:
                continue
            # `self._factory.get_or_create(...)` and `factory.get_or_create(...)`
            # both matter, and so does a workspace manager's: the receiver is a
            # different port, but the argument is the same session key and the
            # state behind it belongs to the same session.
            name = ast.unparse(node.func.value)
            offenders.append(f"{path.relative_to(root)}:{node.lineno} — {name}.{node.func.attr}")

    assert not offenders, (
        "these reach a session without passing the ownership boundary:\n  "
        + "\n  ".join(offenders)
    )


def test_no_step_activates_a_resource_around_the_guard() -> None:
    """One authorization boundary, and no way past it.

    The seven step-internal activations were the same twenty lines copied, none
    of which asked a policy — so a workflow could reach any resource in the
    database by naming it in its own configuration, and `data_index_writer`,
    whose only protection was declaring no signatures, was reachable by exactly
    the thing that uses it.

    This holds the property rather than the refactor: a step that resolves
    resources itself, or calls the loader directly, has built a second way in.
    """
    steps = Path(__file__).resolve().parents[3] / "src" / "nlght" / "adapters" / "outbound" / "workflow" / "steps"
    offenders = {
        str(path.relative_to(steps)): marker
        for path in steps.rglob("*.py")
        for marker in ("find_by_kind", ".instantiate(")
        if marker in path.read_text(encoding="utf-8")
    }

    assert offenders == {}, (
        f"a step resolves or builds a resource without the activator: {offenders}"
    )


def test_tool_implementations_execute_only_behind_the_action_gate() -> None:
    """A new in-repo catalog cannot grow a second unchecked execution path."""
    root = Path(__file__).resolve().parents[3] / "src" / "nlght"
    offenders: list[str] = []
    allowed = Path("adapters/outbound/tools/action_gate.py")

    for path in root.rglob("*.py"):
        relative = path.relative_to(root)
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "_execute_bound"
                and relative != allowed
            ):
                offenders.append(f"{relative}:{node.lineno}")

    assert offenders == [], (
        "tool implementation invoked outside ActionGate:\n  " + "\n  ".join(offenders)
    )
