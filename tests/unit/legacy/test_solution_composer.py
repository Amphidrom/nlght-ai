# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Check archived synthesis instructions without prototype dependencies."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace


def test_intent_text_uses_readable_language_instruction() -> None:
    """Include the stored language code in a correctly spelled instruction."""
    source = (
        Path(__file__).resolve().parents[3]
        / "next-integration/old-jarvis-steps/meta_core/solution_composer.py"
    )
    tree = ast.parse(source.read_text())
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "SolutionComposerStep"
    )
    method = next(
        node
        for node in owner.body
        if isinstance(node, ast.FunctionDef) and node.name == "_get_intent_text"
    )
    namespace = {"WorkflowStepContext": object, "_safe_json": json.loads}
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    exec(compile(module, str(source), "exec"), namespace)
    runtime_state = SimpleNamespace(
        get_intent=lambda: '{"objective":"Summarize","language":"de"}'
    )
    ctx = SimpleNamespace(runtime_state=runtime_state)

    text = namespace["_get_intent_text"](None, ctx)

    assert text == (
        "Summarize Please answer using the language that matches "
        "the language code: de."
    )
