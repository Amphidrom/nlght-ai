# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Check categorized output from the archived knowledge formatter."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


def test_artifact_label_reaches_artifact_category(monkeypatch) -> None:
    """Place a rendered artifact in the artifact list."""
    helpers = ModuleType("json_helper")
    helpers.loads = json.loads
    helpers.dumps = json.dumps
    monkeypatch.setitem(sys.modules, "json_helper", helpers)

    workflows = ModuleType("workflows")
    workflows.__path__ = []
    intel = ModuleType("workflows.intel")
    intel.__path__ = []
    atoms = ModuleType("workflows.intel.atoms")
    atoms.KnowledgeAtom = object
    atoms.ContentType = object
    for module in (workflows, intel, atoms):
        monkeypatch.setitem(sys.modules, module.__name__, module)

    source = (
        Path(__file__).resolve().parents[3]
        / "next-integration/old-jarvis-steps/intel/formatting.py"
    )
    spec = importlib.util.spec_from_file_location("legacy_context_formatter", source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)

    artifact = SimpleNamespace(
        content=json.dumps({"artifact_type": "Artifact", "artifact_name": "report"}),
        source_step="write_report",
    )
    result = module.KnowledgeContextFormatter().format_categorized([artifact])

    assert result.artifacts == ["[Artifact from Task: report]"]
    assert result.diagnostics == []
    assert result.context == []
