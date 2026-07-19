# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from nlght.adapters.outbound.tools.builtin.memory_artifact import LoadMemoryArtifactTool
from nlght.ports.outbound.store_coordinator import ArtifactContent


def _tool(*, workspace=None, store=None, artifacts_dir=None) -> LoadMemoryArtifactTool:
    config = {"artifacts_dir": str(artifacts_dir)} if artifacts_dir else {}
    return LoadMemoryArtifactTool(name="memory", config=config, workspace=workspace, store_coordinator=store)


def test_memory_artifact_signatures_describe_full_and_range_operations() -> None:
    signatures = LoadMemoryArtifactTool.signatures()
    assert [signature.name for signature in signatures] == ["load_memory_artifact", "load_memory_artifact_range"]
    assert [parameter.name for parameter in signatures[1].parameters] == ["id", "from_line", "to_line"]


async def test_memory_artifact_loads_full_content_and_ranges_from_store() -> None:
    store = MagicMock()
    store.load_artifact.return_value = ArtifactContent("one\ntwo\nthree", ".py")
    tool = _tool(store=store)

    full = json.loads(await tool.load_artifact(id=" artifact "))
    excerpt = json.loads(await tool.load_artifact_range(id="artifact", from_line=2, to_line=99))

    assert full == {"id": " artifact ", "total_lines": 3, "content": "one\ntwo\nthree", "extension": ".py"}
    assert excerpt["content"] == "two\nthree"
    assert excerpt["from_line"] == 2
    assert excerpt["to_line"] == 3
    store.load_artifact.assert_any_call("artifact")


async def test_memory_artifact_reports_missing_store_content() -> None:
    store = MagicMock()
    store.load_artifact.return_value = None
    tool = _tool(store=store)
    assert json.loads(await tool.load_artifact(id="missing"))["error"] == "Artifact not found."
    assert json.loads(await tool.load_artifact_range(id="missing", from_line=1, to_line=2))["content"] == ""


async def test_memory_artifact_uses_workspace_before_configured_directory(tmp_path) -> None:
    configured = tmp_path / "configured"
    configured.mkdir()
    (configured / "id.txt").write_text("wrong")
    workspace_dir = tmp_path / "workspace"
    artifacts = workspace_dir / "memory_artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "id.md").write_text("alpha\nbeta\ngamma")
    tool = _tool(workspace=SimpleNamespace(root_path=workspace_dir), artifacts_dir=configured)

    full = json.loads(await tool.load_artifact(id="id"))
    excerpt = json.loads(await tool.load_artifact_range(id="id", from_line=0, to_line=2))
    assert full["content"] == "alpha\nbeta\ngamma"
    assert full["extension"] == ".md"
    assert excerpt["content"] == "alpha\nbeta"
    assert excerpt["from_line"] == 1


async def test_memory_artifact_reports_missing_files_and_directory(tmp_path) -> None:
    tool = _tool()
    assert json.loads(await tool.load_artifact(id="none"))["error"] == "Artifact not found."
    configured = tmp_path / "artifacts"
    configured.mkdir()
    configured_tool = _tool(artifacts_dir=configured)
    assert json.loads(await configured_tool.load_artifact_range(id="none", from_line=1, to_line=2))["error"] == "Artifact not found."
