# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Check archived artifact reference hydration without prototype dependencies."""

from __future__ import annotations

import ast
import dataclasses
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace


@dataclass
class _Atom:
    id: str
    content: str
    content_type: str


def _resolver(read_result: SimpleNamespace):
    """Load the archived resolver with a controlled workspace read.

    Args:
        read_result (SimpleNamespace): Result returned by the fake workspace.

    Returns:
        type: Archived resolver class with its real method bodies.
    """
    source = (
        Path(__file__).resolve().parents[3]
        / "next-integration/old-jarvis-steps/intel/integration.py"
    )
    tree = ast.parse(source.read_text())
    resolver_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ArtifactReferenceResolver"
    )
    workspace = SimpleNamespace(read_file=lambda _operation: read_result)
    adapter = SimpleNamespace(get_workspace=lambda _workspace_id: workspace)
    namespace = {
        "__name__": "legacy_reference_resolver",
        "KnowledgeAtom": _Atom,
        "WorkspaceAdapter": object,
        "ReadOperation": SimpleNamespace,
        "ContentType": SimpleNamespace(TEXT="text", CODE="code", JSON="json"),
        "JSONDecodeError": json.JSONDecodeError,
        "loads": json.loads,
        "dataclasses": dataclasses,
        "get_workspace_adapter": lambda: adapter,
        "logger": logging.getLogger(__name__),
    }
    module = ast.fix_missing_locations(ast.Module(body=[resolver_node], type_ignores=[]))
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["ArtifactReferenceResolver"]


def test_resolve_keeps_original_atom_when_workspace_read_fails() -> None:
    """Keep the reference when the workspace cannot supply its content."""
    resolver = _resolver(SimpleNamespace(success=False, error="missing"))
    atom = _Atom("atom-1", '{"workspace_id":"ws","workspace_path":"a"}', "text")

    assert resolver.resolve(atom) is atom


def test_resolve_uses_payload_string_for_text_atom() -> None:
    """Hydrate text atoms with the payload rather than metadata."""
    resolver = _resolver(
        SimpleNamespace(success=True, content="stored text", size_bytes=11, hash="h")
    )
    atom = _Atom("atom-1", '{"workspace_id":"ws","workspace_path":"a"}', "text")

    hydrated = resolver.resolve(atom)

    assert hydrated is not atom
    assert hydrated.content == "stored text"
    assert isinstance(hydrated.content, str)


def test_workspace_listing_calls_manifest_without_extra_argument() -> None:
    """List workspace entries through the manifest method's actual signature."""
    source = (
        Path(__file__).resolve().parents[3]
        / "next-integration/old-jarvis-steps/persistence/artifact.py"
    )
    tree = ast.parse(source.read_text())
    workspace = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "WorkspaceRef"
    )
    method = next(
        node
        for node in workspace.body
        if isinstance(node, ast.FunctionDef) and node.name == "list_files"
    )
    namespace = {
        "ListOperation": object,
        "ListResult": lambda **fields: SimpleNamespace(**fields),
        "logger": logging.getLogger(__name__),
    }
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    exec(compile(module, str(source), "exec"), namespace)

    entry = SimpleNamespace(path="docs/example.txt")
    manifest = SimpleNamespace(files={entry.path: entry})
    adapter = SimpleNamespace(
        workspace_id="workspace",
        get_manifest=lambda: manifest,
    )

    result = namespace["list_files"](adapter, SimpleNamespace(path="docs"))

    assert result.success is True
    assert result.entries == [entry]
