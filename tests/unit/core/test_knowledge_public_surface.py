# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Keep knowledge package exports tied to actual consumers.

An unused public alias would add an untested API promise.
"""

from __future__ import annotations

import ast
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_PACKAGE = _ROOT / "src" / "nlght" / "core" / "knowledge" / "__init__.py"


def _exported() -> list[str]:
    """The names `__all__` promises, read rather than guessed.

    Returns:
        list[str]: Names declared by the module export list.
    """
    tree = ast.parse(_PACKAGE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        ):
            return [
                element.value
                for element in node.value.elts  # type: ignore[attr-defined]
                if isinstance(element, ast.Constant) and isinstance(element.value, str)
            ]
    raise AssertionError("core.knowledge has no __all__; this guard has gone blind")


def test_no_export_is_dead() -> None:
    """Require a consumer for each public knowledge export."""
    exported = _exported()
    assert len(exported) > 50, "the export list looks truncated; the guard would pass blindly"

    sources = [
        path
        for path in (*(_ROOT / "src").rglob("*.py"), *(_ROOT / "tests").rglob("*.py"))
        if path != _PACKAGE
    ]
    body = "\n".join(path.read_text(encoding="utf-8") for path in sources)

    dead = [name for name in exported if name not in body]

    assert dead == [], (
        f"exported and used nowhere: {dead}. Either something should be using it, "
        f"or it should not be promised. An export with no consumer is a public "
        f"name with no test behind it."
    )
