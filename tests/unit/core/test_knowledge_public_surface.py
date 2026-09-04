# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Everything the knowledge package exports is used by something.

`ProjectionResult` was defined, exported and consumed by nobody — the module that
would have used it spelled the union out longhand instead, so the alias never
earned its keep and nothing said so. One dead export is trivial; the habit is
not, because a public name that nothing exercises is a promise with no test
behind it and it is where "we might need this later" accumulates.

This is the same shape of guard as the provenance contract next door: it does not
name the export that was wrong, it makes the *class* of mistake fail.
"""

from __future__ import annotations

import ast
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_PACKAGE = _ROOT / "src" / "nlght" / "core" / "knowledge" / "__init__.py"


def _exported() -> list[str]:
    """The names `__all__` promises, read rather than guessed."""
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
    """A name nothing outside its own module uses does not belong in `__all__`.

    Counted across `src` and `tests` together, because a name exercised only by a
    test is still exercised — the point is that *something* depends on it, not
    where the dependency lives.

    Removing one is safe here in a way it would not be in a published library:
    this package has no entry points and no plugin surface, so `__all__` is a
    statement about this repository and nothing else can be relying on it.
    """
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
        f"or it should not be promised — an export with no consumer is a public "
        f"name with no test behind it."
    )
