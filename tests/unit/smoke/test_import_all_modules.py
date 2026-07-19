# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

import importlib
import pkgutil

import nlght


def test_all_nlght_modules_are_importable() -> None:
    failed: list[tuple[str, str]] = []

    for module_info in pkgutil.walk_packages(nlght.__path__, prefix="nlght."):
        try:
            importlib.import_module(module_info.name)
        except Exception as exc:  # pragma: no cover - this branch should stay empty
            failed.append((module_info.name, str(exc)))

    assert failed == []
