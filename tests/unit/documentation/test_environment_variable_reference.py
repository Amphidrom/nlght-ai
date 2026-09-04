# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Keep the main runtime environment-variable reference aligned with code."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
REFERENCE = ROOT / "docs/documentation/deployment/environment-vars.mdx"
SOURCE = ROOT / "src/nlght"
START = "{/* runtime-environment-variables:start */}"
END = "{/* runtime-environment-variables:end */}"


def test_runtime_environment_variable_table_matches_code() -> None:
    documented_section = REFERENCE.read_text().split(START, 1)[1].split(END, 1)[0]
    documented = set(re.findall(r"^\| `(?P<name>NLGHT_[A-Z0-9_]+)`", documented_section, re.MULTILINE))

    read_by_runtime: set[str] = set()
    pattern = re.compile(r'os\.environ\.get\("(?P<name>NLGHT_[A-Z0-9_]+)"')
    for source_file in SOURCE.rglob("*.py"):
        read_by_runtime.update(pattern.findall(source_file.read_text()))

    assert documented == read_by_runtime
