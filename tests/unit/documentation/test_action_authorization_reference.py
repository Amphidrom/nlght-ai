# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import re
from pathlib import Path

from nlght.core.tools.action import (
    ActionDecisionReason,
    DataEgressClass,
    ExecutionClass,
    ScopeClass,
    SideEffectClass,
)

REFERENCE = Path(__file__).parents[3] / "docs/documentation/action-authorization.mdx"

#: Declared on the enum, but neither the gate nor the built-in policy produces
#: them: such failures surface as argument or semantics errors before a decision.
_NOT_DECIDED = {
    ActionDecisionReason.UNKNOWN_ACTION_SEMANTICS,
    ActionDecisionReason.PARAMETER_SCOPE_VIOLATION,
    ActionDecisionReason.INVALID_ARGUMENTS,
    ActionDecisionReason.ACTION_RESOLUTION_ERROR,
}


def _section(name: str) -> str:
    text = REFERENCE.read_text(encoding="utf-8")
    start = f"{{/* {name}:start */}}"
    end = f"{{/* {name}:end */}}"
    assert text.count(start) == 1
    assert text.count(end) == 1
    return text.split(start, 1)[1].split(end, 1)[0]


def test_reason_table_lists_every_decision_reason() -> None:
    table = _section("action-decision-reasons")
    first_column = re.findall(r"^\| (.+?) \|", table, flags=re.MULTILINE)
    documented = {
        reason
        for cell in first_column
        for reason in re.findall(r"`([a-z_]+)`", cell)
    }

    assert documented == {reason.value for reason in ActionDecisionReason} - {
        reason.value for reason in _NOT_DECIDED
    }


def test_default_policy_matrix_names_every_class() -> None:
    table = _section("default-action-policy-matrix")
    rows = {
        row.split("|")[1].strip(): row
        for row in table.splitlines()
        if row.startswith("| ") and not row.startswith("| Dimension")
    }

    for dimension, enum in (
        ("execution", ExecutionClass),
        ("side effect", SideEffectClass),
        ("egress", DataEgressClass),
        ("scope", ScopeClass),
    ):
        named = set(re.findall(r"`([a-z_]+)`", rows[dimension]))
        missing = {member.value for member in enum} - named
        # Execution names only `none` and `confined`. Everything else denies.
        if enum is ExecutionClass:
            missing.discard(ExecutionClass.UNCONFINED.value)
        assert not missing, f"{dimension}: {sorted(missing)}"
