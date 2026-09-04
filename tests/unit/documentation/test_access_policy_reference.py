# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import re
from pathlib import Path
from typing import get_args

from nlght.core.access.rule import SubjectType, tool_subject

REFERENCE = Path(__file__).parents[3] / "docs/documentation/access-policies.mdx"
START = "{/* access-policy-subject-types:start */}"
END = "{/* access-policy-subject-types:end */}"


def test_access_policy_reference_covers_every_subject_type_and_address_shape() -> None:
    text = REFERENCE.read_text(encoding="utf-8")
    assert text.count(START) == 1
    assert text.count(END) == 1
    table = text.split(START, 1)[1].split(END, 1)[0]
    documented_types = set(re.findall(r"^\| `([^`]+)` \|", table, flags=re.MULTILINE))

    assert documented_types == set(get_args(SubjectType))
    assert "`<kind>/<name>`" in table
    assert "`<kind>/<name>::<operation>`" in table
    assert tool_subject("data_store/data-main", "data-search") in text
