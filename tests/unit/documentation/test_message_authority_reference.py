# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""The documented authority table is the security contract users read.

A canonical message type added without a row here would be a type whose
provider representation nobody ever decided in public — which is the same
failure the adapters guard against, one layer up.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import get_args

from nlght.core.model.messages import (
    CanonicalMessage,
    compose_messages,
    compose_prompt,
)

REFERENCE = Path(__file__).parents[3] / "docs/documentation/workflows/custom-steps.mdx"
START = "{/* message-authority-types:start */}"
END = "{/* message-authority-types:end */}"


def test_the_reference_documents_every_canonical_message_type() -> None:
    text = REFERENCE.read_text(encoding="utf-8")
    assert text.count(START) == 1
    assert text.count(END) == 1
    table = text.split(START, 1)[1].split(END, 1)[0]
    documented = set(re.findall(r"^\| `([^`]+)` \|", table, flags=re.MULTILINE))

    assert documented == {member.__name__ for member in get_args(CanonicalMessage)}


def test_the_documented_composition_helpers_take_the_documented_arguments() -> None:
    # The page tells a step author to call these two by name; a rename that
    # left the prose behind would be found by a reader, not by a test.
    assert list(inspect.signature(compose_prompt).parameters) == [
        "trusted_instructions", "context",
    ]
    assert list(inspect.signature(compose_messages).parameters) == [
        "envelope", "conversation",
    ]


def test_the_page_does_not_teach_a_role_dict_prompt() -> None:
    text = REFERENCE.read_text(encoding="utf-8")

    # The old pattern, kept only inside the warning that names it as wrong.
    assert '{"role": "system"' not in text
