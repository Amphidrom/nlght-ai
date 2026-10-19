# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Include the extraction contract in model input and versioning.

A changed contract must produce a different extraction version so documents
are processed under the new instructions.
"""

from __future__ import annotations

from nlght.adapters.outbound.workflow.steps.knowledge.extract import _PROMPT
from nlght.core.knowledge import BUILTIN_KINDS, extraction_version
from nlght.core.knowledge.classification import (
    AMBIGUITY_GATES,
    BOUNDARIES,
    DEFINITIONS,
    GATE_MATRIX,
    contract,
)


def _version(prompt: str) -> str:
    return extraction_version(
        workflow="w", model="p/m", prompt=prompt, settings={"min_confidence": 0.0}
    )


def test_the_prompt_carries_the_whole_contract() -> None:
    rendered = _PROMPT.format(kinds=", ".join(BUILTIN_KINDS), text="TEXT")

    for item in DEFINITIONS:
        assert item.definition in rendered
    for line in BOUNDARIES:
        assert line in rendered
    for item in (*GATE_MATRIX, *AMBIGUITY_GATES):
        assert item.sentence in rendered
        assert item.because in rendered


def test_the_contract_is_in_the_text_the_version_hashes() -> None:
    """Include the extraction contract in the prompt text used for versioning."""
    for item in DEFINITIONS:
        assert item.definition in _PROMPT
    assert GATE_MATRIX[0].sentence in _PROMPT


def test_changing_the_contract_moves_the_extraction_version() -> None:
    # Not a tautology about hashing: it holds that the contract is part of the
    # hashed input, which is the property a skipped document depends on.
    before = _version(_PROMPT)
    after = _version(_PROMPT.replace(DEFINITIONS[0].definition, "something else"))

    assert before != after


def test_the_prompt_still_substitutes_its_own_fields() -> None:
    # The contract is inserted with its braces escaped, so it cannot disturb the
    # per-call substitution. A contract containing a brace would otherwise raise
    # or silently swallow part of the text.
    rendered = _PROMPT.format(kinds="fact, rule", text="THE DOCUMENT")

    assert "THE DOCUMENT" in rendered
    assert "{text}" not in rendered
    assert "{kinds}" not in rendered


def test_only_the_four_defined_kinds_are_accepted() -> None:
    # The contract defines what the extractor is allowed to return. A kind
    # defined in one place and accepted in the other is a gap either way.
    assert {item.kind for item in DEFINITIONS} == set(BUILTIN_KINDS)


def test_the_contract_is_assembled_from_the_data_and_not_pasted() -> None:
    # If the prompt held its own copy, editing the definitions would leave the
    # model reading the stale one while every test above still passed.
    assert contract() in _PROMPT.replace("{{", "{").replace("}}", "}")
