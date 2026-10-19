# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Check extraction-version and document-state comparisons.

An unchanged document stays current only when its content hash and complete
extraction version match. Changes to the schema, prompt, segmentation, model,
or workflow input must change that version.
"""

from __future__ import annotations

import pytest

from nlght.core.knowledge import ExtractionState, extraction_version


def _version(**overrides: object) -> str:
    parts: dict[str, object] = {
        "workflow": "ingest-knowledge",
        "model": "ollama/llama3",
        "prompt": "Extract assertions from: {text}",
        "settings": {"min_confidence": 0.3},
    }
    parts.update(overrides)
    return extraction_version(**parts)  # type: ignore[arg-type]


def test_the_same_inputs_give_the_same_version() -> None:
    # Otherwise nothing is ever skipped and the whole mechanism is decoration.
    assert _version() == _version()


@pytest.mark.parametrize(
    "changed",
    [
        {"workflow": "ingest-knowledge-v2"},
        {"model": "ollama/llama3.1"},
        {"prompt": "Extract assertions, atomically, from: {text}"},
        {"settings": {"min_confidence": 0.8}},
    ],
    ids=["workflow", "model", "prompt", "settings"],
)
def test_every_supplied_part_changes_the_version(changed: dict[str, object]) -> None:
    """Change the extraction version when any supplied influence changes."""
    assert _version(**changed) != _version()


@pytest.mark.parametrize(
    "owned",
    ["SCHEMA_VERSION", "SEGMENTATION_VERSION"],
)
def test_the_parts_the_core_owns_change_it_without_a_caller_saying_so(
    owned: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reason the version is assembled here rather than at the call site.

    The assertion schema and the segmentation decide what an extraction
    produces just as much as the model does, and no caller passes them. Folding
    them in centrally is what stops a new influence being added to the extractor
    while the version quietly keeps its old value.
    """
    import nlght.core.knowledge.extraction as extraction_module

    before = _version()
    monkeypatch.setattr(extraction_module, owned, "99")

    assert _version() != before


def test_reordering_the_settings_is_not_a_change() -> None:
    # A config block written in another order is the same configuration, and
    # re-extracting a corpus for it would be an expensive way to say nothing.
    first = _version(settings={"min_confidence": 0.3, "kinds": ["fact", "rule"]})
    second = _version(settings={"kinds": ["fact", "rule"], "min_confidence": 0.3})

    assert first == second


def test_adding_a_setting_is_a_change() -> None:
    first = _version(settings={"min_confidence": 0.3})
    second = _version(settings={"min_confidence": 0.3, "max_assertions": 5})

    assert first != second


@pytest.mark.parametrize("blank", ["workflow", "model", "prompt"])
def test_an_incomplete_version_is_refused_rather_than_silently_weak(blank: str) -> None:
    # A blank part would make two genuinely different extractions look alike,
    # which is exactly the failure this key exists to prevent.
    with pytest.raises(ValueError, match=blank):
        _version(**{blank: "  "})


def _state(content_hash: str = "c-1", version: str | None = None) -> ExtractionState:
    return ExtractionState(
        document_id="doc-1",
        content_hash=content_hash,
        extraction_version=version or _version(),
    )


def test_an_unchanged_document_under_an_unchanged_extraction_is_current() -> None:
    assert _state().is_current(content_hash="c-1", extraction_version=_version())


def test_changed_content_is_not_current() -> None:
    assert not _state().is_current(content_hash="c-2", extraction_version=_version())


def test_a_changed_extraction_version_is_not_current() -> None:
    """Mark unchanged content stale when its extraction version changes."""
    assert not _state().is_current(
        content_hash="c-1", extraction_version=_version(model="ollama/llama3.1")
    )


def test_both_changing_is_not_current() -> None:
    assert not _state().is_current(
        content_hash="c-2", extraction_version=_version(model="ollama/llama3.1")
    )


def test_a_document_never_extracted_has_no_state_to_be_current() -> None:
    # The absent case is the caller's: no row means nothing was skipped, which
    # is the safe direction. Stated so the contract is not read as "missing
    # means current".
    assert _state().document_id == "doc-1"


@pytest.mark.parametrize(
    ("field", "value"),
    [("document_id", " "), ("content_hash", ""), ("extraction_version", "   ")],
)
def test_a_state_missing_part_of_its_key_is_refused(field: str, value: str) -> None:
    fields = {
        "document_id": "doc-1",
        "content_hash": "c-1",
        "extraction_version": "v-1",
        field: value,
    }

    with pytest.raises(ValueError):
        ExtractionState(**fields)
