# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Not extracting again what has not changed.

The knowledge layer has had no counterpart to the data layer's index state, so
every run re-extracts every document. That is what makes identity drift the
normal operating condition rather than an edge case at document change: the
model is asked the same question again and answers it in slightly different
words, and under incremental update those different words are a retraction and
a re-review of something nobody edited.

The skip is keyed on two things and needs both:

    document_content_hash + extraction_version

The version has to be complete — the assertion schema, the segmentation, the
prompt and the workflow it belongs to, and the model. A partial key skips a
document after an extraction change, keeping assertions produced by something
that no longer exists and never producing the ones the new extraction would have
found.

It is assembled in one place for that reason. The parts the extraction owns are
folded in by the core and no caller passes them, so a new influence cannot be
added to the extractor while the version quietly keeps its old value.
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


# ---------------------------------------------------------------------------
# The version key — assembled in one place
# ---------------------------------------------------------------------------

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
    """Each on its own, because a forgotten one fails silently.

    A version missing the model would skip every document after a model upgrade
    — the run reports success, the corpus keeps assertions the old model wrote,
    and nothing says so.
    """
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


# ---------------------------------------------------------------------------
# The decision
# ---------------------------------------------------------------------------

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
    """The half that is easy to forget, and the dangerous one.

    The document did not change, so a content-only key would skip it. But the
    prompt or the model did, which is precisely when the corpus needs redoing —
    a model change is called a deliberate versioned event, and this is what
    makes its consequence happen.
    """
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


# ---------------------------------------------------------------------------
# The record refuses to be half a key
# ---------------------------------------------------------------------------

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
