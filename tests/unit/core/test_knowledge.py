# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Knowledge domain invariants and the confidence merge."""

from __future__ import annotations

import pytest

from nlght.core.knowledge import (
    BUILTIN_KINDS,
    DecisionPayload,
    FactPayload,
    KnowledgeWrite,
    PatternPayload,
    RulePayload,
    kind_of,
    reinforce_confidence,
)


def test_each_payload_maps_to_its_builtin_kind() -> None:
    kinds = [
        kind_of(FactPayload(subject="a", predicate="b", object="c")),
        kind_of(RulePayload(rule_text="r")),
        kind_of(PatternPayload(pattern_name="p", description="d")),
        kind_of(DecisionPayload(decision="d", effect="e")),
    ]
    assert kinds == list(BUILTIN_KINDS)


@pytest.mark.parametrize(
    "payload",
    [
        lambda: FactPayload(subject=" ", predicate="b", object="c"),
        lambda: RulePayload(rule_text="  "),
        lambda: PatternPayload(pattern_name="p", description=""),
        lambda: DecisionPayload(decision="d", effect=" "),
    ],
)
def test_empty_payload_fields_are_rejected(payload) -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        payload()


def _write(**overrides) -> KnowledgeWrite:
    base = {
        "identity": "id-1",
        "payload": FactPayload(subject="a", predicate="b", object="c"),
        "type": "dependency",
        "confidence": 0.5,
        "source_id": "s",
        "run_id": "r",
    }
    return KnowledgeWrite(**{**base, **overrides})


def test_confidence_outside_the_unit_interval_is_rejected() -> None:
    with pytest.raises(ValueError, match="within"):
        _write(confidence=1.2)
    with pytest.raises(ValueError, match="within"):
        _write(confidence=-0.1)


def test_quarantine_requires_a_reason() -> None:
    # An assertion held back from retrieval must say why, or the review queue
    # is unusable.
    with pytest.raises(ValueError, match="review reason"):
        _write(review_required=True)
    assert _write(review_required=True, review_reason="conflict").review_required


def test_independent_sightings_reinforce_each_other() -> None:
    assert reinforce_confidence(0.6, 0.6) > 0.6
    assert reinforce_confidence(0.9, 0.9) > 0.9


def test_a_weak_sighting_cannot_reach_certainty_alone() -> None:
    assert reinforce_confidence(0.5, 0.5) == pytest.approx(0.5)
    assert reinforce_confidence(0.5, 0.2) < 0.5
    assert 0.0 < reinforce_confidence(1.0, 1.0) <= 1.0


def test_the_merge_is_order_independent() -> None:
    assert reinforce_confidence(0.3, 0.8) == pytest.approx(reinforce_confidence(0.8, 0.3))


def test_a_unit_must_know_which_document_it_came_from() -> None:
    """The guard that makes the parsers agree.

    Provenance used to be a convention: every parser was expected to put
    `document_id` and `processing_revision_id` into the metadata dict, and one of two put
    in half of it. Every candidate then reached the graph and none reached the
    lineage, because an assertion with no document revision cannot answer what
    document D at revision R asserted.

    It is a field now, and a unit without it does not get built — so a third
    parser cannot repeat the mistake.
    """
    from nlght.core.knowledge import KnowledgeUnit

    with pytest.raises(ValueError, match="which document"):
        KnowledgeUnit(unit_ordinal="s:d:0", source_id="s", content="text")


def test_a_unit_hands_its_provenance_over_in_one_piece() -> None:
    # Read as a set, so a caller cannot take one part and forget another. The
    # path joined them once one parser carried it in `metadata` and the other
    # carried nothing, which left half a corpus unnameable.
    from nlght.core.knowledge import KnowledgeUnit

    unit = KnowledgeUnit(
        unit_ordinal="s:d:0", source_id="s", content="text",
        document_id="doc-1", processing_revision_id="rev-1", document_path="docs/a.md",
    )

    assert unit.provenance == {
        "document_id": "doc-1",
        "processing_revision_id": "rev-1",
        "document_path": "docs/a.md",
    }
