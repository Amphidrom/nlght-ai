# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Check that extraction fingerprints address exact normalized content.

The fingerprint is stable across field order, casing, and source location.
It separates different kinds, types, and content values. Assertion identity
is resolved separately from stored lineage and propositions; a changed
fingerprint alone does not determine whether a sighting continues a claim.
"""

from __future__ import annotations

from nlght.core.knowledge.pipeline import ExtractedItem


def _item(kind: str, type_: str, **content: object) -> ExtractedItem:
    return ExtractedItem(kind=kind, type=type_, content=dict(content), confidence=0.9)


def _fact(subject: str = "spring-boot", obj: str = "3.2") -> ExtractedItem:
    return _item("fact", "dependency", subject=subject, predicate="version", object=obj)


def test_the_same_claim_extracted_twice_has_one_fingerprint() -> None:
    # Stable content gives retries the same address for the extracted item.
    assert _fact().fingerprint == _fact().fingerprint


def test_different_content_has_a_different_fingerprint() -> None:
    assert _fact(obj="3.2").fingerprint != _fact(obj="3.3").fingerprint


def test_source_document_does_not_change_the_fingerprint() -> None:
    # Source location must not change the fingerprint of the extracted content.
    from_a = _fact()
    from_b = _fact()
    from_a.evidence["document_id"] = "doc-a"
    from_b.evidence["document_id"] = "doc-b"

    assert from_a.fingerprint == from_b.fingerprint


def test_field_order_does_not_change_the_fingerprint() -> None:
    ordered = _item("fact", "dependency", subject="s", predicate="p", object="o")
    shuffled = _item("fact", "dependency", object="o", subject="s", predicate="p")

    assert ordered.fingerprint == shuffled.fingerprint


def test_capitalisation_does_not_change_the_fingerprint() -> None:
    assert _fact(subject="Spring-Boot").fingerprint == _fact(subject="spring-boot").fingerprint


def test_type_contributes_to_the_fingerprint() -> None:
    # "3.2" as a dependency version and as a release date would otherwise merge.
    assert _item("fact", "dependency", subject="s", predicate="p", object="o").fingerprint != \
           _item("fact", "release", subject="s", predicate="p", object="o").fingerprint


def test_kind_contributes_to_the_fingerprint() -> None:
    assert _item("rule", "security", subject="tokens").fingerprint != \
           _item("decision", "security", subject="tokens").fingerprint


def test_two_rules_about_different_subjects_stay_two_rules() -> None:
    # A changed subject changes the content address even when wording matches.
    tokens = _item("rule", "security", subject="access-token-logging", rule_text="Same words.")
    passwords = _item("rule", "security", subject="password-storage", rule_text="Same words.")

    assert tokens.fingerprint != passwords.fingerprint


def _extracted(rule_wording: str, fact_object: str) -> set[str]:
    """Collect fingerprints for three extracted items.

    Args:
        rule_wording (str): Wording of the extracted rule.
        fact_object (str): Object of the extracted fact.

    Returns:
        set[str]: Content fingerprints for the rule, fact, and pattern.
    """
    return {
        _item(
            "rule", "security",
            subject="access-token-logging",
            rule_text=rule_wording,
        ).fingerprint,
        _item("fact", "dependency", subject="spring-boot", predicate="version",
              object=fact_object).fingerprint,
        _item(
            "pattern", "architecture",
            pattern_name="hexagonal-ports",
            description="Adapters depend on ports.",
        ).fingerprint,
    }


def test_two_runs_over_unchanged_content_produce_the_same_fingerprints() -> None:
    # Identical extraction output keeps the same content addresses across runs.
    first = _extracted("Access tokens must not be logged.", "3.2")
    second = _extracted("Access tokens must not be logged.", "3.2")

    assert first == second


def test_a_real_change_is_still_visible_as_a_change() -> None:
    # A changed fact value must have a different content address.
    unchanged = _extracted("Access tokens must not be logged.", "3.2")
    upgraded = _extracted("Access tokens must not be logged.", "3.3")

    assert unchanged != upgraded
