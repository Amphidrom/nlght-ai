# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""What a candidate's content hash is for, now that it is not identity.

`ExtractedItem.fingerprint` was called `identity` and was doing two jobs. One it
can hold: addressing the graph node, so the same claim found on two pages
reinforces one node instead of duplicating it. One it cannot: deciding which
assertion a sighting continues. A claim whose wording moved hashed differently,
so the corpus opened a second assertion and left the approval on the first.

That second job now belongs to the resolver, and the promises that used to be
asserted here against the hash — a reworded rule stays one rule, a rephrased
decision or pattern keeps its assertion, two runs survive the model rewording
one of them — moved to `tests/integration/test_knowledge_identity_core.py`,
where they are properties of stored state rather than of a hash. They could
never have held here: a hash of the whole content block moves when any of it
moves, which is what a hash is for.

What remains is the contract the hash really has. It is stable, order- and
case-independent, and it separates kinds and types that share words — so two
sightings of one claim address one node, and two different claims never do.
"""

from __future__ import annotations

from nlght.core.knowledge.pipeline import ExtractedItem


def _item(kind: str, type_: str, **content: object) -> ExtractedItem:
    return ExtractedItem(kind=kind, type=type_, content=dict(content), confidence=0.9)


def _fact(subject: str = "spring-boot", obj: str = "3.2") -> ExtractedItem:
    return _item("fact", "dependency", subject=subject, predicate="version", object=obj)


# ---------------------------------------------------------------------------
# What already holds. The fingerprint keeps these properties whatever else changes.
# ---------------------------------------------------------------------------

def test_the_same_claim_extracted_twice_is_one_assertion() -> None:
    # The whole point of a content-derived identity, and the precondition for
    # a second run over an unchanged document changing nothing.
    assert _fact().fingerprint == _fact().fingerprint


def test_a_different_claim_is_a_different_assertion() -> None:
    assert _fact(obj="3.2").fingerprint != _fact(obj="3.3").fingerprint


def test_the_same_claim_from_two_documents_reinforces_one_assertion() -> None:
    # Identity is deliberately independent of where the claim was found, so two
    # pages asserting the same thing support one node instead of two.
    from_a = _fact()
    from_b = _fact()
    from_a.evidence["document_id"] = "doc-a"
    from_b.evidence["document_id"] = "doc-b"

    assert from_a.fingerprint == from_b.fingerprint


def test_field_order_does_not_change_identity() -> None:
    ordered = _item("fact", "dependency", subject="s", predicate="p", object="o")
    shuffled = _item("fact", "dependency", object="o", subject="s", predicate="p")

    assert ordered.fingerprint == shuffled.fingerprint


def test_capitalisation_does_not_change_identity() -> None:
    assert _fact(subject="Spring-Boot").fingerprint == _fact(subject="spring-boot").fingerprint


def test_the_same_words_under_a_different_type_are_a_different_assertion() -> None:
    # "3.2" as a dependency version and as a release date would otherwise merge.
    assert _item("fact", "dependency", subject="s", predicate="p", object="o").fingerprint != \
           _item("fact", "release", subject="s", predicate="p", object="o").fingerprint


def test_the_same_words_under_a_different_kind_are_a_different_assertion() -> None:
    assert _item("rule", "security", subject="tokens").fingerprint != \
           _item("decision", "security", subject="tokens").fingerprint


# ---------------------------------------------------------------------------
# What must hold and does not. Each of these is a run-to-run drift today,
# and under incremental update a drift is a retraction plus a lost approval.
# ---------------------------------------------------------------------------

def test_two_rules_about_different_subjects_stay_two_rules() -> None:
    # The other direction of the same specification, and it already holds —
    # today because the subject is hashed along with everything else, later
    # because it is one of the identifying fields. It has to keep holding: a
    # specification that only merged would be satisfied by hashing nothing.
    tokens = _item("rule", "security", subject="access-token-logging", rule_text="Same words.")
    passwords = _item("rule", "security", subject="password-storage", rule_text="Same words.")

    assert tokens.fingerprint != passwords.fingerprint


# ---------------------------------------------------------------------------
# The acceptance metric of the design, in the half that identity decides.
# ---------------------------------------------------------------------------

def _extracted(rule_wording: str, fact_object: str) -> set[str]:
    """One document's assertions, as two runs of the extractor might phrase them."""
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


def test_two_runs_over_an_unchanged_document_produce_the_same_assertions() -> None:
    # The design's acceptance metric: a second run over an unchanged source must
    # change nothing. With a deterministic extractor this already holds — it is
    # the baseline the next test departs from.
    first = _extracted("Access tokens must not be logged.", "3.2")
    second = _extracted("Access tokens must not be logged.", "3.2")

    assert first == second


def test_a_real_change_is_still_visible_as_a_change() -> None:
    # The guard on all of the above: none of this may make the corpus blind to
    # an edit that matters. A different version is a different fact, whatever
    # the rules do.
    unchanged = _extracted("Access tokens must not be logged.", "3.2")
    upgraded = _extracted("Access tokens must not be logged.", "3.3")

    assert unchanged != upgraded
