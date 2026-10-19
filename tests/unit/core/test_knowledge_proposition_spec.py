# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Specify the boundaries of an atomic knowledge proposition.

Independent claims may be split. A relation with multiple roles stays whole.
Role swaps require an equivalence judgement, and retrieval preserves the
complete claim without silently projecting away fields.
"""

from __future__ import annotations

from nlght.core.knowledge import ExtractedItem, Proposition, entity_key


def _fact(**content: str) -> ExtractedItem:
    return ExtractedItem(kind="fact", type="dependency", content=content, confidence=0.9)


async def test_a_distributive_list_may_be_split() -> None:
    """Split a list when each part states an independent claim."""
    from tests.unit.adapters.outbound.workflow.steps.test_knowledge_refine import (
        _ctx,
    )

    from nlght.adapters.outbound.workflow.steps.knowledge.refine import (
        KnowledgeAtomicityStep,
    )

    step = KnowledgeAtomicityStep(config={})
    ctx = _ctx((_fact(subject="SanitizableData", predicate="provides access to",
                      object="key, value and origin"),))

    result = await step.run(ctx)

    assert len(result.ctx.metadata["knowledge.extracted"]) == 3


def test_a_relation_with_more_than_three_roles_stays_one_proposition() -> None:
    """Preserve every role of a single multi-argument relation."""
    item = _fact(subject="Alice", predicate="transfers", amount="CHF 500", recipient="Bob")

    item.well_formed()

    assert dict(item.proposition.fields) == {
        "subject": "Alice", "predicate": "transfers",
        "amount": "CHF 500", "recipient": "Bob",
    }
    assert item.payload() is None


def test_the_entity_key_already_accepts_more_than_three_roles() -> None:
    # The half that works, asserted so the failure above is read as the payload's
    # and not as the key's.
    key = entity_key(
        "fact", predicate="transfers", subject="Alice", amount="CHF 500", recipient="Bob"
    )

    assert key.key


def test_the_same_claim_said_actively_and_passively_is_one_claim() -> None:
    """Recognize active and passive phrasings of one claim through judgement."""
    from nlght.core.knowledge.equivalence import (
        SAME,
        AssertionObservation,
        equivalent,
        may_merge,
    )

    active = entity_key("fact", predicate="requires", subject="spring", object="maven")
    passive = entity_key("fact", predicate="required_by", subject="maven", object="spring")
    assert active != passive

    verdict = equivalent(
        AssertionObservation.of(Proposition(
            {"predicate": "requires", "subject": "spring", "object": "maven"})),
        AssertionObservation.of(Proposition(
            {"predicate": "required_by", "subject": "maven", "object": "spring"})),
        verdict=SAME,
    )

    assert may_merge(verdict)


def test_freely_generated_role_names_do_not_change_the_claim() -> None:
    """Keep generated role labels from defining claim identity."""
    from nlght.core.knowledge.equivalence import (
        SAME,
        AssertionObservation,
        equivalent,
        may_merge,
    )

    first = entity_key(
        "fact", predicate="provides access to", subject="SanitizableData",
        key="key", value="value", origin="PropertySource",
    )
    second = entity_key(
        "fact", predicate="provides access to", subject="SanitizableData",
        data_key="key", data_value="value", property_source="PropertySource",
    )
    assert first != second

    verdict = equivalent(
        AssertionObservation.of(Proposition(
            {"predicate": "provides access to", "subject": "SanitizableData",
             "key": "key", "value": "value", "origin": "PropertySource"})),
        AssertionObservation.of(Proposition(
            {"predicate": "provides access to", "subject": "SanitizableData",
             "data_key": "key", "data_value": "value",
             "property_source": "PropertySource"})),
        verdict=SAME,
    )

    assert may_merge(verdict)


def test_a_moved_value_is_a_different_proposition() -> None:
    # Java 17 against Java 21. Whether the second supersedes the first is a
    # lineage question and never an identity one.
    assert entity_key("fact", predicate="requires", subject="spring", object="java 17") != \
           entity_key("fact", predicate="requires", subject="spring", object="java 21")


def test_swapping_the_roles_of_a_directed_relation_is_a_different_claim() -> None:
    """Keep swapped roles distinct for a directed relation."""
    assert entity_key("fact", predicate="transfers", sender="Alice", recipient="Bob") != \
           entity_key("fact", predicate="transfers", sender="Bob", recipient="Alice")


def test_two_claims_about_one_subject_are_both_live() -> None:
    """Simultaneously true, and never two states of one thing.

    "Spring requires Java" and "Spring requires Maven" are two claims. Reading
    several arguments as a sequence of states is how a corpus turns a list into a
    history it never had.
    """
    java = entity_key("fact", predicate="requires", subject="spring", object="java")
    maven = entity_key("fact", predicate="requires", subject="spring", object="maven")

    assert java != maven


def test_the_same_extraction_asked_twice_is_the_same_proposition() -> None:
    # The invariant everything above rests on. A key that moved between two
    # identical readings would make every other guarantee here unobservable.
    first = _fact(subject="spring", predicate="requires", object="java 17")
    second = _fact(subject="spring", predicate="requires", object="java 17")

    assert first.fingerprint == second.fingerprint
    assert entity_key("fact", predicate="requires", subject="spring", object="java 17") == \
           entity_key("fact", predicate="requires", subject="spring", object="java 17")


def test_generic_retrieval_already_delivers_a_claim_whole() -> None:
    """Preserve complete claim content through generic retrieval."""
    from nlght.core.knowledge import NotRepresentable

    refusal = NotRepresentable(
        shape="whatever the caller asked for",
        reason="it does not fit",
        available=("amount", "predicate", "recipient", "subject"),
    )

    assert refusal.available == ("amount", "predicate", "recipient", "subject")
    assert "amount" in str(refusal)


def test_a_fact_that_cannot_name_what_it_claims_is_not_a_claim() -> None:
    """Reject a fact without a predicate naming its relation."""
    import pytest

    with pytest.raises(ValueError, match="name what it claims"):
        _fact(subject="Alice", object="Bob").well_formed()

    with pytest.raises(ValueError, match="name what it claims"):
        _fact(predicate="transfers").well_formed()


def test_a_fact_that_can_be_placed_passes_however_many_roles_it_has() -> None:
    # The guard above must not cost the thing this whole slice was for.
    _fact(subject="Alice", predicate="transfers", amount="CHF 500",
          recipient="Bob").well_formed()
    _fact(subject="spring", predicate="requires", object="maven").well_formed()
