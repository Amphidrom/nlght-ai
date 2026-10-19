# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Specify schema-based entity identity for knowledge assertions.

Entity keys identify business questions independently of value revisions.
Scope belongs to a variant, while similarity may link an observation to an
existing entity without defining its database identity.
"""

from __future__ import annotations

import pytest

from nlght.core.knowledge import ENTITY_KEY_VERSION, EntityKey, EntityKeyUnavailable, VariantMatch, resolve_variant
from nlght.core.knowledge import entity_key as _entity_key
from nlght.core.knowledge.entity import AMBIGUOUS, EXACT, NEW, WIDENED


def _variant_id(entity: object, scope: object, known: dict[str, object]) -> str:
    """Resolve a variant and return only its identifier.

    Args:
        entity (object): Entity owning the variant.
        scope (object): Applicability scope to resolve.
        known (dict[str, object]): Existing variant scopes keyed by ID.

    Returns:
        str: Resolved variant identifier.
    """
    return resolve_variant(entity, scope, known).variant_id

#
# The platform has no ontology for deciding which roles represent a fact's
# identity and which represent its state. The entire proposition forms the key.
# Material succession is recorded explicitly through `superseded_by`.


def test_a_fact_is_its_whole_proposition() -> None:
    """Give facts with different role values different entity keys."""
    alice = _entity_key("fact", subject="openai", predicate="ceo", object="alice")
    bob = _entity_key("fact", subject="openai", predicate="ceo", object="bob")

    assert alice != bob


def test_facts_holding_at_once_are_separate_assertions() -> None:
    # Spring requires Java and Spring requires Maven are both true, and neither
    # is a state of the other.
    java = _entity_key("fact", subject="spring", predicate="requires", object="java 17")
    maven = _entity_key("fact", subject="spring", predicate="requires", object="maven")

    assert java != maven


def test_the_same_proposition_written_twice_is_one_assertion() -> None:
    # The property everything rests on, and the reason the key is canonical:
    # casing and spacing must not make two claims out of one.
    first = _entity_key("fact", subject="OpenAI", predicate="CEO", object="Alice")
    second = _entity_key("fact", subject="openai", predicate="ceo", object="alice")

    assert first == second


def test_a_fact_about_a_different_predicate_is_a_different_entity() -> None:
    ceo = _entity_key("fact", subject="openai", predicate="ceo", object="alice")
    founded = _entity_key("fact", subject="openai", predicate="founded_in", object="2015")

    assert ceo != founded


def test_a_fact_about_a_different_subject_is_a_different_entity() -> None:
    openai = _entity_key("fact", subject="openai", predicate="ceo", object="alice")
    anthropic = _entity_key("fact", subject="anthropic", predicate="ceo", object="alice")

    assert openai != anthropic


def test_a_fact_without_a_predicate_cannot_be_identified() -> None:
    # The predicate names the relation. Without it the roles stand in nothing.
    with pytest.raises(EntityKeyUnavailable, match="predicate"):
        _entity_key("fact", subject="openai", object="alice")


def test_a_predicate_with_nothing_standing_in_it_cannot_be_identified() -> None:
    with pytest.raises(EntityKeyUnavailable, match="names nothing"):
        _entity_key("fact", predicate="ceo")


def test_the_order_roles_arrive_in_does_not_change_a_fact_key() -> None:
    first = _entity_key("fact", predicate="requires", subject="spring", dependency="java")
    second = _entity_key("fact", dependency="java", predicate="requires", subject="spring")

    assert first == second


def test_a_changed_threshold_is_the_same_rule() -> None:
    """Keep a rule identity when its approval threshold changes."""
    five_hundred = _entity_key(
        "rule", subject="expense", rule_property="approval_threshold",
        operator=">", amount=500, currency="CHF",
    )
    five_fifty = _entity_key(
        "rule", subject="expense", rule_property="approval_threshold",
        operator=">", amount=550, currency="CHF",
    )

    assert five_hundred == five_fifty


def test_a_different_rule_property_about_one_subject_is_a_different_entity() -> None:
    """The boundary the value dimension must not blur.

        expenses over CHF 500 require approval
        expenses over CHF 500 are prohibited

    Same subject, same number, and a different business question:
    `approval_threshold` against `prohibition_threshold`. One entity would mean
    a prohibition silently superseding an approval rule as its next revision.
    """
    approval = _entity_key(
        "rule", subject="expense", rule_property="approval_threshold", amount=500
    )
    prohibition = _entity_key(
        "rule", subject="expense", rule_property="prohibition_threshold", amount=500
    )

    assert approval != prohibition


def test_scope_never_reaches_the_entity_key() -> None:
    """Keep applicability scope out of the entity key."""
    narrow = _entity_key(
        "rule", subject="expense", rule_property="approval_threshold",
        applies_to=["employee"],
    )
    wide = _entity_key(
        "rule", subject="expense", rule_property="approval_threshold",
        applies_to=["employee", "contractor"],
    )

    assert narrow == wide


def test_the_wording_never_reaches_the_entity_key() -> None:
    # The defect that started all of this: a rule identified by a sentence the
    # model phrased freely.
    german = _entity_key(
        "rule", subject="expense", rule_property="approval_threshold",
        rule_text="Spesen über CHF 500 benötigen eine Genehmigung.",
    )
    rephrased = _entity_key(
        "rule", subject="expense", rule_property="approval_threshold",
        rule_text="Ausgaben von mehr als CHF 500 müssen genehmigt werden.",
    )

    assert german == rephrased


def test_a_decision_is_identified_by_its_subject_and_type() -> None:
    first = _entity_key(
        "decision", subject="execution-queue", decision_type="technology_choice",
        outcome="PostgreSQL", effect="No extra broker to operate.",
    )
    reversed_later = _entity_key(
        "decision", subject="execution-queue", decision_type="technology_choice",
        outcome="Redis", effect="A broker has to be operated.",
    )

    # Reversing a decision is a new state of the same decision, and the history
    # of what was chosen before is exactly what an ADR is for.
    assert first == reversed_later


def test_two_decisions_of_different_type_about_one_subject_stay_apart() -> None:
    technology = _entity_key(
        "decision", subject="execution-queue", decision_type="technology_choice",
    )
    retention = _entity_key(
        "decision", subject="execution-queue", decision_type="retention_policy",
    )

    assert technology != retention


def test_a_pattern_is_identified_by_its_name_alone() -> None:
    # `pattern` already carries the split. Only the key has to use it.
    first = _entity_key(
        "pattern", pattern_name="hexagonal-ports",
        description="Adapters depend on ports; the core depends on nothing.",
    )
    second = _entity_key(
        "pattern", pattern_name="hexagonal-ports",
        description="The core defines ports and adapters implement them.",
    )

    assert first == second


def test_the_key_is_built_from_fields_and_not_from_a_generated_phrase() -> None:
    """Build entity keys from schema fields instead of model wording."""
    assembled = _entity_key(
        "rule", subject="expense", rule_property="approval_threshold"
    )
    phrased = _entity_key(
        "rule", subject="expense", rule_property="approval_threshold",
        summary="expense approval threshold",
    )
    also_phrased = _entity_key(
        "rule", subject="expense", rule_property="approval_threshold",
        summary="the limit above which expenses need signing off",
    )

    assert assembled == phrased == also_phrased


def test_a_kind_change_is_never_the_same_entity() -> None:
    rule = _entity_key("rule", subject="expense", rule_property="approval_threshold")
    fact = _entity_key("fact", subject="expense", predicate="approval_threshold")

    assert rule != fact


def test_a_rule_written_before_the_fields_has_no_entity_key() -> None:
    """The corpus that already exists, and what may not be done to it.

    A rule stored as a sentence cannot say which business quantity it is, and
    there is no way to derive one from the wording that is not the guessing this
    whole design removes. So it has no key and says so, rather than being given
    an invented one that a later run would then fail to match.
    """
    with pytest.raises(EntityKeyUnavailable, match="rule_property"):
        _entity_key("rule", rule_text="Expenses above CHF 500 require approval.")


def test_half_the_pair_is_not_a_key_either() -> None:
    with pytest.raises(EntityKeyUnavailable, match="rule_property"):
        _entity_key("rule", subject="expense")


def test_an_unknown_kind_has_no_key() -> None:
    # A kind nobody has defined identifying fields for cannot be identified.
    # Falling back to hashing everything would be the old defect returning
    # through a side door.
    with pytest.raises(EntityKeyUnavailable, match="unknown"):
        _entity_key("prophecy", subject="tomorrow")


#
# Three levels, not two:
#
#     entity   which business quantity is this?
#     variant  for which simultaneously valid case?
#     revision how has that case changed over time?
#
#     Entity: expense + approval_threshold
#       Variant A: employees    rev1: CHF 500   rev2: CHF 550
#       Variant B: executives   rev1: CHF 5000
#
# Revisions are a sequence of states, so two values that are true at the same
# time cannot be revisions of one thing. Without the variant level they would
# have to be, and each run would overwrite the other's number.


def test_two_scopes_holding_at_once_are_two_variants_of_one_entity() -> None:
    """Keep concurrent scopes as variants of one entity."""
    entity = _entity_key("rule", subject="expense", rule_property="approval_threshold")
    employees = _variant_id(entity, ["employee"], known={})
    executives = _variant_id(entity, ["executive"], known={employees: ["employee"]})

    assert employees != executives


def test_a_changed_amount_stays_within_one_variant() -> None:
    # The threshold changes from CHF 500 to CHF 550, but the employee case
    # remains the same variant. The amount belongs to its revision.
    entity = _entity_key("rule", subject="expense", rule_property="approval_threshold")
    first = _variant_id(entity, ["employee"], known={})
    second = _variant_id(entity, ["employee"], known={first: ["employee"]})

    assert first == second


def test_a_widened_scope_continues_the_variant_it_replaced() -> None:
    """Continue a variant when its scope widens and replaces the old scope."""
    entity = _entity_key("rule", subject="expense", rule_property="approval_threshold")
    original = _variant_id(entity, ["employee"], known={})
    widened = _variant_id(entity, ["employee", "contractor"], known={original: ["employee"]})

    assert widened == original


def test_a_disjoint_scope_opens_a_variant_and_leaves_the_other_alone() -> None:
    # No overlap means nothing was replaced. Executives are an additional case,
    # and the employees variant keeps its own number.
    entity = _entity_key("rule", subject="expense", rule_property="approval_threshold")
    employees = _variant_id(entity, ["employee"], known={})
    executives = _variant_id(entity, ["executive"], known={employees: ["employee"]})

    assert executives != employees


def test_a_variant_id_is_a_surrogate_and_not_derived_from_its_scope() -> None:
    """Keep variant identity stable when scope wording changes."""
    entity = _entity_key("rule", subject="expense", rule_property="approval_threshold")
    original = _variant_id(entity, ["employee"], known={})
    widened = _variant_id(entity, ["employee", "contractor"], known={original: ["employee"]})
    from_scratch = _variant_id(entity, ["employee", "contractor"], known={})

    # Same scope, but one continues an existing variant and one is new. A
    # derivation cannot tell those apart. Only a surrogate plus a match can.
    assert widened == original
    assert from_scratch != original


#
#     exact match
#     then a single unambiguous widening or narrowing
#     otherwise a new variant, marked ambiguous
#
# Not "the largest overlap wins". Overlapping scopes that hold at the same time
# are ordinary, and picking the biggest intersection between them is a heuristic
# deciding identity. The surrogate ID prevents that.


def test_an_exact_scope_is_resolved_as_exact() -> None:
    match = resolve_variant("e", ["employee"], {"var-1": ["employee"]})

    assert match == VariantMatch("var-1", EXACT)


def test_a_single_widening_is_a_continuation() -> None:
    match = resolve_variant("e", ["employee", "contractor"], {"var-1": ["employee"]})

    assert match.variant_id == "var-1"
    assert match.resolution == WIDENED
    assert match.is_continuation


def test_a_single_narrowing_is_a_continuation() -> None:
    # The other direction: a rule that covered everyone now covers employees.
    # Still one case, now describing fewer people.
    match = resolve_variant("e", ["employee"], {"var-1": ["employee", "contractor"]})

    assert match.variant_id == "var-1"
    assert match.resolution == "narrowed"


def test_two_possible_widenings_claim_nothing() -> None:
    """Leave ambiguous scope continuations unclaimed."""
    known = {"var-1": ["employee"], "var-2": ["contractor"]}

    match = resolve_variant("e", ["employee", "contractor"], known)

    assert match.variant_id not in known
    assert match.resolution == AMBIGUOUS
    assert not match.is_continuation


def test_an_overlap_that_is_not_containment_claims_nothing() -> None:
    # Partly the same people, partly not: neither a continuation nor plainly
    # separate. Asserting lineage here would merge two cases that were never
    # one, and merging is the direction nobody sees.
    known = {"var-1": ["employee", "manager"]}

    match = resolve_variant("e", ["employee", "contractor"], known)

    assert match.variant_id != "var-1"
    assert match.resolution == AMBIGUOUS


def test_a_disjoint_scope_is_plainly_new() -> None:
    # Nothing shared means this is a second case. Saying so
    # is different from saying "I could not tell".
    match = resolve_variant("e", ["executive"], {"var-1": ["employee"]})

    assert match.resolution == NEW


def test_two_variants_already_claiming_one_scope_claim_nothing() -> None:
    # Something upstream went wrong. Picking one of them would make it
    # permanent. Opening a third leaves the mess visible.
    known = {"var-1": ["employee"], "var-2": ["employee"]}

    match = resolve_variant("e", ["employee"], known)

    assert match.variant_id not in known
    assert match.resolution == AMBIGUOUS


def test_a_key_says_which_scheme_produced_it() -> None:
    """Version beside the key, not folded into it.

    Folded in, a key of the old scheme would simply differ from a new one, and
    nothing could tell that from a key naming a different quantity. Kept apart,
    a migration to a richer proposition can read the old keys, decide what each
    means, and rewrite them deliberately.
    """
    key = _entity_key("rule", subject="expense", rule_property="approval_threshold")

    assert key.version == ENTITY_KEY_VERSION
    assert key.key


def test_a_key_of_another_scheme_is_not_equal_to_this_one() -> None:
    key = _entity_key("rule", subject="expense", rule_property="approval_threshold")

    assert EntityKey(version="2", key=key.key) != key
