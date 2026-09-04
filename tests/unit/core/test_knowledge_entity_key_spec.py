# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Executable specification for the entity key — what an assertion *is*.

The decision this file encodes: **identity is a schema decision, not a
similarity decision.** An `assertion_id` does not identify the current truth
value of a claim; it identifies the business question whose value may change
over time. So the entity key is built from a small set of schema-defined fields
and looked up deterministically, and the similarity matcher is left with entity
*linking* — which existing entity does this newly extracted assertion probably
correspond to — and never with deciding what the database considers identity.

The dividing rule, per kind:

    into the entity key go exactly those dimensions whose change means we are
    talking about a different business quantity — not those whose change merely
    describes a new state of the same quantity.

| kind | entity key | value (revision payload) |
|---|---|---|
| `fact` | subject + predicate | object |
| `rule` | subject + rule_property | condition, consequence, parameters, scope |
| `decision` | decision_subject + decision_type | outcome, effect, scope |
| `pattern` | pattern_name | description, attributes |

Two consequences worth stating plainly, because both change today's behaviour:

*`fact` narrows.* Identity is subject and predicate; the object is the value.
`CEO(OpenAI) = Alice` and `CEO(OpenAI) = Bob` are one entity with two revisions,
where today they are two unrelated assertions. That is the point — a corpus that
holds both as equals is the failure the whole design exists to remove.

*An assertion may have no key at all.* One written before the identifying fields
existed cannot say which business quantity it is, and `EntityKeyUnavailable`
says so rather than inventing one from the wording.

Scope is deliberately *not* in the key. It answers "for which case", which is a
third level between the entity and its revisions:

    entity   which business quantity is this?
    variant  for which simultaneously valid case?
    revision how has that case changed over time?

Two values that are true at the same time — CHF 500 for employees, CHF 5000 for
executives — cannot be revisions of one thing, because revisions are a sequence.
They are two variants of one entity. Specified in the last section of this
file.
"""

from __future__ import annotations

import pytest

from nlght.core.knowledge import ENTITY_KEY_VERSION, EntityKey, EntityKeyUnavailable, VariantMatch, resolve_variant
from nlght.core.knowledge import entity_key as _entity_key
from nlght.core.knowledge.entity import AMBIGUOUS, EXACT, NEW, WIDENED


def _variant_id(entity: object, scope: object, known: dict[str, object]) -> str:
    """The id alone, for the cases that only care which variant it is."""
    return resolve_variant(entity, scope, known).variant_id

# ---------------------------------------------------------------------------
# fact — the whole proposition, with succession recorded rather than inferred
# ---------------------------------------------------------------------------
#
# This section said `subject + predicate` first, which collapsed Spring's Java
# and Maven requirements into one entity; then a per-predicate role projection,
# which needed the platform to know which relations are functional. Both tried
# to turn arbitrary language into a state-variable model, and that needs an
# ontology nobody has.
#
# A fact is its whole proposition. Two claims that say different things are two
# assertions, and `superseded_by` records that one took the other's place —
# which is §7.4 as it was originally written: a rewording keeps an identity, a
# material change makes a new assertion.


def test_a_fact_is_its_whole_proposition() -> None:
    """Two claims that say different things are two assertions.

    An earlier version tried to split a fact's roles so that OpenAI's chief
    executive would be one thing whose value moved from Alice to Bob. Deciding
    which roles identify a relation and which are its state is domain knowledge
    no platform has without an ontology, and neither a hardcoded vocabulary nor
    asking a customer to declare one before ingesting is an answer.

    So the platform does not decide it. Alice and Bob are two assertions, and
    that the second replaced the first is recorded between them rather than
    inferred from the words.
    """
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
    # The predicate names the relation; without it the roles stand in nothing.
    with pytest.raises(EntityKeyUnavailable, match="predicate"):
        _entity_key("fact", subject="openai", object="alice")


def test_a_predicate_with_nothing_standing_in_it_cannot_be_identified() -> None:
    with pytest.raises(EntityKeyUnavailable, match="names nothing"):
        _entity_key("fact", predicate="ceo")


def test_the_order_roles_arrive_in_does_not_change_a_fact_key() -> None:
    first = _entity_key("fact", predicate="requires", subject="spring", dependency="java")
    second = _entity_key("fact", dependency="java", predicate="requires", subject="spring")

    assert first == second


# ---------------------------------------------------------------------------
# rule — subject + rule_property identify; thresholds and scope are the value
# ---------------------------------------------------------------------------

def test_a_changed_threshold_is_the_same_rule() -> None:
    """CHF 500 → CHF 550. The worked example the decision was made on.

    The rule is *the approval threshold for expenses*. Five hundred is what it
    currently says, not what it is. Deriving identity from the value would
    retire the rule every time finance adjusts a number, taking the approval
    with it.
    """
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
    """Scope answers "for which case", which is the variant's question.

        employees need approval above CHF 500
        employees and contractors need approval above CHF 500

    The quantity did not change, its applicability did. Scope belongs to the
    variant below the entity — see the variant section further down — and never
    to the entity key, or one business rule would fragment into one entity per
    group it mentions.
    """
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


# ---------------------------------------------------------------------------
# decision and pattern
# ---------------------------------------------------------------------------

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
    # `pattern` already carries the split; only the key has to use it.
    first = _entity_key(
        "pattern", pattern_name="hexagonal-ports",
        description="Adapters depend on ports; the core depends on nothing.",
    )
    second = _entity_key(
        "pattern", pattern_name="hexagonal-ports",
        description="The core defines ports and adapters implement them.",
    )

    assert first == second


# ---------------------------------------------------------------------------
# The key is schema-defined, not model-phrased
# ---------------------------------------------------------------------------

def test_the_key_is_built_from_fields_and_not_from_a_generated_phrase() -> None:
    """Why the field is not called `semantic_subject`.

    A name like that invites a model-generated string — "expense approval
    threshold" — back into the identity, which is the defect wearing a better
    label. The key is assembled from schema fields, so two extractions agree
    because the schema made them agree, not because the model happened to phrase
    the summary the same way.
    """
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


# ---------------------------------------------------------------------------
# An assertion that cannot say what it is
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Variants — one entity, several cases holding at the same time
# ---------------------------------------------------------------------------
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
    """The case that was open, and the reason the variant level exists.

        expense approval threshold, employees:  CHF 500
        expense approval threshold, executives: CHF 5000

    Both hold. One entity, because it is one business quantity; two variants,
    because neither replaced the other; and each carries its own revisions.
    """
    entity = _entity_key("rule", subject="expense", rule_property="approval_threshold")
    employees = _variant_id(entity, ["employee"], known={})
    executives = _variant_id(entity, ["executive"], known={employees: ["employee"]})

    assert employees != executives


def test_a_changed_amount_stays_within_one_variant() -> None:
    # CHF 500 → 550 for employees. The variant is the same case; only its state
    # moved, which is what a revision is.
    entity = _entity_key("rule", subject="expense", rule_property="approval_threshold")
    first = _variant_id(entity, ["employee"], known={})
    second = _variant_id(entity, ["employee"], known={first: ["employee"]})

    assert first == second


def test_a_widened_scope_continues_the_variant_it_replaced() -> None:
    """employees → employees and contractors, with the narrow one gone.

    The old applicability was replaced rather than joined by a second rule, so
    this is Variant A at a new revision: the same case, now covering more people.
    Minting a new variant would retract the employees rule and put a fresh one
    into review for what is an edit to one word.
    """
    entity = _entity_key("rule", subject="expense", rule_property="approval_threshold")
    original = _variant_id(entity, ["employee"], known={})
    widened = _variant_id(entity, ["employee", "contractor"], known={original: ["employee"]})

    assert widened == original


def test_a_disjoint_scope_opens_a_variant_and_leaves_the_other_alone() -> None:
    # No overlap, so nothing was replaced — executives are an additional case,
    # and the employees variant keeps its own number.
    entity = _entity_key("rule", subject="expense", rule_property="approval_threshold")
    employees = _variant_id(entity, ["employee"], known={})
    executives = _variant_id(entity, ["executive"], known={employees: ["employee"]})

    assert executives != employees


def test_a_variant_id_is_a_surrogate_and_not_derived_from_its_scope() -> None:
    """Why the variant cannot be a hash of the scope, which follows from above.

    A derived key changes when the scope changes, so widening employees to
    employees and contractors would produce a different key and retire the
    variant it was meant to continue — contradicting the test above. Scope is
    therefore a matching signal for the variant, exactly as content is a
    matching signal for the assertion, and the id itself is minted once.

    Stated as a test rather than a comment because it is the property that a
    later refactor to "just hash the scope" would quietly break, while every
    other test here would keep passing.
    """
    entity = _entity_key("rule", subject="expense", rule_property="approval_threshold")
    original = _variant_id(entity, ["employee"], known={})
    widened = _variant_id(entity, ["employee", "contractor"], known={original: ["employee"]})
    from_scratch = _variant_id(entity, ["employee", "contractor"], known={})

    # Same scope, but one continues an existing variant and one is new. A
    # derivation cannot tell those apart; only a surrogate plus a match can.
    assert widened == original
    assert from_scratch != original




# ---------------------------------------------------------------------------
# The ladder, and where it refuses to decide
# ---------------------------------------------------------------------------
#
#     exact match
#     → a single unambiguous widening or narrowing
#     → otherwise a new variant, marked ambiguous
#
# Not "the largest overlap wins". Overlapping scopes that hold at the same time
# are ordinary, and picking the biggest intersection between them is a heuristic
# deciding identity — the thing the surrogate id exists to prevent.


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
    """The tie the old rule would have decided by counting.

    Employees and contractors both exist as their own cases, and a rule now
    covering both could be a widening of either. Choosing the bigger overlap
    would attach it to one and silently retire the other's continuity; choosing
    neither leaves a case a person can merge.
    """
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
    # Nothing shared, nothing ambiguous — this is a second case, and saying so
    # is different from saying "I could not tell".
    match = resolve_variant("e", ["executive"], {"var-1": ["employee"]})

    assert match.resolution == NEW


def test_two_variants_already_claiming_one_scope_claim_nothing() -> None:
    # Something upstream went wrong. Picking one of them would make it
    # permanent; opening a third leaves the mess visible.
    known = {"var-1": ["employee"], "var-2": ["employee"]}

    match = resolve_variant("e", ["employee"], known)

    assert match.variant_id not in known
    assert match.resolution == AMBIGUOUS


# ---------------------------------------------------------------------------
# The key carries the scheme that produced it
# ---------------------------------------------------------------------------

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
