# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Check canonical identity fields on rule and decision payloads.

Rule entity keys use subject and rule_property. Decision entity keys use
subject and decision_type. Their body text remains separate from those fields.
Older payloads without the field pairs still load, though they cannot supply
those keys until the missing identity fields are resolved.
"""

from __future__ import annotations

import pytest

from nlght.core.knowledge import (
    DecisionPayload,
    RulePayload,
    canonical_term,
)


@pytest.mark.parametrize(
    "written",
    [
        "approval_threshold",
        "Approval Threshold",
        "  approval threshold  ",
        "APPROVAL-THRESHOLD",
        "approval   threshold",
    ],
)
def test_the_same_property_written_differently_is_one_value(written: str) -> None:
    """A model asked for a short field still varies its casing and spacing.

    The whole point of a short field is that two runs agree on it. Left raw it
    would carry exactly the drift it was introduced to remove, only shorter.
    """
    assert canonical_term(written) == "approval_threshold"


def test_canonicalisation_keeps_genuinely_different_terms_apart() -> None:
    # It must not flatten so far that two business questions merge.
    assert canonical_term("approval_threshold") != canonical_term("prohibition_threshold")
    assert canonical_term("approval threshold") != canonical_term("approval deadline")


def test_a_blank_term_is_refused() -> None:
    # An empty property would put every rule about one subject into one bucket,
    # which is worse than having no field at all.
    with pytest.raises(ValueError):
        canonical_term("   ")


def test_a_rule_carries_a_subject_and_a_property_beside_its_text() -> None:
    payload = RulePayload(
        rule_text="Expenses above CHF 500 require approval.",
        subject="Expense",
        rule_property="Approval Threshold",
    )

    assert payload.subject == "expense"
    assert payload.rule_property == "approval_threshold"
    # The wording stays exactly as written: it is the body, and a citation has
    # to be able to quote it.
    assert payload.rule_text == "Expenses above CHF 500 require approval."


def test_two_rules_about_one_subject_keep_their_properties_apart() -> None:
    approval = RulePayload(
        rule_text="Expenses above CHF 500 require approval.",
        subject="expense", rule_property="approval_threshold",
    )
    prohibition = RulePayload(
        rule_text="Expenses above CHF 500 are prohibited.",
        subject="expense", rule_property="prohibition_threshold",
    )

    assert approval.subject == prohibition.subject
    assert approval.rule_property != prohibition.rule_property


def test_a_reworded_rule_keeps_the_same_subject_and_property() -> None:
    # The pair names one business question across two phrasings.
    monday = RulePayload(
        rule_text="Access tokens must not be written to application logs.",
        subject="access_token", rule_property="logging_prohibition",
    )
    tuesday = RulePayload(
        rule_text="Access tokens may never be logged by the application.",
        subject="access_token", rule_property="logging_prohibition",
    )

    assert (monday.subject, monday.rule_property) == (tuesday.subject, tuesday.rule_property)


def test_a_decision_carries_a_subject_and_a_type() -> None:
    payload = DecisionPayload(
        decision="Use PostgreSQL for the execution queue.",
        effect="No extra broker has to be operated.",
        subject="Execution Queue",
        decision_type="Technology Choice",
    )

    assert payload.subject == "execution_queue"
    assert payload.decision_type == "technology_choice"


def test_two_decisions_about_one_subject_keep_their_types_apart() -> None:
    technology = DecisionPayload(
        decision="PostgreSQL.", effect="No broker.",
        subject="execution_queue", decision_type="technology_choice",
    )
    retention = DecisionPayload(
        decision="Thirty days.", effect="Smaller tables.",
        subject="execution_queue", decision_type="retention_policy",
    )

    assert technology.decision_type != retention.decision_type


def test_a_rule_written_before_the_fields_still_loads() -> None:
    """The reason the fields are optional rather than required.

    A corpus exists. Requiring the new pair would make every assertion in it
    unreadable, and backfilling one by guessing at the wording is the drift
    this whole design is trying to remove. They are absent, and absent is a
    state the model has to be able to express.
    """
    payload = RulePayload(rule_text="Expenses above CHF 500 require approval.")

    assert payload.subject is None
    assert payload.rule_property is None


def test_a_decision_written_before_the_fields_still_loads() -> None:
    payload = DecisionPayload(decision="Use PostgreSQL.", effect="No extra broker.")

    assert payload.subject is None
    assert payload.decision_type is None


def test_a_rule_may_not_carry_half_the_pair() -> None:
    """Either both or neither, because half of it identifies nothing.

    A subject without a property cannot say which question about that subject
    this is, and a property without a subject cannot say what it is a property
    of. Allowing half would let an entity key be built from something that is
    not one.
    """
    with pytest.raises(ValueError, match="rule_property"):
        RulePayload(rule_text="Expenses require approval.", subject="expense")

    with pytest.raises(ValueError, match="subject"):
        RulePayload(rule_text="Expenses require approval.", rule_property="approval_threshold")


def test_a_decision_may_not_carry_half_the_pair() -> None:
    with pytest.raises(ValueError, match="decision_type"):
        DecisionPayload(decision="PostgreSQL.", effect="No broker.", subject="execution_queue")

    with pytest.raises(ValueError, match="subject"):
        DecisionPayload(
            decision="PostgreSQL.", effect="No broker.", decision_type="technology_choice"
        )


def test_fingerprint_includes_body_even_with_identity_fields() -> None:
    """Change the content fingerprint when a rule body is reworded.

    Rule entity identity uses the canonical field pair. A fingerprint still
    addresses the complete extracted content.
    """
    from nlght.core.knowledge import ExtractedItem

    with_fields = ExtractedItem(
        kind="rule", type="security",
        content={
            "rule_text": "Expenses above CHF 500 require approval.",
            "subject": "expense",
            "rule_property": "approval_threshold",
        },
        confidence=0.9,
    )
    reworded = ExtractedItem(
        kind="rule", type="security",
        content={
            "rule_text": "Expenses over CHF 500 need signing off.",
            "subject": "expense",
            "rule_property": "approval_threshold",
        },
        confidence=0.9,
    )

    assert with_fields.fingerprint != reworded.fingerprint
