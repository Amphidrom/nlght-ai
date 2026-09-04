# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""What generic retrieval already does with a claim, measured rather than assumed.

Written because the opposite was assumed once. "Retrieval loses n-ary
propositions" was recorded as a gap and used to justify an abstraction, and the
serialiser had never lost anything: it hands over whatever the claim carries.

So the open question is narrower than it looked, and this file is what keeps it
narrow:

    generic retrieval          delivers the claim whole — measured below
    an explicit projection     into a form that cannot hold the claim is where a
                               refusal is needed

Which forms exist at all is not decided, and nothing here decides it.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from nlght.adapters.outbound.stores.knowledge import _serialize
from nlght.core.knowledge import KnowledgeObject


def _object(**payload: str) -> KnowledgeObject:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    return KnowledgeObject(
        identity="k-1",
        kind="fact",
        type="dependency",
        confidence=0.9,
        payload=dict(payload),
        review_required=False,
        review_reason=None,
        product=None,
        version=None,
        metadata={},
        created_at=now,
        updated_at=now,
    )


def test_every_field_of_a_claim_reaches_the_caller() -> None:
    """Three roles, and all three arrive."""
    [item] = json.loads(_serialize([_object(subject="spring", predicate="requires", object="java")]))

    assert item["subject"] == "spring"
    assert item["predicate"] == "requires"
    assert item["object"] == "java"


def test_a_claim_with_more_roles_than_three_also_arrives_whole() -> None:
    """The measurement that made an assumed gap go away.

    The serialiser does not know how many roles a claim has and does not care —
    it hands over what the claim carries. A proposition the payload cannot yet
    *store* would nonetheless be delivered complete if it could be, so retrieval
    is not where that loss would happen.
    """
    [item] = json.loads(
        _serialize([
            _object(subject="Alice", predicate="transfers", amount="CHF 500", recipient="Bob")
        ])
    )

    assert item["amount"] == "CHF 500"
    assert item["recipient"] == "Bob"
    assert {"subject", "predicate", "amount", "recipient"} <= set(item)


def test_nothing_is_dropped_for_being_unfamiliar() -> None:
    # No allow-list, no known-roles filter. A field the serialiser has never seen
    # is not a field it may quietly remove.
    [item] = json.loads(_serialize([_object(**{"a": "1", "zeta": "2", "role_47": "3"})]))

    assert item["a"] == "1"
    assert item["zeta"] == "2"
    assert item["role_47"] == "3"
