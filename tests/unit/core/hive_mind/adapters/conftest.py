# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""The contract every adapter owes, whatever domain type it starts from.

An adapter is the last place domain knowledge is allowed to be. What it produces
must be complete enough for the reduction and the renderer to work without ever
asking what the thing was — so these checks are run against each adapter's output
rather than written out five times.
"""

from __future__ import annotations

from nlght.core.hive_mind.models import Level, MentalElement, estimate_tokens


def assert_element_contract(element: MentalElement) -> None:
    """What must hold for any element, from any adapter."""
    # It can be said at all, and it can be said at exactly one level per name.
    assert element.representations
    assert len(element.levels) == len({item.level for item in element.representations})

    # Costs are the shared estimate and nothing else. An adapter pricing its own
    # text differently would make the budget measure one thing and reserve
    # another.
    for representation in element.representations:
        if representation.level is Level.OMIT:
            assert representation.cost == 0
            assert representation.text == ""
        else:
            assert representation.cost == estimate_tokens(representation.text), (
                f"{element.element_id} prices {representation.level} itself"
            )

    # Cheaper as it gets shorter — the condition the monotonicity guarantee
    # rests on, checked here too because an adapter is where it can be broken.
    offered = [element.at(level) for level in element.levels]
    for fuller, shorter in zip(offered, offered[1:], strict=False):
        assert shorter is not None and fuller is not None
        assert shorter.cost <= fuller.cost

    # A full representation is what the source said; only a shortened one is
    # derived (ADR-0053).
    full = element.at(Level.FULL)
    if full is not None:
        assert full.derived is False

    # Somewhere to render it. An element with no section would be invisible.
    assert element.presentation.section


def assert_no_invented_short_form(element: MentalElement) -> None:
    """No adapter may fabricate a compact wording.

    Truncating ends a claim mid-sentence and extracting one is a judgement about
    which sentence carries it. Where no honest short form exists, the element
    offers none — and the reduction copes, because levels are optional.
    """
    assert Level.COMPACT not in element.levels, (
        f"{element.element_id} offers a compact form; was it written or invented?"
    )
