# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Apply shared mental-element invariants to each domain adapter.

Adapters carry domain meaning into a generic form before reduction and
rendering.
"""

from __future__ import annotations

from nlght.core.hive_mind.models import Level, MentalElement, estimate_tokens


def assert_element_contract(element: MentalElement) -> None:
    """What must hold for any element, from any adapter.

    Args:
        element (MentalElement): Mental element whose adapter contract is checked.
    """
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

    # Shorter forms must not cost more. An adapter can otherwise break the
    # reduction's monotonicity guarantee.
    offered = [element.at(level) for level in element.levels]
    for fuller, shorter in zip(offered, offered[1:], strict=False):
        assert shorter is not None and fuller is not None
        assert shorter.cost <= fuller.cost

    # A full representation is what the source said. Only a shortened one is
    # derived (ADR-0053).
    full = element.at(Level.FULL)
    if full is not None:
        assert full.derived is False

    # Somewhere to render it. An element with no section would be invisible.
    assert element.presentation.section


def assert_no_invented_short_form(element: MentalElement) -> None:
    """Reject an adapter representation that invents abbreviated source text.

    Args:
        element (MentalElement): Adapted element to inspect.
        """
    assert Level.COMPACT not in element.levels, (
        f"{element.element_id} offers a compact form; was it written or invented?"
    )
