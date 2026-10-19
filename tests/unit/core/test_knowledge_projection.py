# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Report when a complete proposition does not fit a requested shape.

The caller supplies the shape. Projection must not shorten a claim to make it
fit that shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nlght.core.knowledge import NotRepresentable, Represented


@dataclass(frozen=True, slots=True)
class _Pair:
    """A shape a caller might want. Defined here, on purpose."""

    left: str
    right: str


def _as_pair(content: dict[str, Any]) -> Represented[_Pair] | NotRepresentable:
    """One caller's projection, with one caller's idea of what fits.

    Args:
        content (dict[str, Any]): Content supplied to the projection or step.

    Returns:
        Represented[_Pair] | NotRepresentable: Projected pair or the reason it cannot be represented.
    """
    if set(content) == {"left", "right"}:
        return Represented(_Pair(left=content["left"], right=content["right"]))
    return NotRepresentable(
        shape="pair",
        reason="the claim does not have exactly a left and a right",
        available=tuple(sorted(content)),
    )


def test_a_claim_that_fits_comes_back_in_the_shape() -> None:
    result = _as_pair({"left": "a", "right": "b"})

    assert result == Represented(_Pair(left="a", right="b"))


def test_a_claim_that_does_not_fit_is_refused_rather_than_trimmed() -> None:
    """The whole point: no third option.

    Trimming would produce something well-formed that asserts what the source did
    not. The caller is told instead, and has to do something about it.
    """
    result = _as_pair({"left": "a", "right": "b", "middle": "c"})

    assert isinstance(result, NotRepresentable)
    assert result.shape == "pair"


def test_a_refusal_says_what_the_claim_does_have() -> None:
    # A caller that cannot render it still has to report something useful.
    result = _as_pair({"left": "a", "right": "b", "middle": "c"})

    assert isinstance(result, NotRepresentable)
    assert result.available == ("left", "middle", "right")


def test_a_refusal_reads_as_a_sentence() -> None:
    # It ends up in a tool result a model reads and in a log a person reads.
    printed = str(_as_pair({"only": "a"}))

    assert "not representable as pair" in printed
    assert "only" in printed


def test_core_knows_no_shapes_of_its_own() -> None:
    """Keep caller-specific projection shapes out of the knowledge core."""
    import nlght.core.knowledge.projection as projection

    exported = {name for name in dir(projection) if not name.startswith("_")}

    assert not exported & {"Triple", "TRIPLE", "as_triple", "Relation", "Event"}
