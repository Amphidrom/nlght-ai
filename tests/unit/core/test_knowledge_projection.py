# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Being told a shape does not fit, rather than being handed a shortened claim.

Generic retrieval already delivers a claim whole — measured separately, in
`tests/unit/adapters/outbound/stores/test_knowledge_delivers_whole.py`. What was
missing is narrower: a way for a consumer that asked for a particular *shape* to
be told, as a value it must handle, that this claim does not fit in it.

**The shape is the caller's, never this module's.** An earlier version defined
`as_triple` and sorted claims by whether they had parts "beyond subject,
predicate and object" — which made the legacy payload the privileged form inside
core, during a slice whose entire point is that no representation has been
chosen. So the tests below invent their own shape and do their own deciding; core
only carries the answer.
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
    """One caller's projection, with one caller's idea of what fits."""
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
    """The guard on the correction that produced this file.

    Nothing in core may name a privileged form while the representation question
    is open — a shape sitting here is how that decision gets made without anybody
    taking it.
    """
    import nlght.core.knowledge.projection as projection  # noqa: PLC0415

    exported = {name for name in dir(projection) if not name.startswith("_")}

    assert not exported & {"Triple", "TRIPLE", "as_triple", "Relation", "Event"}
