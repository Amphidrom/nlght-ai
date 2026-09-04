# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Asking for a claim in a particular shape, and being refused in one.

Generic retrieval already delivers a claim whole — measured in
`tests/unit/adapters/outbound/stores/test_knowledge_delivers_whole.py`, where the
serialiser hands over every field a claim carries and drops nothing for being
unfamiliar. So the open question is narrower than "retrieval loses things":

    what happens when a consumer asks for a **shape** that cannot hold this
    claim?

The answer is a value, not a log line. Handing back a claim shortened to fit is
worse than handing back nothing: it looks like a claim, it is not the one that
was made, and nothing downstream can tell. A log line is something nobody sees; a
returned state is something a caller has to handle, and the difference is whether
the guarantee survives contact with the next programmer.

**What this module does not know is deliberate.** It has no idea which shapes
exist, what a proposition is made of, how many arguments one has, or which of
them matter. An earlier version of this file did: it offered `as_triple` and
classified claims by whether they had roles "beyond subject, predicate and
object" — which quietly made the legacy payload's shape the privileged one, in
the middle of a slice whose whole point is that no representation has been
chosen. It is removed rather than kept as a convenience, because a privileged
shape sitting in core is exactly how the decision gets made without anybody
taking it.

A caller that has a shape names it. This says only whether the claim arrived in
it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(slots=True, frozen=True)
class Represented(Generic[T]):
    """The claim, in the shape that was asked for."""

    value: T


@dataclass(slots=True, frozen=True)
class NotRepresentable:
    """The claim cannot be expressed in the shape that was asked for.

    Carries what the claim *does* have, because a caller that cannot render it
    still has to say something useful — "this assertion has parts your view does
    not model" is actionable, and a bare failure is not.

    `shape` is whatever the caller called the thing it wanted. This module does
    not maintain a list of them.
    """

    shape: str
    reason: str
    #: What the claim carries, in the caller's own terms, so it can report or
    #: degrade deliberately instead of guessing.
    available: tuple[str, ...] = ()

    def __str__(self) -> str:
        parts = ", ".join(self.available) or "nothing named"
        return f"not representable as {self.shape}: {self.reason} (has {parts})"

