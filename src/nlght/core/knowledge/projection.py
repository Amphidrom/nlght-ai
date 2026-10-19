# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Define explicit results for a requested claim representation."""

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
    """Report why a claim cannot fit a caller-defined shape.

    Available fields let the caller explain the missing representation without
    truncating the claim.
    """

    shape: str
    reason: str
    #: Preserve available fields so callers can explain the failed projection.
    available: tuple[str, ...] = ()

    def __str__(self) -> str:
        parts = ", ".join(self.available) or "nothing named"
        return f"not representable as {self.shape}: {self.reason} (has {parts})"

