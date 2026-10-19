# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Hold complete extracted claims without imposing field semantics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from nlght.core.ingestion.document import stable_digest

FINGERPRINT_VERSION = "1"
"""Keep the fingerprint scheme separate to distinguish version changes from claim changes."""


PropositionValue = (
    str | int | float | bool | None | Mapping[str, Any] | Sequence[Any]
)
"""Preserve extracted scalar, nested, and ordered values without imposing field semantics."""


def _freeze(value: Any) -> Any:  # noqa: ANN401 (it holds whatever was extracted)
    """Copy extracted values into stable nested structures.

    Args:
        value (Any): Scalar, mapping, list, or tuple value.

    Returns:
        Any: Copied mappings and tuples that retain sequence order and values.
    """
    if isinstance(value, Mapping):
        return {str(key): _freeze(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:  # noqa: ANN401
    """Convert stored proposition values to JSON-compatible structures.

    Args:
        value (Any): Nested proposition value.

    Returns:
        Any: Mappings and lists containing JSON-compatible child values.
    """
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(slots=True, frozen=True)
class Proposition:
    """Hold the extracted field values of any knowledge kind.

    Equality ignores field-name order and preserves sequence-element order.
    No field name has a privileged semantic role.
    """

    fields: Mapping[str, PropositionValue]

    def __post_init__(self) -> None:
        if not self.fields:
            # An empty structure cannot state a claim.
            raise ValueError("a proposition needs at least one field")
        object.__setattr__(self, "fields", _freeze(self.fields))

    @property
    def fingerprint(self) -> str:
        """Name the exact structure without treating it as claim identity.

        Returns:
            str: Stable digest of sorted field names and values.
        """
        return stable_digest(
            *(f"{key}={_thaw(value)!r}" for key, value in sorted(self.fields.items()))
        )

    def as_dict(self) -> dict[str, Any]:
        return {str(key): _thaw(value) for key, value in self.fields.items()}

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> Proposition:
        return cls(fields=dict(values))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Proposition):
            return NotImplemented
        return dict(self.fields) == dict(other.fields)

    def __hash__(self) -> int:
        return hash(tuple(sorted((key, str(value)) for key, value in self.fields.items())))
