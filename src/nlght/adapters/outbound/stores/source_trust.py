# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Which ingestion sources a store's operator trusts as content (ADR-0074).

The store resource carries the decision as ``trusted_sources``: a list of
``source_id``s, the IDs stamped on ingested documents. The ingestion
configuration identifies each source. This setting marks trusted sources.
Unlisted sources and hits without a source are untrusted.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from nlght.core.model.trust import (
    PLATFORM_TEXT,
    UNANNOTATED,
    ContentOrigin,
    ContentTrust,
    SourceRef,
)
from nlght.core.tools.tool import ToolOption

TRUSTED_SOURCES_OPTION = ToolOption(
    "trusted_sources",
    "array",
    "Ingestion source ids whose documents the operator trusts as content. "
    "Unlisted sources are untrusted. Trust never makes text an instruction.",
    default=[],
    placeholder='["confluence-handbook"]',
)


def trusted_sources(config: Mapping[str, Any], *, where: str) -> frozenset[str]:
    """Validate and normalize a store's trusted source IDs.

    Args:
        config (Mapping[str, Any]): Store resource configuration.
        where (str): Location named in validation errors.

    Returns:
        frozenset[str]: Trimmed trusted source IDs, or an empty set when unset.

    Raises:
        ValueError: The value is not a list of nonempty strings.
    """
    value = config.get("trusted_sources", [])
    if value is None:
        return frozenset()
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise ValueError(
            f"{where}: 'trusted_sources' must be a list of non-empty source ids"
        )
    return frozenset(item.strip() for item in value)


class SourceLabeller:
    """Labels content of one store activation by the source it came from."""

    def __init__(self, *, kind: str, resource_id: str, trusted: frozenset[str]) -> None:
        self._kind = kind
        self._resource_id = resource_id
        self._trusted = trusted

    def ref(self, source_id: str) -> SourceRef:
        trust = (
            ContentTrust.TRUSTED
            if source_id and source_id in self._trusted
            else ContentTrust.UNTRUSTED
        )
        return SourceRef(self._kind, self._resource_id, source_id, trust, source_id)

    def origin(self, source_ids: Iterable[str]) -> ContentOrigin:
        return ContentOrigin.of(*(self.ref(source_id) for source_id in source_ids))

    def result_origin(self, per_item: Iterable[Iterable[str]]) -> ContentOrigin:
        """Combine source origins for all items in a result.

        Args:
            per_item (Iterable[Iterable[str]]): Source IDs supporting each item.

        Returns:
            ContentOrigin: Combined origin. An empty result is platform text.
            An item without a source is unannotated.
        """
        origins = [
            self.origin(sources) if (sources := tuple(item)) else UNANNOTATED
            for item in per_item
        ]
        return ContentOrigin.combine(*origins) if origins else PLATFORM_TEXT
