# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Track which extraction produced each document claim."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from nlght.core.ingestion.document import stable_digest
from nlght.core.knowledge.knowledge import BUILTIN_KINDS
from nlght.core.knowledge.pipeline import SEGMENTATION_VERSION

SCHEMA_VERSION = "1"
"""The shape of an extracted assertion.

Bumped when a kind gains, loses or redefines a field, so a schema change
re-extracts the corpus instead of leaving assertions behind in a shape nothing
produces any more.
"""


def extraction_version(
    *,
    workflow: str,
    model: str,
    prompt: str,
    settings: Mapping[str, Any],
) -> str:
    """Bind an extraction to every factor that can change its result.

    Args:
        workflow (str): Workflow identity.
        model (str): Model identity.
        prompt (str): Extraction prompt text.
        settings (Mapping[str, Any]): Caller-owned extraction settings.

    Returns:
        str: Stable digest of the complete extraction contract.

    Raises:
        ValueError: A required workflow, model, or prompt value is blank.
    """
    supplied = {"workflow": workflow, "model": model, "prompt": prompt}
    missing = [name for name, value in supplied.items() if not value.strip()]
    if missing:
        raise ValueError(
            f"extraction version is incomplete: {', '.join(sorted(missing))} must not be empty"
        )
    return stable_digest(
        # Owned here, so they cannot be left out by a caller.
        SCHEMA_VERSION,
        SEGMENTATION_VERSION,
        ",".join(sorted(BUILTIN_KINDS)),
        # Supplied, because only the step knows them.
        *(supplied[name].strip() for name in sorted(supplied)),
        json.dumps(dict(settings), sort_keys=True, separators=(",", ":"), default=str),
    )


@dataclass(slots=True, frozen=True)
class ExtractionState:
    """That one document was extracted, under one extraction, at one content.

    The absence of a record is not "current": a document nobody has extracted
    has no state, and the caller extracts it. Only a record that matches on both
    halves licenses a skip.
    """

    document_id: str
    content_hash: str
    extraction_version: str
    extracted_at: datetime | None = None

    def __post_init__(self) -> None:
        missing = [
            name
            for name in ("document_id", "content_hash", "extraction_version")
            if not str(getattr(self, name)).strip()
        ]
        if missing:
            raise ValueError(
                f"extraction state is missing {', '.join(missing)}"
            )

    def is_current(self, *, content_hash: str, extraction_version: str) -> bool:
        """Permit a skip only when content and extraction both match.

        Args:
            content_hash (str): Hash of the current source content.
            extraction_version (str): Current extraction contract digest.

        Returns:
            bool: True only when the stored content and extraction contract both match.
        """
        return (
            self.content_hash == content_hash
            and self.extraction_version == extraction_version
        )


@dataclass(slots=True, frozen=True)
class ExtractionStateWrite:
    """Records that a document has been extracted under a given extraction."""

    document_id: str
    content_hash: str
    extraction_version: str
    run_id: str
    assertion_count: int = 0

    def __post_init__(self) -> None:
        missing = [
            name
            for name in ("document_id", "content_hash", "extraction_version", "run_id")
            if not str(getattr(self, name)).strip()
        ]
        if missing:
            raise ValueError(
                f"extraction state write is missing {', '.join(missing)}"
            )
        if self.assertion_count < 0:
            raise ValueError("assertion_count must not be negative")
