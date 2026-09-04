# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""What was extracted from a document, and by which extraction.

The knowledge layer had no counterpart to the data layer's index state, so every
run asked the model about every document again. That is not merely wasteful: the
model answers the same question in slightly different words each time, and once
assertions are updated incrementally those different words are a retraction and
a re-review of something nobody edited. Identity drift stops being an edge case
at document change and becomes the normal operating condition.

Two things decide whether extraction may be skipped, and it takes both:

    document_content_hash + extraction_version

The version covers everything that can change what the extraction produces — the
assertion schema, the prompt and the workflow it belongs to, and the model.
Leave one out and a document is skipped after that thing changed: the run reports
success, the corpus keeps assertions written by an extraction that no longer
exists, and the ones the new extraction would have found are never produced.
"""

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
    """Everything that decides what an extraction produces, folded in one place.

    Assembled here and nowhere else. Composing it at the call site is how a new
    influence gets forgotten — the factor is added to the extractor, the version
    keeps its old value, and every document is skipped after a change that
    should have redone them. So the parts this module owns are folded in here,
    and a caller supplies only what it alone knows.

    ``settings`` is whatever of the caller's configuration shapes the result. It
    is serialised canonically, so reordering a config block is not a change and
    adding a key is.

    The caller's parts must not be blank: a blank one would make two genuinely
    different extractions look alike, which is the single failure this key
    exists to prevent, so it is refused rather than hashed.
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
        """Whether this document may be left alone.

        Both halves, and not either: unchanged content under a changed model is
        exactly when the corpus needs redoing, and unchanged extraction over
        changed content is the ordinary reason to run at all.
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
