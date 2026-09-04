# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from dataclasses import dataclass

from nlght.core.ingestion.document import SourceDocument


@dataclass(slots=True, frozen=True)
class AcquisitionDiagnostic:
    code: str
    location: str
    detail: str


@dataclass(slots=True, frozen=True)
class SourceSnapshot:
    source_id: str
    documents: tuple[SourceDocument, ...]
    observed_external_ids: tuple[str, ...]
    complete: bool
    diagnostics: tuple[AcquisitionDiagnostic, ...] = ()

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise ValueError("source_id must not be empty")
        observed = set(self.observed_external_ids)
        if len(observed) != len(self.observed_external_ids):
            raise ValueError("observed_external_ids must be unique")
        document_ids = [document.external_id for document in self.documents]
        if len(set(document_ids)) != len(document_ids):
            raise ValueError("acquired document external ids must be unique")
        if any(document.source != self.source_id for document in self.documents):
            raise ValueError("acquired documents must belong to the snapshot source")
        if any(document.external_id not in observed for document in self.documents):
            raise ValueError("every acquired document must be present in observed_external_ids")
