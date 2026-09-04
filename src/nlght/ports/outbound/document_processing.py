# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from nlght.core.ingestion import DocumentClassification, Enrichment


class DocumentClassifier(Protocol):
    def classify(self, *, path: Path, content: bytes) -> tuple[DocumentClassification, str | None, tuple[str, ...]]: ...


class DocumentEnricher(Protocol):
    def enrich(
        self,
        *,
        path: Path,
        text: str,
        classification: DocumentClassification,
    ) -> Enrichment: ...
