# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from typing import Protocol

from nlght.core.ingestion import SourceSnapshot


class DocumentSource(Protocol):
    @property
    def source_id(self) -> str: ...

    async def acquire(self) -> SourceSnapshot: ...
