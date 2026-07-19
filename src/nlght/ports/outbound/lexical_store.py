# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class LexicalResult:
    id: str
    score: float
    content: str
    metadata: dict[str, object] = field(default_factory=dict)


@runtime_checkable
class LexicalStore(Protocol):
    async def search(self, query: str, limit: int = 10) -> list[LexicalResult]: ...
    async def search_symbols(self, query: str, limit: int = 10) -> list[LexicalResult]: ...
    async def search_multi(self, queries: list[str], limit: int = 10) -> list[LexicalResult]: ...
