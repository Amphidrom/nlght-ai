# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class VectorResult:
    id: str
    score: float
    content: str
    metadata: dict[str, object] = field(default_factory=dict)


@runtime_checkable
class VectorStore(Protocol):
    async def search(self, query_vector: list[float], limit: int = 10) -> list[VectorResult]: ...
    async def search_symbols(self, query_vector: list[float], limit: int = 10) -> list[VectorResult]: ...
    async def search_multi(self, query_vectors: list[list[float]], limit: int = 10) -> list[VectorResult]: ...
