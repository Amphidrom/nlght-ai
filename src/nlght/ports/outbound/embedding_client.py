# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class EmbeddingResult:
    """Vectors for one batch of texts, with the provenance to persist alongside.

    ``provider``, ``model``, and ``dimension`` are stored with every written
    vector so a later change of embedding model is detectable and a mixed index
    can be reconciled rather than silently returning incomparable distances.
    """

    vectors: tuple[tuple[float, ...], ...]
    provider: str
    model: str
    dimension: int

    def __post_init__(self) -> None:
        if not self.provider.strip() or not self.model.strip():
            raise ValueError("embedding provider and model must not be empty")
        if self.dimension <= 0:
            raise ValueError("embedding dimension must be positive")
        if any(len(vector) != self.dimension for vector in self.vectors):
            raise ValueError("every embedding vector must have the declared dimension")


@runtime_checkable
class EmbeddingClient(Protocol):
    """Outbound port for turning text into vectors.

    Deliberately separate from ``ModelClient``: that contract is chat-shaped
    (``call``/``stream``/``append_tool_turn``), while the validated embedding
    implementation is a local model with no conversation at all. Keeping them
    apart lets an embedding provider be chosen, configured, and versioned
    independently of the chat provider.

    Implementations must preserve input order and return exactly one vector per
    input text.
    """

    async def embed(self, texts: Sequence[str]) -> EmbeddingResult: ...
