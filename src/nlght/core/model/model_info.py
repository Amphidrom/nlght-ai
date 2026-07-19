# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class ModelInfo:
    """Normalized model descriptor returned by any ModelProviderBackend."""

    name: str
    size: int | None = None
    modified_at: datetime | None = None
    digest: str | None = None


@dataclass(frozen=True)
class RunningModelInfo:
    """Descriptor for a model that is currently loaded into VRAM.

    Only providers that implement ``RunningModelsCapable`` can return these.
    Ollama is currently the only such provider.
    """

    name: str
    size_vram: int
    expires_at: datetime | None = None
    digest: str | None = None
    size: int | None = None
