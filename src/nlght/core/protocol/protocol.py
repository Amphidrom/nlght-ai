# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ProtocolKind(StrEnum):
    OPENAI_CHAT_COMPLETIONS = "openai.chat_completions"
    OPENAI_MODELS = "openai.models"
    OLLAMA_CHAT = "ollama.chat"
    OLLAMA_MODELS = "ollama.models"
    GENERIC_JSON = "generic.json"
    UNKNOWN = "unknown"


@dataclass(slots=True, frozen=True)
class DetectedProtocol:
    kind: ProtocolKind
    confidence: float
    reason: str
    version: str | None = None