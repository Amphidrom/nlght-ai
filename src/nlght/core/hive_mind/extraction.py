# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Core vocabulary for memory extraction.

The classes in this module describe what an extractor found in a user request.
They intentionally do not depend on a concrete NLP implementation. spaCy/Stanza
NER, LLM extraction, or a hybrid NLP pipeline can all produce the same
structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class MemoryCandidateConfidence(StrEnum):
    VALIDATED = "validated_candidate"
    UNCERTAIN = "uncertain_candidate"
    TRANSIENT = "transient_candidate"


class MemoryCandidateKind(StrEnum):
    FACT = "fact"
    ARTIFACT = "artifact"
    RELATION = "relation"
    REQUEST = "request"
    DIRECTIVE = "directive"


@dataclass(frozen=True)
class ExtractedEntity:
    text: str
    kind: str = "entity"
    canonical: str | None = None
    confidence: float = 1.0

    @property
    def key(self) -> str:
        return self.canonical or self.text


@dataclass(frozen=True)
class ExtractedArtifact:
    label: str
    content: str
    content_type: str = "text"
    confidence: float = 1.0
    language: str = "text"


@dataclass(frozen=True)
class ExtractedRelation:
    subject: str
    predicate: str
    object: str
    confidence: float = 1.0


@dataclass(frozen=True)
class MemoryCandidate:
    key: str
    kind: MemoryCandidateKind
    content: str
    entities: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    confidence: MemoryCandidateConfidence = MemoryCandidateConfidence.TRANSIENT


@dataclass(frozen=True)
class ExtractionResult:
    intent: str
    entities: list[ExtractedEntity] = field(default_factory=list)
    artifacts: list[ExtractedArtifact] = field(default_factory=list)
    relations: list[ExtractedRelation] = field(default_factory=list)
    candidates: list[MemoryCandidate] = field(default_factory=list)
    referenced_labels: list[str] = field(default_factory=list)
