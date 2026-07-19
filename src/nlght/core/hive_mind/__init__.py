# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Core Hive-Mind vocabulary: data models, stores, directives, and logic.

No framework dependencies.  No persistence logic.
"""

from nlght.core.hive_mind.builder import ContextSnapshot, MentalModelBuilder
from nlght.core.hive_mind.directives import KnownDirective
from nlght.core.hive_mind.extraction import (
    ExtractedArtifact,
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
    MemoryCandidate,
    MemoryCandidateConfidence,
    MemoryCandidateKind,
)
from nlght.core.hive_mind.models import (
    AtomType,
    Directive,
    DirectivePriority,
    PromotionStatus,
    SessionResult,
    TurnSummary,
    WorkingAtom,
    WriteIntent,
)
from nlght.core.hive_mind.relevance import RelevanceEngine
from nlght.core.hive_mind.stores import (
    ConversationStore,
    DirectiveStore,
    SessionResultStore,
    WorkingMemory,
)
from nlght.core.hive_mind.system_prompt import PromptInputPolicy, SystemPromptBuilder

__all__ = [
    # Builder
    "ContextSnapshot",
    "MentalModelBuilder",
    # Directives
    "KnownDirective",
    # Extraction
    "ExtractedArtifact",
    "ExtractedEntity",
    "ExtractedRelation",
    "ExtractionResult",
    "MemoryCandidate",
    "MemoryCandidateConfidence",
    "MemoryCandidateKind",
    # Models
    "AtomType",
    "Directive",
    "DirectivePriority",
    "PromotionStatus",
    "SessionResult",
    "TurnSummary",
    "WorkingAtom",
    "WriteIntent",
    # Relevance
    "RelevanceEngine",
    # Stores
    "ConversationStore",
    "DirectiveStore",
    "SessionResultStore",
    "WorkingMemory",
    # System prompt
    "PromptInputPolicy",
    "SystemPromptBuilder",
]
