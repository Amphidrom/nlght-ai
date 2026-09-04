# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Core Hive-Mind vocabulary: data models, stores, and logic.

No framework dependencies.  No persistence logic.
"""

from nlght.core.hive_mind.builder import ContextSnapshot, MentalModelBuilder
from nlght.core.hive_mind.extraction import (
    ExtractedArtifact,
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
    MemoryCandidate,
    MemoryCandidateConfidence,
    MemoryCandidateKind,
)
from nlght.core.hive_mind.model_context import ModelContextBuilder, PromptInputPolicy
from nlght.core.hive_mind.models import (
    AtomType,
    Level,
    MentalElement,
    PromotionStatus,
    Representation,
    Retention,
    SessionResult,
    TurnSummary,
    WorkingAtom,
    WriteIntent,
)
from nlght.core.hive_mind.relevance import (
    ReducedView,
    RelevanceEngine,
    reduce_to_budget,
)
from nlght.core.hive_mind.stores import (
    ConversationStore,
    SessionResultStore,
    WorkingMemory,
)
from nlght.core.hive_mind.system_prompt import SystemPromptBuilder

__all__ = [
    # Builder
    "ContextSnapshot",
    "MentalModelBuilder",
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
    "Level",
    "MentalElement",
    "PromotionStatus",
    "Representation",
    "Retention",
    "SessionResult",
    "TurnSummary",
    "WorkingAtom",
    "WriteIntent",
    # Relevance
    "RelevanceEngine",
    "ReducedView",
    "reduce_to_budget",
    # Stores
    "ConversationStore",
    "SessionResultStore",
    "WorkingMemory",
    # Model context and trusted system prompt
    "ModelContextBuilder",
    "PromptInputPolicy",
    "SystemPromptBuilder",
]
