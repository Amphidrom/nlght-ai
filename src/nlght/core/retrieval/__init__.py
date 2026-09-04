# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Asking the index a question.

Search, and only search. Three systems are asked, each keeps its own
granularity and its own score semantics, and their *rankings* are fused — never
their numbers. What comes out is "these systems found these things", ranked.

Which of those findings becomes text a model is shown is a different question,
with a token budget and a context window in it, and it is not answered here.
"""

from nlght.core.retrieval.fusion import (
    DEFAULT_DAMPING,
    DEFAULT_SOURCE_WEIGHTS,
    FusedHit,
    FusionWeights,
    fuse,
    rank_within_sources,
)
from nlght.core.retrieval.hit import (
    ASSERTION,
    CARRIERS,
    CHUNK,
    DOCUMENT,
    KNOWLEDGE,
    LEXICAL,
    SOURCES,
    VECTOR,
    Provenance,
    RetrievalHit,
    as_fields,
)
from nlght.core.retrieval.plan import RetrievalPlan, RetrievalPlanError, plan_from_config

__all__ = [
    "ASSERTION",
    "CARRIERS",
    "CHUNK",
    "DEFAULT_DAMPING",
    "DEFAULT_SOURCE_WEIGHTS",
    "DOCUMENT",
    "KNOWLEDGE",
    "LEXICAL",
    "SOURCES",
    "VECTOR",
    "FusedHit",
    "FusionWeights",
    "Provenance",
    "RetrievalHit",
    "RetrievalPlan",
    "RetrievalPlanError",
    "as_fields",
    "fuse",
    "plan_from_config",
    "rank_within_sources",
]
