# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""PipelineExtractor — composes multiple MemoryExtractor stages.

Runs each stage in sequence and merges their ExtractionResult into one.
Deduplicates by entity key and candidate key across stages.
Intent is taken from the first non-"store" stage result.
"""

from __future__ import annotations

from typing import Any

from nlght.core.hive_mind.extraction import (
    ExtractedArtifact,
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
    MemoryCandidate,
)


class PipelineExtractor:
    """Composes multiple extractor stages into a single MemoryExtractor.

    Usage:
        PipelineExtractor(stages=[
            EntityExtractor(client=ctx.llm),
            ArtifactExtractor(),
        ])
    """

    def __init__(self, stages: list[Any]) -> None:
        self._stages = stages

    async def extract(self, text: str) -> ExtractionResult:
        intent = "store"
        entities: list[ExtractedEntity] = []
        artifacts: list[ExtractedArtifact] = []
        relations: list[ExtractedRelation] = []
        candidates: list[MemoryCandidate] = []
        referenced_labels: list[str] = []

        seen_entity_keys: set[str] = set()
        seen_candidate_keys: set[str] = set()
        seen_labels: set[str] = set()

        for stage in self._stages:
            result: ExtractionResult = await stage.extract(text)

            if result.intent != "store" and intent == "store":
                intent = result.intent

            for e in result.entities:
                if e.key not in seen_entity_keys:
                    seen_entity_keys.add(e.key)
                    entities.append(e)

            artifacts.extend(result.artifacts)
            relations.extend(result.relations)

            for c in result.candidates:
                if c.key not in seen_candidate_keys:
                    seen_candidate_keys.add(c.key)
                    candidates.append(c)

            for label in result.referenced_labels:
                if label not in seen_labels:
                    seen_labels.add(label)
                    referenced_labels.append(label)

        return ExtractionResult(
            intent=intent,
            entities=entities,
            artifacts=artifacts,
            relations=relations,
            candidates=candidates,
            referenced_labels=referenced_labels,
        )
