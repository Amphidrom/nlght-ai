# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""LLM-backed entity extractor — Stage 1 of the memory extraction pipeline.

One LLM call → list[ExtractedEntity].
No artifact markers, no MIME detection, no relations, no intent classification.
Those are separate pipeline stages.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from nlght.core.hive_mind.extraction import (
    ExtractedEntity,
    ExtractionResult,
    MemoryCandidate,
    MemoryCandidateConfidence,
    MemoryCandidateKind,
)
from nlght.ports.outbound.model_client import ModelClient

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are an entity extractor. Identify all meaningful named entities in the user text.
Return ONLY valid JSON — no explanation, no markdown.

Schema:
{
  "entities": [
    {"name": "...", "kind": "..."}
  ]
}

Rules:
- name: canonical name as it appears in or can be derived from the text
- kind: one of: person, class, function, file, module, concept, tool, service, artifact, other
- Include all explicitly named things: variables, classes, files, people, services, concepts
- Empty list [] if no distinct named entities are found
"""


@dataclass
class EntityExtractor:
    """Single-stage LLM entity extractor.

    Inject the step-bound ModelClient:
        EntityExtractor(client=ctx.llm)
    """

    client: ModelClient

    async def extract(self, text: str) -> ExtractionResult:
        raw = await self._call_llm(text)
        entities = _parse_entities(raw)
        candidates = [_entity_candidate(e) for e in entities]
        return ExtractionResult(
            intent="store",
            entities=entities,
            candidates=candidates,
            referenced_labels=[e.key for e in entities],
        )

    async def _call_llm(self, text: str) -> dict[str, Any]:
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]
        tokens: list[str] = []
        async for event in self.client.stream(messages):
            if event.kind == "token" and event.content:
                tokens.append(event.content)
        return _parse_json("".join(tokens))


def _parse_json(raw: str) -> dict[str, Any]:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        first_newline = cleaned.find("\n")
        cleaned = cleaned[first_newline + 1:] if first_newline >= 0 else ""
        cleaned = cleaned.rsplit("```", 1)[0].strip()
    try:
        result = json.loads(cleaned)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass
    return {"entities": []}


def _parse_entities(raw: dict[str, Any]) -> list[ExtractedEntity]:
    entities: list[ExtractedEntity] = []
    seen: set[str] = set()
    for item in raw.get("entities", []):
        name = str(item.get("name") or "").strip()
        kind = str(item.get("kind") or "entity").strip() or "entity"
        if not name:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        entities.append(ExtractedEntity(text=name, kind=kind, canonical=name))
    return entities


def _name_slug(name: str) -> str:
    chars = []
    for ch in name.strip().lower():
        if ch.isalnum() or ch in {"_", ".", "-"}:
            chars.append(ch)
        elif chars and chars[-1] != "_":
            chars.append("_")
    return "".join(chars).strip("_") or "entity"


def _entity_candidate(entity: ExtractedEntity) -> MemoryCandidate:
    slug = _name_slug(entity.text)
    return MemoryCandidate(
        key=f"entity.{slug}",
        kind=MemoryCandidateKind.FACT,
        content=f"{entity.text} ({entity.kind})",
        entities=[entity.text],
        tags=["entity", entity.kind],
        confidence=MemoryCandidateConfidence.VALIDATED,
    )
