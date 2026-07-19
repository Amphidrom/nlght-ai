# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Artifact extractor — Stage 2 of the memory extraction pipeline.

Two-tier detection:
  1. Structural (fast, free): recognises backtick-fenced blocks with a
     derivable label.  O(n) line-by-line state machine, no regex.
  2. LLM fallback (only when structural finds nothing): a single LLM call
     asks the model to identify and extract any file/artifact content
     verbatim — handles raw pastes, triple-quote docs, unlabelled blocks,
     or any other format the structural parser misses.

Inject ``client`` to enable the fallback:
    ArtifactExtractor(client=ctx.llm)

Without a client only backtick-fenced blocks are recognised.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from nlght.core.hive_mind.extraction import (
    ExtractedArtifact,
    ExtractedEntity,
    ExtractionResult,
    MemoryCandidate,
    MemoryCandidateConfidence,
    MemoryCandidateKind,
)

if TYPE_CHECKING:
    from nlght.ports.outbound.model_client import ModelClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM system prompt — fallback path only
# ---------------------------------------------------------------------------

_FALLBACK_SYSTEM_PROMPT = """\
You are a file and artifact extractor. Given a user message, identify any file content, \
code snippets, scripts, or named documents — whether wrapped in fences, quotes, \
triple-quotes, or pasted raw without any delimiter.

Return ONLY valid JSON — no explanation, no markdown.

Schema:
{
  "artifacts": [
    {
      "label": "filename or short descriptive name",
      "content": "verbatim content exactly as it appears — do not modify or summarise"
    }
  ]
}

Rules:
- label: use the filename when mentioned (e.g. "debug_runner.py"); otherwise a short name
- content: copy verbatim — preserve indentation, line breaks, and all characters
- Only include clearly intentional file/code content, not every inline snippet
- Return [] if no distinct file or artifact content is present
"""

# ---------------------------------------------------------------------------
# Structural helpers
# ---------------------------------------------------------------------------

_LABEL_STRIP_PREFIXES = (
    "hier hast du ",
    "hier ist ",
    "das ist ",
    "this is ",
    "here is ",
    "here's ",
    "speichere ",
    "merke dir ",
    "remember ",
    "store ",
)

_CODE_LANGS = frozenset({
    "python", "py", "javascript", "js", "typescript", "ts",
    "java", "go", "rust", "rs", "bash", "sh", "zsh",
    "sql", "json", "yaml", "yml", "toml", "css", "html",
    "xml", "c", "cpp", "csharp", "cs", "ruby", "rb", "php",
})


@dataclass(frozen=True)
class _ParsedBlock:
    label: str
    language: str
    content: str


def _parse_fence_header(rest: str) -> tuple[str, str]:
    """Parse the part after ``` — returns (language, inline_label).

    Handles:
      ```python                 → ("python", "")
      ```python "service.py"   → ("python", "service.py")
      ```"service.py"          → ("", "service.py")
    """
    rest = rest.strip()
    label = ""
    if '"' in rest:
        q1 = rest.index('"') + 1
        q2 = rest.find('"', q1)
        if q2 > q1:
            label = rest[q1:q2].strip()
            lang_part = rest[:rest.index('"')].strip()
        else:
            lang_part = rest
    else:
        lang_part = rest
    return lang_part.lower(), label


def _label_from_preceding(lines: list[str], fence_idx: int) -> str:
    """Derive a label from the nearest non-empty line before the fence."""
    for j in range(fence_idx - 1, -1, -1):
        raw = lines[j].strip()
        if not raw:
            continue
        if raw.endswith(":"):
            raw = raw[:-1].strip()
        lo = raw.lower()
        for prefix in _LABEL_STRIP_PREFIXES:
            if lo.startswith(prefix):
                raw = raw[len(prefix):].strip()
                break
        raw = raw.strip("\"'")
        return raw if len(raw) >= 2 else ""
    return ""


def _parse_fenced_blocks(text: str) -> list[_ParsedBlock]:
    """Line-by-line state machine — O(n), no regex backtracking."""
    result: list[_ParsedBlock] = []
    lines = text.split("\n")
    n = len(lines)
    i = 0
    while i < n:
        line = lines[i]
        if line.startswith("```"):
            lang, inline_label = _parse_fence_header(line[3:])
            fence_idx = i
            i += 1
            body_lines: list[str] = []
            while i < n and not lines[i].startswith("```"):
                body_lines.append(lines[i])
                i += 1
            content = "\n".join(body_lines).strip()
            if content:
                label = inline_label or _label_from_preceding(lines, fence_idx)
                if label:
                    result.append(_ParsedBlock(label=label, language=lang, content=content))
        i += 1
    return result


def _content_type(lang: str) -> str:
    return "code" if lang in _CODE_LANGS else "text"


def _label_slug(label: str) -> str:
    chars = []
    for ch in label.strip().lower():
        if ch.isalnum() or ch in {"_", ".", "-"}:
            chars.append(ch)
        elif chars and chars[-1] != "_":
            chars.append("_")
    return "".join(chars).strip("_") or "artifact"


def _artifact_candidate(artifact: ExtractedArtifact) -> MemoryCandidate:
    slug = _label_slug(artifact.label)
    return MemoryCandidate(
        key=f"artifact.{slug}",
        kind=MemoryCandidateKind.ARTIFACT,
        content=f"Artefakt {artifact.label}:\n{artifact.content}",
        entities=[artifact.label],
        tags=["user_fact", "artifact", slug],
        confidence=MemoryCandidateConfidence.VALIDATED,
    )


def _build_result(blocks: list[_ParsedBlock]) -> ExtractionResult:
    artifacts = [
        ExtractedArtifact(
            label=b.label,
            content=b.content,
            content_type=_content_type(b.language),
        )
        for b in blocks
    ]
    entities = [
        ExtractedEntity(text=a.label, kind="artifact_label", canonical=a.label)
        for a in artifacts
    ]
    candidates = [_artifact_candidate(a) for a in artifacts]
    return ExtractionResult(
        intent="store",
        entities=entities,
        artifacts=artifacts,
        candidates=candidates,
        referenced_labels=[a.label for a in artifacts],
    )


# ---------------------------------------------------------------------------
# LLM fallback helpers
# ---------------------------------------------------------------------------

def _parse_llm_json(raw: str) -> dict[str, Any]:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        first_nl = cleaned.find("\n")
        cleaned = cleaned[first_nl + 1:] if first_nl >= 0 else ""
        cleaned = cleaned.rsplit("```", 1)[0].strip()
    try:
        result = json.loads(cleaned)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass
    return {"artifacts": []}


def _blocks_from_llm(raw: dict[str, Any]) -> list[_ParsedBlock]:
    blocks: list[_ParsedBlock] = []
    seen: set[str] = set()
    for item in raw.get("artifacts", []):
        label = str(item.get("label") or "").strip()
        content = str(item.get("content") or "").strip()
        if not label or not content:
            continue
        key = label.lower()
        if key in seen:
            continue
        seen.add(key)
        blocks.append(_ParsedBlock(label=label, language="", content=content))
    return blocks


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

@dataclass
class ArtifactExtractor:
    """Stage 2: artifact extractor with structural parsing and LLM fallback.

    Structural path (no LLM): backtick-fenced blocks with a derivable label.
    LLM fallback (when structural finds nothing and client is provided):
    one LLM call extracts file/code content from any format verbatim.
    """

    client: ModelClient | None = field(default=None)

    async def extract(self, text: str) -> ExtractionResult:
        blocks = _parse_fenced_blocks(text)
        if blocks:
            logger.debug("artifact_extractor.structural | found=%d", len(blocks))
            return _build_result(blocks)

        if self.client is None:
            return ExtractionResult(intent="store")

        logger.debug("artifact_extractor.llm_fallback | structural_found=0")
        raw = await self._call_llm(text)
        llm_blocks = _blocks_from_llm(raw)
        if llm_blocks:
            logger.info("artifact_extractor.llm_fallback.found | count=%d", len(llm_blocks))
        return _build_result(llm_blocks)

    async def _call_llm(self, text: str) -> dict[str, Any]:
        # Only called from extract() after its self.client is None guard.
        assert self.client is not None
        messages = [
            {"role": "system", "content": _FALLBACK_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]
        tokens: list[str] = []
        async for event in self.client.stream(messages):
            if event.kind == "token" and event.content:
                tokens.append(event.content)
        return _parse_llm_json("".join(tokens))
