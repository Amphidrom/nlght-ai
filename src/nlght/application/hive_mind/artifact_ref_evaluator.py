# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Inline artifact-reference evaluator.

Resolves ``@artifact-ref(ARTIFACT_ID)`` markers embedded by the LLM in its
markdown response, replacing them with fenced code blocks whose language tag
matches the artifact's file extension.

Two modes:
  - ``evaluate_artifact_refs`` — pure text replacement, no I/O side effects.
  - ``stream_response`` — streams the full response word-by-word via an
    emitter, substituting artifact refs with their fenced content inline.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator

from nlght.core.signals.signal import Signal
from nlght.ports.outbound.signal_emitter import SignalEmitter
from nlght.ports.outbound.store_coordinator import StoreCoordinator

logger = logging.getLogger(__name__)

_REF_RE = re.compile(r"@artifact-ref\(([^)]+)\)")
_WORD_RE = re.compile(r"\S+\s*")
_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

_EXT_TO_LANG: dict[str, str] = {
    ".py":   "python",
    ".js":   "javascript",
    ".ts":   "typescript",
    ".tsx":  "tsx",
    ".jsx":  "jsx",
    ".json": "json",
    ".yaml": "yaml",
    ".yml":  "yaml",
    ".toml": "toml",
    ".md":   "markdown",
    ".html": "html",
    ".css":  "css",
    ".sh":   "bash",
    ".bash": "bash",
    ".zsh":  "bash",
    ".sql":  "sql",
    ".rs":   "rust",
    ".go":   "go",
    ".java": "java",
    ".cpp":  "cpp",
    ".c":    "c",
    ".cs":   "csharp",
    ".rb":   "ruby",
    ".php":  "php",
    ".xml":  "xml",
}


def _fence(content: str, extension: str) -> str:
    """Wrap *content* in a fenced code block with the appropriate language tag."""
    lang = _EXT_TO_LANG.get(extension.lower(), "")
    return f"```{lang}\n{content.rstrip()}\n```"


def _iter_words(text: str) -> Iterator[str]:
    for m in _WORD_RE.finditer(text):
        yield m.group()


def _validate_artifact_id(artifact_id: str) -> bool:
    """Accept only well-formed UUIDs to prevent prompt-injection via crafted IDs."""
    return bool(_UUID_RE.match(artifact_id))


async def evaluate_artifact_refs(text: str, store: StoreCoordinator) -> str:
    """Replace every ``@artifact-ref(id)`` in *text* with a fenced code block.

    IDs that are not valid UUIDs are rejected; unknown IDs get a bracketed marker.
    """
    refs = _REF_RE.findall(text)
    if not refs:
        return text

    for raw_id in refs:
        artifact_id = raw_id.strip()
        if not _validate_artifact_id(artifact_id):
            logger.warning("artifact_ref.rejected | id=%r (not a UUID)", artifact_id)
            replacement = f"[invalid artifact ref: {artifact_id}]"
        else:
            result = store.load_artifact(artifact_id)
            if result is not None:
                logger.info("artifact_ref.resolved | id=%s chars=%d ext=%s", artifact_id, len(result.content), result.extension)
                replacement = _fence(result.content, result.extension)
            else:
                logger.warning("artifact_ref.not_found | id=%s", artifact_id)
                replacement = f"[artifact not found: {artifact_id}]"
        text = text.replace(f"@artifact-ref({raw_id})", replacement, 1)

    return text


async def stream_response(
    text: str,
    store: StoreCoordinator,
    emitter: SignalEmitter,
) -> str:
    """Stream *text* word-by-word via *emitter*, resolving ``@artifact-ref(id)`` inline.

    Splits the text at artifact-ref markers. Regular segments and fenced artifact
    content are both emitted as ``kind="token"`` signals, word by word.

    Returns the fully resolved text (suitable for TurnSummary storage).
    """
    parts = _REF_RE.split(text)   # alternating: segment, id, segment, id, …
    ids = _REF_RE.findall(text)
    resolved: list[str] = []

    for i, part in enumerate(parts):
        for chunk in _iter_words(part):
            await emitter.emit(Signal(role="assistant", kind="token", content=chunk))
        resolved.append(part)

        if i < len(ids):
            artifact_id = ids[i].strip()
            if not _validate_artifact_id(artifact_id):
                logger.warning("artifact_ref.rejected | id=%r (not a UUID)", artifact_id)
                marker = f"[invalid artifact ref: {artifact_id}]"
                await emitter.emit(Signal(role="assistant", kind="token", content=marker))
                resolved.append(marker)
                continue

            result = store.load_artifact(artifact_id)
            if result is not None:
                logger.info("artifact_ref.stream | id=%s chars=%d ext=%s", artifact_id, len(result.content), result.extension)
                fenced = _fence(result.content, result.extension)
                for chunk in _iter_words(fenced):
                    await emitter.emit(Signal(role="assistant", kind="token", content=chunk))
                resolved.append(fenced)
            else:
                logger.warning("artifact_ref.not_found | id=%s", artifact_id)
                marker = f"[artifact not found: {artifact_id}]"
                await emitter.emit(Signal(role="assistant", kind="token", content=marker))
                resolved.append(marker)

    return "".join(resolved)
