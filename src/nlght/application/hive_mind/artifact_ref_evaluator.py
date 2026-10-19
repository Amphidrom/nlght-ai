# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Resolve artifact references in assistant text."""

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
    """Wrap artifact content in a language-tagged code fence.

    Args:
        content (str): Artifact text to enclose.
        extension (str): Filename extension used for the language tag.

    Returns:
        str: Fenced text with a known language tag when available.
    """
    lang = _EXT_TO_LANG.get(extension.lower(), "")
    return f"```{lang}\n{content.rstrip()}\n```"


def _iter_words(text: str) -> Iterator[str]:
    for m in _WORD_RE.finditer(text):
        yield m.group()


def _validate_artifact_id(artifact_id: str) -> bool:
    """Restrict artifact lookups to well-formed UUIDs.

    Args:
        artifact_id (str): Candidate artifact ID.

    Returns:
        bool: True when the ID matches the UUID syntax guard.
    """
    return bool(_UUID_RE.match(artifact_id))


async def evaluate_artifact_refs(text: str, store: StoreCoordinator) -> str:
    """Resolve inline artifact references into fenced content.

    Args:
        text (str): Assistant text containing references.
        store (StoreCoordinator): Session store containing referenced artifacts.

    Returns:
        str: Text with resolved content or explicit invalid/missing markers.

    Side Effects:
        Reads artifacts from the store and logs lookup failures.
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
    """Stream assistant text while resolving artifact references.

    Args:
        text (str): Assistant text to stream.
        store (StoreCoordinator): Session store containing referenced artifacts.
        emitter (SignalEmitter): Signal emitter for resolved token output.

    Returns:
        str: Fully resolved text for session storage.

    Side Effects:
        Reads artifacts, emits token signals, and logs lookup failures.
    """
    parts = _REF_RE.split(text)  # Preserve text between reference IDs.
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
