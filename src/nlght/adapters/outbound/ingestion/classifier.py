# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from pathlib import Path

from pygments.lexers import get_lexer_for_filename, guess_lexer
from pygments.util import ClassNotFound

from nlght.core.ingestion import DocumentClassification

_VERSION = "pygments-v1"
_OVERRIDES = {
    ".adoc": "asciidoc",
    ".asciidoc": "asciidoc",
    ".env": "dotenv",
    ".gradle": "gradle",
    ".ini": "ini",
    ".java": "java",
    ".js": "javascript",
    ".json": "json",
    ".jsx": "jsx",
    ".kts": "gradle.kts",
    ".markdown": "markdown",
    ".md": "markdown",
    ".proto": "protobuf",
    ".py": "python",
    ".rst": "restructuredtext",
    ".toml": "toml",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".txt": "text",
    ".vue": "vue",
    ".xml": "xml",
    ".yaml": "yaml",
    ".yml": "yaml",
}
_DOCUMENTATION = {"asciidoc", "markdown", "restructuredtext", "text"}
_CONFIG = {"dotenv", "ini", "json", "toml", "yaml"}
_CODE_MARKERS = (";", "{", "}", "=>", "import ", "class ", "def ", "package ")
_DOC_MARKERS = ("====", "----", "= ", "# ", "[source", "```", ":toc:")


def _canonical_name(lexer: object) -> str:
    aliases = getattr(lexer, "aliases", ())
    value = aliases[0] if aliases else getattr(lexer, "name", "unknown")
    return str(value or "unknown").lower().replace(" ", "-").replace("/", "-")


class PygmentsDocumentClassifier:
    def classify(
        self,
        *,
        path: Path,
        content: bytes,
    ) -> tuple[DocumentClassification, str | None, tuple[str, ...]]:
        if b"\0" in content[:2048]:
            return DocumentClassification("binary", "binary", _VERSION), None, ("binary-content",)
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            return (
                DocumentClassification("text", "unknown", _VERSION),
                None,
                (f"invalid-utf8:{exc.start}",),
            )

        language = "dotenv" if path.name == ".env" or path.name.startswith(".env.") else _OVERRIDES.get(path.suffix.lower()) or self._detect(path, text)
        if language in _DOCUMENTATION:
            kind = "documentation"
        elif language in _CONFIG:
            kind = "config"
        elif language == "unknown":
            kind = "text"
        else:
            kind = "code"
        return DocumentClassification(kind, language, _VERSION), text, ()

    def _detect(self, path: Path, text: str) -> str:
        try:
            return _canonical_name(get_lexer_for_filename(path.name, text))
        except ClassNotFound:
            pass
        sample = text[:8192]
        if any(marker in sample for marker in _DOC_MARKERS):
            return "unknown"
        if sum(marker in sample for marker in _CODE_MARKERS) < 2:
            return "unknown"
        try:
            return _canonical_name(guess_lexer(sample))
        except ClassNotFound:
            return "unknown"
