# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Format-aware parsers, ported from the prototype pipeline.

``knowledge.parse`` splits generic prose. These handle formats where the markup
itself carries structure, so a unit ends at a real section boundary rather than
wherever a blank line happened to fall — which is what makes extraction from
specifications and cheat sheets usable.

The parsing logic for each format is a standalone function registered in
``FORMAT_PARSERS`` by name. Three things consume that registry:

- the explicit per-format steps (``knowledge.parse_markdown``, ``.parse_asciidoc``,
  ``.parse_html``, ``.parse_pdf``), which each claim one format by file extension;
- the dispatching step (``knowledge.parse_auto``), which routes each document to
  the parser its extension names and falls back to prose for the rest;
- and any future caller that wants the parsing without a step.

Every parser keeps the section heading in the unit's evidence, so the noise
filter can later drop assertions whose subject merely echoes their heading.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from nlght.core.errors.errors import WorkflowConfigurationError
from nlght.core.ingestion import ProcessedDocument, SourceSnapshot
from nlght.core.knowledge import (
    AUTHORED_ANCHOR,
    DERIVED_ANCHOR,
    NO_ANCHOR,
    KnowledgeUnit,
)
from nlght.core.workflow.step import (
    StepBase,
    StepOption,
    StepResult,
    WorkflowStepContext,
)

logger = logging.getLogger(__name__)

EMPTY = "empty"

try:  # pragma: no cover - import guard
    from pdfminer.high_level import extract_text as _extract_pdf_text

    _HAS_PDFMINER = True
except ImportError:  # pragma: no cover - import guard
    _extract_pdf_text = None
    _HAS_PDFMINER = False

try:  # pragma: no cover - import guard
    from bs4 import BeautifulSoup as _BeautifulSoup

    _HAS_BS4 = True
except ImportError:  # pragma: no cover - import guard
    _BeautifulSoup = None
    _HAS_BS4 = False

_ADOC_NOISE = ("//", "ifdef::", "ifndef::", "endif::", "include::")


class _UnitBuilder:
    """Accumulates lines into units, remembering the current section."""

    def __init__(
        self,
        source_id: str,
        document_id: str,
        tags: dict[str, Any],
        processing_revision_id: str = "",
        document_path: str = "",
    ) -> None:
        self._source_id = source_id
        self._document_id = document_id
        self._revision_id = processing_revision_id
        # Provenance carried through to the evidence. It was the other parser's
        # `metadata["path"]` and this one had none, so half the corpus had no
        # path at all — the same shape of gap as the missing `processing_revision_id` that
        # made `KnowledgeUnit` use fields instead of a metadata dict.
        self._document_path = document_path
        self._tags = tags
        self._buffer: list[str] = []
        self.section: str | None = None
        #: The heading path, by level, so a derived anchor names a section's
        #: place in the document rather than only its last heading. Two sections
        #: called "Overview" under different parents are different sections.
        self._path: list[str] = []
        self._anchor = ""
        self._strength = NO_ANCHOR
        self.units: list[KnowledgeUnit] = []

    def heading(self, level: int, text: str, anchor: str = "") -> None:
        """A new section starts here, with whatever the format could offer.

        `anchor` is what the *source* wrote. When there is one it is taken
        unchanged — an authored id is the strongest thing a document can give,
        and normalising it would be inventing. When there is none the heading
        path stands in, and says so.
        """
        self.flush()
        self.section = text
        del self._path[max(level - 1, 0):]
        self._path.append(text)
        if anchor:
            self._anchor, self._strength = anchor, AUTHORED_ANCHOR
        else:
            self._anchor = " > ".join(part for part in self._path if part)
            self._strength = DERIVED_ANCHOR

    def add(self, line: str) -> None:
        self._buffer.append(line)

    def flush(self) -> None:
        content = "\n".join(self._buffer).strip()
        self._buffer.clear()
        if not content:
            return
        self.units.append(
            KnowledgeUnit(
                unit_ordinal=f"{self._source_id}:{self._document_id}:{len(self.units)}",
                source_id=self._source_id,
                content=content,
                document_id=self._document_id,
                processing_revision_id=self._revision_id,
                document_path=self._document_path,
                anchor=self._anchor,
                anchor_strength=self._strength,
                metadata={"section": self.section or ""},
                tags=dict(self._tags),
            )
        )


# ---------------------------------------------------------------------------
# Parsing functions — one per format, appending to a _UnitBuilder. Text parsers
# take the document's text; the PDF parser takes the original bytes.
# ---------------------------------------------------------------------------

def parse_markdown(text: str, builder: _UnitBuilder) -> None:
    """Split Markdown on headings, dropping code, quotes, images, and rules.

    They carry no assertion, and feeding them to a model produces confident
    nonsense about syntax rather than about the subject.
    """
    in_code = False
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("```"):
            in_code = not in_code
            continue
        if in_code or not line or line.startswith((">", "!", "---")):
            continue
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            builder.heading(level, line.lstrip("#").strip())
            continue
        builder.add(line)


_ADOC_MACRO = re.compile(r"(?<![\w:])([a-z][\w-]*):{1,2}(\S*?)\[([^\[\]]*)\]")


def _macro_text(match: re.Match[str]) -> str:
    """Reduce one AsciiDoc macro to the words a reader actually sees.

    `parse_html` has always done this — it takes an element's text and discards
    its attributes — while this parser passed macros through verbatim. So a
    navigation partial reached the model as

        xref:api:java/index.html[Spring Boot,role=link-external, window=_blank]

    and the model, asked what that asserts, answered with `link-external` and
    `_blank`. It was not wrong; it was shown markup and asked to read it as
    prose. Eight such assertions were persisted as approved knowledge.

    An attribute list is positional entries first, then `name=value` pairs, so
    the visible text is the positional part and the rest is presentation. When
    there is no positional part the target carries the meaning — `javadoc:`,
    `configprop:` and `include-code::` are written with an empty list precisely
    because their target *is* the subject — so dropping it would delete the one
    identifier the sentence was about.

    This is reading the macro's own syntax, not matching known attribute names:
    a list of `role`, `window`, `format` and whatever the next markup version
    adds would never be finished.
    """
    target, attributes = match.group(2), match.group(3)
    positional = [
        part.strip() for part in attributes.split(",") if part.strip() and "=" not in part
    ]
    if positional:
        return " ".join(positional)
    # Anchors and query strings are addressing, not words: `monitoring.adoc#foo`
    # names a place in a document, and the place is not what the sentence claims.
    return target.split("#", 1)[0].rstrip("/").rsplit("/", 1)[-1] or target


def parse_asciidoc(text: str, builder: _UnitBuilder) -> None:
    """Split AsciiDoc on ``=`` headings, dropping build directives.

    Attribute lines, comments, conditionals, includes, and anchors are build
    directives, not statements about the subject.
    """
    pending_anchor = ""
    for raw in text.splitlines():
        line = raw.lstrip("﻿").strip()
        if not line:
            continue
        if line.startswith(":") or line.startswith(_ADOC_NOISE):
            continue
        if line.startswith("[[") and line.endswith("]]"):
            # An authored id, and the parser used to drop it with the build
            # directives. `[[howto.actuator.customizing-sanitization]]` survives
            # a rename, a reorder and a rewrite, which is everything a section's
            # position does not — see `working/knowledge-slot-identity`. It
            # precedes its heading, so it waits for it.
            pending_anchor = line[2:-2].split(",", 1)[0].strip()
            continue
        # Block attributes — `[source,xml]`, `[NOTE]` — configure how the block
        # below is rendered. They say nothing about the subject, and a line that
        # is nothing but an attribute list has no words in it to begin with.
        if line.startswith("[") and line.endswith("]"):
            continue
        if line.startswith("="):
            level = len(line) - len(line.lstrip("="))
            builder.heading(
                level,
                _ADOC_MACRO.sub(_macro_text, line.lstrip("=").strip()),
                pending_anchor,
            )
            pending_anchor = ""
            continue
        line = _ADOC_MACRO.sub(_macro_text, line)
        # A line that was only markup is now empty, and an empty line is not a
        # statement about anything.
        if not line.strip(" *-.,;:|"):
            continue
        builder.add(line)


def parse_html(text: str, builder: _UnitBuilder) -> None:
    """Split HTML on headings after stripping chrome.

    Scripts, styles, navigation, headers, and footers are removed first: on a
    specification page they repeat on every document and would otherwise
    dominate extraction. Assumes ``beautifulsoup4`` is importable — the caller
    checks the dependency and decides whether to raise or skip.
    """
    soup = _BeautifulSoup(text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    for element in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "pre", "td"]):
        content = " ".join(element.get_text(" ", strip=True).split())
        if not content:
            continue
        if element.name.startswith("h"):
            # The id is authored and taken unchanged; without one the heading
            # path stands in and says it is derived. Nothing is invented for a
            # heading that has no id — a made-up identifier would be a position
            # wearing a better name.
            level = int(element.name[1:]) if element.name[1:].isdigit() else 1
            builder.heading(level, content, str(element.get("id") or "").strip())
            continue
        builder.add(content)


def parse_pdf(content: bytes, builder: _UnitBuilder) -> None:
    """Extract PDF text and split it on blank lines.

    Takes the original bytes because processing turns documents into text and a
    PDF has none until it is extracted here. Assumes ``pdfminer.six`` is
    importable — the caller checks the dependency.
    """
    import io  # noqa: PLC0415 (only needed on the PDF path)

    text = _extract_pdf_text(io.BytesIO(content))
    for block in re.split(r"\n\s*\n", text):
        cleaned = " ".join(block.split())
        if cleaned:
            builder.add(cleaned)
            builder.flush()


def parse_prose(text: str, builder: _UnitBuilder) -> None:
    """Generic fallback: one unit per blank-line-separated block.

    Blank lines are the most reliable structural signal across plain text and
    any format without its own parser, so an unrecognised document still yields
    coherent units rather than nothing.
    """
    for block in re.split(r"\n\s*\n", text):
        cleaned = block.strip()
        if cleaned:
            builder.add(cleaned)
            builder.flush()


# ---------------------------------------------------------------------------
# Registry — named parsers with their default extensions and any optional
# dependency. Both the explicit per-format steps and the dispatching step use it.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Requirement:
    """An optional dependency a parser needs, checked at call time.

    ``available`` is a callable rather than a captured bool so tests (and a
    late install) see the current state of the module-level import guard.
    """

    available: Callable[[], bool]
    package: str
    extra: str


@dataclass(frozen=True)
class _Parser:
    parse: Callable[..., None]
    default_extensions: tuple[str, ...]
    requires: _Requirement | None = None
    #: The processed text is useless to this parser — PDF has no text until it
    #: is extracted from the bytes. Without the snapshot the document is skipped.
    needs_raw: bool = False
    #: The processed text works, and the source's own form works better. HTML on
    #: disk arrives as its own markup either way; HTML from Confluence arrives as
    #: text with the tags removed, and the tags are the structure. Preferring
    #: rather than requiring keeps the parser usable where no snapshot is at hand.
    prefers_raw: bool = False


FORMAT_PARSERS: dict[str, _Parser] = {
    "markdown": _Parser(parse_markdown, (".md", ".markdown")),
    "asciidoc": _Parser(parse_asciidoc, (".adoc", ".asciidoc")),
    "html": _Parser(
        parse_html,
        (".html", ".htm"),
        requires=_Requirement(lambda: _HAS_BS4, "beautifulsoup4", "web-search"),
        prefers_raw=True,
    ),
    "pdf": _Parser(
        parse_pdf,
        (".pdf",),
        requires=_Requirement(lambda: _HAS_PDFMINER, "pdfminer.six", "pdf"),
        needs_raw=True,
    ),
    # No default extensions: prose is the fallback, chosen explicitly, never by
    # matching a suffix.
    "prose": _Parser(parse_prose, ()),
}


def _raw_content(ctx: WorkflowStepContext, document: ProcessedDocument) -> bytes | None:
    """The document as its source holds it, from the acquired snapshot.

    `markup` rather than `content`: a source that transformed its document to
    produce the searchable text keeps the original beside it, and a parser wants
    that one.
    """
    snapshot = ctx.metadata.get("ingestion.snapshot")
    if not isinstance(snapshot, SourceSnapshot):
        return None
    for acquired in snapshot.documents:
        if acquired.external_id == document.external_id:
            return acquired.markup
    return None


def _apply_parser(
    spec: _Parser,
    document: ProcessedDocument,
    builder: _UnitBuilder,
    ctx: WorkflowStepContext,
    *,
    step_type: str,
    on_missing_dependency: str,
) -> bool:
    """Run one parser against a document.

    Returns ``True`` when the parser ran, ``False`` when the document was
    skipped. ``on_missing_dependency`` is ``"raise"`` (the explicit per-format
    steps: a configured format with a missing backend is an error) or
    ``"skip"`` (the dispatcher: one unparseable document must not fail a mixed
    corpus).
    """
    if spec.requires is not None and not spec.requires.available():
        if on_missing_dependency == "raise":
            raise WorkflowConfigurationError(
                f"Step '{step_type}' requires {spec.requires.package}. "
                f"Install it with: pip install 'nlght-ai[{spec.requires.extra}]'"
            )
        logger.warning(
            "[%s] %s.dependency_missing | package=%s document=%s — skipped",
            ctx.correlation_id, step_type, spec.requires.package, document.document_id,
        )
        return False

    if spec.needs_raw:
        content = _raw_content(ctx, document)
        if content is None:
            logger.warning(
                "[%s] %s.missing_content | document=%s",
                ctx.correlation_id, step_type, document.document_id,
            )
            return False
        spec.parse(content, builder)
        return True

    text = document.text or ""
    if spec.prefers_raw:
        raw = _raw_content(ctx, document)
        if raw is not None:
            text = raw.decode("utf-8", errors="replace")
    spec.parse(text, builder)
    return True


def _require_processed(step_type: str, ctx: WorkflowStepContext) -> tuple[ProcessedDocument, ...]:
    documents = ctx.metadata.get("ingestion.processed")
    if not isinstance(documents, tuple):
        raise WorkflowConfigurationError(
            f"Step '{step_type}' requires processed documents; "
            f"place an 'ingestion.process' step before it."
        )
    return documents


# ---------------------------------------------------------------------------
# Explicit per-format steps — each claims one format by extension.
# ---------------------------------------------------------------------------

class _FormatParseStep(StepBase):
    """Shared plumbing: claim documents by extension, parse, append units."""

    EXTENSIONS: tuple[str, ...] = ()
    PARSER: str = ""  # key into FORMAT_PARSERS

    @classmethod
    def options(cls) -> list[StepOption]:
        return [
            StepOption("tags", "object", "Static tags attached to every unit",
                       placeholder='{"product": "spring", "version": "3.2"}'),
            StepOption("extensions", "array",
                       f"File extensions this parser claims (default {list(cls.EXTENSIONS)})",
                       default=list(cls.EXTENSIONS)),
        ]

    def _claims(self, document: ProcessedDocument) -> bool:
        extensions = tuple(
            str(item).lower() for item in self.config.get("extensions", self.EXTENSIONS)
        )
        return document.path.lower().endswith(extensions) if extensions else True

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        documents = _require_processed(self.TYPE, ctx)
        tags = dict(self.config.get("tags", {}))
        spec = FORMAT_PARSERS[self.PARSER]

        existing = ctx.metadata.get("knowledge.units")
        units: list[KnowledgeUnit] = list(existing) if isinstance(existing, tuple) else []
        parsed = 0

        for document in documents:
            if not self._claims(document):
                continue
            builder = _UnitBuilder(
                document.source, document.document_id, tags, document.processing_revision_id,
                document.path,
            )
            if _apply_parser(
                spec, document, builder, ctx,
                step_type=self.TYPE, on_missing_dependency="raise",
            ):
                builder.flush()
                units.extend(builder.units)
            parsed += 1

        ctx.metadata["knowledge.units"] = tuple(units)
        logger.info(
            "[%s] %s.done | documents=%d units=%d",
            ctx.correlation_id, self.TYPE, parsed, len(units),
        )
        return StepResult(ctx=ctx, verdict="DEFAULT" if units else EMPTY)


class MarkdownParseStep(_FormatParseStep):
    """Parses Markdown, splitting on headings and dropping non-assertive markup."""

    TYPE = "knowledge.parse_markdown"
    EXTENSIONS = (".md", ".markdown")
    PARSER = "markdown"


class AsciiDocParseStep(_FormatParseStep):
    """Parses AsciiDoc, splitting on ``=`` headings and dropping build directives."""

    TYPE = "knowledge.parse_asciidoc"
    EXTENSIONS = (".adoc", ".asciidoc")
    PARSER = "asciidoc"


class HtmlParseStep(_FormatParseStep):
    """Parses HTML, splitting on headings after stripping chrome."""

    TYPE = "knowledge.parse_html"
    EXTENSIONS = (".html", ".htm")
    PARSER = "html"


class PdfParseStep(_FormatParseStep):
    """Extracts text from PDFs and splits it on blank lines."""

    TYPE = "knowledge.parse_pdf"
    EXTENSIONS = (".pdf",)
    PARSER = "pdf"


# ---------------------------------------------------------------------------
# Dispatching step — routes each document to the parser its extension names.
# ---------------------------------------------------------------------------

_DEFAULT_AUTO_PARSERS = ("markdown", "asciidoc", "html", "pdf")
_SKIP_FALLBACK = {"", "none", "skip"}


class KnowledgeParseAutoStep(StepBase):
    """Routes each document to the parser its file extension names.

    One step instead of a hand-wired chain of per-format parsers: a mixed
    corpus of Markdown, AsciiDoc, HTML, and PDF is parsed correctly without the
    pipeline having to know in advance which formats it will meet. A document
    whose extension matches nothing falls back to the prose splitter, so an
    unrecognised file still yields units rather than silently nothing.

    Step config:
        parsers:    named parsers to enable (default markdown, asciidoc, html,
                    pdf), each claiming its own default extensions.
        extensions: map of extension -> parser name, to add or override which
                    parser an extension routes to (e.g. ``{".txt": "prose"}``).
        fallback:   parser for unmatched extensions (default ``prose``;
                    ``none`` skips them).
        tags:       static tags attached to every unit.

    Reads ``ctx.metadata['ingestion.processed']`` and appends to
    ``ctx.metadata['knowledge.units']``, so it can share a workflow with the
    explicit per-format steps.
    """

    TYPE = "knowledge.parse_auto"

    @classmethod
    def options(cls) -> list[StepOption]:
        return [
            StepOption("parsers", "array",
                       f"Named parsers to enable (default {list(_DEFAULT_AUTO_PARSERS)}); "
                       f"available: {sorted(FORMAT_PARSERS)}",
                       default=list(_DEFAULT_AUTO_PARSERS)),
            StepOption("extensions", "object",
                       'Extension -> parser overrides, e.g. {".txt": "prose"}'),
            StepOption("fallback", "string",
                       'Parser for unmatched extensions; "none" to skip them',
                       default="prose"),
            StepOption("tags", "object", "Static tags attached to every unit",
                       placeholder='{"product": "spring", "version": "3.2"}'),
        ]

    def _extension_map(self) -> dict[str, str]:
        enabled = self.config.get("parsers") or list(_DEFAULT_AUTO_PARSERS)
        mapping: dict[str, str] = {}
        for name in enabled:
            if str(name) not in FORMAT_PARSERS:
                raise WorkflowConfigurationError(
                    f"Step '{self.TYPE}' enables unknown parser '{name}'. "
                    f"Known parsers: {sorted(FORMAT_PARSERS)}"
                )
            for extension in FORMAT_PARSERS[str(name)].default_extensions:
                mapping[extension] = str(name)
        for extension, name in (self.config.get("extensions") or {}).items():
            if str(name) not in FORMAT_PARSERS:
                raise WorkflowConfigurationError(
                    f"Step '{self.TYPE}' maps '{extension}' to unknown parser '{name}'. "
                    f"Known parsers: {sorted(FORMAT_PARSERS)}"
                )
            mapping[str(extension).lower()] = str(name)
        return mapping

    def _fallback(self) -> str | None:
        fallback = str(self.config.get("fallback", "prose")).strip().lower()
        if fallback in _SKIP_FALLBACK:
            return None
        if fallback not in FORMAT_PARSERS:
            raise WorkflowConfigurationError(
                f"Step '{self.TYPE}' has unknown fallback parser '{fallback}'. "
                f"Known parsers: {sorted(FORMAT_PARSERS)}"
            )
        return fallback

    @staticmethod
    def _match(path: str, mapping: dict[str, str]) -> str | None:
        lowered = path.lower()
        best_ext: str | None = None
        best_name: str | None = None
        for extension, name in mapping.items():
            # Longest matching extension wins, so a compound suffix is not
            # shadowed by a shorter registered one.
            if lowered.endswith(extension) and (best_ext is None or len(extension) > len(best_ext)):
                best_ext, best_name = extension, name
        return best_name

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        documents = _require_processed(self.TYPE, ctx)
        mapping = self._extension_map()
        fallback = self._fallback()
        tags = dict(self.config.get("tags", {}))

        existing = ctx.metadata.get("knowledge.units")
        units: list[KnowledgeUnit] = list(existing) if isinstance(existing, tuple) else []
        parsed = 0

        for document in documents:
            name = self._match(document.path, mapping) or fallback
            if name is None:
                continue
            builder = _UnitBuilder(
                document.source, document.document_id, tags, document.processing_revision_id,
                document.path,
            )
            if _apply_parser(
                FORMAT_PARSERS[name], document, builder, ctx,
                step_type=self.TYPE, on_missing_dependency="skip",
            ):
                builder.flush()
                units.extend(builder.units)
            parsed += 1

        ctx.metadata["knowledge.units"] = tuple(units)
        logger.info(
            "[%s] %s.done | documents=%d units=%d",
            ctx.correlation_id, self.TYPE, parsed, len(units),
        )
        return StepResult(ctx=ctx, verdict="DEFAULT" if units else EMPTY)
