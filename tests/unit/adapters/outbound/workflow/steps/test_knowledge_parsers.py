# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Format parsers: structure-aware splitting and markup that carries no assertion."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from nlght.adapters.outbound.workflow.steps.knowledge import (
    AsciiDocParseStep,
    HtmlParseStep,
    KnowledgeParseAutoStep,
    MarkdownParseStep,
    PdfParseStep,
)
from nlght.core.entry.context import RequestContext
from nlght.core.errors.errors import WorkflowConfigurationError
from nlght.core.ingestion import (
    DocumentClassification,
    Enrichment,
    ProcessedDocument,
)
from nlght.core.protocol.protocol import ProtocolKind
from nlght.core.trigger.trigger import Trigger, TriggerKind
from nlght.core.workflow.step import WorkflowStepContext


class _Emitter:
    async def emit(self, signal: object) -> None: ...


def _document(path: str, text: str) -> ProcessedDocument:
    return ProcessedDocument(
        document_id=f"doc-{path}",
        source_revision_id="src-1",
        processing_revision_id="rev-1",
        source="wiki",
        external_id=f"wiki:{path}",
        path=path,
        text=text,
        classification=DocumentClassification(kind="text", language="x", classifier_version="1"),
        enrichment=Enrichment(enricher="none", version="1"),
        chunks=(),
        metadata={},
    )


def _ctx(*documents: ProcessedDocument) -> WorkflowStepContext:
    context = RequestContext(
        correlation_id="run-1", request_id="rid", received_at=datetime(2026, 1, 1, tzinfo=UTC),
        path="/", method="POST", headers={}, query_params={}, client_host=None,
    )
    ctx = WorkflowStepContext(
        correlation_id="run-1",
        trigger=Trigger(
            kind=TriggerKind.INBOUND_EVENT, protocol=ProtocolKind.GENERIC_JSON,
            operation="ingest", payload={}, context=context,
        ),
        model="", messages=[], stream=False, emitter=_Emitter(),
    )
    ctx.metadata["ingestion.processed"] = documents
    return ctx


async def _units(step, *documents):
    result = await step.run(_ctx(*documents))
    return result.ctx.metadata["knowledge.units"]


# -- markdown ----------------------------------------------------------------

_MARKDOWN = """# Session Management

Use short-lived tokens.

## Storage

> quoted advice
![diagram](x.png)
---
Store secrets in a vault.

```python
print("not an assertion")
```
"""


async def test_markdown_splits_on_headings_and_records_the_section() -> None:
    units = await _units(MarkdownParseStep(config={}), _document("cheat.md", _MARKDOWN))

    assert [unit.content for unit in units] == [
        "Use short-lived tokens.",
        "Store secrets in a vault.",
    ]
    assert units[1].metadata["section"] == "Storage"


async def test_markdown_drops_code_quotes_images_and_rules() -> None:
    units = await _units(MarkdownParseStep(config={}), _document("cheat.md", _MARKDOWN))

    joined = " ".join(unit.content for unit in units)
    assert "not an assertion" not in joined
    assert "quoted advice" not in joined
    assert "diagram" not in joined


async def test_a_parser_only_claims_its_own_extensions() -> None:
    units = await _units(
        MarkdownParseStep(config={}),
        _document("notes.txt", "# Heading\n\nSomething."),
    )

    assert units == ()


async def test_extensions_are_configurable() -> None:
    units = await _units(
        MarkdownParseStep(config={"extensions": [".txt"]}),
        _document("notes.txt", "# Heading\n\nSomething."),
    )

    assert [unit.content for unit in units] == ["Something."]


# -- asciidoc ----------------------------------------------------------------

_ADOC = """:toc: left
// a comment
ifdef::backend-html5[]
= Spring Boot
[[anchor]]
Spring Boot requires Java 17.

== Configuration
include::other.adoc[]
Properties live in application.yaml.
"""


async def test_asciidoc_splits_on_headings_and_drops_build_directives() -> None:
    units = await _units(AsciiDocParseStep(config={}), _document("guide.adoc", _ADOC))

    contents = [unit.content for unit in units]
    assert "Spring Boot requires Java 17." in contents
    assert "Properties live in application.yaml." in contents
    assert not any("include::" in content or "ifdef" in content for content in contents)
    assert units[-1].metadata["section"] == "Configuration"


_ADOC_MACROS = """= API navigation

* Java APIs
** xref:api:java/index.html[Spring Boot,role=link-external, window=_blank]
** xref:gradle-plugin:api/java/index.html[Gradle Plugin,role=link-external, window=_blank]

== Details
[source,xml]
Set configprop:management.server.port[] to move the endpoints.
Define a javadoc:org.springframework.boot.actuate.endpoint.SanitizingFunction[] bean.
See xref:reference:actuator/monitoring.adoc#actuator.monitoring.port[Customizing the Port] for more.
"""


async def test_asciidoc_reduces_a_macro_to_the_words_a_reader_sees() -> None:
    """The presentation attributes never reach the model.

    They did, and it read them: a navigation partial produced eight approved
    assertions whose objects were `link-external`, `_blank` and `index.html`.
    The model was not wrong — it was shown markup and asked what it asserts.
    `parse_html` had always taken an element's text and dropped its attributes;
    this parser did not, and the two disagreed about the same class of input.
    """
    units = await _units(AsciiDocParseStep(config={}), _document("nav.adoc", _ADOC_MACROS))

    body = " ".join(unit.content for unit in units)
    assert "Spring Boot" in body
    assert "Gradle Plugin" in body
    for attribute in ("role=", "window=", "_blank", "link-external", "index.html"):
        assert attribute not in body, f"{attribute} reached the model"


async def test_asciidoc_keeps_the_target_when_a_macro_has_no_text() -> None:
    # `javadoc:` and `configprop:` are written with an empty attribute list
    # precisely because their target is the subject. Dropping it would delete
    # the one identifier the sentence is about.
    units = await _units(AsciiDocParseStep(config={}), _document("nav.adoc", _ADOC_MACROS))

    body = " ".join(unit.content for unit in units)
    assert "management.server.port" in body
    assert "org.springframework.boot.actuate.endpoint.SanitizingFunction" in body


async def test_asciidoc_prefers_the_link_text_over_the_path() -> None:
    # An anchor names a place in a document, and a place is not what a sentence
    # claims. The words the reader sees are.
    units = await _units(AsciiDocParseStep(config={}), _document("nav.adoc", _ADOC_MACROS))

    body = " ".join(unit.content for unit in units)
    assert "Customizing the Port" in body
    assert "monitoring.adoc" not in body


async def test_asciidoc_drops_a_line_that_was_only_markup() -> None:
    # A block attribute configures how the block below renders. After the macros
    # are reduced, a line that held nothing else has no words left in it, and an
    # empty line is not a statement about anything.
    units = await _units(AsciiDocParseStep(config={}), _document("nav.adoc", _ADOC_MACROS))

    assert not any("[source" in unit.content for unit in units)


# -- html --------------------------------------------------------------------

_HTML = """<html><head><style>body{}</style></head>
<body>
<nav>Home | Docs</nav>
<h1>JEP 400</h1>
<p>UTF-8 becomes the default charset.</p>
<footer>Copyright</footer>
</body></html>"""


async def test_html_strips_chrome_and_splits_on_headings() -> None:
    pytest.importorskip("bs4")

    units = await _units(HtmlParseStep(config={}), _document("jep400.html", _HTML))

    joined = " ".join(unit.content for unit in units)
    assert "UTF-8 becomes the default charset." in joined
    # Navigation and footers repeat on every page and would swamp extraction.
    assert "Home | Docs" not in joined
    assert "Copyright" not in joined
    assert units[0].metadata["section"] == "JEP 400"


# -- pdf ---------------------------------------------------------------------


async def test_pdf_without_the_optional_dependency_says_what_to_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nlght.adapters.outbound.workflow.steps.knowledge import parsers

    monkeypatch.setattr(parsers, "_HAS_PDFMINER", False)

    with pytest.raises(WorkflowConfigurationError, match=r"nlght-ai\[pdf\]"):
        await PdfParseStep(config={}).run(_ctx(_document("spec.pdf", "")))


async def test_pdf_without_the_original_bytes_is_skipped_not_fatal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Processing produces no text for a PDF, and the raw bytes live on the
    # snapshot; without it there is nothing to extract, but the run continues.
    from nlght.adapters.outbound.workflow.steps.knowledge import parsers

    monkeypatch.setattr(parsers, "_HAS_PDFMINER", True)

    result = await PdfParseStep(config={}).run(_ctx(_document("spec.pdf", "")))

    assert result.ctx.metadata["knowledge.units"] == ()


# -- shared ------------------------------------------------------------------


@pytest.mark.parametrize(
    "step_cls", [MarkdownParseStep, AsciiDocParseStep, HtmlParseStep, PdfParseStep]
)
async def test_every_parser_requires_processed_documents(step_cls) -> None:
    ctx = _ctx()
    del ctx.metadata["ingestion.processed"]

    with pytest.raises(WorkflowConfigurationError, match="ingestion.process"):
        await step_cls(config={}).run(ctx)


@pytest.mark.parametrize(
    "step_cls", [MarkdownParseStep, AsciiDocParseStep, HtmlParseStep, PdfParseStep]
)
def test_every_parser_declares_its_options(step_cls) -> None:
    names = {option.name for option in step_cls.options()}

    assert names == {"tags", "extensions"}


async def test_parsers_accumulate_so_several_formats_can_share_a_workflow() -> None:
    ctx = _ctx(_document("a.md", "# H\n\nMarkdown claim."), _document("b.adoc", "= H\nAdoc claim."))

    await MarkdownParseStep(config={}).run(ctx)
    await AsciiDocParseStep(config={}).run(ctx)

    contents = [unit.content for unit in ctx.metadata["knowledge.units"]]
    assert contents == ["Markdown claim.", "Adoc claim."]


# -- parse_auto (dispatching step) -------------------------------------------


async def test_parse_auto_routes_each_document_to_its_format_parser() -> None:
    units = await _units(
        KnowledgeParseAutoStep(config={}),
        _document("a.md", "# H\n\nMarkdown claim."),
        _document("b.adoc", "= H\nAdoc claim."),
    )

    contents = [unit.content for unit in units]
    assert contents == ["Markdown claim.", "Adoc claim."]


async def test_parse_auto_falls_back_to_prose_for_unmatched_extensions() -> None:
    units = await _units(
        KnowledgeParseAutoStep(config={}),
        _document("notes.txt", "First claim.\n\nSecond claim."),
    )

    assert [unit.content for unit in units] == ["First claim.", "Second claim."]


async def test_parse_auto_fallback_none_skips_unmatched() -> None:
    units = await _units(
        KnowledgeParseAutoStep(config={"parsers": ["markdown"], "fallback": "none"}),
        _document("a.md", "# H\n\nMarkdown claim."),
        _document("b.adoc", "= H\nAdoc claim."),
    )

    # AsciiDoc is neither enabled nor caught by a fallback, so only the Markdown
    # claim survives.
    assert [unit.content for unit in units] == ["Markdown claim."]


async def test_parse_auto_extension_override_reroutes_a_suffix() -> None:
    units = await _units(
        KnowledgeParseAutoStep(config={"extensions": {".txt": "markdown"}}),
        _document("notes.txt", "# Heading\n\nSomething."),
    )

    # Parsed as Markdown (heading dropped) rather than prose (heading kept).
    assert [unit.content for unit in units] == ["Something."]


async def test_parse_auto_skips_a_document_whose_parser_dependency_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nlght.adapters.outbound.workflow.steps.knowledge import parsers

    monkeypatch.setattr(parsers, "_HAS_BS4", False)

    units = await _units(
        KnowledgeParseAutoStep(config={}),
        _document("page.html", _HTML),
        _document("a.md", "# H\n\nMarkdown claim."),
    )

    # The HTML document is skipped (not fatal); the Markdown one still parses.
    contents = [unit.content for unit in units]
    assert contents == ["Markdown claim."]


async def test_parse_auto_rejects_an_unknown_parser_name() -> None:
    with pytest.raises(WorkflowConfigurationError, match="unknown parser"):
        await KnowledgeParseAutoStep(config={"parsers": ["nope"]}).run(
            _ctx(_document("a.md", "# H\n\nX."))
        )


async def test_parse_auto_requires_processed_documents() -> None:
    ctx = _ctx()
    del ctx.metadata["ingestion.processed"]

    with pytest.raises(WorkflowConfigurationError, match="ingestion.process"):
        await KnowledgeParseAutoStep(config={}).run(ctx)


def test_parse_auto_declares_its_options() -> None:
    names = {option.name for option in KnowledgeParseAutoStep.options()}

    assert names == {"parsers", "extensions", "fallback", "tags"}
