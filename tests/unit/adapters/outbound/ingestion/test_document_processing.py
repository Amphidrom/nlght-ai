# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from dataclasses import replace

import pytest

from nlght.adapters.outbound.ingestion import (
    BuiltinDocumentEnricher,
    PygmentsDocumentClassifier,
)
from nlght.core.ingestion import SourceDocument
from nlght.core.ingestion.processing import ChunkingSettings, IngestionDocumentProcessor


def _processor(*, size: int = 4000, overlap: int = 200) -> IngestionDocumentProcessor:
    return IngestionDocumentProcessor(
        classifier=PygmentsDocumentClassifier(),
        enricher=BuiltinDocumentEnricher(),
        chunking=ChunkingSettings(size=size, overlap=overlap),
    )


@pytest.mark.parametrize(
    ("path", "text", "kind", "language", "enricher"),
    [
        ("src/app.ts", "import {x} from './x'; export class App {}", "code", "typescript", "typescript"),
        ("src/app.tsx", "import X from './x'; export function App() { return <X/>; }", "code", "tsx", "typescript"),
        ("src/app.js", "const x = require('./x'); class App {}", "code", "javascript", "typescript"),
        ("src/app.jsx", "import X from './x'; function App() { return <X/>; }", "code", "jsx", "typescript"),
        ("src/app.py", "import pathlib\nclass App:\n    pass\n", "code", "python", "python"),
        ("src/App.java", "package org.example; import app.Api; public class App {}", "code", "java", "java"),
        ("docs/index.adoc", "= Guide\n== Start\nxref:other.adoc[Other]", "documentation", "asciidoc", "asciidoc"),
        ("api/service.proto", 'package api.v1; import "base.proto"; message Request {}', "code", "protobuf", "protobuf"),
        ("README.md", "# Guide\nSee [other](other.md) and `app.Api`.", "documentation", "markdown", "markdown"),
        ("ui/App.vue", "<template><Other/></template><script>import X from './x'</script>", "code", "vue", "vue"),
        ("pom.xml", "<project><groupId>org.example</groupId></project>", "code", "xml", "xml"),
        ("build.gradle", "implementation 'org.example:api:1'; task verify", "code", "gradle", "gradle"),
        ("build.gradle.kts", 'implementation("org.example:api:1")', "code", "gradle.kts", "gradle"),
        ("data/schema.json", '{"type":"org.example.Api","items":[{"ref":"other.json"}]}', "config", "json", "json"),
        ("config/settings.yaml", "service: app\nenabled: true\n", "config", "yaml", "generic"),
    ],
)
def test_supported_formats_are_classified_and_enriched(
    path: str,
    text: str,
    kind: str,
    language: str,
    enricher: str,
) -> None:
    result = _processor().process(SourceDocument("filesystem", path, path, text.encode()))

    assert result.accepted
    assert result.classification.kind == kind
    assert result.classification.language == language
    assert result.enrichment.enricher == enricher
    assert result.chunks[0].content == text


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("README.rst", "restructuredtext"),
        ("notes.txt", "text"),
        ("settings.toml", "toml"),
        ("settings.ini", "ini"),
        (".env", "dotenv"),
        ("settings.yml", "yaml"),
    ],
)
def test_deterministic_classifier_overrides(path: str, expected: str) -> None:
    result = _processor().process(SourceDocument("filesystem", path, path, b"key=value\n"))
    assert result.classification.language == expected


def test_binary_and_invalid_utf8_are_rejected_with_diagnostics() -> None:
    binary = _processor().process(SourceDocument("filesystem", "binary", "image.bin", b"PNG\0data"))
    invalid = _processor().process(SourceDocument("filesystem", "invalid", "broken.txt", b"valid\xffinvalid"))

    assert not binary.accepted
    assert binary.classification.kind == "binary"
    assert binary.diagnostics == ("binary-content",)
    assert not invalid.accepted
    assert invalid.diagnostics[0].startswith("invalid-utf8:")


def test_identity_revision_and_chunks_are_deterministic() -> None:
    document = SourceDocument(
        "filesystem",
        "repo:guide",
        "guide.txt",
        b"first line\nsecond line\nthird line",
        {"repository": "fixture"},
    )
    processor = _processor(size=18, overlap=4)

    first = processor.process(document)
    repeated = processor.process(document)
    changed = processor.process(replace(document, content=document.content + b"!"))

    assert first == repeated
    assert first.document_id == changed.document_id
    assert first.source_revision_id != changed.source_revision_id
    assert first.processing_revision_id != changed.processing_revision_id
    assert len(first.chunks) > 1
    assert len({chunk.chunk_id for chunk in first.chunks}) == len(first.chunks)
    assert [chunk.position for chunk in first.chunks] == list(range(len(first.chunks)))
    assert first.metadata == {"repository": "fixture"}


def test_processing_revision_tracks_path_and_chunking_configuration() -> None:
    document = SourceDocument("filesystem", "repo:guide", "guide.txt", b"same content")

    baseline = _processor(size=18, overlap=4).process(document)
    moved = _processor(size=18, overlap=4).process(replace(document, path="docs/guide.txt"))
    rechunked = _processor(size=20, overlap=4).process(document)

    assert baseline.source_revision_id == moved.source_revision_id
    assert baseline.source_revision_id == rechunked.source_revision_id
    assert baseline.processing_revision_id != moved.processing_revision_id
    assert baseline.processing_revision_id != rechunked.processing_revision_id


def test_source_revision_participates_in_source_and_processing_identity() -> None:
    first = SourceDocument(
        "wiki",
        "confluence:10",
        "ENG/10.html",
        b"unchanged text",
        source_revision="7",
    )
    next_version = replace(first, source_revision="8")

    assert first.document_id == next_version.document_id
    assert first.source_revision_id != next_version.source_revision_id
    assert (
        _processor().process(first).processing_revision_id
        != _processor().process(next_version).processing_revision_id
    )


@pytest.mark.parametrize(
    "settings",
    [ChunkingSettings, lambda: ChunkingSettings(size=0), lambda: ChunkingSettings(size=4, overlap=4)],
)
def test_invalid_chunking_settings(settings) -> None:
    if settings is ChunkingSettings:
        assert settings().size == 4000
    else:
        with pytest.raises(ValueError):
            settings()
