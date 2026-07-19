# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nlght.adapters.outbound.signals.buffering import BufferingSignalEmitter
from nlght.application.hive_mind.artifact_extractor import (
    ArtifactExtractor,
    _blocks_from_llm,
    _parse_fence_header,
    _parse_fenced_blocks,
    _parse_llm_json,
)
from nlght.application.hive_mind.artifact_ref_evaluator import (
    evaluate_artifact_refs,
    stream_response,
)
from nlght.application.hive_mind.attachment_observer import (
    DetectedAttachment,
    ingest_attachments,
    scan_attachments,
)
from nlght.application.hive_mind.entity_extractor import EntityExtractor, _parse_json
from nlght.application.hive_mind.layered_observe_extractor import (
    LayeredObserveExtractor,
    RuleBasedIntentClassifier,
    _Deduplicator,
    _extract_fenced_blocks,
    _extract_label_from_prefix,
    _label_slug,
    _ner_entities_from_text,
)
from nlght.application.hive_mind.memory_ingestion import (
    MemoryIngestionPipeline,
    MemoryTransaction,
    _parse_artifact_ref,
    _SessionDedup,
)
from nlght.application.hive_mind.pipeline_extractor import PipelineExtractor
from nlght.core.hive_mind.extraction import (
    ExtractedArtifact,
    ExtractedEntity,
    ExtractionResult,
    MemoryCandidate,
    MemoryCandidateConfidence,
    MemoryCandidateKind,
)


class _StreamingClient:
    def __init__(self, *chunks: str) -> None:
        self.chunks = chunks
        self.messages: list[dict[str, str]] | None = None

    async def stream(self, messages, **_):
        self.messages = messages
        for chunk in self.chunks:
            yield SimpleNamespace(kind="token", content=chunk)
        yield SimpleNamespace(kind="done", content="")


def _candidate(
    key: str,
    content: str,
    *,
    confidence: MemoryCandidateConfidence = MemoryCandidateConfidence.VALIDATED,
    kind: MemoryCandidateKind = MemoryCandidateKind.FACT,
    entities: list[str] | None = None,
) -> MemoryCandidate:
    return MemoryCandidate(
        key=key,
        kind=kind,
        content=content,
        entities=entities or [],
        tags=["test"],
        confidence=confidence,
    )


def test_layered_structural_helpers_reject_requests_and_normalize_labels() -> None:
    assert _extract_label_from_prefix("") == ""
    assert _extract_label_from_prefix("Vergleiche diese Dateien:") == ""
    assert _extract_label_from_prefix("Hier ist Mein Service:") == "Mein Service"
    assert _label_slug(" A fancy / file! ") == "a_fancy_file"
    assert _label_slug("!!!") == "artifact"

    blocks = _extract_fenced_blocks(
        "Hier ist App:\n```PYTHON\nclass App: pass\n```\n"
        "Hier ist Empty:\n```py\n```\n"
        "Hier ist Notes:\n```unknown\nhello\n```"
    )
    assert [(b.label, b.language) for b in blocks] == [("App", "python"), ("Notes", "text")]


def test_rule_classifier_and_ner_filtering() -> None:
    classifier = RuleBasedIntentClassifier()
    assert classifier.classify("compare these") == "compare"
    assert classifier.classify("recall this") == "recall"
    assert classifier.classify("save this") == "store"
    assert classifier.has_uncertainty("maybe later")

    doc = SimpleNamespace(ents=[
        SimpleNamespace(text="Berlin", label_="LOC"),
        SimpleNamespace(text="berlin", label_="LOC"),
        SimpleNamespace(text="if", label_="OTHER"),
        SimpleNamespace(text="", label_=""),
    ])
    assert [entity.text for entity in _ner_entities_from_text("places", lambda _: doc)] == ["Berlin"]
    assert _ner_entities_from_text(" ", lambda _: doc) == []


async def test_layered_extractor_builds_artifacts_relations_requests_and_deduplicates() -> None:
    extractor = LayeredObserveExtractor(use_ner=False)
    result = await extractor.extract(
        "Hier ist Service A:\n```python\nclass Worker:\n    async def run(self): pass\n```\n"
        "Vergleiche Service A mit der gespeicherten Version"
    )

    assert result.intent == "compare"
    assert result.artifacts[0].label == "Service A"
    assert {relation.object for relation in result.relations} == {"Worker", "run"}
    assert result.referenced_labels == ["Service A"]
    assert result.candidates[-1].key == "comparison.request"

    invalid = ExtractedArtifact(label="bad", content="def (", content_type="code", language="python")
    other = ExtractedArtifact(label="js", content="function x() {}", content_type="code", language="js")
    assert extractor._python_symbols(invalid) == []
    assert extractor._python_symbols(other) == []

    dedup = _Deduplicator()
    candidate = _candidate("one", "same")
    assert dedup.filter([candidate, candidate]) == [candidate]


async def test_layered_extractor_recall_and_optional_ner_failure(monkeypatch) -> None:
    def fail_load(_):
        raise ImportError("optional dependency unavailable")

    monkeypatch.setattr("nlght.application.hive_mind.layered_observe_extractor._load_spacy", fail_load)
    result = await LayeredObserveExtractor().extract("Recall Berlin")
    assert result.intent == "recall"
    assert result.candidates[0].key == "recall.request"


def test_artifact_parser_handles_headers_labels_dedup_and_invalid_json() -> None:
    assert _parse_fence_header('python "app.py" trailing') == ("python", "app.py")
    assert _parse_fence_header('"notes.md"') == ("", "notes.md")
    assert _parse_fence_header('python "unterminated') == ('python "unterminated', "")

    blocks = _parse_fenced_blocks(
        "Hier ist app.py:\n```python\nprint('x')\n```\n"
        "```json \"data.json\"\n{}\n```\n```py\n```"
    )
    assert [(block.label, block.language) for block in blocks] == [("app.py", "python"), ("data.json", "json")]
    assert _parse_llm_json('```json\n{"artifacts": []}\n```') == {"artifacts": []}
    assert _parse_llm_json("[]") == {"artifacts": []}
    assert _parse_llm_json("bad") == {"artifacts": []}
    assert [block.label for block in _blocks_from_llm({"artifacts": [
        {"label": " A ", "content": "one"},
        {"label": "a", "content": "duplicate"},
        {"label": "", "content": "ignored"},
    ]})] == ["A"]


async def test_artifact_extractor_prefers_structural_then_uses_llm_fallback() -> None:
    unused = _StreamingClient("should not be called")
    structural = await ArtifactExtractor(client=unused).extract("File:\n```python\nprint(1)\n```")
    assert structural.artifacts[0].label == "File"
    assert unused.messages is None

    client = _StreamingClient('{"artifacts":[{"label":"raw.py",', '"content":"print(2)"}]}')
    fallback = await ArtifactExtractor(client=client).extract("raw paste")
    assert fallback.artifacts[0].label == "raw.py"
    assert client.messages[0]["role"] == "system"

    empty = await ArtifactExtractor().extract("ordinary prose")
    assert empty.artifacts == []


async def test_entity_extractor_parses_streamed_json_and_deduplicates() -> None:
    client = _StreamingClient(
        '```json\n{"entities":[{"name":"ServiceA","kind":"class"},',
        '{"name":"servicea","kind":"class"},{"name":"","kind":"tool"}]}\n```',
    )
    result = await EntityExtractor(client=client).extract("ServiceA")
    assert [entity.text for entity in result.entities] == ["ServiceA"]
    assert result.candidates[0].key == "entity.servicea"
    assert _parse_json("not json") == {"entities": []}
    assert _parse_json("[]") == {"entities": []}


async def test_pipeline_extractor_merges_in_order_and_deduplicates() -> None:
    entity = ExtractedEntity("Service", canonical="service")
    artifact = ExtractedArtifact("app.py", "pass")
    first = SimpleNamespace(extract=MagicMock())
    second = SimpleNamespace(extract=MagicMock())

    async def first_extract(_):
        return ExtractionResult("compare", entities=[entity], candidates=[_candidate("same", "first")], referenced_labels=["A"])

    async def second_extract(_):
        return ExtractionResult(
            "recall",
            entities=[entity],
            artifacts=[artifact],
            candidates=[_candidate("same", "second"), _candidate("other", "other")],
            referenced_labels=["A", "B"],
        )

    first.extract = first_extract
    second.extract = second_extract
    result = await PipelineExtractor([first, second]).extract("input")
    assert result.intent == "compare"
    assert len(result.entities) == 1
    assert result.artifacts == [artifact]
    assert [candidate.content for candidate in result.candidates] == ["first", "other"]
    assert result.referenced_labels == ["A", "B"]


def test_attachment_scanning_uses_last_file_and_ignores_unrelated_messages() -> None:
    attachments = scan_attachments([
        {"role": "assistant", "content": 'Attached file\n```txt "ignore.txt"\nx\n```'},
        {"role": "user", "content": 'ordinary\n```txt "ignore.txt"\nx\n```'},
        {"role": "system", "content": 'Attached documents\n```py "app.py"\nold\n```'},
        {"role": "user", "content": 'Attached file\n```py "app.py"\nnew\n```'},
    ])
    assert [(item.filename, item.content, item.extension) for item in attachments] == [("app.py", "new", ".py")]
    assert len(attachments[0].content_hash) == 16


def test_attachment_ingestion_stores_changes_and_skips_identical(tmp_path) -> None:
    attachment = DetectedAttachment("notes.md", "hello")
    store = MagicMock()
    store.get_payload.side_effect = [None, SimpleNamespace(tags=[f"content_hash:{attachment.content_hash}"])]
    workspace = SimpleNamespace(root_path=tmp_path)

    assert ingest_attachments([attachment], store=store, turn_nr=4, workspace=workspace) == ["notes.md"]
    assert ingest_attachments([attachment], store=store, turn_nr=5) == []
    promoted = store.promote_payload.call_args.kwargs
    assert promoted["entities"] == ["notes.md"]
    assert "type=md" in promoted["content"]
    assert list((tmp_path / "memory_artifacts").iterdir())[0].read_text() == "hello"


@pytest.mark.parametrize("artifact_id", ["not-a-uuid", "123"])
async def test_artifact_ref_evaluation_rejects_invalid_ids(artifact_id: str) -> None:
    store = MagicMock()
    text = await evaluate_artifact_refs(f"See @artifact-ref({artifact_id})", store)
    assert text == f"See [invalid artifact ref: {artifact_id}]"
    store.load_artifact.assert_not_called()


async def test_artifact_ref_evaluation_resolves_and_reports_missing() -> None:
    known = "12345678-1234-1234-1234-123456789abc"
    missing = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    store = MagicMock()
    store.load_artifact.side_effect = [SimpleNamespace(content="print(1)\n", extension=".py"), None]
    text = await evaluate_artifact_refs(f"A @artifact-ref({known}) B @artifact-ref({missing})", store)
    assert "```python\nprint(1)\n```" in text
    assert f"[artifact not found: {missing}]" in text
    assert await evaluate_artifact_refs("plain", store) == "plain"


async def test_artifact_ref_stream_emits_regular_resolved_missing_and_invalid_tokens() -> None:
    known = "12345678-1234-1234-1234-123456789abc"
    missing = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
    store = MagicMock()
    store.load_artifact.side_effect = [SimpleNamespace(content="x = 1", extension=".py"), None]
    emitter = BufferingSignalEmitter()
    resolved = await stream_response(
        f"before @artifact-ref({known}) middle @artifact-ref({missing}) end @artifact-ref(bad)",
        store,
        emitter,
    )
    assert "```python\nx = 1\n```" in resolved
    assert "artifact not found" in resolved
    assert "invalid artifact ref" in resolved
    assert all(signal.kind == "token" for signal in emitter.collected())


def test_memory_transaction_report_and_dedup_indexing() -> None:
    committed = _candidate("one", "known", entities=["A", "B"])
    discarded = _candidate("two", "maybe", confidence=MemoryCandidateConfidence.UNCERTAIN, entities=["B"])
    tx = MemoryTransaction(
        extraction=ExtractionResult("compare", candidates=[committed, discarded], referenced_labels=["A"]),
        committed=[committed],
        discarded=[discarded],
        promoted_atom_count=2,
        stored_result_count=1,
    )
    assert tx.committed_entities() == ["A", "B"]
    assert "Intent: compare" in tx.summary()

    store = MagicMock()
    store.get_context.return_value = {"known_results": [
        SimpleNamespace(content="known"),
        SimpleNamespace(content="[memory_artifact] id=1 | label=app.py | type=code"),
        SimpleNamespace(content=""),
    ]}
    dedup = _SessionDedup(store)
    assert dedup.is_duplicate("known")
    assert dedup.is_duplicate("[memory_artifact] id=2 | label=app.py | type=code")
    assert not dedup.is_duplicate("new")
    dedup.register("new")
    assert dedup.is_duplicate("new")
    assert _parse_artifact_ref("[memory_artifact] malformed | label=x") == {"label": "x"}


async def test_memory_ingestion_commits_validated_discards_uncertain_and_writes_artifact(tmp_path) -> None:
    artifact_candidate = _candidate(
        "artifact.app",
        "large body",
        kind=MemoryCandidateKind.ARTIFACT,
        entities=["app.py"],
    )
    uncertain = _candidate("maybe", "uncertain", confidence=MemoryCandidateConfidence.UNCERTAIN)
    extraction = ExtractionResult(
        "store",
        artifacts=[ExtractedArtifact("app.py", "print(1)", "code")],
        candidates=[artifact_candidate, uncertain],
    )
    extractor = SimpleNamespace(extract=MagicMock())

    async def extract(_):
        return extraction

    extractor.extract = extract
    store = MagicMock()
    store.get_context.return_value = {"known_results": []}
    store.close_branch.return_value = [object(), object()]
    store.store_promoted_atoms.return_value = [object()]

    tx = await MemoryIngestionPipeline(extractor).ingest(
        store=store,
        correlation_id="cid",
        user_text="input",
        turn_nr=3,
        workspace=SimpleNamespace(root_path=tmp_path),
    )
    assert tx.committed == [artifact_candidate]
    assert tx.discarded == [uncertain]
    assert tx.promoted_atom_count == 2
    assert tx.stored_result_count == 1
    assert any("[memory_artifact]" in call.args[0].content for call in store.write.call_args_list)
    assert list((tmp_path / "memory_artifacts").iterdir())


async def test_memory_ingestion_rolls_back_outer_branch_on_failure() -> None:
    extractor = SimpleNamespace(extract=MagicMock())

    async def extract(_):
        return ExtractionResult("store", candidates=[_candidate("x", "x")])

    extractor.extract = extract
    store = MagicMock()
    store.get_context.return_value = {"known_results": []}
    store.write.side_effect = RuntimeError("store failed")

    with pytest.raises(RuntimeError, match="store failed"):
        await MemoryIngestionPipeline(extractor).ingest(
            store=store,
            correlation_id="cid",
            user_text="input",
            turn_nr=1,
        )
    store.close_branch.assert_called_with(promote=False)
