# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""`knowledge.extract` does not ask the model about a document twice.

Without this the model is asked the same question on every run and answers it in
slightly different words each time. Once assertions are updated incrementally
rather than rebuilt, those different words are a retraction and a re-review of
something nobody edited — so the skip is not an optimisation, it is what keeps
identity drift out of normal operation.

Both halves of the key are exercised here, because each fails in its own
direction: content alone skips a document after the model changed, version alone
skips one that was edited.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from tests.extraction_stub import quote_from

from nlght.adapters.outbound.tools.activator import ResourceActivator
from nlght.adapters.outbound.workflow.steps.knowledge import (
    KnowledgeExtractStep,
    KnowledgeParseStep,
)
from nlght.adapters.outbound.workflow.steps.knowledge.extract import UNCHANGED
from nlght.core.entry.context import RequestContext
from nlght.core.ingestion import (
    DocumentClassification,
    Enrichment,
    ProcessedDocument,
)
from nlght.core.knowledge import ExtractionState, ExtractionStateWrite
from nlght.core.protocol.protocol import ProtocolKind
from nlght.core.runtime.resource import ResourceDef
from nlght.core.trigger.trigger import Trigger, TriggerKind
from nlght.core.workflow.step import WorkflowStepContext
from nlght.ports.outbound.model_client import ModelStreamEvent

_FACT = '[{"kind":"fact","type":"dependency","confidence":0.9,' \
        '"observed_text":"{quote}",' \
        '"content":{"subject":"spring","predicate":"requires","object":"java 17"}}]'


class _Emitter:
    async def emit(self, signal: object) -> None: ...


class _Llm:
    """Counts, because "was the model asked" is the whole assertion here."""

    def __init__(self, response: str = _FACT) -> None:
        self.response = response
        self.calls = 0

    def stream(self, messages, tools=None, tool_choice=None, *, temperature=None):  # noqa: ANN001, ANN201
        self.calls += 1
        response = self.response.replace(
            "{quote}", quote_from(str(messages[0]["content"]))
        )

        async def _gen():  # noqa: ANN202
            yield ModelStreamEvent(kind="token", content=response)
            yield ModelStreamEvent(kind="done")

        return _gen()

    async def call(self, messages, *, temperature=None) -> None: ...  # noqa: ANN001

    def append_tool_turn(self, messages, tool_calls_raw, results, assistant_text=""):  # noqa: ANN001, ANN201
        return messages


class _Repo:
    def __init__(self, states: dict[str, ExtractionState] | None = None) -> None:
        self.states = dict(states or {})
        self.recorded: list[ExtractionStateWrite] = []

    async def extraction_state(self, document_id: str) -> ExtractionState | None:
        return self.states.get(document_id)

    async def record_extraction(self, write: ExtractionStateWrite) -> ExtractionState:
        self.recorded.append(write)
        state = ExtractionState(
            document_id=write.document_id,
            content_hash=write.content_hash,
            extraction_version=write.extraction_version,
            extracted_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        self.states[write.document_id] = state
        return state


class _Writer:
    KIND = "knowledge_index_writer"

    def __init__(self, repository: _Repo) -> None:
        self.repository = repository


class _Resources:
    def __init__(self, resources: list[ResourceDef]) -> None:
        self._resources = resources

    async def find_by_kind(self, kind: str) -> list[ResourceDef]:
        return [r for r in self._resources if r.kind == kind]


class _Loader:
    def __init__(self, instance: object) -> None:
        self._instance = instance

    def instantiate(self, *, kind, provider, name, config, **runtime_deps):  # noqa: ANN001, ANN003, ANN201
        return self._instance


def _resource() -> ResourceDef:
    import uuid as _uuid

    return ResourceDef(
        resource_id=_uuid.uuid4(), name="knowledge-index",
        kind="knowledge_index_writer", provider="opensearch", config={},
    )


def _ctx(llm: _Llm) -> WorkflowStepContext:
    context = RequestContext(
        correlation_id="run-1", request_id="rid-1",
        received_at=datetime(2026, 1, 1, tzinfo=UTC),
        path="/", method="POST", headers={}, query_params={}, client_host=None,
        workflow="ingest-knowledge",
    )
    return WorkflowStepContext(
        correlation_id="run-1",
        trigger=Trigger(
            kind=TriggerKind.INBOUND_EVENT, protocol=ProtocolKind.GENERIC_JSON,
            operation="ingest", payload={}, context=context,
        ),
        model="llama3", messages=[], stream=False, emitter=_Emitter(), llm=llm,
    )


def _document(text: str, document_id: str = "doc-1") -> ProcessedDocument:
    return ProcessedDocument(
        document_id=document_id, source_revision_id="src-1", processing_revision_id="rev-1",
        source="wiki", external_id="wiki:1", path="page.md", text=text,
        classification=DocumentClassification(
            kind="text", language="markdown", classifier_version="1"
        ),
        enrichment=Enrichment(enricher="none", version="1"),
        chunks=(), metadata={},
    )


@pytest.fixture(autouse=True)
def _writer_type(monkeypatch: pytest.MonkeyPatch) -> None:
    # The step checks that its resource really is a writer, so the fake has to
    # be what it checks against — the same substitution the persist tests make.
    from nlght.adapters.outbound.workflow.steps.knowledge import extract as extract_module

    monkeypatch.setattr(extract_module, "KnowledgeIndexWriterTool", _Writer)


def _step(repo: _Repo | None, **config: object) -> KnowledgeExtractStep:
    settings: dict[str, Any] = {"model_provider": "ollama", "model": "llama3"}
    settings.update(config)
    if repo is None:
        return KnowledgeExtractStep(config=settings)
    settings["writer"] = "knowledge-index"
    return KnowledgeExtractStep(
        config=settings,
        resource_repository=_Resources([_resource()]),
        tool_loader=_Loader(_Writer(repo)),
        resource_activator=ResourceActivator(resources=_Resources([_resource()]), loader=_Loader(_Writer(repo))),
    )


async def _run(step: KnowledgeExtractStep, llm: _Llm, text: str, document_id: str = "doc-1"):
    ctx = _ctx(llm)
    ctx.metadata["ingestion.processed"] = (_document(text, document_id),)
    await KnowledgeParseStep(config={}).run(ctx)
    return await step.run(ctx)


_TEXT = "Spring requires Java 17."


def _acquired(content: str, source_form: str | None, external_id: str = "wiki:1"):  # noqa: ANN201
    from nlght.core.ingestion import SourceDocument

    return SourceDocument(
        source="wiki",
        external_id=external_id,
        path="page.html",
        content=content.encode("utf-8"),
        source_form=source_form.encode("utf-8") if source_form is not None else None,
    )


async def _run_acquired(step, llm, *, content: str, source_form: str | None):  # noqa: ANN001, ANN201
    """A run where the snapshot is present, so a source form can exist."""
    from nlght.core.ingestion import SourceSnapshot

    ctx = _ctx(llm)
    document = _document(content)
    acquired = _acquired(content, source_form, external_id=document.external_id)
    ctx.metadata["ingestion.snapshot"] = SourceSnapshot(
        source_id="wiki", documents=(acquired,),
        observed_external_ids=(acquired.external_id,), complete=True
    )
    ctx.metadata["ingestion.processed"] = (document,)
    await KnowledgeParseStep(config={}).run(ctx)
    return await step.run(ctx)


async def test_markup_that_changed_is_re_extracted_even_when_the_text_did_not() -> None:
    """The gap the three representations opened, and why the skip had to follow.

    A heading given an `id` changes the markup and not one word of the text. The
    skip hashed the text, so the document was skipped and the new anchor arrived
    only with the next real edit — and slot identity resolves against exactly
    those anchors. Hashing what the *parser* is handed closes it.
    """
    repo = _Repo()
    await _run_acquired(
        _step(repo), _Llm(), content=_TEXT, source_form=f"<h1>Stack</h1><p>{_TEXT}</p>"
    )

    second = _Llm()
    await _run_acquired(
        _step(repo), second, content=_TEXT,
        source_form=f'<h1 id="stack">Stack</h1><p>{_TEXT}</p>',
    )

    assert second.calls == 1, "the markup moved and the parser was not asked again"


async def test_an_unchanged_markup_is_still_skipped() -> None:
    # The other direction: following the parser's input must not make every run
    # re-read everything.
    repo = _Repo()
    markup = f'<h1 id="stack">Stack</h1><p>{_TEXT}</p>'
    await _run_acquired(_step(repo), _Llm(), content=_TEXT, source_form=markup)

    second = _Llm()
    result = await _run_acquired(_step(repo), second, content=_TEXT, source_form=markup)

    assert second.calls == 0
    assert result.verdict == UNCHANGED


async def test_a_document_without_a_source_form_hashes_its_text_as_before() -> None:
    """Most documents, and the reason an existing corpus is not re-extracted.

    A source that hands over what it holds has no separate form, so the answer is
    the one it has always been. Only a source that transforms its documents gets
    a different hash than before.
    """
    repo = _Repo()
    await _run_acquired(_step(repo), _Llm(), content=_TEXT, source_form=None)
    recorded = repo.recorded[0].content_hash

    plain = _Repo()
    await _run(_step(plain), _Llm(), _TEXT)

    assert plain.recorded[0].content_hash == recorded


# ---------------------------------------------------------------------------
# The first run, and the record it leaves
# ---------------------------------------------------------------------------

async def test_a_document_never_extracted_is_extracted_and_recorded() -> None:
    repo, llm = _Repo(), _Llm()

    result = await _run(_step(repo), llm, _TEXT)

    assert llm.calls == 1
    assert result.verdict == "DEFAULT"
    assert [write.document_id for write in repo.recorded] == ["doc-1"]
    assert repo.recorded[0].run_id == "run-1"


async def test_the_second_run_over_an_unchanged_document_does_not_ask_the_model() -> None:
    """The acceptance metric of the design, at the step that costs the most.

    Two consecutive runs over an unchanged source must change nothing. The
    strongest way to hold that is not to run the extraction at all — a model
    that is never asked cannot phrase an answer differently.
    """
    repo = _Repo()
    first_llm = _Llm()
    await _run(_step(repo), first_llm, _TEXT)

    second_llm = _Llm()
    result = await _run(_step(repo), second_llm, _TEXT)

    assert first_llm.calls == 1
    assert second_llm.calls == 0
    assert result.verdict == UNCHANGED


async def test_the_skip_verdict_is_distinct_from_finding_nothing() -> None:
    # `empty` means the model read the text and had nothing to say; `unchanged`
    # means it was not asked. A workflow routes those differently — one is worth
    # an alert, the other is the normal case.
    repo = _Repo()
    await _run(_step(repo), _Llm(), _TEXT)

    result = await _run(_step(repo), _Llm(), _TEXT)

    assert result.verdict == UNCHANGED
    assert result.verdict != "empty"


# ---------------------------------------------------------------------------
# Both halves of the key
# ---------------------------------------------------------------------------

async def test_an_edited_document_is_extracted_again() -> None:
    repo = _Repo()
    await _run(_step(repo), _Llm(), _TEXT)

    llm = _Llm()
    result = await _run(_step(repo), llm, "Spring requires Java 21.")

    assert llm.calls == 1
    assert result.verdict == "DEFAULT"


async def test_a_changed_model_re_extracts_an_unchanged_document() -> None:
    """The half a content-only key would get wrong, and the dangerous one.

    Nothing about the document changed, so a hash of its text still matches.
    But the model did — and a model change is meant to be a deliberate versioned
    event whose consequence actually happens, rather than one that silently
    leaves the corpus as the old model wrote it.
    """
    repo = _Repo()
    await _run(_step(repo, model="llama3"), _Llm(), _TEXT)

    llm = _Llm()
    await _run(_step(repo, model="llama3.1"), llm, _TEXT)

    assert llm.calls == 1


async def test_a_changed_confidence_floor_re_extracts_an_unchanged_document() -> None:
    # The step setting shapes what the extraction yields, so it belongs to the
    # version. Leaving it out would keep candidates the new setting rejects.
    repo = _Repo()
    await _run(_step(repo, min_confidence=0.0), _Llm(), _TEXT)

    llm = _Llm()
    await _run(_step(repo, min_confidence=0.8), llm, _TEXT)

    assert llm.calls == 1


async def test_a_changed_segmentation_re_extracts_an_unchanged_document() -> None:
    """The part that changes what the model sees without changing the text.

    `split_into_units` decides where a document is cut, and the model answers a
    different question about a differently cut block. The bytes are identical,
    so a content hash cannot see it — `SEGMENTATION_VERSION` is what makes a
    change to the splitter re-extract the corpus instead of skipping all of it.
    """
    import nlght.core.knowledge.extraction as extraction_module

    repo = _Repo()
    await _run(_step(repo), _Llm(), _TEXT)

    llm = _Llm()
    with pytest.MonkeyPatch.context() as patch:
        # Patched where the value lives. The step does not name it at all — the
        # core folds it in, which is the point: a factor the step cannot forget.
        patch.setattr(extraction_module, "SEGMENTATION_VERSION", "2")
        await _run(_step(repo), llm, _TEXT)

    assert llm.calls == 1


# ---------------------------------------------------------------------------
# The safe directions
# ---------------------------------------------------------------------------

async def test_without_a_writer_nothing_is_skipped() -> None:
    # No writer means no knowledge database, so there is nowhere to record what
    # was extracted and nothing may be assumed about it. Existing pipelines that
    # never configured one behave exactly as they did.
    llm = _Llm()
    await _run(_step(None), llm, _TEXT)
    second = _Llm()
    await _run(_step(None), second, _TEXT)

    assert llm.calls == 1
    assert second.calls == 1


async def test_units_without_processed_documents_are_never_skipped() -> None:
    """A step that cannot see the text cannot claim it is unchanged.

    Units may reach extraction from somewhere other than `ingestion.process`.
    There is then no content hash to compare, and the absent case has to mean
    "extract", not "assume current".
    """
    repo = _Repo()
    llm = _Llm()
    ctx = _ctx(llm)
    ctx.metadata["ingestion.processed"] = (_document(_TEXT),)
    await KnowledgeParseStep(config={}).run(ctx)
    del ctx.metadata["ingestion.processed"]

    await _step(repo).run(ctx)

    assert llm.calls == 1
    assert repo.recorded == []


async def test_one_document_being_current_does_not_skip_another() -> None:
    repo = _Repo()
    await _run(_step(repo), _Llm(), _TEXT, document_id="doc-1")

    llm = _Llm()
    result = await _run(_step(repo), llm, "Quarkus requires Java 17.", document_id="doc-2")

    assert llm.calls == 1
    assert result.verdict == "DEFAULT"


@pytest.mark.parametrize("batch_size", [1, 2])
async def test_nothing_is_recorded_until_every_unit_has_been_read(batch_size: int) -> None:
    """Recording per batch would mark a document extracted while half of it was not.

    A crash between batches would then skip the remainder for good — the run
    after it sees a current record and asks nothing. This is the ordering
    `ingestion.write` already keeps for the index: do the work, then record it.
    """
    repo, llm = _Repo(), _Llm()
    step = _step(repo, unit_batch_size=batch_size)
    ctx = _ctx(llm)
    ctx.metadata["ingestion.processed"] = (
        _document("First paragraph about Spring.\n\nSecond paragraph about Java.\n"),
    )
    await KnowledgeParseStep(config={}).run(ctx)

    result = await step.run(ctx)
    if result.verdict == "more":
        assert repo.recorded == []
        result = await step.run(result.ctx)

    assert result.verdict == "DEFAULT"
    assert len(repo.recorded) == 1


async def test_the_extraction_version_is_carried_forward_for_persistence() -> None:
    # `knowledge.persist` records it on every revision it writes, so a stored
    # assertion can say which extraction produced it. Recomputing it there would
    # be the second assembly site the whole design avoids.
    repo, llm = _Repo(), _Llm()
    ctx = _ctx(llm)
    ctx.metadata["ingestion.processed"] = (_document(_TEXT),)
    await KnowledgeParseStep(config={}).run(ctx)

    await _step(repo).run(ctx)

    assert ctx.metadata["knowledge.extraction_version"] == repo.recorded[0].extraction_version


async def test_a_fully_skipped_run_still_leaves_the_chain_something_to_read() -> None:
    """The rerun failed here, and the fault was the early return.

    Every step after extraction reads `knowledge.extracted` and refuses to run
    without it — that guard is right, it catches a workflow wired without an
    extract step. But the skip returned before establishing the key, so a second
    run over an unchanged corpus reached `knowledge.canonicalize` with nothing
    in the context and the whole run failed on a configuration error that was
    not one.

    `empty` never had the problem: the model was asked, found nothing, and the
    step still wrote an empty tuple. The skip has to leave the same trace — the
    contract is that this step establishes the key, whether or not it did any
    work.
    """
    repo = _Repo()
    await _run(_step(repo), _Llm(), _TEXT)

    ctx = _ctx(_Llm())
    ctx.metadata["ingestion.processed"] = (_document(_TEXT),)
    await KnowledgeParseStep(config={}).run(ctx)
    result = await _step(repo).run(ctx)

    assert result.verdict == UNCHANGED
    assert result.ctx.metadata["knowledge.extracted"] == ()


async def test_the_refinement_chain_runs_over_a_fully_skipped_run() -> None:
    # The end-to-end shape of the failure: the first step of the chain used to
    # raise here. It has nothing to do and must say so quietly.
    from nlght.adapters.outbound.workflow.registry import step_registry

    repo = _Repo()
    await _run(_step(repo), _Llm(), _TEXT)

    ctx = _ctx(_Llm())
    ctx.metadata["ingestion.processed"] = (_document(_TEXT),)
    await KnowledgeParseStep(config={}).run(ctx)
    await _step(repo).run(ctx)

    canonicalize = step_registry._registry["knowledge.canonicalize"]
    result = await canonicalize(config={}).run(ctx)

    assert result.ctx.metadata["knowledge.extracted"] == ()


async def test_a_document_with_no_units_also_leaves_the_key_behind() -> None:
    # The same trap through the other early return: an empty page, or one whose
    # text the classifier could not read, produces no units — and the chain
    # behind it still expects `knowledge.extracted` to exist.
    from nlght.adapters.outbound.workflow.registry import step_registry

    ctx = _ctx(_Llm())
    ctx.metadata["knowledge.units"] = ()

    result = await _step(None).run(ctx)

    assert result.verdict == "empty"
    assert result.ctx.metadata["knowledge.extracted"] == ()
    canonicalize = step_registry._registry["knowledge.canonicalize"]
    await canonicalize(config={}).run(result.ctx)


@pytest.mark.parametrize("parser", ["knowledge.parse", "knowledge.parse_auto"])
async def test_every_parser_gives_an_assertion_a_document_and_a_revision(parser: str) -> None:
    """The gap that produced a corpus with no lineage at all.

    `document_id` was added to `knowledge.parse` and the pipeline uses
    `knowledge.parse_auto`, which set the document but not its revision — so
    every candidate reached the graph and none reached the lineage. Seventeen
    assertions, zero history, and the counter said only that something was
    missing.

    Parametrised over both parsers because the fault was exactly that the two
    disagreed about what a unit carries. Any further parser has to pass this
    too.
    """
    from nlght.adapters.outbound.workflow.registry import step_registry
    from nlght.adapters.outbound.workflow.steps.knowledge.persist import KnowledgePersistStep

    ctx = _ctx(_Llm())
    ctx.metadata["ingestion.processed"] = (_document(_TEXT),)
    await step_registry._registry[parser](config={}).run(ctx)
    await _step(None).run(ctx)

    item = ctx.metadata["knowledge.extracted"][0]
    lineage, gap = KnowledgePersistStep._lineage(item, "run-1", "ver-1")

    assert gap is None, f"{parser} left a candidate without {gap}"
    assert lineage is not None
    assert (lineage.document_id, lineage.document_revision) == ("doc-1", "rev-1")


async def test_the_funnel_says_where_candidates_were_lost() -> None:
    """Four candidates where a run before produced eighteen is a behaviour change.

    "candidates=4" cannot say whether the model said less, said it wrongly, or
    said it below the floor, and those want three different fixes. Each stage is
    counted instead, so the log answers the question rather than raising it.
    """
    response = """[
      {"kind": "fact", "type": "d", "confidence": 0.9, "observed_text": "{quote}",
       "content": {"predicate": "requires", "subject": "spring", "object": "java"}},
      {"kind": "fact", "type": "d", "confidence": 0.1, "observed_text": "{quote}",
       "content": {"predicate": "requires", "subject": "spring", "object": "maven"}},
      {"kind": "prophecy", "type": "d", "confidence": 0.9, "content": {"a": "b"}},
      {"kind": "fact", "type": "d", "confidence": 0.9, "observed_text": "{quote}", "content": {"subject": "only"}},
      "not an object"
    ]"""
    llm = _Llm(response)
    ctx = _ctx(llm)
    ctx.metadata["ingestion.processed"] = (_document(_TEXT),)
    await KnowledgeParseStep(config={}).run(ctx)

    parsed, tally = await _step(None, min_confidence=0.5)._extract(ctx, "some text")

    assert tally["returned"] == 5
    # One element that is not an object, and one whose content cannot form a fact.
    assert tally["malformed"] == 2
    assert tally["unknown_kind"] == 1
    # Two survive parsing; the confidence floor is applied by the caller, so it
    # is not counted here.
    assert len(parsed) == 2


async def test_the_confidence_floor_is_counted_where_it_is_applied() -> None:
    response = """[
      {"kind": "fact", "type": "d", "confidence": 0.9, "observed_text": "{quote}", "content": {"predicate": "requires", "subject": "spring", "object": "java"}},
      {"kind": "fact", "type": "d", "confidence": 0.1, "observed_text": "{quote}", "content": {"predicate": "requires", "subject": "spring", "object": "maven"}}
    ]"""
    ctx = _ctx(_Llm(response))
    ctx.metadata["ingestion.processed"] = (_document(_TEXT),)
    await KnowledgeParseStep(config={}).run(ctx)

    result = await _step(None, min_confidence=0.5).run(ctx)

    assert len(result.ctx.metadata["knowledge.extracted"]) == 1
