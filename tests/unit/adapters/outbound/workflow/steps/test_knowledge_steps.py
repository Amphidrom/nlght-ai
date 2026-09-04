# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Knowledge steps: unit splitting, extraction robustness, quarantine, persistence."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from tests.extraction_stub import quote_from

from nlght.adapters.outbound.tools.activator import ResourceActivator
from nlght.adapters.outbound.workflow.steps.knowledge import (
    KnowledgeExtractStep,
    KnowledgeParseStep,
    KnowledgePersistStep,
    KnowledgeReviewFlagStep,
)
from nlght.core.entry.context import RequestContext
from nlght.core.errors.errors import WorkflowConfigurationError
from nlght.core.ingestion import (
    DocumentClassification,
    Enrichment,
    ProcessedDocument,
)
from nlght.core.knowledge import ExtractedItem, split_into_units
from nlght.core.protocol.protocol import ProtocolKind
from nlght.core.trigger.trigger import Trigger, TriggerKind
from nlght.core.workflow.step import WorkflowStepContext
from nlght.ports.outbound.model_client import ModelStreamEvent


class _Emitter:
    async def emit(self, signal: object) -> None: ...


class _Llm:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def stream(self, messages, tools=None, tool_choice=None, *, temperature=None):
        self.prompts.append(messages[0]["content"])
        response = self.response.replace(
            "{quote}", quote_from(str(messages[0]["content"]))
        )

        async def _gen():
            yield ModelStreamEvent(kind="token", content=response)
            yield ModelStreamEvent(kind="done")

        return _gen()

    async def call(self, messages, *, temperature=None) -> None: ...

    def append_tool_turn(self, messages, tool_calls_raw, results, assistant_text=""):
        return messages


def _ctx(llm: _Llm | None = None) -> WorkflowStepContext:
    context = RequestContext(
        correlation_id="run-1",
        request_id="rid-1",
        received_at=datetime(2026, 1, 1, tzinfo=UTC),
        path="/",
        method="POST",
        headers={},
        query_params={},
        client_host=None,
    )
    return WorkflowStepContext(
        correlation_id="run-1",
        trigger=Trigger(
            kind=TriggerKind.INBOUND_EVENT,
            protocol=ProtocolKind.GENERIC_JSON,
            operation="ingest",
            payload={},
            context=context,
        ),
        model="",
        messages=[],
        stream=False,
        emitter=_Emitter(),
        llm=llm,
    )


def _document(text: str) -> ProcessedDocument:
    return ProcessedDocument(
        document_id="doc-1",
        source_revision_id="src-1",
        processing_revision_id="rev-1",
        source="wiki",
        external_id="wiki:1",
        path="page.md",
        text=text,
        classification=DocumentClassification(
            kind="text", language="markdown", classifier_version="1"
        ),
        enrichment=Enrichment(enricher="none", version="1"),
        chunks=(),
        metadata={},
    )


# -- parsing -----------------------------------------------------------------


def test_blank_lines_and_headings_separate_units() -> None:
    units = split_into_units(
        "SECTION ONE\nSpring requires Java 17.\n\nNever log secrets.\n",
        source_id="wiki",
        document_id="doc-1",
    )

    assert [unit.content for unit in units] == [
        "Spring requires Java 17.",
        "Never log secrets.",
    ]
    assert [unit.unit_ordinal for unit in units] == ["wiki:doc-1:0", "wiki:doc-1:1"]


def test_unstructured_text_still_yields_one_unit() -> None:
    units = split_into_units("one single line", source_id="s", document_id="d")

    assert len(units) == 1


async def test_parsing_attaches_configured_tags() -> None:
    ctx = _ctx()
    ctx.metadata["ingestion.processed"] = (_document("Spring requires Java 17.\n"),)

    result = await KnowledgeParseStep(
        config={"tags": {"product": "spring", "version": "3.2"}}
    ).run(ctx)

    units = result.ctx.metadata["knowledge.units"]
    assert units[0].tags == {"product": "spring", "version": "3.2"}


async def test_parsing_without_processed_documents_names_the_missing_step() -> None:
    with pytest.raises(WorkflowConfigurationError, match="ingestion.process"):
        await KnowledgeParseStep(config={}).run(_ctx())


# -- extraction --------------------------------------------------------------

_VALID = """[
  {"kind": "fact", "type": "dependency", "confidence": 0.9,
   "observed_text": "{quote}",
   "content": {"subject": "Spring", "predicate": "requires", "object": "Java 17"}}
]"""


async def _extracted(response: str, **config) -> tuple:
    ctx = _ctx(_Llm(response))
    ctx.metadata["ingestion.processed"] = (_document("Spring requires Java 17.\n"),)
    await KnowledgeParseStep(config={}).run(ctx)
    result = await KnowledgeExtractStep(config=config).run(ctx)
    return result.ctx.metadata["knowledge.extracted"]


async def test_a_well_formed_response_becomes_a_candidate() -> None:
    items = await _extracted(_VALID)

    assert len(items) == 1
    assert items[0].kind == "fact"
    assert items[0].evidence["unit_ordinal"] == "wiki:doc-1:0"
    assert items[0].evidence["excerpt"].startswith("Spring requires")


async def test_prose_around_the_json_is_tolerated() -> None:
    items = await _extracted(f"Sure, here you go:\n{_VALID}\nHope that helps!")

    assert len(items) == 1


async def test_an_unparsable_response_yields_nothing_rather_than_guessing() -> None:
    assert await _extracted("I could not find anything useful.") == ()


async def test_one_malformed_element_does_not_lose_the_valid_ones() -> None:
    response = """[
      {"kind": "fact", "type": "x", "confidence": 0.9,
       "observed_text": "{quote}", "content": {"subject": "only"}},
      {"kind": "rule", "type": "security", "confidence": 0.8,
       "observed_text": "{quote}",
       "content": {"rule_text": "Never log secrets"}}
    ]"""

    items = await _extracted(response)

    assert [item.kind for item in items] == ["rule"]


async def test_an_unknown_kind_is_dropped() -> None:
    response = '[{"kind": "prophecy", "type": "x", "confidence": 0.9, "content": {}}]'

    assert await _extracted(response) == ()


async def test_low_confidence_candidates_can_be_filtered_out() -> None:
    response = """[
      {"kind": "rule", "type": "s", "confidence": 0.2,
       "observed_text": "{quote}", "content": {"rule_text": "weak"}},
      {"kind": "rule", "type": "s", "confidence": 0.9,
       "observed_text": "{quote}", "content": {"rule_text": "strong"}}
    ]"""

    items = await _extracted(response, min_confidence=0.5)

    assert [item.content["rule_text"] for item in items] == ["strong"]


async def test_extraction_processes_one_unit_per_invocation_by_default() -> None:
    llm = _Llm(_VALID)
    ctx = _ctx(llm)
    ctx.metadata["ingestion.processed"] = (
        _document("Spring requires Java 17.\n\nNever log secrets.\n"),
    )
    await KnowledgeParseStep(config={}).run(ctx)
    step = KnowledgeExtractStep(config={})

    first = await step.run(ctx)

    assert first.verdict == "more"
    assert len(first.ctx.metadata["knowledge.extracted"]) == 1
    assert first.ctx.metadata["knowledge.extract.next_unit"] == 1
    assert len(llm.prompts) == 1

    second = await step.run(first.ctx)

    assert second.verdict == "DEFAULT"
    assert len(second.ctx.metadata["knowledge.extracted"]) == 2
    assert second.ctx.metadata["knowledge.extract.next_unit"] == 2
    assert [item.evidence["unit_ordinal"] for item in second.ctx.metadata["knowledge.extracted"]] == [
        "wiki:doc-1:0",
        "wiki:doc-1:1",
    ]
    assert len(llm.prompts) == 2


async def test_extraction_batch_size_limits_each_invocation() -> None:
    llm = _Llm(_VALID)
    ctx = _ctx(llm)
    ctx.metadata["ingestion.processed"] = (
        _document("one\n\ntwo\n\nthree\n"),
    )
    await KnowledgeParseStep(config={}).run(ctx)
    step = KnowledgeExtractStep(config={"unit_batch_size": 2})

    first = await step.run(ctx)

    assert first.verdict == "more"
    assert len(first.ctx.metadata["knowledge.extracted"]) == 2
    assert first.ctx.metadata["knowledge.extract.next_unit"] == 2
    assert len(llm.prompts) == 2


async def test_extraction_rejects_invalid_unit_batch_size() -> None:
    ctx = _ctx(_Llm(_VALID))
    ctx.metadata["knowledge.units"] = ()

    with pytest.raises(WorkflowConfigurationError, match="unit_batch_size"):
        await KnowledgeExtractStep(config={"unit_batch_size": 0}).run(ctx)


async def test_extraction_without_a_model_is_a_configuration_error() -> None:
    ctx = _ctx(None)
    ctx.metadata["knowledge.units"] = ()

    with pytest.raises(WorkflowConfigurationError, match="requires a model"):
        await KnowledgeExtractStep(config={}).run(ctx)


def test_the_same_claim_always_derives_the_same_identity() -> None:
    def item(subject: str) -> ExtractedItem:
        return ExtractedItem(
            kind="fact",
            type="dependency",
            content={"subject": subject, "predicate": "requires", "object": "Java 17"},
            confidence=0.9,
            evidence={"source_id": "a"},
        )

    # Identity is content-derived, so the same claim from another source
    # reinforces one node instead of duplicating it.
    assert item("Spring").fingerprint == item("Spring").fingerprint
    assert item("Spring").fingerprint != item("Quarkus").fingerprint


# -- review classification ---------------------------------------------------


#: What extraction always attaches, and what a fact now needs to be stored at
#: all: a claim keeps its structure on the revision a sighting produces, and a
#: sighting with no document produces none. The fixture used to omit it, which
#: made these tests depend on a candidate the pipeline cannot actually emit.
_EVIDENCE = {
    "source_id": "corpus",
    "document_id": "doc-1",
    "processing_revision_id": "rev-1",
    "unit_ordinal": "unit-1",
}


def _items() -> tuple[ExtractedItem, ...]:
    return (
        ExtractedItem(kind="rule", type="security", content={"rule_text": "a"},
                      confidence=0.9, observed_text="Secrets are never logged.",
                      evidence=dict(_EVIDENCE)),
        ExtractedItem(kind="fact", type="dependency",
                      content={"subject": "s", "predicate": "p", "object": "o"},
                      confidence=0.3, observed_text="s p o.",
                      evidence=dict(_EVIDENCE)),
    )


async def _classified(**config):
    ctx = _ctx()
    ctx.metadata["knowledge.extracted"] = _items()
    result = await KnowledgeReviewFlagStep(config=config).run(ctx)
    return result.ctx.metadata["knowledge.classified"]


async def test_the_default_policy_quarantines_everything() -> None:
    classified = await _classified()

    assert all(reason for _, reason in classified)


async def test_thresholds_quarantine_only_what_matches() -> None:
    classified = await _classified(policy="thresholds", min_confidence=0.5)

    reasons = {item.kind: reason for item, reason in classified}
    assert reasons["rule"] is None
    assert "below threshold" in reasons["fact"]


async def test_a_kind_or_type_can_always_require_review() -> None:
    by_kind = await _classified(policy="thresholds", kinds=["rule"])
    assert {item.kind: bool(reason) for item, reason in by_kind}["rule"] is True

    by_type = await _classified(policy="thresholds", types=["dependency"])
    assert {item.type: bool(reason) for item, reason in by_type}["dependency"] is True


async def test_an_unknown_policy_is_refused() -> None:
    with pytest.raises(WorkflowConfigurationError, match="unknown policy"):
        await _classified(policy="vibes")


# -- persistence -------------------------------------------------------------


class _Repo:
    def __init__(self) -> None:
        self.writes = []
        self.lineage = []

    async def upsert(self, write):
        self.writes.append(write)
        return write

    async def persist(self, write, lineage=None):
        # The graph and the lineage go together now, so the fake records both
        # and hands back the pair the step expects.
        self.lineage.append(lineage)
        return await self.upsert(write), None

    async def resolve(self, identities, *, include_quarantined=False): return []
    async def pending_review(self, *, limit=50): return []
    async def record_review(self, **kwargs): ...



def _persist_step(monkeypatch, repo, **config):
    """A persist step reaching its knowledge database the only way there is.

    Through the writer activation. There is no injected repository any more: a
    connection reached past the activation would let one platform-wide URL
    decide where every pipeline writes.
    """
    from nlght.adapters.outbound.workflow.steps.knowledge import persist as persist_module

    monkeypatch.setattr(persist_module, "KnowledgeIndexWriterTool", _IndexWriter)
    writer = _IndexWriter(repository=repo)
    step = KnowledgePersistStep(
        config={"writer": "knowledge-index", **config},
        resource_repository=_Resources([_resource()]),
        tool_loader=_Loader(writer),
        resource_activator=ResourceActivator(resources=_Resources([_resource()]), loader=_Loader(writer)),
    )
    return step, writer


async def test_persisting_carries_the_review_decision_into_the_graph(monkeypatch) -> None:
    ctx = _ctx()
    ctx.metadata["knowledge.extracted"] = _items()
    await KnowledgeReviewFlagStep(config={"policy": "thresholds", "min_confidence": 0.5}).run(ctx)
    repo = _Repo()

    step, _ = _persist_step(monkeypatch, repo)
    await step.run(ctx)

    flags = {write.payload.__class__.__name__: write.review_required for write in repo.writes}
    assert flags["RulePayload"] is False
    assert flags["FactPayload"] is True


async def test_skipping_classification_quarantines_everything(monkeypatch) -> None:
    # Persisting unreviewed assertions as retrievable would defeat the review
    # boundary, so the fallback must be the safe direction.
    ctx = _ctx()
    ctx.metadata["knowledge.extracted"] = _items()
    repo = _Repo()

    step, _ = _persist_step(monkeypatch, repo)
    await step.run(ctx)

    assert all(write.review_required for write in repo.writes)
    assert all("no review classification" in write.review_reason for write in repo.writes)


async def test_persisting_without_a_knowledge_database_is_a_configuration_error() -> None:
    ctx = _ctx()
    ctx.metadata["knowledge.extracted"] = _items()

    with pytest.raises(WorkflowConfigurationError, match="requires a knowledge database"):
        await KnowledgePersistStep(config={}).run(ctx)


async def test_persisting_without_candidates_names_the_missing_step(monkeypatch) -> None:
    step, _ = _persist_step(monkeypatch, _Repo())
    with pytest.raises(WorkflowConfigurationError, match="knowledge.extract"):
        await step.run(_ctx())


# -- lexical indexing --------------------------------------------------------


class _Resources:
    def __init__(self, resources=()) -> None:
        self._resources = list(resources)

    async def find_by_kind(self, kind):
        return [r for r in self._resources if r.kind == kind]

    async def find_by_id(self, resource_id): return None
    async def list_enabled(self): return list(self._resources)


class _IndexWriter:
    KIND = "knowledge_index_writer"

    def __init__(self, repository=None) -> None:
        self.ready = False
        self.indexed: list[str] = []
        self.removed: list[str] = []
        # The writer owns the knowledge database on the write side.
        self.repository = repository

    def ensure_ready(self) -> None:
        self.ready = True

    async def index(self, item, *, keywords=None) -> None:
        self.indexed.append(item.identity)

    async def remove(self, *, identity, kind) -> None:
        # Persisting a quarantined assertion withdraws it from the index rather
        # than skipping it: re-ingestion can quarantine something a reviewer had
        # already approved, and it has to leave the index at that moment.
        self.removed.append(identity)


class _Loader:
    def __init__(self, instance) -> None:
        self._instance = instance

    def instantiate(self, *, kind, provider, name, config, **runtime_deps):
        # The real loader forwards whatever the caller hands it — store
        # connections today, something else tomorrow. Absorbing them keeps this
        # fake from breaking every time a step gains a dependency.
        self.runtime_deps = runtime_deps
        return self._instance


def _resource(kind="knowledge_index_writer", name="knowledge-index"):
    import uuid as _uuid

    from nlght.core.runtime.resource import ResourceDef

    return ResourceDef(
        resource_id=_uuid.uuid4(), name=name, kind=kind, provider="opensearch", config={}
    )


class _StoringRepo(_Repo):
    async def upsert(self, write):
        from nlght.core.knowledge import KnowledgeObject

        self.writes.append(write)
        now = datetime(2026, 1, 1, tzinfo=UTC)
        return KnowledgeObject(
            identity=write.identity, kind=write.kind, type=write.type,
            confidence=write.confidence, payload={}, review_required=write.review_required,
            review_reason=write.review_reason, product=None, version=None,
            metadata={}, created_at=now, updated_at=now,
        )


async def test_persisting_indexes_assertions_for_lexical_search(monkeypatch) -> None:
    from nlght.adapters.outbound.workflow.steps.knowledge import persist as persist_module

    ctx = _ctx()
    ctx.metadata["knowledge.extracted"] = _items()
    # Only approved assertions are indexed, so something has to approve them.
    # Without a classification step the persist step quarantines everything,
    # which is the safe direction but would make this test about nothing.
    await KnowledgeReviewFlagStep(
        config={"policy": "thresholds", "min_confidence": 0.1}
    ).run(ctx)
    repo = _StoringRepo()
    writer = _IndexWriter(repository=repo)
    monkeypatch.setattr(persist_module, "KnowledgeIndexWriterTool", _IndexWriter)

    await KnowledgePersistStep(
        config={"writer": "knowledge-index"},
        resource_repository=_Resources([_resource()]),
        tool_loader=_Loader(writer),
        resource_activator=ResourceActivator(resources=_Resources([_resource()]), loader=_Loader(writer)),
    ).run(ctx)

    # Without this the knowledge store's search side stays empty.
    assert writer.ready is True
    assert len(writer.indexed) == 2
    assert writer.removed == []


async def test_persisting_without_a_writer_is_refused() -> None:
    # There is no central connection to fall back on, so a step without a
    # writer resource has no database at all — it must say so rather than
    # appear to succeed while storing nothing.
    ctx = _ctx()
    ctx.metadata["knowledge.extracted"] = _items()

    with pytest.raises(WorkflowConfigurationError, match="requires a knowledge database"):
        await KnowledgePersistStep(config={}).run(ctx)


async def test_an_unknown_writer_resource_is_refused(monkeypatch) -> None:
    from nlght.adapters.outbound.workflow.steps.knowledge import persist as persist_module

    monkeypatch.setattr(persist_module, "KnowledgeIndexWriterTool", _IndexWriter)
    ctx = _ctx()
    ctx.metadata["knowledge.extracted"] = _items()

    with pytest.raises(WorkflowConfigurationError, match="No enabled .* named 'missing'"):
        await KnowledgePersistStep(
            config={"writer": "missing"},
            resource_repository=_Resources([]),
            tool_loader=_Loader(_IndexWriter()),
            resource_activator=ResourceActivator(resources=_Resources([]), loader=_Loader(_IndexWriter())),
        ).run(ctx)


# -- persistence writes the lineage too --------------------------------------


async def test_persisting_records_the_lineage_of_an_identifiable_assertion(monkeypatch) -> None:
    """Both records of an assertion, from one step, in one transaction.

    The step hands the repository the graph write and the lineage write
    together. Which of them the repository writes first is its business; that
    they cannot land apart is the point.
    """
    ctx = _ctx()
    ctx.metadata["knowledge.extraction_version"] = "ver-1"
    ctx.metadata["knowledge.extracted"] = (
        ExtractedItem(
            kind="rule", type="security",
            content={
                "rule_text": "Expenses above CHF 500 require approval.",
                "subject": "expense", "rule_property": "approval_threshold",
            },
            confidence=0.9,
            observed_text="Expenses above CHF 500 require approval.",
            evidence={"document_id": "doc-1", "processing_revision_id": "rev-1", "unit_ordinal": "u-0"},
        ),
    )
    await KnowledgeReviewFlagStep(
        config={"policy": "thresholds", "min_confidence": 0.1}
    ).run(ctx)
    repo = _Repo()
    step, _ = _persist_step(monkeypatch, repo)

    await step.run(ctx)

    lineage = repo.lineage[0]
    assert lineage is not None
    assert lineage.kind == "rule"
    assert (lineage.document_id, lineage.document_revision) == ("doc-1", "rev-1")
    assert lineage.slot_id == "u-0"
    assert lineage.extraction_version == "ver-1"


async def test_an_assertion_without_the_identifying_fields_gets_no_lineage(monkeypatch) -> None:
    """Explicitly no lineage, rather than a key made up from the wording.

    A rule extracted before `subject` and `rule_property` were asked for cannot
    name a business question. Deriving a key from its sentence would give it one
    a later run cannot reproduce, so every run would open a new assertion for the
    same claim — the drift this design removes, arriving through the migration
    meant to end it. It is still persisted to the graph; only its lineage is
    absent.
    """
    ctx = _ctx()
    ctx.metadata["knowledge.extracted"] = (
        ExtractedItem(
            kind="rule", type="security",
            content={"rule_text": "Expenses above CHF 500 require approval."},
            confidence=0.9,
            evidence={"document_id": "doc-1", "processing_revision_id": "rev-1"},
        ),
    )
    await KnowledgeReviewFlagStep(
        config={"policy": "thresholds", "min_confidence": 0.1}
    ).run(ctx)
    repo = _Repo()
    step, _ = _persist_step(monkeypatch, repo)

    result = await step.run(ctx)

    assert repo.lineage == [None]
    assert len(repo.writes) == 1
    assert result.ctx.metadata["knowledge.persisted"] == 1


async def test_an_assertion_without_a_document_gets_no_lineage(monkeypatch) -> None:
    # Evidence with no document cannot answer "what did document D at revision R
    # assert", which is the question the diff is built on. A lineage row that
    # could not be diffed would be worse than none.
    ctx = _ctx()
    ctx.metadata["knowledge.extracted"] = (
        ExtractedItem(
            kind="rule", type="security",
            content={
                "rule_text": "Expenses above CHF 500 require approval.",
                "subject": "expense", "rule_property": "approval_threshold",
            },
            confidence=0.9,
            evidence={"source_id": "wiki"},
        ),
    )
    await KnowledgeReviewFlagStep(
        config={"policy": "thresholds", "min_confidence": 0.1}
    ).run(ctx)
    repo = _Repo()
    step, _ = _persist_step(monkeypatch, repo)

    await step.run(ctx)

    assert repo.lineage == [None]


# -- quarantine is not a retraction ------------------------------------------


async def test_a_quarantined_assertion_never_reaches_the_index(monkeypatch) -> None:
    """Nothing written, and nothing deleted either.

    Quarantine means the claim does not go to OpenSearch. It used to be
    *withdrawn* instead, which was wrong twice over: quarantine is not a
    retraction, and on a corpus whose policy quarantines everything it meant one
    pointless delete per assertion per run against a document that had never
    been written — an initial run over a fresh index issued a delete for every
    single candidate.
    """
    ctx = _ctx()
    ctx.metadata["knowledge.extracted"] = _items()
    repo = _StoringRepo()
    writer = _IndexWriter(repository=repo)
    from nlght.adapters.outbound.workflow.steps.knowledge import persist as persist_module

    monkeypatch.setattr(persist_module, "KnowledgeIndexWriterTool", _IndexWriter)

    await KnowledgePersistStep(
        config={"writer": "knowledge-index"},
        resource_repository=_Resources([_resource()]),
        tool_loader=_Loader(writer),
        resource_activator=ResourceActivator(resources=_Resources([_resource()]), loader=_Loader(writer)),
    ).run(ctx)

    # No classification step ran, so everything is quarantined — the safe
    # fallback, and the case an initial run hits.
    assert writer.indexed == []
    assert writer.removed == []


async def test_only_the_released_half_of_a_batch_is_indexed(monkeypatch) -> None:
    # One approved, one quarantined. The approved one is written, the
    # quarantined one is neither written nor deleted.
    ctx = _ctx()
    ctx.metadata["knowledge.extracted"] = _items()
    await KnowledgeReviewFlagStep(
        config={"policy": "thresholds", "min_confidence": 0.5}
    ).run(ctx)
    repo = _StoringRepo()
    writer = _IndexWriter(repository=repo)
    from nlght.adapters.outbound.workflow.steps.knowledge import persist as persist_module

    monkeypatch.setattr(persist_module, "KnowledgeIndexWriterTool", _IndexWriter)

    await KnowledgePersistStep(
        config={"writer": "knowledge-index"},
        resource_repository=_Resources([_resource()]),
        tool_loader=_Loader(writer),
        resource_activator=ResourceActivator(resources=_Resources([_resource()]), loader=_Loader(writer)),
    ).run(ctx)

    assert len(writer.indexed) == 1
    assert writer.removed == []


async def test_a_fact_that_cannot_be_placed_is_counted_and_not_stored(monkeypatch) -> None:
    """The step's half of the invariant, for the route extraction cannot see.

    A fact whose evidence names no document gets no lineage and therefore no
    revision, and a fact with no revision has nowhere to keep what it says.
    Storing it anyway wrote a graph row asserting nothing — so it is skipped,
    counted, and logged with enough to find the producer that emitted it.

    Skipping is only safe because it is *counted*: a claim silently absent and a
    claim silently unreadable are the same fault, and the number is what makes
    this one visible in a run report.
    """
    from nlght.adapters.outbound.workflow.steps.knowledge import persist as persist_module

    ctx = _ctx()
    ctx.metadata["knowledge.extracted"] = (
        ExtractedItem(kind="fact", type="dependency",
                      content={"subject": "s", "predicate": "p", "object": "o"},
                      confidence=0.9, observed_text="s p o.",
                      evidence={"source_id": "corpus"}),
    )
    await KnowledgeReviewFlagStep(
        config={"policy": "thresholds", "min_confidence": 0.1}
    ).run(ctx)
    repo = _StoringRepo()
    writer = _IndexWriter(repository=repo)
    monkeypatch.setattr(persist_module, "KnowledgeIndexWriterTool", _IndexWriter)

    result = await KnowledgePersistStep(
        config={"writer": "knowledge-index"},
        resource_repository=_Resources([_resource()]),
        tool_loader=_Loader(writer),
        resource_activator=ResourceActivator(resources=_Resources([_resource()]), loader=_Loader(writer)),
    ).run(ctx)

    # Placeable after all: a three-field fact projects losslessly onto the
    # legacy row, so the payload holds the whole claim even without a revision.
    # What cannot be stored is a claim with *no* home — neither a revision nor a
    # payload that fits — which the case below is.
    assert result.ctx.metadata["knowledge.unplaceable"] == 0
    assert result.ctx.metadata["knowledge.persisted"] == 1


async def test_a_claim_with_nowhere_to_live_is_counted_and_not_stored(monkeypatch) -> None:
    """The guard, stated as what a claim *has* rather than what kind it is.

    A proposition lives on the revision a sighting produces, so a candidate with
    no lineage has no revision to keep it on. That is survivable while the legacy
    payload can hold the whole claim. It is not survivable when both are missing:
    the graph row then has no payload, no proposition and no revision, and says
    nothing at all.

    Four roles is the case — for a `fact` or, now that every kind has a
    proposition, for any of them.
    """
    from nlght.adapters.outbound.workflow.steps.knowledge import persist as persist_module

    ctx = _ctx()
    ctx.metadata["knowledge.extracted"] = (
        ExtractedItem(
            kind="fact", type="transfer",
            content={"subject": "Alice", "predicate": "transfers",
                     "amount": "CHF 500", "recipient": "Bob"},
            confidence=0.9, observed_text="Alice transfers CHF 500 to Bob.",
            evidence={"source_id": "corpus"},
        ),
    )
    await KnowledgeReviewFlagStep(
        config={"policy": "thresholds", "min_confidence": 0.1}
    ).run(ctx)
    repo = _StoringRepo()
    writer = _IndexWriter(repository=repo)
    monkeypatch.setattr(persist_module, "KnowledgeIndexWriterTool", _IndexWriter)

    result = await KnowledgePersistStep(
        config={"writer": "knowledge-index"},
        resource_repository=_Resources([_resource()]),
        tool_loader=_Loader(writer),
        resource_activator=ResourceActivator(resources=_Resources([_resource()]), loader=_Loader(writer)),
    ).run(ctx)

    assert result.ctx.metadata["knowledge.unplaceable"] == 1
    assert result.ctx.metadata["knowledge.persisted"] == 0
    assert repo.writes == []


async def test_a_rule_without_evidence_is_still_stored(monkeypatch) -> None:
    # The guard is about claims that need a revision to be readable at all. A
    # rule *is* its payload, so a lineage-less one loses nothing — and a corpus
    # written before entity keys existed is full of them.
    from nlght.adapters.outbound.workflow.steps.knowledge import persist as persist_module

    ctx = _ctx()
    ctx.metadata["knowledge.extracted"] = (
        ExtractedItem(kind="rule", type="security", content={"rule_text": "a"},
                      confidence=0.9, observed_text="Secrets are never logged.",
                      evidence={"source_id": "corpus"}),
    )
    await KnowledgeReviewFlagStep(
        config={"policy": "thresholds", "min_confidence": 0.1}
    ).run(ctx)
    repo = _StoringRepo()
    writer = _IndexWriter(repository=repo)
    monkeypatch.setattr(persist_module, "KnowledgeIndexWriterTool", _IndexWriter)

    result = await KnowledgePersistStep(
        config={"writer": "knowledge-index"},
        resource_repository=_Resources([_resource()]),
        tool_loader=_Loader(writer),
        resource_activator=ResourceActivator(resources=_Resources([_resource()]), loader=_Loader(writer)),
    ).run(ctx)

    assert result.ctx.metadata["knowledge.unplaceable"] == 0
    assert result.ctx.metadata["knowledge.persisted"] == 1


async def test_readiness_is_established_before_the_first_write(monkeypatch) -> None:
    """`ready → writes allowed` / `not ready → no knowledge persistence`.

    The ordering half of the invariant. Bootstrapping lazily — on the first
    document, or per batch — would leave a window where candidates are committed
    to PostgreSQL before the mapping exists, and an OpenSearch index with no
    mapping stores what it is given and makes it unsearchable.

    So a writer that cannot make itself ready stops the step before any
    candidate is written, and the repository is untouched. Asserted at the step
    rather than trusted from reading it, because the call sits in `_writer()`
    and someone moving it into the loop would look like a tidy-up.
    """
    from nlght.adapters.outbound.workflow.steps.knowledge import persist as persist_module

    class _UnreadyWriter(_IndexWriter):
        def ensure_ready(self) -> None:
            raise RuntimeError("opensearch refused the create")

    ctx = _ctx()
    ctx.metadata["knowledge.extracted"] = _items()
    await KnowledgeReviewFlagStep(
        config={"policy": "thresholds", "min_confidence": 0.1}
    ).run(ctx)
    repo = _StoringRepo()
    writer = _UnreadyWriter(repository=repo)
    monkeypatch.setattr(persist_module, "KnowledgeIndexWriterTool", _UnreadyWriter)

    with pytest.raises(RuntimeError, match="refused the create"):
        await KnowledgePersistStep(
            config={"writer": "knowledge-index"},
            resource_repository=_Resources([_resource()]),
            tool_loader=_Loader(writer),
            resource_activator=ResourceActivator(resources=_Resources([_resource()]), loader=_Loader(writer)),
        ).run(ctx)

    assert repo.writes == []
