# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Exercise knowledge ingestion across repeated edits to a real file corpus.

The workflow uses real files and a database with scripted model responses and
an in-memory search-index substitute. It checks identity, revisions, evidence,
and retraction across the complete pipeline.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import event, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from tests.extraction_stub import quote_from

from nlght.adapters.outbound.persistence.knowledge_models import (
    KnowledgeAssertion,
    KnowledgeBase,
    KnowledgeEvidence,
    KnowledgeRevision,
    KnowledgeVariant,
)
from nlght.adapters.outbound.persistence.knowledge_repository import (
    SqlAlchemyKnowledgeRepository,
)
from nlght.adapters.outbound.tools.activator import ResourceActivator
from nlght.adapters.outbound.workflow.steps.ingestion import (
    IngestionProcessStep,
    IngestionSourceStep,
)
from nlght.adapters.outbound.workflow.steps.knowledge import (
    KnowledgeExtractStep,
    KnowledgePersistStep,
    KnowledgeReviewFlagStep,
)
from nlght.adapters.outbound.workflow.steps.knowledge.extract import MORE_UNITS
from nlght.adapters.outbound.workflow.steps.knowledge.parsers import KnowledgeParseAutoStep
from nlght.adapters.outbound.workflow.steps.knowledge.refine import (
    KnowledgeAtomicityStep,
    KnowledgeCanonicalizeStep,
    KnowledgeDedupStep,
    KnowledgeNormalizeStep,
)
from nlght.core.entry.context import RequestContext
from nlght.core.knowledge import compare
from nlght.core.protocol.protocol import ProtocolKind
from nlght.core.runtime.resource import ResourceDef
from nlght.core.trigger.trigger import Trigger, TriggerKind
from nlght.core.workflow.step import WorkflowStepContext
from nlght.ports.outbound.model_client import ModelStreamEvent

pytestmark = pytest.mark.e2e


STACK = "stack.md"
POLICY = "policy.md"

#: The claim has two wordings with the same subject, predicate, and object.
#: Only the surrounding sentence changed.
JAVA_17 = "# Stack\n\nSpring requires Java 17.\n"
JAVA_17_REWORDED = "# Stack\n\nFor Spring, Java 17 is a requirement.\n"
#: The claim moved.
JAVA_21 = "# Stack\n\nSpring requires Java 21.\n"

#: The rule keeps its subject and property while its body wording changes.
EXPENSES = "# Expenses\n\nExpenses above CHF 500 require approval.\n"
#: Keep "Expenses" as the subject. "Anything" would broaden the rule's scope.
EXPENSES_REWORDED = "# Expenses\n\nExpenses over CHF 500 must be signed off.\n"
#: Keep all four transfer roles together because the legacy row cannot hold them.
TRANSFER = "# Transfer\n\nAlice transfers CHF 500 to Bob.\n"
#: Renamed roles require a judge because structure alone cannot distinguish a
#: renamed role from a swapped one.
TRANSFER_RENAMED = "# Transfer\n\nAlice sends CHF 500 to the beneficiary Bob.\n"


def _fact(obj: str) -> str:
    return (
        '[{"kind":"fact","type":"dependency","confidence":0.92,'
        '"observed_text":"{quote}",'
        f'"content":{{"subject":"spring","predicate":"requires","object":"{obj}"}}}}]'
    )


def _rule(text: str) -> str:
    return (
        '[{"kind":"rule","type":"policy","confidence":0.88,'
        '"observed_text":"{quote}",'
        '"content":{"subject":"expense","rule_property":"approval_threshold",'
        f'"rule_text":"{text}"}}}}]'
    )


#: What the model answers, keyed by a phrase that only appears in one wording.
#: Scripted rather than sampled, because the point of the reworded cases is that
#: the *proposition* is identical. A model free to phrase its answer differently
#: would make every run a new claim and prove nothing about the pipeline.
SCRIPT: tuple[tuple[str, str], ...] = (
    (
        "Alice sends",
        '[{"kind":"fact","type":"transfer","confidence":0.92,'
        '"observed_text":"{quote}",'
        '"content":{"sender":"Alice","predicate":"sends",'
        '"amount":"CHF 500","beneficiary":"Bob"}}]',
    ),
    (
        "Alice transfers",
        '[{"kind":"fact","type":"transfer","confidence":0.92,'
        '"observed_text":"{quote}",'
        '"content":{"subject":"Alice","predicate":"transfers",'
        '"amount":"CHF 500","recipient":"Bob"}}]',
    ),
    ("Java 21", _fact("java 21")),
    ("Java 17", _fact("java 17")),
    ("signed off", _rule("Expenses over CHF 500 must be signed off.")),
    ("CHF 500", _rule("Expenses above CHF 500 require approval.")),
)


class _ScriptedModel:
    """Answers by what the prompt contains, and counts how often it was asked."""

    def __init__(self) -> None:
        self.calls = 0
        self.classifications = 0
        self.equivalences = 0
        self.prompts: list[str] = []

    def stream(self, messages, tools=None, tool_choice=None):
        self.calls += 1
        prompt = str(messages[0]["content"])
        self.prompts.append(prompt)
        # Keyed on the question itself, not on the sentence introducing it: a
        # reworded preamble once turned this branch into a no-op, and the
        # `equivalences == 1` assertion below is what noticed.
        if "Do A and B assert the same thing" in prompt:
            # A role rename and a role swap look alike structurally. The judge
            # answers `yes` because both wordings below express the same claim.
            self.equivalences += 1
            response = "yes"
        elif "Answer with exactly one word" in prompt:
            # The changed rule bodies in this corpus are rewrites. An unreadable
            # judge answer would exercise the ambiguity fallback instead of the
            # corpus lifecycle.
            self.classifications += 1
            response = "rewrite"
        else:
            response = "[]"
            for marker, scripted in SCRIPT:
                if marker in prompt:
                    response = scripted
                    break
            # A candidate's wording has to occur in the unit it came from, so a
            # stub answering one canned sentence for every unit would model a
            # model that misquotes, which the pipeline is right to reject.
            response = response.replace("{quote}", quote_from(prompt))

        async def _gen():
            yield ModelStreamEvent(kind="token", content=response)
            yield ModelStreamEvent(kind="done")

        return _gen()

    async def call(self, messages) -> None: ...

    def append_tool_turn(self, messages, tool_calls_raw, results, assistant_text=""):
        return messages


class _Writer:
    """Provide the real knowledge repository through a search-index substitute.

    The workflow obtains its repository from the writer resource, so this test
    keeps that activation boundary while replacing only network indexing.
    """

    KIND = "knowledge_index_writer"

    def __init__(self, repository: SqlAlchemyKnowledgeRepository) -> None:
        self.repository = repository
        self.indexed: list[str] = []
        self.removed: list[str] = []

    def ensure_ready(self) -> None: ...

    async def index(self, stored: object, keywords: object = None) -> None:
        self.indexed.append(str(getattr(stored, "identity", "")))

    async def remove(self, *, identity: str, kind: str) -> None:
        self.removed.append(identity)


class _Resources:
    def __init__(self, resources: list[ResourceDef]) -> None:
        self._resources = resources

    async def find_by_kind(self, kind: str) -> list[ResourceDef]:
        return [r for r in self._resources if r.kind == kind]


class _Loader:
    def __init__(self, instance: object) -> None:
        self._instance = instance

    def instantiate(self, *, kind, provider, name, config, **runtime_deps):
        return self._instance


class _Emitter:
    async def emit(self, signal: object) -> None: ...


def _resource() -> ResourceDef:
    import uuid

    return ResourceDef(
        resource_id=uuid.uuid4(),
        name="knowledge-index",
        kind="knowledge_index_writer",
        provider="opensearch",
        config={},
    )


@pytest.fixture(autouse=True)
def _substitute_writer(monkeypatch: pytest.MonkeyPatch) -> None:
    from nlght.adapters.outbound.workflow.steps.knowledge import extract as extract_module
    from nlght.adapters.outbound.workflow.steps.knowledge import persist as persist_module

    monkeypatch.setattr(extract_module, "KnowledgeIndexWriterTool", _Writer)
    monkeypatch.setattr(persist_module, "KnowledgeIndexWriterTool", _Writer)


@pytest_asyncio.fixture
async def engine(tmp_path: Path):
    """A file-backed database, so the runs really are separate runs.

    Args:
        tmp_path (Path): Temporary test directory.
    """
    url = f"sqlite+aiosqlite:///{(tmp_path / 'knowledge.db').as_posix()}"
    engine = create_async_engine(url)

    @event.listens_for(engine.sync_engine, "connect")
    def _enforce_foreign_keys(dbapi_connection, _record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    async with engine.begin() as conn:
        await conn.run_sync(KnowledgeBase.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / STACK).write_text(JAVA_17, encoding="utf-8")
    (root / POLICY).write_text(EXPENSES, encoding="utf-8")
    return root


@pytest.fixture
def writer(engine) -> _Writer:
    return _Writer(SqlAlchemyKnowledgeRepository(engine))


def _ctx(model: _ScriptedModel, run_id: str) -> WorkflowStepContext:
    context = RequestContext(
        correlation_id=run_id,
        request_id=f"rid-{run_id}",
        received_at=datetime(2026, 1, 1, tzinfo=UTC),
        path="/",
        method="POST",
        headers={},
        query_params={},
        client_host=None,
        workflow="ingest-knowledge",
    )
    return WorkflowStepContext(
        correlation_id=run_id,
        trigger=Trigger(
            kind=TriggerKind.INBOUND_EVENT,
            protocol=ProtocolKind.GENERIC_JSON,
            operation="ingest",
            payload={},
            context=context,
        ),
        model="scripted",
        messages=[],
        stream=False,
        emitter=_Emitter(),
        llm=model,
    )


async def ingest(corpus: Path, writer: _Writer, model: _ScriptedModel, run_id: str):
    """One full pass over the corpus, through the steps a workflow would run.

    The refinement chain is included rather than skipped: canonicalisation and
    atomicity rewrite the content that identity is derived from, so a chain that
    left them out would be testing an identity nothing in production computes.

    Extraction is looped on its ``more`` verdict, because that is the transition
    a configured workflow draws back to it: the step takes one batch of units
    per invocation so a long document cannot hold a lease open. Running it once
    would extract the first batch and quietly drop the rest.

    Args:
        corpus (Path): Corpus fixture used for ingestion.
        writer (_Writer): Knowledge writer used by the test.
        model (_ScriptedModel): Model stub used during ingestion.
        run_id (str): Ingestion run identifier.
    """
    ctx = _ctx(model, run_id)
    extract = KnowledgeExtractStep(
        config={
            "model_provider": "scripted",
            "model": "scripted",
            "writer": "knowledge-index",
        },
        resource_repository=_Resources([_resource()]),
        tool_loader=_Loader(writer),
        resource_activator=ResourceActivator(resources=_Resources([_resource()]), loader=_Loader(writer)),
    )
    steps = [
        IngestionSourceStep(
            config={
                "type": "filesystem",
                "source_id": "corpus",
                "roots": [{"path": str(corpus), "alias": "corpus"}],
                "include": ["**/*.md"],
                "respect_ignore_files": False,
            }
        ),
        IngestionProcessStep(config={"document_batch_size": 10}),
        KnowledgeParseAutoStep(config={}),
        extract,
        KnowledgeCanonicalizeStep(config={}),
        KnowledgeNormalizeStep(config={}),
        KnowledgeAtomicityStep(config={}),
        KnowledgeDedupStep(config={}),
        # The seed policy: trust what the model is confident about, review the
        # rest. Reviewing everything would leave the corpus quarantined and the
        # approval-carrying case untestable.
        KnowledgeReviewFlagStep(config={"policy": "thresholds", "min_confidence": 0.6}),
        KnowledgePersistStep(
            config={"writer": "knowledge-index"},
            resource_repository=_Resources([_resource()]),
            tool_loader=_Loader(writer),
            resource_activator=ResourceActivator(resources=_Resources([_resource()]), loader=_Loader(writer)),
        ),
    ]
    for step in steps:
        result = await step.run(ctx)
        ctx = result.ctx
        if step is extract:
            hops = 0
            while result.verdict == MORE_UNITS:
                hops += 1
                assert hops < 50, "extraction never finished its batches"
                result = await step.run(ctx)
                ctx = result.ctx
    return ctx


async def _rows(engine, model):
    async with AsyncSession(engine) as session:
        return list((await session.execute(select(model))).scalars().all())


async def _assertion(engine, *, kind: str):
    rows = [row for row in await _rows(engine, KnowledgeAssertion) if row.kind == kind]
    assert len(rows) == 1, f"expected exactly one {kind}, found {len(rows)}"
    return rows[0]


async def _revisions(engine, assertion_id: str):
    """The revisions of an assertion, reached the way the chain stores them.

    Revisions hang off the variant, not off the assertion: a claim can hold in
    one scope and not another, and each scope carries its own history.

    Args:
        engine (Any): Database engine used by the test.
        assertion_id (str): Assertion whose revisions are read.
    """
    variants = {
        row.variant_id
        for row in await _rows(engine, KnowledgeVariant)
        if row.assertion_id == assertion_id
    }
    rows = [row for row in await _rows(engine, KnowledgeRevision) if row.variant_id in variants]
    return sorted(rows, key=lambda row: row.revision)


async def _evidence(engine, assertion_id: str):
    revisions = {row.revision_id for row in await _revisions(engine, assertion_id)}
    return [row for row in await _rows(engine, KnowledgeEvidence) if row.revision_id in revisions]


async def test_a_file_on_disk_becomes_an_assertion_with_lineage(
    corpus: Path, writer: _Writer, engine
) -> None:
    """The chain end to end: two files in, two assertions with full lineage out."""
    model = _ScriptedModel()

    ctx = await ingest(corpus, writer, model, "run-1")

    assert ctx.metadata["knowledge.persisted"] == 2
    report = await writer.repository.lineage_report()
    assert report.assertions == 2
    assert report.revisions == 2
    assert report.evidence == 2
    # The gap counters are the ones that say the lineage is real rather than
    # merely present: a candidate that could not name a business question, or
    # evidence that could not name a document, is silently useless to the diff.
    assert report.graph_without_lineage == 0
    assert report.evidence_without_document == 0
    assert report.review_states == {"approved": 2}
    # Approved means readable, and readable means indexed.
    assert len(writer.indexed) == 2


async def test_the_document_id_survives_an_edit_but_the_revision_does_not(
    corpus: Path, writer: _Writer, engine
) -> None:
    """Keep document identity stable while changing its revision after an edit.

    Claims can be tracked across edits only when the document ID remains stable.
    """
    await ingest(corpus, writer, _ScriptedModel(), "run-1")
    before = await _rows(engine, KnowledgeEvidence)
    assert len({row.document_id for row in before}) == 2

    (corpus / STACK).write_text(JAVA_17_REWORDED, encoding="utf-8")
    await ingest(corpus, writer, _ScriptedModel(), "run-2")
    after = await _rows(engine, KnowledgeEvidence)

    assert {row.document_id for row in after} == {row.document_id for row in before}, (
        "the edited file came back as a different document"
    )
    edited = {row.document_id: row.document_revision for row in before}
    moved = {
        row.document_id
        for row in after
        if row.document_id in edited and row.document_revision != edited[row.document_id]
    }
    assert len(moved) == 1, "exactly the edited file should have a new revision"


async def test_a_second_pass_over_an_untouched_corpus_changes_nothing(
    corpus: Path, writer: _Writer, engine
) -> None:
    """The acceptance metric, taken over the real pipeline.

    Not one number but the whole picture: the digest folds every assertion's
    key, case and state, so an unchanged digest means literally nothing moved.
    """
    await ingest(corpus, writer, _ScriptedModel(), "run-1")
    baseline = await writer.repository.lineage_report()

    second_model = _ScriptedModel()
    await ingest(corpus, writer, second_model, "run-2")
    current = await writer.repository.lineage_report()

    # The strongest form of "nothing moved": the model was never asked, so it
    # could not have phrased anything differently.
    assert second_model.calls == 0, "an unchanged corpus must ask the model nothing at all"
    result = compare(baseline, current)
    assert result.unchanged, str(result)
    assert current.state_digest == baseline.state_digest
    assert current.evidence == baseline.evidence


async def test_a_reworded_fact_keeps_its_claim_and_records_the_new_wording(
    corpus: Path, writer: _Writer, engine
) -> None:
    """Keep a fact assertion and record new evidence after source rewording.

    Its structured proposition is unchanged, while the new document revision
    establishes that the source still supports it.
    """
    await ingest(corpus, writer, _ScriptedModel(), "run-1")
    before = await _assertion(engine, kind="fact")

    (corpus / STACK).write_text(JAVA_17_REWORDED, encoding="utf-8")
    model = _ScriptedModel()
    await ingest(corpus, writer, model, "run-2")

    # The edited file was re-read. The untouched one was not.
    assert model.calls - model.classifications == 1
    after = await _assertion(engine, kind="fact")
    assert after.assertion_id == before.assertion_id
    assert after.entity_key == before.entity_key

    # One claim, two observed forms. This asserted a single revision until a
    # live run showed the cost: a fact's fingerprint is its structured claim and
    # holds no sentence, so a reworded source hashed identically, no state was
    # recorded, and the corpus went on quoting a sentence the file no longer
    # had. A revision is the state that was observed (ADR-0048).
    revisions = sorted(
        await _revisions(engine, after.assertion_id), key=lambda row: row.revision
    )
    assert [row.revision for row in revisions] == [1, 2]
    assert len({row.fingerprint for row in revisions}) == 1, "the claim did not move"
    assert revisions[-1].review_state == "approved", "a rewording spends no approval"

    # Two sightings of one claim: the same document, at two of its revisions.
    evidence = await _evidence(engine, after.assertion_id)
    assert len({row.document_id for row in evidence}) == 1
    assert len({row.document_revision for row in evidence}) == 2


async def test_a_reworded_rule_writes_a_revision_and_keeps_its_approval(
    corpus: Path, writer: _Writer, engine
) -> None:
    """Add a rule revision while preserving assertion identity and approval.

    A rule's subject and property identify it, while observed wording records a
    new state of that same rule.
    """
    await ingest(corpus, writer, _ScriptedModel(), "run-1")
    before = await _assertion(engine, kind="rule")

    (corpus / POLICY).write_text(EXPENSES_REWORDED, encoding="utf-8")
    await ingest(corpus, writer, _ScriptedModel(), "run-2")

    after = await _assertion(engine, kind="rule")
    assert after.assertion_id == before.assertion_id
    revisions = await _revisions(engine, after.assertion_id)
    assert [row.revision for row in revisions] == [1, 2]
    assert revisions[1].supersedes == 1
    assert revisions[1].review_state == "approved"


async def test_a_changed_claim_opens_a_new_assertion_and_retires_the_old(
    corpus: Path, writer: _Writer, engine
) -> None:
    """Create a new assertion and retire the old one after a claim changes.

    Java 17 and Java 21 are different propositions. Preserve the old assertion's
    evidence without inferring succession from wording alone.
    """
    await ingest(corpus, writer, _ScriptedModel(), "run-1")
    before = await _assertion(engine, kind="fact")

    (corpus / STACK).write_text(JAVA_21, encoding="utf-8")
    await ingest(corpus, writer, _ScriptedModel(), "run-2")

    facts = [row for row in await _rows(engine, KnowledgeAssertion) if row.kind == "fact"]
    assert len(facts) == 2
    old = next(row for row in facts if row.assertion_id == before.assertion_id)
    fresh = next(row for row in facts if row.assertion_id != before.assertion_id)

    assert old.retired_at is not None
    assert fresh.retired_at is None
    assert len(await _evidence(engine, before.assertion_id)) == 1
    assert await writer.repository.supersessions_of(before.assertion_id) == ()
    # Retired means unfindable: it leaves the index in the same breath.
    assert writer.removed


TRANSFER_CLAIM = {
    "subject": "Alice",
    "predicate": "transfers",
    "amount": "CHF 500",
    "recipient": "Bob",
}


async def test_a_four_role_claim_survives_the_whole_chain(
    corpus: Path, writer: _Writer, engine
) -> None:
    """Preserve a four-role claim from source file through retrieval.

    The proposition retains every role on its revision. It participates in
    lineage and retrieval without being reduced to a legacy three-field row.
    """
    from nlght.adapters.outbound.persistence.knowledge_models import (
        KnowledgeFact,
        KnowledgeProposition,
    )

    (corpus / STACK).write_text(TRANSFER, encoding="utf-8")
    (corpus / POLICY).unlink()

    ctx = await ingest(corpus, writer, _ScriptedModel(), "run-1")

    assert ctx.metadata["knowledge.persisted"] == 1

    [proposition] = await _rows(engine, KnowledgeProposition)
    assert proposition.fields == TRANSFER_CLAIM

    assertion = await _assertion(engine, kind="fact")
    assert assertion.retired_at is None

    [revision] = await _revisions(engine, assertion.assertion_id)
    assert revision.proposition_id == proposition.proposition_id

    [stored] = await writer.repository.resolve([writer.indexed[0]])
    assert stored.payload == TRANSFER_CLAIM

    assert await _rows(engine, KnowledgeFact) == []


async def test_a_renamed_role_continues_the_assertion_once_a_judge_says_so(
    corpus: Path, writer: _Writer, engine
) -> None:
    """Continue an assertion only after equivalence judgment confirms renamed roles.

    Record the assessment first. A yes verdict joins the new observation to the
    existing assertion while both revision propositions remain readable.
    """
    from nlght.adapters.outbound.persistence.knowledge_models import (
        KnowledgeEquivalenceAssessment,
        KnowledgeProposition,
    )

    (corpus / STACK).write_text(TRANSFER, encoding="utf-8")
    (corpus / POLICY).unlink()
    await ingest(corpus, writer, _ScriptedModel(), "run-1")
    before = await _assertion(engine, kind="fact")

    (corpus / STACK).write_text(TRANSFER_RENAMED, encoding="utf-8")
    model = _ScriptedModel()
    await ingest(corpus, writer, model, "run-2")

    # Asked, and asked once, not once per differing fingerprint.
    assert model.equivalences == 1

    # Recorded before anything was joined, with the judge that reached it.
    [assessment] = await _rows(engine, KnowledgeEquivalenceAssessment)
    assert assessment.verdict == "yes"
    assert assessment.classifier == "proposition-equivalence"
    assert assessment.run_id == "run-2"

    # One claim, continued rather than duplicated, and nothing retired.
    after = await _assertion(engine, kind="fact")
    assert after.assertion_id == before.assertion_id
    assert after.retired_at is None

    # Two states of it, each keeping the structure it recorded.
    revisions = await _revisions(engine, after.assertion_id)
    assert len(revisions) == 2
    fields = {row.proposition_id: row.fields for row in await _rows(engine, KnowledgeProposition)}
    assert [sorted(fields[row.proposition_id]) for row in revisions] == [
        ["amount", "predicate", "recipient", "subject"],
        ["amount", "beneficiary", "predicate", "sender"],
    ]


async def test_without_a_yes_the_two_claims_stay_apart(
    corpus: Path, writer: _Writer, engine
) -> None:
    """Keep two claims separate unless the equivalence judge returns yes.

    No, ambiguous, and unjudged outcomes remain distinct audit records, but none
    merges the assertions.
    """
    from nlght.adapters.outbound.persistence.knowledge_models import (
        KnowledgeEquivalenceAssessment,
    )

    class _Unsure(_ScriptedModel):
        def stream(self, messages, tools=None, tool_choice=None):
            if "Do A and B assert the same thing" in str(messages[0]["content"]):
                async def _gen():
                    yield ModelStreamEvent(kind="token", content="uncertain")
                    yield ModelStreamEvent(kind="done")

                return _gen()
            return super().stream(messages, tools, tool_choice)

    (corpus / STACK).write_text(TRANSFER, encoding="utf-8")
    (corpus / POLICY).unlink()
    await ingest(corpus, writer, _ScriptedModel(), "run-1")

    (corpus / STACK).write_text(TRANSFER_RENAMED, encoding="utf-8")
    await ingest(corpus, writer, _Unsure(), "run-2")

    [assessment] = await _rows(engine, KnowledgeEquivalenceAssessment)
    assert assessment.verdict == "ambiguous"

    facts = [row for row in await _rows(engine, KnowledgeAssertion) if row.kind == "fact"]
    assert len(facts) == 2


async def test_the_report_can_name_every_document_it_saw(
    corpus: Path, writer: _Writer, engine
) -> None:
    """Carry document identities through knowledge-only ingestion into reports.

    The report cannot resolve claim sources through data-ingestion tables that a
    knowledge-only workflow never writes.
    """
    await ingest(corpus, writer, _ScriptedModel(), "run-1")

    report = await writer.repository.lineage_report()

    # The source's own path, alias and all, which is why an expectation names a
    # file by suffix rather than having to know where the corpus was mounted.
    assert sorted(report.document_paths.values()) == [f"corpus/{POLICY}", f"corpus/{STACK}"]
    # And the id side is what the diff uses, so both halves are present.
    assert set(report.document_paths) == set(report.documents)

    # End to end: a case naming the bare file finds it.
    from nlght.core.knowledge.expectations import (
        CaseExpectation,
        check_expectations,
    )

    outcome = check_expectations(
        report, report, compare(report, report), [CaseExpectation(document=STACK)]
    )

    assert outcome.unknown_documents == ()
