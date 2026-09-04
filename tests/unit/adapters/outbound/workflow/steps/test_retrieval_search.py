# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Layer B — the adapter contract: what each store's answer becomes.

The ranking is decided in `core.retrieval` and proved there. What only the step
can get wrong is the translation and the routing: reading the payload the
ingestion actually writes, giving each hit the right carrier and provenance,
asking the stores this platform has, and surviving one of them being down.

The store shapes below are the ones `data_writer` really produces — one
OpenSearch document per document, one Qdrant point per chunk — so a change to
the index that this step does not follow shows up here rather than live.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import pytest

from nlght.adapters.outbound.workflow.steps.retrieval import (
    NOTHING_FOUND,
    RetrievalSearchStep,
)
from nlght.core.entry.context import RequestContext
from nlght.core.errors.errors import WorkflowConfigurationError
from nlght.core.protocol.protocol import ProtocolKind
from nlght.core.retrieval import (
    ASSERTION,
    CHUNK,
    DOCUMENT,
    KNOWLEDGE,
    LEXICAL,
    VECTOR,
    Provenance,
    RetrievalHit,
)
from nlght.core.trigger.trigger import Trigger, TriggerKind
from nlght.core.workflow.step import WorkflowStepContext


class _Emitter:
    async def emit(self, signal: object) -> None: ...


def test_every_supported_retrieval_option_is_declared_for_the_admin_ui() -> None:
    options = {option.name: option for option in RetrievalSearchStep.options()}

    assert set(options) == {
        "sources", "required_sources", "fetch_k", "fusion_k", "final_k",
        "source_weights", "damping", "kinds", "lexical_floor", "lexical_ceiling",
        "data_store", "knowledge_store",
    }
    assert options["sources"].type == "array"
    assert options["required_sources"].type == "array"
    assert options["fetch_k"].default == {"lexical": 50, "vector": 50, "knowledge": 20}
    assert options["source_weights"].default == {
        "lexical": 1.0, "vector": 1.0, "knowledge": 1.0,
    }
    assert options["kinds"].default == ["fact", "rule", "pattern", "decision"]


def _ctx(question: str = "keystore password for server.ssl") -> WorkflowStepContext:
    context = RequestContext(
        correlation_id="run-1", request_id="rid", received_at=datetime(2026, 1, 1, tzinfo=UTC),
        path="/", method="POST", headers={}, query_params={}, client_host=None,
    )
    ctx = WorkflowStepContext(
        correlation_id="run-1",
        trigger=Trigger(
            kind=TriggerKind.INBOUND_EVENT, protocol=ProtocolKind.GENERIC_JSON,
            operation="ask", payload={}, context=context,
        ),
        model="", messages=[], stream=False, emitter=_Emitter(),
    )
    ctx.metadata["retrieval.question"] = question
    return ctx


@dataclass
class _Hit:
    id: str
    score: float
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _document(identity: str = "d1", score: float = 7.2) -> RetrievalHit:
    return RetrievalHit(
        source=LEXICAL,
        carrier=DOCUMENT,
        carrier_id=identity,
        content="Server certificates are configured through server.ssl.",
        raw_score=score,
        provenance=Provenance(
            document_id=identity, processing_revision_id="r1", path="tls.adoc",
            source_name="docs", external_id="tls",
        ),
        fields={"symbols": ("KeyStore",), "fqns": ("server.ssl.key-store",),
                "semantics": ("tls", "server")},
    )


class _Lexical:
    """The data store's keyword capability.

    It answers with findings rather than backend rows, and everything it returns
    is already on the published revision — the store drops the rest before the
    step can rank it (ADR-0062). So there is no translation left here to test;
    that lives with the store, where the payload does.
    """

    def __init__(self, hits: list[RetrievalHit] | None = None, *, fails: bool = False) -> None:
        self._hits = hits if hits is not None else [_document()]
        self._fails = fails
        self.asked = 0
        self.limit_seen = 0

    async def search_keyword_candidates(
        self, *, query: str, limit: int = 10
    ) -> list[RetrievalHit]:
        self.asked += 1
        self.limit_seen = limit
        if self._fails:
            raise RuntimeError("opensearch is down")
        return self._hits[:limit]


class _Vector(_Lexical):
    """The same activation, seen through its semantic capability.

    One object, because in production it *is* one: `data_store` answers both
    capabilities over one corpus. Two stubs would let the step look correct
    while asking two different stores, which is the arrangement this slice
    retired.
    """

    def __init__(
        self, hits: list[RetrievalHit] | None = None, *,
        fails: bool = False, answers: bool = True,
    ) -> None:
        super().__init__(hits, fails=fails)
        self.asked_semantic = 0
        self._answers = answers

    async def search_semantic_candidates(
        self, *, query: str, limit: int = 10
    ) -> list[RetrievalHit]:
        self.asked_semantic += 1
        if not self._answers:
            return []
        return [
            RetrievalHit(
                source=VECTOR,
                carrier=CHUNK,
                carrier_id="c7",
                content="A keystore password is required whenever a keystore is configured.",
                raw_score=0.81,
                provenance=Provenance(
                    document_id="d1", chunk_id="c7", processing_revision_id="r1",
                    path="tls.adoc", start_offset=10, end_offset=80,
                ),
                fields={"semantics": ("tls",)},
            )
        ][:limit]


@dataclass
class _Claim:
    identity: str
    type: str
    confidence: float
    payload: Any


@dataclass
class _RulePayload:
    rule_text: str


class _Knowledge:
    """Only what `resolve()` let through — the quarantine is upstream of here."""

    def __init__(self) -> None:
        self.kinds: list[str] = []

    async def query_rules(self, query: str, limit: int = 10) -> list[_Claim]:
        self.kinds.append("rule")
        return [_Claim("fp-1", "security", 0.9, _RulePayload(
            "A keystore password is required whenever a keystore is configured."))]

    async def query_facts(self, query: str, limit: int = 10) -> list[_Claim]:
        self.kinds.append("fact")
        return []

    async def query_patterns(self, query: str, limit: int = 10) -> list[_Claim]:
        self.kinds.append("pattern")
        return []

    async def query_decisions(self, query: str, limit: int = 10) -> list[_Claim]:
        self.kinds.append("decision")
        return []


@dataclass
class _Embedded:
    vectors: tuple[tuple[float, ...], ...]


class _Embedding:
    def __init__(self) -> None:
        self.asked = 0

    async def embed(self, texts: list[str]) -> _Embedded:
        self.asked += 1
        return _Embedded(vectors=((0.1, 0.2, 0.3),))


def _step(
    stores: dict[str, object], *, embedding: _Embedding | None = None, **config: object
) -> RetrievalSearchStep:
    """A step whose activations are the stubs.

    The resource lookup is plumbing shared with every other step; what this file
    is about is what the retrieval does with what comes back.
    """
    # Both discovery capabilities come from the one data store activation.
    option = {LEXICAL: "data_store", VECTOR: "data_store", KNOWLEDGE: "knowledge_store"}
    settings: dict[str, object] = {}
    for name in stores:
        settings.setdefault(option[name], option[name])
    settings.update(config)
    step = RetrievalSearchStep(config=settings, embedding_client=embedding)

    async def _activate(source: str, caller=None) -> object | None:
        return stores.get(source)

    # `_stores` activates once per named resource, so a data store serving both
    # capabilities is activated once and shared — as it is in production.
    async def _activated(caller=None) -> dict[str, object]:
        data = stores.get(VECTOR) or stores.get(LEXICAL)
        activated: dict[str, object] = {}
        if LEXICAL in stores:
            activated[LEXICAL] = data
        if VECTOR in stores:
            activated[VECTOR] = data
        if KNOWLEDGE in stores:
            activated[KNOWLEDGE] = stores[KNOWLEDGE]
        return activated

    step._stores = _activated  # type: ignore[method-assign]  # noqa: SLF001

    step._activate = _activate  # type: ignore[method-assign]  # noqa: SLF001
    return step


# --- translation -----------------------------------------------------------

async def test_each_store_answers_at_its_own_granularity() -> None:
    """Three systems, three carriers, and none of them collapsed into another."""
    step = _step(
        {LEXICAL: _Lexical(), VECTOR: _Vector(), KNOWLEDGE: _Knowledge()},
        embedding=_Embedding(), lexical_floor=5,
    )

    hits = (await step.run(_ctx())).ctx.metadata["retrieval.hits"]

    assert {item.carrier for item in hits} == {DOCUMENT, CHUNK, ASSERTION}
    assert len(hits) == 3, "a document, a chunk of it and a claim from it are three findings"


async def test_a_lexical_answer_becomes_a_document_with_its_indexed_terms() -> None:
    step = _step({LEXICAL: _Lexical()}, sources="lexical")

    [found] = (await step.run(_ctx())).ctx.metadata["retrieval.hits"]
    hit = found.best

    assert (hit.carrier, hit.carrier_id) == (DOCUMENT, "d1")
    assert hit.raw_score == 7.2
    assert hit.provenance.path == "tls.adoc"
    assert hit.provenance.processing_revision_id == "r1"
    assert hit.provenance.source_name == "docs"
    assert hit.terms("fqns") == ("server.ssl.key-store",)
    assert hit.terms("symbols") == ("KeyStore",)


async def test_a_vector_answer_becomes_a_chunk_that_knows_its_document() -> None:
    # The document id is what later lets context building see the relationship —
    # see it, not act on it.
    step = _step({VECTOR: _Vector()}, embedding=_Embedding(), sources="vector")

    [found] = (await step.run(_ctx())).ctx.metadata["retrieval.hits"]
    hit = found.best

    assert (hit.carrier, hit.carrier_id) == (CHUNK, "c7")
    assert hit.provenance.document_id == "d1"
    assert hit.provenance.chunk_id == "c7"


async def test_a_claim_is_quoted_in_the_sources_own_words() -> None:
    # The observed wording is the authority (ADR-0048); nothing is reassembled
    # from the structured fields.
    step = _step({KNOWLEDGE: _Knowledge()}, sources="knowledge")

    [found] = (await step.run(_ctx())).ctx.metadata["retrieval.hits"]
    hit = found.best

    assert (hit.carrier, hit.carrier_id) == (ASSERTION, "fp-1")
    assert hit.content.startswith("A keystore password is required")
    assert hit.terms("kind") == ("rule",)


# --- what gets asked -------------------------------------------------------

async def test_the_sources_default_to_the_stores_this_platform_has() -> None:
    """A platform without a vector index has not misconfigured anything."""
    step = _step({LEXICAL: _Lexical(), KNOWLEDGE: _Knowledge()})

    stats = (await step.run(_ctx())).ctx.metadata["retrieval.stats"]

    assert stats["sources"] == [LEXICAL, KNOWLEDGE]
    assert stats["unavailable"] == []


async def test_a_requested_store_that_is_absent_is_reported_and_not_an_error() -> None:
    step = _step({LEXICAL: _Lexical()}, sources="lexical,vector")

    stats = (await step.run(_ctx())).ctx.metadata["retrieval.stats"]

    assert stats["sources"] == [LEXICAL]
    assert stats["unavailable"] == [VECTOR]


async def test_a_store_a_workflow_declares_indispensable_must_be_there() -> None:
    """The distinction that stops the "every store must exist" trap returning.

    Wanting a store and depending on one are different, and only the second is
    a reason to refuse to run.
    """
    step = _step({LEXICAL: _Lexical()}, sources="lexical,vector", required_sources="vector")

    with pytest.raises(WorkflowConfigurationError, match="required retrieval source"):
        await step.run(_ctx())


async def test_semantic_discovery_is_not_asked_when_keyword_answered() -> None:
    lexical, vector, embedding = _Lexical(), _Vector(), _Embedding()
    step = _step({LEXICAL: lexical, VECTOR: vector}, embedding=embedding,
                 lexical_floor=1, lexical_ceiling=100)

    stats = (await step.run(_ctx())).ctx.metadata["retrieval.stats"]

    assert vector.asked_semantic == 0
    assert embedding.asked == 0, "a query embedded for a call never made is money spent"
    assert stats["routes"][VECTOR]["because"] == "lexical_answered"


async def test_without_an_embedding_provider_the_other_capabilities_still_answer() -> None:
    """The store decides whether it can embed, and says so by answering nothing.

    The step used to hold the embedding client and skip the call itself. It
    embeds nothing now — the store does, because the store owns the collection
    the vector has to match — so an unembeddable question is an empty semantic
    answer rather than a route the step declined to take.
    """
    step = _step({LEXICAL: _Lexical(), VECTOR: _Vector(answers=False)}, lexical_floor=5)

    result = await step.run(_ctx())

    assert result.ctx.metadata["retrieval.stats"]["routes"][VECTOR]["hits"] == 0
    assert result.ctx.metadata["retrieval.hits"]


async def test_a_store_that_is_down_does_not_take_the_others_with_it() -> None:
    step = _step({LEXICAL: _Lexical(fails=True), KNOWLEDGE: _Knowledge()})

    result = await step.run(_ctx())
    routes = result.ctx.metadata["retrieval.stats"]["routes"]

    assert routes[LEXICAL]["ok"] is False
    assert "opensearch is down" in routes[LEXICAL]["error"]
    assert result.ctx.metadata["retrieval.hits"], "the knowledge store still answered"


async def test_the_knowledge_store_is_asked_for_the_kinds_configured() -> None:
    knowledge = _Knowledge()
    step = _step({KNOWLEDGE: knowledge}, kinds=["rule", "decision"])

    await step.run(_ctx())

    assert knowledge.kinds == ["rule", "decision"]


# --- the three cut-offs ----------------------------------------------------

async def test_how_deep_a_store_is_asked_is_not_how_much_leaves_the_step() -> None:
    """Setting fetch_k to final_k looks tidy and destroys evidence.

    A hit one store put twentieth and another put first is exactly what fusion
    exists to catch, and fetching ten from each throws away the half of it that
    would have made the point.
    """
    lexical = _Lexical()
    step = _step({LEXICAL: lexical}, fetch_k={"lexical": 50}, final_k=5)

    await step.run(_ctx())

    assert lexical.limit_seen == 50


async def test_only_what_leaves_the_step_is_capped_by_final_k() -> None:
    many = [_document(f"d{i}", 9.0 - i) for i in range(10)]
    step = _step({LEXICAL: _Lexical(many)}, final_k=3)

    hits = (await step.run(_ctx())).ctx.metadata["retrieval.hits"]

    assert len(hits) == 3
    assert [item.best.carrier_id for item in hits] == ["d0", "d1", "d2"]


async def test_fusion_k_cuts_each_stores_list_before_the_ranking() -> None:
    many = [_document(f"d{i}", 9.0 - i) for i in range(10)]
    step = _step({LEXICAL: _Lexical(many)}, fusion_k=2, final_k=20)

    hits = (await step.run(_ctx())).ctx.metadata["retrieval.hits"]

    assert len(hits) == 2


# --- the question ----------------------------------------------------------

async def test_a_search_with_no_question_is_refused() -> None:
    step = _step({LEXICAL: _Lexical()})
    ctx = _ctx()
    del ctx.metadata["retrieval.question"]

    with pytest.raises(WorkflowConfigurationError, match="no question to ask"):
        await step.run(ctx)


async def test_the_question_falls_back_to_the_last_user_message() -> None:
    step = _step({LEXICAL: _Lexical()})
    ctx = _ctx()
    del ctx.metadata["retrieval.question"]
    ctx.messages = [{"role": "system", "content": "be helpful"},
                    {"role": "user", "content": "keystore password"}]

    stats = (await step.run(ctx)).ctx.metadata["retrieval.stats"]

    assert stats["question"] == "keystore password"


async def test_a_step_with_no_store_at_all_is_refused() -> None:
    step = _step({})

    with pytest.raises(WorkflowConfigurationError, match="no store to search"):
        await step.run(_ctx())


async def test_an_empty_index_says_so_rather_than_looking_like_a_failure() -> None:
    step = _step({LEXICAL: _Lexical(hits=[])})

    result = await step.run(_ctx())

    assert result.verdict == NOTHING_FOUND
    assert result.ctx.metadata["retrieval.hits"] == ()
