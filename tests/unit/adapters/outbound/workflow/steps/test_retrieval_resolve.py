# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""What a model is shown comes from the corpus, never from a projection.

The two properties this slice exists for, at the seam where they become visible:
a chunk is answered with the span cut from the stored revision, and a finding the
corpus cannot answer leaves rather than being answered from the index payload.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from nlght.adapters.outbound.stores.data import ResolvedText
from nlght.adapters.outbound.workflow.steps.retrieval import (
    NOTHING_RESOLVED,
    RetrievalResolveStep,
)
from nlght.core.context import ContextBudget, build_context
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
    fuse,
    rank_within_sources,
)
from nlght.core.trigger.trigger import Trigger, TriggerKind
from nlght.core.workflow.step import WorkflowStepContext


class _Emitter:
    async def emit(self, signal: object) -> None: ...


def _ctx() -> WorkflowStepContext:
    context = RequestContext(
        correlation_id="run-1", request_id="rid", received_at=datetime(2026, 1, 1, tzinfo=UTC),
        path="/", method="POST", headers={}, query_params={}, client_host=None,
    )
    return WorkflowStepContext(
        correlation_id="run-1",
        trigger=Trigger(
            kind=TriggerKind.INBOUND_EVENT, protocol=ProtocolKind.GENERIC_JSON,
            operation="ask", payload={}, context=context,
        ),
        model="", messages=[], stream=False, emitter=_Emitter(),
    )


def _chunk(chunk_id: str, *, document: str = "d1", revision: str = "r1") -> RetrievalHit:
    return RetrievalHit(
        source=VECTOR, carrier=CHUNK, carrier_id=chunk_id,
        content="WHAT THE PAYLOAD HAPPENS TO SAY",
        raw_score=0.9,
        provenance=Provenance(
            document_id=document, chunk_id=chunk_id, processing_revision_id=revision,
            start_offset=4, end_offset=10, path="Foo.java",
        ),
    )


def _document(document: str = "d2", revision: str = "r2") -> RetrievalHit:
    return RetrievalHit(
        source=LEXICAL, carrier=DOCUMENT, carrier_id=document,
        content="THE PAYLOAD'S COPY OF THE WHOLE FILE",
        raw_score=7.0,
        provenance=Provenance(
            document_id=document, processing_revision_id=revision, path="notes.md",
        ),
    )


def _claim() -> RetrievalHit:
    return RetrievalHit(
        source=KNOWLEDGE, carrier=ASSERTION, carrier_id="a1",
        content="A keystore password is required.",
        raw_score=0.8,
        provenance=Provenance(assertion_id="a1", knowledge_revision_id="k1"),
    )


class _Corpus:
    """The data store, seen through the one operation this step uses."""

    def __init__(self, texts: dict[str, str] | None = None) -> None:
        self._texts = texts or {}
        self.asked: list[str] = []

    async def resolve(self, hits: list[RetrievalHit]) -> dict[str, ResolvedText]:
        self.asked.extend(hit.carrier_id for hit in hits)
        return {
            hit.carrier_id: ResolvedText(
                carrier_id=hit.carrier_id,
                document_id=hit.provenance.document_id,
                processing_revision_id=hit.provenance.processing_revision_id,
                text=self._texts[hit.carrier_id],
            )
            for hit in hits
            if hit.carrier_id in self._texts
        }


def _step(corpus: _Corpus, **config: object) -> RetrievalResolveStep:
    step = RetrievalResolveStep(config={"data_store": "data-main", **config})

    async def _store(caller=None) -> object:
        return corpus

    step._store = _store  # type: ignore[method-assign]  # noqa: SLF001
    return step


def _ranked(*hits: RetrievalHit) -> tuple[Any, ...]:
    return tuple(fuse(rank_within_sources(list(hits))))


async def test_a_chunk_is_answered_with_the_span_from_the_stored_revision() -> None:
    """The payload says one thing and the corpus another; the corpus wins."""
    corpus = _Corpus({"c7": "the authoritative span"})
    ctx = _ctx()
    ctx.metadata["retrieval.hits"] = _ranked(_chunk("c7"))

    result = await _step(corpus).run(ctx)

    [group] = result.ctx.metadata["retrieval.hits"]
    assert group.best.content == "the authoritative span"
    assert result.ctx.metadata["retrieval.resolved"] == {
        "resolved": 1, "dropped": 0, "claims": 0,
    }


async def test_a_finding_the_corpus_cannot_answer_is_dropped_not_substituted() -> None:
    """Falling back to the payload would hand a model a projection.

    A citation would then name the corpus for text the corpus does not hold. The
    finding leaves instead, and the count says how many did.
    """
    corpus = _Corpus({"c7": "the authoritative span"})
    ctx = _ctx()
    ctx.metadata["retrieval.hits"] = _ranked(_chunk("c7"), _chunk("c9"))

    result = await _step(corpus).run(ctx)

    kept = result.ctx.metadata["retrieval.hits"]
    assert [group.best.carrier_id for group in kept] == ["c7"]
    assert result.ctx.metadata["retrieval.resolved"]["dropped"] == 1
    # And emphatically not the payload's text under a different id.
    assert all(
        "PAYLOAD" not in hit.content for group in kept for hit in group.hits
    )


async def test_a_claim_passes_through_because_it_was_resolved_at_discovery() -> None:
    """`knowledge_store` reads PostgreSQL before it ranks, so this is already true."""
    corpus = _Corpus()
    ctx = _ctx()
    ctx.metadata["retrieval.hits"] = _ranked(_claim())

    result = await _step(corpus).run(ctx)

    [group] = result.ctx.metadata["retrieval.hits"]
    assert group.best.content == "A keystore password is required."
    assert corpus.asked == [], "a claim must not be looked up in the document corpus"
    assert result.ctx.metadata["retrieval.resolved"]["claims"] == 1


async def test_a_document_is_answered_whole_and_a_chunk_by_its_span() -> None:
    """Two carriers, two right answers, neither invented."""
    corpus = _Corpus({"c7": "the span", "d2": "the whole revision"})
    ctx = _ctx()
    ctx.metadata["retrieval.hits"] = _ranked(_chunk("c7"), _document())

    result = await _step(corpus).run(ctx)

    answers = {
        group.best.carrier_id: group.best.content
        for group in result.ctx.metadata["retrieval.hits"]
    }
    assert answers == {"c7": "the span", "d2": "the whole revision"}


async def test_the_passage_a_model_sees_is_the_corpus_text() -> None:
    """End to end through the pure layer, which still reads no store.

    `core/context` is unchanged and unaware of any of this: it receives findings
    whose text is already true, which is exactly what ADR-0052 asked for when it
    left the fetch to a later slice.
    """
    corpus = _Corpus({"c7": "the authoritative span"})
    ctx = _ctx()
    ctx.metadata["retrieval.hits"] = _ranked(_chunk("c7"))

    result = await _step(corpus).run(ctx)
    selection = build_context(result.ctx.metadata["retrieval.hits"], ContextBudget(1000))

    assert [passage.text for passage in selection.passages] == ["the authoritative span"]


async def test_nothing_resolvable_is_its_own_verdict() -> None:
    """Found something, and the corpus cannot say what it says — not the same as
    having found nothing, so a pipeline can notice it."""
    ctx = _ctx()
    ctx.metadata["retrieval.hits"] = _ranked(_chunk("c7"))

    result = await _step(_Corpus()).run(ctx)

    assert result.verdict == NOTHING_RESOLVED
    assert result.ctx.metadata["retrieval.hits"] == ()


async def test_resolving_without_a_search_before_it_is_refused() -> None:
    with pytest.raises(WorkflowConfigurationError, match="retrieval.search"):
        await _step(_Corpus()).run(_ctx())
