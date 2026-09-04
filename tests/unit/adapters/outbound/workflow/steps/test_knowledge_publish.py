# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""knowledge.publish: what a review decided reaches the index, and only that."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from nlght.adapters.outbound.tools.activator import ResourceActivator
from nlght.adapters.outbound.workflow.steps.knowledge.publish import (
    MORE,
    NOTHING_TO_PUBLISH,
    KnowledgePublishStep,
)
from nlght.core.entry.context import RequestContext
from nlght.core.errors.errors import WorkflowConfigurationError
from nlght.core.knowledge import KnowledgeObject
from nlght.core.protocol.protocol import ProtocolKind
from nlght.core.trigger.trigger import Trigger, TriggerKind
from nlght.core.workflow.step import WorkflowStepContext


class _Emitter:
    async def emit(self, signal: object) -> None: ...


def _ctx(payload: dict[str, Any] | None = None) -> WorkflowStepContext:
    context = RequestContext(
        correlation_id="cid-1",
        request_id="rid-1",
        received_at=datetime(2026, 1, 1, tzinfo=UTC),
        path="/",
        method="POST",
        headers={},
        query_params={},
        client_host=None,
    )
    return WorkflowStepContext(
        correlation_id="cid-1",
        trigger=Trigger(
            kind=TriggerKind.INBOUND_EVENT,
            protocol=ProtocolKind.GENERIC_JSON,
            operation="knowledge.publish",
            payload=payload or {},
            context=context,
        ),
        model="",
        messages=[],
        stream=False,
        emitter=_Emitter(),
    )


def _assertion(identity: str, *, quarantined: bool = False, merged: str | None = None) -> KnowledgeObject:
    now = datetime(2026, 8, 27, tzinfo=UTC)
    return KnowledgeObject(
        identity=identity,
        kind="fact",
        type="capability",
        confidence=0.8,
        payload={"subject": "s", "predicate": "p", "object": "o"},
        review_required=quarantined,
        review_reason="unsure" if quarantined else None,
        merged_into=merged,
        product=None,
        version=None,
        metadata={"keywords": ["k"]},
        created_at=now,
        updated_at=now,
    )


class _Repository:
    def __init__(
        self,
        canonical: list[KnowledgeObject] | None = None,
        withdrawn: list[tuple[str, str]] | None = None,
        resolved: list[KnowledgeObject] | None = None,
    ) -> None:
        self._canonical = canonical or []
        self._withdrawn = withdrawn or []
        self._resolved = resolved or []
        self.resolve_calls: list[tuple[list[str], bool]] = []

    async def canonical_page(self, *, limit: int = 200, offset: int = 0) -> list[KnowledgeObject]:
        return self._canonical[offset : offset + limit]

    async def withdrawn_page(self, *, limit: int = 200, offset: int = 0) -> list[tuple[str, str]]:
        return self._withdrawn[offset : offset + limit]

    async def resolve(
        self, identities: list[str], *, include_quarantined: bool = False
    ) -> list[KnowledgeObject]:
        self.resolve_calls.append((list(identities), include_quarantined))
        return self._resolved


class _Writer:
    KIND = "knowledge_index_writer"
    name = "knowledge-index"

    def __init__(self, repository: _Repository) -> None:
        self.repository = repository
        self.indexed: list[str] = []
        self.removed: list[tuple[str, str]] = []
        self.prepared = 0

    def ensure_ready(self) -> None:
        self.prepared += 1

    async def index(self, item: KnowledgeObject, *, keywords: list[str] | None = None) -> None:
        self.indexed.append(item.identity)

    async def remove(self, *, identity: str, kind: str) -> None:
        self.removed.append((identity, kind))


def _step(writer: _Writer, **config: object) -> KnowledgePublishStep:
    step = KnowledgePublishStep(
        config={"writer": "knowledge-index", **config},
        resource_repository=object(),
        tool_loader=object(),
        resource_activator=ResourceActivator(resources=object(), loader=object()),
    )

    async def _resolved() -> _Writer:
        return writer

    step._writer = lambda caller=None: _resolved()  # type: ignore[assignment]
    return step


async def test_approved_knowledge_reaches_the_index() -> None:
    # The whole point: an approval changes a flag in PostgreSQL and nothing
    # else. This is what makes it findable.
    writer = _Writer(_Repository(canonical=[_assertion("fact-1"), _assertion("fact-2")]))

    result = await _step(writer).run(_ctx())

    assert writer.indexed == ["fact-1", "fact-2"]
    assert result.ctx.metadata["knowledge.published"] == 2


async def test_refused_and_merged_knowledge_is_taken_back_out() -> None:
    writer = _Writer(
        _Repository(withdrawn=[("fact-3", "fact"), ("rule-1", "rule")])
    )

    result = await _step(writer).run(_ctx())

    assert writer.removed == [("fact-3", "fact"), ("rule-1", "rule")]
    assert result.ctx.metadata["knowledge.withdrawn"] == 2


async def test_the_mapping_exists_before_the_first_write() -> None:
    # Indexing into an unmapped OpenSearch index stores the document and makes
    # it unsearchable — the exact failure this step exists to prevent.
    writer = _Writer(_Repository(canonical=[_assertion("fact-1")]))

    await _step(writer).run(_ctx())

    assert writer.prepared == 1


async def test_a_full_sweep_works_through_the_graph_in_batches() -> None:
    everything = [_assertion(f"fact-{n}") for n in range(5)]
    writer = _Writer(_Repository(canonical=everything))
    step = _step(writer, batch_size=2)

    result = await step.run(_ctx())
    assert result.verdict == MORE
    result = await step.run(result.ctx)
    assert result.verdict == MORE
    result = await step.run(result.ctx)

    assert result.verdict == "DEFAULT"
    assert writer.indexed == [f"fact-{n}" for n in range(5)]
    assert result.ctx.metadata["knowledge.published"] == 5


async def test_an_empty_graph_is_not_a_failure() -> None:
    writer = _Writer(_Repository())

    result = await _step(writer).run(_ctx())

    assert result.verdict == NOTHING_TO_PUBLISH


async def test_named_identities_are_brought_in_line_one_by_one() -> None:
    # For a caller that wants one decision to take effect immediately rather
    # than waiting for the next sweep.
    repository = _Repository(
        resolved=[
            _assertion("approved"),
            _assertion("refused", quarantined=True),
            _assertion("folded", merged="other"),
        ]
    )
    writer = _Writer(repository)

    result = await _step(writer).run(_ctx({"identities": ["approved", "refused", "folded"]}))

    assert writer.indexed == ["approved"]
    assert writer.removed == [("refused", "fact"), ("folded", "fact")]
    # Quarantined ones have to be asked for, or the step could not tell an
    # assertion that must be withdrawn from one that no longer exists.
    assert repository.resolve_calls[0][1] is True
    assert result.verdict == "DEFAULT"


async def test_identities_that_are_not_a_list_of_strings_are_refused() -> None:
    writer = _Writer(_Repository())

    with pytest.raises(WorkflowConfigurationError, match="list of strings"):
        await _step(writer).run(_ctx({"identities": "fact-1"}))


async def test_a_batch_size_below_one_is_refused() -> None:
    writer = _Writer(_Repository())

    with pytest.raises(WorkflowConfigurationError, match="batch_size"):
        await _step(writer, batch_size=0).run(_ctx())
