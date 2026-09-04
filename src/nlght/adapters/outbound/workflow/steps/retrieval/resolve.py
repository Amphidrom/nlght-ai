# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Findings become what they actually say, read from the authority.

Search says "these systems found these things" and hands over locators; this
turns each locator into the text its source really holds. It exists as its own
step for two reasons, and both are boundaries rather than preferences.

**`core/context` reads no store.** ADR-0052 made that a property rather than a
habit: passage building is pure, synchronous, and adds nothing to a finding. A
reader is I/O and could not live there without taking that property away. So the
fetch happens before it, and `passages_from` keeps receiving findings whose
`content` is simply already true.

**And it happens after the ranking, not during discovery.** Deciding whether a
candidate is still current needs identity only, and the store already did that
before anything was ranked (ADR-0062). Reading *content* is the expensive half —
whole documents, and a document can be an eight-thousand-line file — so it is
spent on the findings a ranking selected rather than on every candidate a search
looked at. With the default `fetch_k` that is twenty reads instead of a hundred.

Assertions pass through untouched. `knowledge_store` resolves its identities
against PostgreSQL during discovery, because a claim is small and its quarantine
has to be enforced before it is ranked at all — so a knowledge hit's `content`
is already authoritative when it arrives here.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from nlght.core.errors.errors import WorkflowConfigurationError
from nlght.core.retrieval import ASSERTION, FusedHit, RetrievalHit
from nlght.core.workflow.step import (
    StepBase,
    StepOption,
    StepResult,
    WorkflowStepContext,
)

if TYPE_CHECKING:
    from nlght.adapters.outbound.stores.connections import StoreConnections
    from nlght.adapters.outbound.tools.activator import ResourceActivator
    from nlght.adapters.outbound.tools.loader import ToolLoader
    from nlght.core.entry.context import RequestContext
    from nlght.ports.outbound.resource_repository import ResourceRepository

logger = logging.getLogger(__name__)

NOTHING_RESOLVED = "unresolved"
"""Verdict when no finding could be answered from the corpus.

Distinct from finding nothing: the search *did* find something and the corpus
cannot say what it says. A pipeline that wants to notice that — a corpus indexed
before its content was stored, say — routes this verdict somewhere.
"""


class RetrievalResolveStep(StepBase):
    """Replaces each finding's text with what its source actually holds.

    Reads and rewrites ``retrieval.hits``; writes ``retrieval.resolved`` with
    the counts. Place it between ``retrieval.search`` and ``prompt.compose``.

    Step config:
        data_store: resource name of the data store activation to read through

    A finding the corpus cannot answer is **dropped**, not answered from the
    index payload. The payload is a projection: it can be rebuilt, re-mapped or
    dropped, it holds a document in two different shapes across two backends,
    and no offset addresses it. Quietly substituting it would hand a model a
    projection while a citation claimed the corpus — so the finding leaves
    instead, and the count says how many did.
    """

    TYPE = "retrieval.resolve"

    def __init__(
        self,
        *,
        config: dict[str, Any],
        resource_repository: ResourceRepository | None = None,
        resource_activator: ResourceActivator | None = None,
        tool_loader: ToolLoader | None = None,
        store_connections: StoreConnections | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(config=config, **kwargs)
        self._resources = resource_repository
        self._activator = resource_activator
        self._tools = tool_loader
        self._connections = store_connections

    @classmethod
    def options(cls) -> list[StepOption]:
        return [
            StepOption("data_store", "string",
                       "Resource name of the data store activation whose corpus "
                       "answers document and chunk findings",
                       required=True, placeholder="data-main"),
        ]

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        found = ctx.metadata.get("retrieval.hits")
        if not isinstance(found, tuple | list):
            raise WorkflowConfigurationError(
                f"Step '{self.TYPE}' requires ranked findings; place a "
                f"'retrieval.search' step before it."
            )
        if not found:
            ctx.metadata["retrieval.resolved"] = {"resolved": 0, "dropped": 0, "claims": 0}
            return StepResult(ctx=ctx, verdict=None)

        store = await self._store(ctx.trigger.context)
        # One read for every finding that names a document, at whatever carrier.
        readable = [
            hit
            for group in found
            for hit in group.hits
            if hit.carrier != ASSERTION
        ]
        texts = await store.resolve(readable) if readable else {}

        groups: list[FusedHit] = []
        resolved = dropped = claims = 0
        for group in found:
            kept: list[RetrievalHit] = []
            for hit in group.hits:
                if hit.carrier == ASSERTION:
                    kept.append(hit)
                    claims += 1
                    continue
                answer = texts.get(hit.carrier_id)
                if answer is None:
                    dropped += 1
                    continue
                kept.append(replace(hit, content=answer.text))
                resolved += 1
            if kept:
                groups.append(replace(group, hits=tuple(kept)))

        ctx.metadata["retrieval.hits"] = tuple(groups)
        ctx.metadata["retrieval.resolved"] = {
            "resolved": resolved,
            "dropped": dropped,
            "claims": claims,
        }
        if dropped:
            logger.warning(
                "[%s] retrieval.resolve.dropped | count=%d — the corpus holds no text "
                "for these revisions, so they were left out rather than answered from "
                "the index payload; reindex to make them readable",
                ctx.correlation_id, dropped,
            )
        logger.info(
            "[%s] retrieval.resolve | resolved=%d claims=%d dropped=%d findings=%d",
            ctx.correlation_id, resolved, claims, dropped, len(groups),
        )
        return StepResult(
            ctx=ctx, verdict=None if groups else NOTHING_RESOLVED
        )

    async def _store(self, caller: RequestContext | None) -> Any:  # noqa: ANN401
        """The data store whose corpus answers these findings.

        Required, so every failure here is an error: unnamed is a
        misconfiguration, unresolvable is a misconfiguration, and denied is a
        denial. None of them is a reason to answer from the index payload.
        """
        name = str(self.config.get("data_store", "")).strip()
        if not name:
            raise WorkflowConfigurationError(
                f"Step '{self.TYPE}' requires a 'data_store' config key naming the "
                f"activation whose corpus answers its findings."
            )
        if self._activator is None:
            raise WorkflowConfigurationError(
                f"Step '{self.TYPE}' has no resource activator, so it cannot reach "
                f"'{name}'."
            )
        return await self._activator.activate(
            kind="data_store", name=name, caller=caller
        )
