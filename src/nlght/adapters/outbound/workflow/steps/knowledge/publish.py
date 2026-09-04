# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Make reviewed knowledge findable, and take back what must not be.

This is how an approved assertion reaches search. Extraction fills PostgreSQL,
review decides what may be read, and this step carries that decision into the
index — nothing else does. Without it an approval changes a flag and nothing
else, and the assertion stays invisible.

It is deliberately not tied to a single decision. A review session works a batch
of assertions and then asks for more; that moment is the natural boundary, and a
sweep is what suits it: it publishes everything approved since the last one and
withdraws everything refused, regardless of who decided it, in which browser, or
whether a tab was closed before anything could be sent. The same sweep rebuilds
an index that was deleted and repairs one that drifted, because there is only one
thing it can do — make the index match what PostgreSQL says may be found.

PostgreSQL holds everything the index needs: the assertion, its payload, and the
metadata carrying its keywords. Nothing here re-reads a source or asks a model,
so a full sweep costs a copy and not an extraction.

Handing it a list of ``identities`` restricts it to those, for a caller that
wants one decision to take effect immediately.
"""

from __future__ import annotations

import logging
from typing import Any

from nlght.adapters.outbound.stores.connections import StoreConnections
from nlght.adapters.outbound.stores.knowledge_writer import KnowledgeIndexWriterTool
from nlght.adapters.outbound.tools.activator import ResourceActivator
from nlght.adapters.outbound.tools.loader import ToolLoader
from nlght.core.entry.context import RequestContext
from nlght.core.errors.errors import WorkflowConfigurationError
from nlght.core.knowledge import KnowledgeObject
from nlght.core.workflow.step import (
    StepBase,
    StepOption,
    StepResult,
    WorkflowStepContext,
)
from nlght.ports.outbound.resource_repository import ResourceRepository

logger = logging.getLogger(__name__)

MORE = "more"
"""Verdict while assertions remain. Route it back to this step."""

NOTHING_TO_PUBLISH = "empty"
"""Verdict when the graph held nothing at all."""


class KnowledgePublishStep(StepBase):
    """Copies canonical assertions into the search index and removes the rest.

    Step config:
        writer:     resource name of the `knowledge_index_writer` activation
        batch_size: assertions handled by one invocation (default 200)

    Reads an optional ``identities`` list from the trigger payload; without one
    it sweeps the whole graph. Emits ``knowledge.published`` and
    ``knowledge.withdrawn``.

    Idempotent by construction: publishing is an upsert keyed by identity and
    withdrawal ignores an absent document, so running it twice changes nothing
    and running it after a failure finishes the job.
    """

    TYPE = "knowledge.publish"

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
            StepOption("writer", "string",
                       "Resource name of the knowledge_index_writer activation to publish through",
                       required=True, placeholder="knowledge-index"),
            StepOption("batch_size", "integer",
                       "Assertions handled by one step invocation", default=200),
        ]

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        if self._resources is None or self._tools is None:
            raise WorkflowConfigurationError(
                f"Step '{self.TYPE}' requires a resource repository and tool loader "
                f"to activate its writer."
            )
        writer = await self._writer(ctx.trigger.context)
        batch_size = self._batch_size()
        repository = writer.repository

        identities = self._identities(ctx)
        if identities is not None:
            return await self._publish_named(ctx, writer, identities)

        # The mapping must exist before the first write: indexing into an
        # OpenSearch index with no mapping stores the document and makes it
        # unsearchable, which is the failure this step exists to prevent.
        writer.ensure_ready()

        published = int(ctx.metadata.get("knowledge.published", 0))
        withdrawn = int(ctx.metadata.get("knowledge.withdrawn", 0))
        offset = int(ctx.metadata.get("knowledge.publish.offset", 0))

        canonical = await repository.canonical_page(limit=batch_size, offset=offset)
        for assertion in canonical:
            await self._index(writer, assertion)
        published += len(canonical)

        # Withdrawals are paged against their own offset, so the two sweeps do
        # not have to be the same length.
        withdrawn_offset = int(ctx.metadata.get("knowledge.withdraw.offset", 0))
        gone = await repository.withdrawn_page(limit=batch_size, offset=withdrawn_offset)
        for identity, kind in gone:
            await writer.remove(identity=identity, kind=kind)
        withdrawn += len(gone)

        ctx.metadata["knowledge.published"] = published
        ctx.metadata["knowledge.withdrawn"] = withdrawn
        ctx.metadata["knowledge.publish.offset"] = offset + len(canonical)
        ctx.metadata["knowledge.withdraw.offset"] = withdrawn_offset + len(gone)

        logger.info(
            "[%s] knowledge.publish.batch | target=%s published=%d withdrawn=%d",
            ctx.correlation_id, writer.name, published, withdrawn,
        )

        if len(canonical) == batch_size or len(gone) == batch_size:
            return StepResult(ctx=ctx, verdict=MORE)
        if not published and not withdrawn:
            return StepResult(ctx=ctx, verdict=NOTHING_TO_PUBLISH)
        return StepResult(ctx=ctx, verdict="DEFAULT")

    async def _publish_named(
        self, ctx: WorkflowStepContext, writer: KnowledgeIndexWriterTool, identities: list[str]
    ) -> StepResult:
        """Bring exactly these assertions in line, in one invocation.

        Each one goes in or comes out according to what PostgreSQL says it is
        now — the same rule as the sweep, applied to a named set rather than to
        everything.
        """
        writer.ensure_ready()
        found = await writer.repository.resolve(identities, include_quarantined=True)
        published = withdrawn = 0
        for assertion in found:
            if assertion.review_required or assertion.merged_into is not None:
                await writer.remove(identity=assertion.identity, kind=assertion.kind)
                withdrawn += 1
            else:
                await self._index(writer, assertion)
                published += 1

        ctx.metadata["knowledge.published"] = published
        ctx.metadata["knowledge.withdrawn"] = withdrawn
        logger.info(
            "[%s] knowledge.publish.named | target=%s asked=%d published=%d withdrawn=%d",
            ctx.correlation_id, writer.name, len(identities), published, withdrawn,
        )
        if not published and not withdrawn:
            return StepResult(ctx=ctx, verdict=NOTHING_TO_PUBLISH)
        return StepResult(ctx=ctx, verdict="DEFAULT")

    @staticmethod
    async def _index(writer: KnowledgeIndexWriterTool, assertion: KnowledgeObject) -> None:
        keywords = assertion.metadata.get("keywords")
        await writer.index(
            assertion, keywords=list(keywords) if isinstance(keywords, list) else None
        )

    def _identities(self, ctx: WorkflowStepContext) -> list[str] | None:
        """The named subset, when the trigger asked for one."""
        raw = ctx.trigger.payload.get("identities")
        if raw is None:
            return None
        if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
            raise WorkflowConfigurationError(
                f"Step '{self.TYPE}' expects 'identities' to be a list of strings; "
                f"omit it entirely to sweep the whole graph."
            )
        return [item for item in raw if item.strip()]

    def _batch_size(self) -> int:
        try:
            batch_size = int(self.config.get("batch_size", 200))
        except ValueError as exc:
            raise WorkflowConfigurationError(
                f"Step '{self.TYPE}' has invalid batch_size: {exc}"
            ) from exc
        if batch_size < 1:
            raise WorkflowConfigurationError(
                f"Step '{self.TYPE}' requires batch_size to be at least 1."
            )
        return batch_size

    async def _writer(self, caller: RequestContext | None) -> KnowledgeIndexWriterTool:
        assert self._resources is not None and self._tools is not None  # noqa: S101 (guarded by run)
        name = str(self.config.get("writer", "")).strip()
        if not name:
            raise WorkflowConfigurationError(
                f"Step '{self.TYPE}' requires a 'writer' config key naming an "
                f"enabled '{KnowledgeIndexWriterTool.KIND}' resource."
            )
        if self._activator is None:
            raise WorkflowConfigurationError(
                f"Step '{self.TYPE}' has no resource activator, so it cannot "
                f"reach '{name}'."
            )
        instance = await self._activator.activate(
            kind=KnowledgeIndexWriterTool.KIND,
            name=name,
            caller=caller,
        )
        if not isinstance(instance, KnowledgeIndexWriterTool):
            raise WorkflowConfigurationError(
                f"Resource '{name}' resolves to {type(instance).__name__}, "
                f"which is not a {KnowledgeIndexWriterTool.__name__}."
            )
        return instance
