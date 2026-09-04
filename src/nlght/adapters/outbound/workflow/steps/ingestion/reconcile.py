# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Compare what the database expects of an index against what the index holds.

Every other guard here prevents divergence. This one finds it. The distinction
matters, because the guards cover the paths the platform controls — a failed
write raises, a recreated index rotates its generation — and none of them covers
what happens outside those paths: a collection dropped by hand, a container
restarted without its volume, a database restored to a different point in time
than the indexes, an index wiped before the generation existed at all.

So the question this answers is deliberately narrow and checkable: for one
target, how many documents does the corpus have, how many does the index state
claim, how many of those claims are on the current generation, and how many
documents does the index actually contain. Those four numbers name every
interesting state without needing to enumerate a corpus.

It reports; it never repairs. Repair is running the pipeline, which is a
decision with a cost attached, and one an operator should take deliberately
rather than as a side effect of asking a question.
"""

from __future__ import annotations

import logging
from typing import Any

from nlght.adapters.outbound.stores.connections import StoreConnections
from nlght.adapters.outbound.stores.data_writer import DataIndexWriterTool
from nlght.adapters.outbound.tools.activator import ResourceActivator
from nlght.adapters.outbound.tools.loader import ToolLoader
from nlght.core.entry.context import RequestContext
from nlght.core.errors.errors import WorkflowConfigurationError
from nlght.core.workflow.step import (
    StepBase,
    StepOption,
    StepResult,
    WorkflowStepContext,
)
from nlght.ports.outbound.resource_repository import ResourceRepository

logger = logging.getLogger(__name__)

DIVERGED = "diverged"
"""Verdict when the index does not hold what the state claims it holds."""

OUTSTANDING = "outstanding"
"""Verdict when the state is consistent but a reindex is unfinished."""


class IngestionReconcileStep(StepBase):
    """Reports one index target's expected state against its actual contents.

    Step config:
        writer:    resource name of the `data_index_writer` activation to check
        tolerance: documents the lexical count may differ by before it counts as
                   divergence (default 0)

    Emits ``ingestion.reconcile`` — the four counts plus the verdict — and
    returns ``diverged``, ``outstanding``, or ``DEFAULT``. Route ``diverged``
    wherever a deployment wants to be told; the step itself only reports.
    """

    TYPE = "ingestion.reconcile"

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
                       "Resource name of the data_index_writer activation to reconcile",
                       required=True, placeholder="data-main-writer"),
            StepOption("tolerance", "integer",
                       "Documents the lexical count may differ by before it counts "
                       "as divergence",
                       default=0),
        ]

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        if self._resources is None or self._tools is None:
            raise WorkflowConfigurationError(
                f"Step '{self.TYPE}' requires a resource repository and tool loader "
                f"to activate the writer it reconciles."
            )
        writer = await self._writer(ctx.trigger.context)
        summary = await writer.repository.index_state_summary(target=writer.target)
        counts = writer.observed_counts()

        tolerance = self._tolerance()
        lexical_gap = summary.indexed - counts.lexical_documents
        # A corpus recorded as indexed while the vector collection is empty is
        # the signature of a wiped index that the generation could not catch —
        # wiped before it existed, or between two runs of one process.
        vectors_missing = summary.indexed > 0 and counts.vector_points == 0
        diverged = abs(lexical_gap) > tolerance or vectors_missing

        report = {
            "target": summary.target,
            "generation": summary.generation,
            "live_documents": summary.live_documents,
            "indexed": summary.indexed,
            "on_current_generation": summary.on_current_generation,
            "outstanding": summary.outstanding,
            "lexical_documents": counts.lexical_documents,
            "vector_points": counts.vector_points,
            "lexical_gap": lexical_gap,
            "diverged": diverged,
        }
        ctx.metadata["ingestion.reconcile"] = report

        logger.info(
            "[%s] ingestion.reconcile | target=%s generation=%s live=%d indexed=%d "
            "on_generation=%d outstanding=%d lexical=%d vectors=%d",
            ctx.correlation_id, summary.target, summary.generation,
            summary.live_documents, summary.indexed, summary.on_current_generation,
            summary.outstanding, counts.lexical_documents, counts.vector_points,
        )

        if diverged:
            stale = await writer.repository.stale_document_ids(target=writer.target, limit=5)
            logger.error(
                "[%s] ingestion.reconcile.diverged | target=%s the index state claims %d "
                "documents, the index holds %d, vectors=%d — rerun the ingestion pipeline "
                "to rebuild it%s",
                ctx.correlation_id, summary.target, summary.indexed,
                counts.lexical_documents, counts.vector_points,
                f"; e.g. {', '.join(stale)}" if stale else "",
            )
            return StepResult(ctx=ctx, verdict=DIVERGED)

        if summary.outstanding:
            logger.warning(
                "[%s] ingestion.reconcile.outstanding | target=%s %d documents are still on an "
                "earlier index generation — a reindex was interrupted; rerunning the pipeline "
                "writes exactly those",
                ctx.correlation_id, summary.target, summary.outstanding,
            )
            return StepResult(ctx=ctx, verdict=OUTSTANDING)

        return StepResult(ctx=ctx, verdict="DEFAULT")

    def _tolerance(self) -> int:
        try:
            tolerance = int(self.config.get("tolerance", 0))
        except ValueError as exc:
            raise WorkflowConfigurationError(
                f"Step '{self.TYPE}' has invalid tolerance: {exc}"
            ) from exc
        if tolerance < 0:
            raise WorkflowConfigurationError(
                f"Step '{self.TYPE}' requires tolerance to be zero or more."
            )
        return tolerance

    async def _writer(self, caller: RequestContext | None) -> DataIndexWriterTool:
        assert self._resources is not None and self._tools is not None  # noqa: S101 (guarded by run)
        name = str(self.config.get("writer", "")).strip()
        if not name:
            raise WorkflowConfigurationError(
                f"Step '{self.TYPE}' requires a 'writer' config key naming an "
                f"enabled '{DataIndexWriterTool.KIND}' resource."
            )
        if self._activator is None:
            raise WorkflowConfigurationError(
                f"Step '{self.TYPE}' has no resource activator, so it cannot "
                f"reach '{name}'."
            )
        instance = await self._activator.activate(
            kind=DataIndexWriterTool.KIND,
            name=name,
            caller=caller,
        )
        if not isinstance(instance, DataIndexWriterTool):
            raise WorkflowConfigurationError(
                f"Resource '{name}' resolves to {type(instance).__name__}, "
                f"which is not a {DataIndexWriterTool.__name__}."
            )
        return instance
