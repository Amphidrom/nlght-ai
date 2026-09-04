# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Spread one acquisition's documents across the worker pool.

The alternative this replaces is a single execution looping over the corpus on
the worker that acquired it: serial, bounded by one step machine's hop budget,
and unable to use a second machine no matter how many are idle. Here the
acquiring run submits one child execution per document batch and ends. Any
capable worker claims any child, so the corpus is processed concurrently, and
the pool — not the run — is the limit.

A child is an ordinary execution: capability matching, leases, fencing tokens,
retries, and at-least-once delivery all apply to it unchanged. It differs only in
carrying ``parent_execution_id``, which is what keeps the work of one logical run
findable after it has been spread out.

What travels to a child is the *names* of its documents, never their content. A
worker that is allowed to run the ingestion workflow can reach the source the
workflow is configured with — that is what its capability means — so the child
acquires its own batch from the same source rather than being fed through the
database. That sentence only holds if the child carries the work flow's declared
capabilities, so it does: they are required of every child, and the step's own
`capabilities` narrow it further.

A deployment either distributes its ingestion or it does not, and the two look
different. Distributing means two workflows: a **distribution flow** that
acquires and hands out shares, and a **work flow** that does the work on one
share. This step is the whole point of the first, and never appears in the
second. A deployment that does not distribute has one flow that acquires and
works, and no fan-out step anywhere.

That is why there is no switch here: which pipeline is seeded says whether the
architecture is distributed, so a flow containing this step is a distribution
flow by construction — with no dead branch for the mode it is not in.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

from nlght.core.errors.errors import WorkflowConfigurationError
from nlght.core.execution import ExecutionSubmission
from nlght.core.ingestion.acquisition import SourceSnapshot
from nlght.core.trigger.trigger import Trigger, TriggerKind
from nlght.core.workflow.step import (
    StepBase,
    StepOption,
    StepResult,
    WorkflowStepContext,
)
from nlght.core.workflow.workflow import WorkflowDef
from nlght.ports.outbound.workflow_repository import WorkflowRepository

logger = logging.getLogger(__name__)

NOTHING_TO_FAN_OUT = "empty"
"""Verdict when the snapshot held no documents; there is nothing to distribute."""


class IngestionFanOutStep(StepBase):
    """Submits one child execution per document batch of the acquired snapshot.

    Step config:
        workflow:            the work flow each child runs
        document_batch_size: documents handed to one child (default 1)
        capabilities:        capabilities required of a worker *in addition to*
                             those the child workflow declares

    Reads ``ingestion.snapshot``; emits ``ingestion.fan_out.children`` (the child
    execution ids) and ``ingestion.fan_out.batches``.

    On success this step ends the run: it does not wait for its children, because
    waiting would hold a worker for as long as the corpus takes — and on a
    single-worker deployment a waiting parent would keep its own children from
    ever being claimed. Whether the corpus is finished is asked of the children,
    through ``ExecutionRepository.fan_out_summary``.
    """

    TYPE = "ingestion.fan_out"

    def __init__(
        self,
        *,
        config: dict[str, Any],
        workflow_repository: WorkflowRepository | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(config=config, **kwargs)
        self._workflows = workflow_repository

    @classmethod
    def options(cls) -> list[StepOption]:
        return [
            StepOption("workflow", "string",
                       "The work flow each child execution runs",
                       required=True, placeholder="ingest-data-document"),
            StepOption("document_batch_size", "integer",
                       "Documents handed to one child execution", default=1),
            StepOption("capabilities", "string",
                       "Comma-separated capabilities a worker needs on top of the ones the "
                       "child workflow itself declares",
                       default=""),
        ]

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        snapshot = ctx.metadata.get("ingestion.snapshot")
        if not isinstance(snapshot, SourceSnapshot):
            raise WorkflowConfigurationError(
                f"Step '{self.TYPE}' requires an acquired snapshot; "
                f"place an 'ingestion.source' step before it."
            )
        child_workflow_name = str(self.config.get("workflow", "")).strip()
        if not child_workflow_name:
            raise WorkflowConfigurationError(
                f"Step '{self.TYPE}' requires 'workflow': the work flow each child runs."
            )
        batch_size = self._batch_size()

        # Fanning out means handing work to other processes. Each of these is a
        # capability this deployment either has or has not; none can be
        # improvised, so name the missing one rather than silently running the
        # corpus serially here.
        if ctx.execution_id is None:
            raise WorkflowConfigurationError(
                f"Step '{self.TYPE}' runs only inside a dispatched execution — a workflow "
                f"executed inline in a request has no execution to parent its children to. "
                f"Submit this workflow through the durable queue."
            )
        if ctx.dispatcher is None:
            raise WorkflowConfigurationError(
                f"Step '{self.TYPE}' requires the durable execution queue; "
                f"configure 'integrations.persistence.workflows.url'."
            )
        if self._workflows is None:
            raise WorkflowConfigurationError(
                f"Step '{self.TYPE}' requires the workflow repository to resolve "
                f"'{child_workflow_name}'."
            )

        child_workflow = await self._workflows.find_by_name(child_workflow_name)
        if child_workflow is None:
            raise WorkflowConfigurationError(
                f"Step '{self.TYPE}' cannot fan out: no workflow named '{child_workflow_name}'."
            )
        version = await self._workflows.find_active_version(child_workflow.workflow_id)
        if version is None:
            raise WorkflowConfigurationError(
                f"Step '{self.TYPE}' cannot fan out: workflow '{child_workflow_name}' "
                f"has no active version."
            )

        documents = snapshot.documents
        if not documents:
            ctx.metadata["ingestion.fan_out.children"] = ()
            ctx.metadata["ingestion.fan_out.batches"] = 0
            return StepResult(ctx=ctx, verdict=NOTHING_TO_FAN_OUT)

        capabilities = self._capabilities(child_workflow)
        children: list[str] = []
        for index in range(0, len(documents), batch_size):
            batch = documents[index : index + batch_size]
            record = await ctx.dispatcher.submit(
                ExecutionSubmission(
                    workflow_id=child_workflow.workflow_id,
                    workflow_version_id=version.version_id,
                    trigger=self._child_trigger(
                        ctx,
                        batch_index=index // batch_size,
                        source_id=snapshot.source_id,
                        external_ids=tuple(document.external_id for document in batch),
                    ),
                    # Derived from the parent and the batch, so a retried fan-out
                    # re-submits the same children instead of duplicating the
                    # corpus. At-least-once delivery makes this the difference
                    # between a rerun and a doubled index.
                    idempotency_key=f"fanout:{ctx.execution_id}:{index // batch_size}",
                    required_capabilities=capabilities,
                    parent_execution_id=ctx.execution_id,
                )
            )
            children.append(str(record.execution_id))

        ctx.metadata["ingestion.fan_out.children"] = tuple(children)
        ctx.metadata["ingestion.fan_out.batches"] = len(children)
        logger.info(
            "[%s] ingestion.fan_out.done | workflow=%s documents=%d children=%d batch=%d",
            ctx.correlation_id, child_workflow_name, len(documents), len(children), batch_size,
        )
        return StepResult(ctx=ctx, verdict="DEFAULT")

    def _batch_size(self) -> int:
        try:
            batch_size = int(self.config.get("document_batch_size", 1))
        except ValueError as exc:
            raise WorkflowConfigurationError(
                f"Step '{self.TYPE}' has invalid document_batch_size: {exc}"
            ) from exc
        if batch_size < 1:
            raise WorkflowConfigurationError(
                f"Step '{self.TYPE}' requires document_batch_size to be at least 1."
            )
        return batch_size

    def _capabilities(self, child_workflow: WorkflowDef) -> tuple[str, ...]:
        """What a worker must have to claim a child.

        The child workflow's own declared capabilities come first, because that
        is what every other way of submitting it already does — the HTTP and
        streaming paths both derive `required_capabilities` from
        `workflow.capabilities`. Reading them only there and not here meant a
        workflow restricted to particular workers was restricted when a caller
        submitted it and open to any worker when a fan-out did, which is
        precisely the machine that cannot reach the source picking up a share.

        The step's own `capabilities` narrow that further, for a requirement
        that belongs to this distribution rather than to the flow itself.
        """
        configured = self.config.get("capabilities", ())
        if isinstance(configured, str):
            items = configured.split(",")
        elif isinstance(configured, list | tuple):
            items = [str(item) for item in configured]
        else:
            raise WorkflowConfigurationError(
                f"Step '{self.TYPE}' requires 'capabilities' to be a list or a "
                f"comma-separated string."
            )
        items = list(items) + list(child_workflow.capabilities or ())
        return tuple(sorted({item.strip() for item in items if item.strip()}))

    @staticmethod
    def _child_trigger(
        ctx: WorkflowStepContext,
        *,
        batch_index: int,
        source_id: str,
        external_ids: tuple[str, ...],
    ) -> Trigger:
        """The child's trigger: which documents of which source are its share.

        Each child gets its **own** correlation id, derived from the parent's so
        the lineage stays greppable (``<parent>:fanout:3``). It must not simply
        inherit the parent's: the runtime treats a correlation id as the identity
        of one run, and children sharing it would collide on every key built from
        it — the ingestion snapshot record, the executor's "already running"
        guard, and the per-run workspace. Deriving it also keeps it deterministic,
        so a retried fan-out submits the same children rather than new ones.
        """
        parent_context = ctx.trigger.context
        suffix = f"fanout:{batch_index}"
        return Trigger(
            # Driven by the platform's own queue rather than by a caller, which
            # is what INBOUND_EVENT already covers.
            kind=TriggerKind.INBOUND_EVENT,
            protocol=ctx.trigger.protocol,
            operation="ingestion.document_batch",
            payload={
                "source_id": source_id,
                "external_ids": list(external_ids),
                "parent_execution_id": str(ctx.execution_id),
                "parent_correlation_id": parent_context.correlation_id,
            },
            context=dataclasses.replace(
                parent_context,
                correlation_id=f"{parent_context.correlation_id}:{suffix}",
                request_id=f"{parent_context.request_id}:{suffix}",
            ),
        )
