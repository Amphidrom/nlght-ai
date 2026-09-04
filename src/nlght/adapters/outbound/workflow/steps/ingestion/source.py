# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import json
import logging

from nlght.adapters.outbound.ingestion.factory import (
    SOURCE_BUILDERS,
    SourceConfigurationError,
    build_source,
)
from nlght.core.errors.errors import WorkflowConfigurationError
from nlght.core.ingestion import SourceSnapshot, stable_digest
from nlght.core.workflow.step import (
    StepBase,
    StepOption,
    StepResult,
    WorkflowStepContext,
)

logger = logging.getLogger(__name__)

INCOMPLETE = "incomplete"
"""Verdict for a snapshot that could not observe its source completely.

Kept distinct from a plain failure: an incomplete snapshot is usable for
indexing what *was* seen, but must never drive deletions. The workflow decides
via its transitions whether to continue or stop.
"""


class IngestionSourceStep(StepBase):
    """Acquires one immutable snapshot from a configured source.

    Step config:
        type:      "filesystem" | "confluence"
        source_id: stable identity of this source
        ... plus the type's own keys (roots/include/exclude, or
            base_url/space_keys/user_email/api_token/cql)

    Emits ``ctx.metadata['ingestion.snapshot']`` plus
    ``ingestion.source_type`` and ``ingestion.config_revision``, which the
    commit step needs to record what produced a snapshot. Returns ``incomplete``
    when the source could not be observed in full, so the workflow can route
    around deletion handling.
    """

    TYPE = "ingestion.source"

    @classmethod
    def options(cls) -> list[StepOption]:
        return [
            StepOption("type", "string", "Source backend", required=True,
                       choices=sorted(SOURCE_BUILDERS), default="filesystem"),
            StepOption("source_id", "string", "Stable identity of this source", required=True,
                       placeholder="internal-docs"),
            StepOption("roots", "array", "filesystem: list of {path, alias?}",
                       placeholder='[{"path": "/data/docs", "alias": "docs"}]'),
            StepOption("include", "array", "filesystem: glob patterns to include",
                       default=["**/*"]),
            StepOption("exclude", "array", "filesystem: glob patterns to exclude", default=[]),
            StepOption("respect_ignore_files", "boolean",
                       "filesystem: honour .gitignore and friends", default=True),
            StepOption("base_url", "string", "confluence: wiki base URL",
                       placeholder="https://example.atlassian.net/wiki"),
            StepOption("space_keys", "array", "confluence: space keys to acquire"),
            StepOption("user_email", "string", "confluence: account e-mail"),
            StepOption("api_token", "string", "confluence: API token"),
            StepOption("cql", "string", "confluence: optional CQL filter"),
            StepOption("max_pages", "integer", "confluence: page cap", default=1000),
            StepOption("page_size", "integer", "confluence: page size", default=50),
            StepOption("timeout_seconds", "number", "confluence: request timeout", default=30.0),
        ]

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        source_type = str(self.config.get("type", "")).strip()
        try:
            source = build_source(source_type, self.config, where=f"Step '{self.TYPE}'")
        except SourceConfigurationError as exc:
            # The factory speaks plainly so both callers can wrap it; a step's
            # caller expects a workflow configuration error.
            raise WorkflowConfigurationError(str(exc)) from exc
        snapshot: SourceSnapshot = await source.acquire()

        # A fanned-out child is told which documents are its share and acquires
        # them from this same source, because a worker allowed to run this
        # workflow can reach what the workflow is configured with. Its view is
        # deliberately partial, so the snapshot it publishes is not complete and
        # can never license a deletion.
        share = _requested_share(ctx)
        if share is not None:
            snapshot = _restrict(snapshot, share)

        ctx.metadata["ingestion.snapshot"] = snapshot
        ctx.metadata["ingestion.source_type"] = source_type
        # A digest rather than the configuration itself: the record needs to
        # show that a source was reconfigured between snapshots, not to hold a
        # copy of credentials that happen to sit in the same step config.
        ctx.metadata["ingestion.config_revision"] = stable_digest(
            json.dumps(self.config, sort_keys=True, default=str)
        )
        logger.info(
            "[%s] ingestion.source.done | source=%s documents=%d complete=%s",
            ctx.correlation_id,
            snapshot.source_id,
            len(snapshot.documents),
            snapshot.complete,
        )
        return StepResult(ctx=ctx, verdict="DEFAULT" if snapshot.complete else INCOMPLETE)


def _requested_share(ctx: WorkflowStepContext) -> tuple[str, ...] | None:
    """The external ids a fan-out gave this execution, if it is a child.

    ``None`` means "acquire the source in full" — the ordinary case. An empty
    list is not the same thing: it is a child that was given nothing, and
    acquiring everything instead would have it index the whole corpus.
    """
    requested = ctx.trigger.payload.get("external_ids")
    if requested is None:
        return None
    if not isinstance(requested, list | tuple):
        raise WorkflowConfigurationError(
            f"Step '{IngestionSourceStep.TYPE}' received an unreadable document share: "
            f"'external_ids' must be a list."
        )
    return tuple(str(item) for item in requested)


def _restrict(snapshot: SourceSnapshot, external_ids: tuple[str, ...]) -> SourceSnapshot:
    """Narrow a snapshot to one share of it, marked incomplete.

    A document named but no longer present at the source is simply absent here:
    it was deleted between the distribution flow's acquisition and this one, and
    a partial view is exactly the wrong place to conclude anything about
    deletion. Losing *every* named document is different — that is not attrition
    but a share that does not line up with what this worker can see, and running
    on with nothing would report an empty success for work never done.
    """
    wanted = set(external_ids)
    documents = tuple(
        document for document in snapshot.documents if document.external_id in wanted
    )
    if external_ids and not documents:
        available = sorted(document.external_id for document in snapshot.documents)
        raise WorkflowConfigurationError(
            f"Step '{IngestionSourceStep.TYPE}' was given {len(external_ids)} document(s) to "
            f"acquire and found none of them in source '{snapshot.source_id}'. This worker sees "
            f"{len(available)} document(s) there, so the share does not match what it can reach — "
            f"check that the distribution flow and the work flow name the same roots. "
            f"Asked for e.g. {external_ids[0]!r}; this source offers e.g. "
            f"{available[0] if available else '(nothing)'!r}."
        )
    return SourceSnapshot(
        source_id=snapshot.source_id,
        documents=documents,
        observed_external_ids=tuple(document.external_id for document in documents),
        complete=False,
        diagnostics=snapshot.diagnostics,
    )
