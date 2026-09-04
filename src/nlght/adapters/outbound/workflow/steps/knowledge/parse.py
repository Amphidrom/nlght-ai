# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import logging

from nlght.core.errors.errors import WorkflowConfigurationError
from nlght.core.knowledge import KnowledgeUnit, split_into_units
from nlght.core.workflow.step import (
    StepBase,
    StepOption,
    StepResult,
    WorkflowStepContext,
)

logger = logging.getLogger(__name__)

EMPTY = "empty"


class KnowledgeParseStep(StepBase):
    """Splits processed documents into knowledge units.

    A unit is one coherent block of prose — the exact text handed to the
    extraction model. Splitting on blank lines and ALL-CAPS headings is the
    prototype's rule: a model reasons better over one block than over a whole
    document, and blank lines survive plain text, Markdown, and converted PDFs
    alike.

    Step config:
        tags: static tags attached to every unit (e.g. product, version)

    Reads ``ctx.metadata['ingestion.processed']``, emits
    ``ctx.metadata['knowledge.units']``.
    """

    TYPE = "knowledge.parse"

    @classmethod
    def options(cls) -> list[StepOption]:
        return [
            StepOption("tags", "object",
                       "Static tags attached to every unit, e.g. product and version",
                       placeholder='{"product": "spring", "version": "3.2"}'),
        ]

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        documents = ctx.metadata.get("ingestion.processed")
        if not isinstance(documents, tuple):
            raise WorkflowConfigurationError(
                f"Step '{self.TYPE}' requires processed documents; "
                f"place an 'ingestion.process' step before it."
            )

        tags = dict(self.config.get("tags", {}))
        units: list[KnowledgeUnit] = []
        for document in documents:
            if not document.text:
                continue
            units.extend(
                split_into_units(
                    document.text,
                    source_id=document.source,
                    document_id=document.document_id,
                    processing_revision_id=document.processing_revision_id,
                    document_path=document.path,
                    tags=tags,
                )
            )

        ctx.metadata["knowledge.units"] = tuple(units)
        logger.info("[%s] knowledge.parse.done | units=%d", ctx.correlation_id, len(units))
        return StepResult(ctx=ctx, verdict="DEFAULT" if units else EMPTY)
