# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from nlght.adapters.outbound.workflow.loader import StepLoader
from nlght.adapters.outbound.workflow.steps.done import DoneStep
from nlght.adapters.outbound.workflow.steps.failed import FailedStep
from nlght.adapters.outbound.workflow.steps.passthrough import PassthroughStep

step_registry: StepLoader = StepLoader()

# Register terminal steps even without database entries.
# The step machine also handles "done"/"failed" as reserved verdicts directly,
# so workflows do not need explicit done/failed nodes unless desired for clarity.
step_registry.register(DoneStep)
step_registry.register(FailedStep)
step_registry.register(PassthroughStep)

# Compose ingestion pipelines from these workflow steps.
from nlght.adapters.outbound.workflow.steps.ingestion import (  # noqa: E402 (registration order)
    IngestionEmbedStep,
    IngestionFanOutStep,
    IngestionProcessStep,
    IngestionReconcileStep,
    IngestionSourceStep,
    IngestionWriteStep,
)

step_registry.register(IngestionSourceStep)
step_registry.register(IngestionFanOutStep)
step_registry.register(IngestionProcessStep)
step_registry.register(IngestionEmbedStep)
step_registry.register(IngestionWriteStep)
step_registry.register(IngestionReconcileStep)

from nlght.adapters.outbound.workflow.steps.knowledge import (  # noqa: E402 (registration order)
    AsciiDocParseStep,
    HtmlParseStep,
    KnowledgeAtomicityStep,
    KnowledgeCanonicalizeStep,
    KnowledgeCollapseStep,
    KnowledgeDedupStep,
    KnowledgeDomainClassifyStep,
    KnowledgeExtractStep,
    KnowledgeGraphQualityStep,
    KnowledgeIdentityMergeStep,
    KnowledgeMetaEnrichmentStep,
    KnowledgeNoiseFilterStep,
    KnowledgeNormalizeStep,
    KnowledgeParseAutoStep,
    KnowledgeParseStep,
    KnowledgePersistStep,
    KnowledgePublishStep,
    KnowledgeQualityScoreStep,
    KnowledgeReviewFlagStep,
    KnowledgeValidateStep,
    MarkdownParseStep,
    PdfParseStep,
)

# Register retrieval steps for data written by ingestion workflows.
from nlght.adapters.outbound.workflow.steps.prompt import (  # noqa: E402 (registration order)
    PromptComposeStep,
)
from nlght.adapters.outbound.workflow.steps.retrieval import (  # noqa: E402 (registration order)
    RetrievalResolveStep,
    RetrievalSearchStep,
)

step_registry.register(RetrievalSearchStep)
step_registry.register(RetrievalResolveStep)
step_registry.register(PromptComposeStep)

step_registry.register(KnowledgeParseStep)
step_registry.register(KnowledgeExtractStep)
step_registry.register(KnowledgeReviewFlagStep)
step_registry.register(KnowledgePersistStep)
step_registry.register(KnowledgePublishStep)

# Keep refinement steps independent so workflows can select the needed stages.
for _refinement in (
    KnowledgeCanonicalizeStep,
    KnowledgeValidateStep,
    KnowledgeDomainClassifyStep,
    KnowledgeNormalizeStep,
    KnowledgeAtomicityStep,
    KnowledgeIdentityMergeStep,
    KnowledgeCollapseStep,
    KnowledgeDedupStep,
    KnowledgeNoiseFilterStep,
    KnowledgeQualityScoreStep,
    KnowledgeGraphQualityStep,
    KnowledgeMetaEnrichmentStep,
):
    step_registry.register(_refinement)

# Select these parsers instead of knowledge.parse for structured markup.
# knowledge.parse_auto dispatches across them by file extension.
for _parser in (
    MarkdownParseStep,
    AsciiDocParseStep,
    HtmlParseStep,
    PdfParseStep,
    KnowledgeParseAutoStep,
):
    step_registry.register(_parser)
