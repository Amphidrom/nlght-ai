# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import logging
from typing import Any

from nlght.core.errors.errors import WorkflowConfigurationError
from nlght.core.knowledge import ExtractedItem
from nlght.core.workflow.step import (
    StepBase,
    StepOption,
    StepResult,
    WorkflowStepContext,
)

logger = logging.getLogger(__name__)

POLICY_ALL = "all"
POLICY_THRESHOLDS = "thresholds"


class KnowledgeReviewFlagStep(StepBase):
    """Marks which candidates must be reviewed before they become retrievable.

    Review is quarantine, never workflow suspension: flagged candidates are
    persisted and surfaced asynchronously while the run continues (ADR-0032).

    Step config:
        policy:         "all" (default) or "thresholds"
        min_confidence: under "thresholds", quarantine below this
        kinds:          under "thresholds", always quarantine these kinds
        types:          under "thresholds", always quarantine these types

    An absent policy defaults to ``all`` — the safe direction, because an
    unreviewed assertion that silently became retrievable is the failure that
    matters here.

    Reads ``knowledge.extracted``, emits ``knowledge.classified`` as
    ``(item, review_reason | None)`` pairs.
    """

    TYPE = "knowledge.review_flag"

    @classmethod
    def options(cls) -> list[StepOption]:
        return [
            StepOption("policy", "string", "Which candidates require review",
                       choices=[POLICY_ALL, POLICY_THRESHOLDS], default=POLICY_ALL),
            StepOption("min_confidence", "number",
                       "thresholds: quarantine below this confidence", default=0.0),
            StepOption("kinds", "array", "thresholds: kinds that always require review",
                       default=[]),
            StepOption("types", "array", "thresholds: types that always require review",
                       default=[]),
        ]

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        items = ctx.metadata.get("knowledge.extracted")
        if not isinstance(items, tuple):
            raise WorkflowConfigurationError(
                f"Step '{self.TYPE}' requires extracted candidates; "
                f"place a 'knowledge.extract' step before it."
            )

        policy = str(self.config.get("policy", POLICY_ALL)).strip().lower()
        if policy not in (POLICY_ALL, POLICY_THRESHOLDS):
            raise WorkflowConfigurationError(
                f"Step '{self.TYPE}' has unknown policy '{policy}'; "
                f"expected '{POLICY_ALL}' or '{POLICY_THRESHOLDS}'."
            )

        classified = tuple((item, self._reason(item, policy)) for item in items)
        ctx.metadata["knowledge.classified"] = classified

        quarantined = sum(1 for _, reason in classified if reason)
        logger.info(
            "[%s] knowledge.review_flag.done | policy=%s total=%d quarantined=%d",
            ctx.correlation_id, policy, len(classified), quarantined,
        )
        return StepResult(ctx=ctx, verdict="DEFAULT")

    def _reason(self, item: ExtractedItem, policy: str) -> str | None:
        if policy == POLICY_ALL:
            return "review policy: all candidates require review"

        minimum = float(self.config.get("min_confidence", 0.0))
        if item.confidence < minimum:
            return f"confidence {item.confidence:.2f} below threshold {minimum:.2f}"

        kinds: list[Any] = list(self.config.get("kinds", []))
        if item.kind in {str(kind) for kind in kinds}:
            return f"kind '{item.kind}' always requires review"

        types: list[Any] = list(self.config.get("types", []))
        if item.type in {str(type_) for type_ in types}:
            return f"type '{item.type}' always requires review"

        return None
