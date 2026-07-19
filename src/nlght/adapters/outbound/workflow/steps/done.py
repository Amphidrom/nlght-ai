# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from nlght.core.workflow.step import StepBase, StepResult, WorkflowStepContext


class DoneStep(StepBase):
    """Terminal step that signals successful workflow completion.

    Emits nothing — it is a structural terminal marker only.
    Any output was already emitted by preceding steps.
    """

    TYPE = "done"

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        return StepResult(ctx=ctx, verdict=None)
