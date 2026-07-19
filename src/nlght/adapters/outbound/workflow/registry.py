# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from nlght.adapters.outbound.workflow.loader import StepLoader
from nlght.adapters.outbound.workflow.steps.done import DoneStep
from nlght.adapters.outbound.workflow.steps.failed import FailedStep
from nlght.adapters.outbound.workflow.steps.passthrough import PassthroughStep

step_registry: StepLoader = StepLoader()

# Built-in terminal steps — always available, DB entries optional.
# The step machine also handles "done"/"failed" as reserved verdicts directly,
# so workflows do not need explicit done/failed nodes unless desired for clarity.
step_registry.register(DoneStep)
step_registry.register(FailedStep)
step_registry.register(PassthroughStep)
