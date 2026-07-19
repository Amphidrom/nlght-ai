# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Workflow adapter — executor, step loader, and built-in step registry.

The ``step_registry`` singleton is the extension point for registering custom steps:

    from nlght.adapters.outbound.workflow import step_registry
    from my_app.steps import MyStep

    step_registry.register(MyStep)
"""

from nlght.adapters.outbound.workflow.executor import StepMachineWorkflowExecutor
from nlght.adapters.outbound.workflow.loader import StepLoader
from nlght.adapters.outbound.workflow.registry import step_registry

__all__ = [
    "StepLoader",
    "StepMachineWorkflowExecutor",
    "step_registry",
]
