# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import logging

from nlght.core.errors.errors import WorkflowConfigurationError
from nlght.core.workflow.step import StepBase
from nlght.core.workflow.workflow import WorkflowStepDef

logger = logging.getLogger(__name__)


class StepLoader:
    """Registry that maps step type names to StepBase subclasses.

    New step types are registered via ``register()``.  The loader is passed to
    ``StepMachineWorkflowExecutor`` at construction time so that steps can be
    added without modifying the executor.
    """

    def __init__(self) -> None:
        self._registry: dict[str, type[StepBase]] = {}

    def register(self, step_cls: type[StepBase]) -> None:
        self._registry[step_cls.TYPE] = step_cls
        logger.debug("step.registered | type=%s", step_cls.TYPE)

    def load(self, step_def: WorkflowStepDef) -> StepBase:
        cls = self._registry.get(step_def.type)
        if cls is None:
            raise WorkflowConfigurationError(
                f"Unknown step type '{step_def.type}'. "
                f"Registered types: {sorted(self._registry)}"
            )
        logger.debug("step.load | name=%s type=%s", step_def.name, step_def.type)
        return cls(config=step_def.config)
