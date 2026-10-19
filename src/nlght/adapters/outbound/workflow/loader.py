# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import logging
from collections.abc import Mapping

from nlght.core.errors.errors import WorkflowConfigurationError
from nlght.core.workflow.step import StepBase
from nlght.core.workflow.workflow import WorkflowStepDef

logger = logging.getLogger(__name__)


class StepLoader:
    """Registry that maps step type names to StepBase subclasses.

    New step types are registered via ``register()``.  The loader is passed to
    ``StepMachineWorkflowExecutor`` at construction time so that steps can be
    added without modifying the executor.

    Some steps need a platform capability that cannot travel in JSON step config
    such as a repository or embedding client. ``bind()`` supplies those, mirroring
    how ``ToolLoader.instantiate`` forwards runtime dependencies into a tool.
    They are passed as keyword arguments to the step constructor. A step that
    does not declare them absorbs them through ``StepBase.__init__``.
    """

    def __init__(self, runtime_deps: Mapping[str, object] | None = None) -> None:
        self._registry: dict[str, type[StepBase]] = {}
        self._runtime_deps: dict[str, object] = dict(runtime_deps or {})

    def register(self, step_cls: type[StepBase]) -> None:
        self._registry[step_cls.TYPE] = step_cls
        logger.debug("step.registered | type=%s", step_cls.TYPE)

    def bind(self, **runtime_deps: object) -> StepLoader:
        """Bind runtime dependencies to a loader sharing this registry.

        Args:
            **runtime_deps (object): Dependencies passed to loaded steps.

        Returns:
            StepLoader: New loader with the shared registry and merged dependencies.

        Side Effects:
            Shares the registry by reference. Keeps per-container dependencies
            out of the global ``step_registry`` singleton.
        """
        bound = StepLoader({**self._runtime_deps, **runtime_deps})
        bound._registry = self._registry
        return bound

    def load(self, step_def: WorkflowStepDef) -> StepBase:
        cls = self._registry.get(step_def.type)
        if cls is None:
            raise WorkflowConfigurationError(
                f"Unknown step type '{step_def.type}'. "
                f"Registered types: {sorted(self._registry)}"
            )
        logger.debug("step.load | name=%s type=%s", step_def.name, step_def.type)
        return cls(config=step_def.config, **self._runtime_deps)
