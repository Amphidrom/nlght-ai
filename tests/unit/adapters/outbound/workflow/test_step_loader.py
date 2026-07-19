# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

import uuid

import pytest

from nlght.adapters.outbound.workflow.loader import StepLoader
from nlght.core.errors.errors import WorkflowConfigurationError
from nlght.core.workflow.step import StepBase, StepResult, WorkflowStepContext
from nlght.core.workflow.workflow import WorkflowStepDef


class _FooStep(StepBase):
    TYPE = "foo"

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        return StepResult(ctx=ctx, verdict="done")


def _make_step_def(type_: str, config: dict | None = None) -> WorkflowStepDef:
    return WorkflowStepDef(
        step_id=uuid.uuid4(),
        position=0,
        name=type_,
        type=type_,
        enabled=True,
        config=config or {},
        transitions={},
        is_start=True,
        is_terminal=True,
        is_resume=False,
    )


def test_register_and_load_returns_instance() -> None:
    loader = StepLoader()
    loader.register(_FooStep)

    step = loader.load(_make_step_def("foo"))

    assert isinstance(step, _FooStep)


def test_load_passes_config_to_step() -> None:
    loader = StepLoader()
    loader.register(_FooStep)

    step = loader.load(_make_step_def("foo", config={"key": "value"}))

    assert step.config == {"key": "value"}


def test_load_raises_for_unknown_type() -> None:
    loader = StepLoader()

    with pytest.raises(WorkflowConfigurationError, match="Unknown step type"):
        loader.load(_make_step_def("not_registered"))


def test_register_overwrites_previous_registration() -> None:
    class _FooStepV2(StepBase):
        TYPE = "foo"

        async def run(self, ctx: WorkflowStepContext) -> StepResult:
            return StepResult(ctx=ctx)

    loader = StepLoader()
    loader.register(_FooStep)
    loader.register(_FooStepV2)

    step = loader.load(_make_step_def("foo"))

    assert isinstance(step, _FooStepV2)
