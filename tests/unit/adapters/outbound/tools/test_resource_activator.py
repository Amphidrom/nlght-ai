# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""A resource is authorized before it is built, or it is not built.

The property these hold is narrow and load-bearing: *instantiation does not
happen* for a caller the policy refuses. Not "the instance is discarded", not
"the call fails afterwards" — the constructor is never reached, so no engine is
opened and no client is created on behalf of somebody who may not use it.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import pytest

from nlght.adapters.outbound.tools.activator import ResourceActivator
from nlght.adapters.outbound.tools.loader import ToolLoader
from nlght.core.entry.context import RequestContext
from nlght.core.errors.errors import (
    ResourceAccessDeniedError,
    WorkflowConfigurationError,
)
from nlght.core.runtime.resource import ResourceDef
from nlght.core.tools.tool import ToolBase, ToolSignature


class _CountingTool(ToolBase):
    """Counts its own construction, which is the thing under test."""

    KIND = "counting"
    PROVIDER = "test"
    built = 0

    def __init__(self, *, name: str, config: dict[str, Any], **_: object) -> None:
        super().__init__(name=name, config=config)
        type(self).built += 1

    @classmethod
    def signatures(cls) -> list[ToolSignature]:
        return []


class _Resources:
    def __init__(self, resources: list[ResourceDef]) -> None:
        self._resources = resources

    async def find_by_kind(self, kind: str) -> list[ResourceDef]:
        return [r for r in self._resources if r.kind == kind]

    async def find_by_id(self, resource_id: uuid.UUID) -> ResourceDef | None:
        return next((r for r in self._resources if r.resource_id == resource_id), None)

    async def list_enabled(self) -> list[ResourceDef]:
        return list(self._resources)


class _Policy:
    """Records what it was asked, and answers as instructed."""

    def __init__(self, allow: bool = True, raises: bool = False) -> None:
        self._allow = allow
        self._raises = raises
        self.asked: list[tuple[str, str | None]] = []

    async def is_allowed(
        self, resource: ResourceDef, caller: RequestContext | None, model: str | None
    ) -> bool:
        self.asked.append((resource.address, caller.workflow if caller else None))
        if self._raises:
            raise RuntimeError("policy backend unreachable")
        return self._allow


def _resource(name: str = "main", kind: str = "counting") -> ResourceDef:
    return ResourceDef(
        resource_id=uuid.uuid4(), name=name, kind=kind, provider="test", config={},
    )


def _caller(workflow: str | None = "answer") -> RequestContext:
    return RequestContext(
        correlation_id="cid", request_id="rid",
        received_at=datetime(2026, 1, 1, tzinfo=UTC),
        path="/", method="POST", headers={}, query_params={}, client_host=None,
        workflow=workflow,
    )


def _activator(resources: list[ResourceDef], policy: object | None) -> ResourceActivator:
    loader = ToolLoader()
    loader.register(_CountingTool)
    return ResourceActivator(
        resources=_Resources(resources), loader=loader, access_policy=policy,
    )


@pytest.fixture(autouse=True)
def _reset_construction_count() -> None:
    _CountingTool.built = 0


async def test_an_allowed_resource_is_built() -> None:
    policy = _Policy(allow=True)
    activator = _activator([_resource()], policy)

    instance = await activator.activate(kind="counting", name="main", caller=_caller())

    assert isinstance(instance, _CountingTool)
    assert _CountingTool.built == 1
    assert policy.asked == [("counting/main", "answer")]


async def test_a_denied_resource_is_never_instantiated() -> None:
    """The whole point of putting the guard before the loader.

    Constructing a store opens engines and clients. A caller who may not use the
    resource must not cause connections to be made on their behalf, so the
    denial has to land before the constructor, not after it.
    """
    activator = _activator([_resource()], _Policy(allow=False))

    with pytest.raises(ResourceAccessDeniedError) as denied:
        await activator.activate(kind="counting", name="main", caller=_caller("other"))

    assert _CountingTool.built == 0, "a denied resource was constructed anyway"
    assert denied.value.address == "counting/main"
    assert denied.value.workflow == "other"


async def test_a_failing_policy_denies_and_does_not_build() -> None:
    """Whatever went wrong, authorization must not be what follows from it."""
    activator = _activator([_resource()], _Policy(raises=True))

    with pytest.raises(ResourceAccessDeniedError):
        await activator.activate(kind="counting", name="main", caller=_caller())

    assert _CountingTool.built == 0


async def test_the_policy_sees_the_workflow_the_executor_is_running() -> None:
    """`workflow` as a condition is only worth anything if it is the real one.

    The executor overwrites `RequestContext.workflow` with the definition it
    resolved, so what arrives here cannot be a workflow a request claimed.
    """
    policy = _Policy(allow=True)
    activator = _activator([_resource()], policy)

    await activator.activate(kind="counting", name="main", caller=_caller("ingest-kep"))

    assert policy.asked == [("counting/main", "ingest-kep")]


async def test_a_caller_outside_any_workflow_is_still_asked_about() -> None:
    """No system bypass: running outside a flow is not a way past a rule.

    The activator does not skip the check for a caller without a workflow; it
    passes it through, and a workflow-scoped rule then denies it on its own
    terms rather than by an exemption written here.
    """
    policy = _Policy(allow=True)
    activator = _activator([_resource()], policy)

    await activator.activate(kind="counting", name="main", caller=None)

    assert policy.asked == [("counting/main", None)]


async def test_without_a_policy_everything_is_allowed() -> None:
    """An unconfigured policy restricts nothing — the documented default."""
    activator = _activator([_resource()], None)

    await activator.activate(kind="counting", name="main", caller=_caller())

    assert _CountingTool.built == 1


async def test_an_unresolvable_address_is_a_configuration_error() -> None:
    activator = _activator([_resource()], _Policy(allow=True))

    with pytest.raises(WorkflowConfigurationError, match="No enabled .* named 'absent'"):
        await activator.activate(kind="counting", name="absent", caller=_caller())

    assert _CountingTool.built == 0


async def test_an_ambiguous_address_is_refused_rather_than_guessed() -> None:
    """The constraint should make this impossible; if it is missing, say so.

    Taking the first row is what this whole slice removes — an address that two
    rows answer to cannot carry an authorization decision, and silently picking
    one is how that stayed invisible.
    """
    activator = _activator([_resource(), _resource()], _Policy(allow=True))

    with pytest.raises(WorkflowConfigurationError, match="matches 2 enabled"):
        await activator.activate(kind="counting", name="main", caller=_caller())

    assert _CountingTool.built == 0
