# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from nlght.core.entry.context import RequestContext
    from nlght.core.runtime.resource import ResourceDef


@runtime_checkable
class ResourceAccessPolicy(Protocol):
    """Controls which callers may use a given resource *at all*.

    A resource is authorized as a whole: whether this caller may activate it.
    That is the only question for a resource used inside a workflow step, where
    no tool call happens, and the first of two for one whose signatures a model
    may call — ``ToolAccessPolicy`` decides the second, and cannot overrule this
    one.

    ``model`` is the effective model of the request (None when no model is bound
    yet); ``caller`` may be None for internal invocations without a request
    context.

    If no policy is configured every resource is accessible (default behaviour).
    """

    async def is_allowed(
        self,
        resource: ResourceDef,
        caller: RequestContext | None,
        model: str | None,
    ) -> bool: ...


@runtime_checkable
class ToolAccessPolicy(Protocol):
    """Controls which single operations of an authorized resource are offered.

    Checked per ``ToolSignature`` during catalog construction; a signature for
    which this returns False is left out of the request-scoped catalog, so a
    model is never told it exists.

    It decides *within* a resource the caller may already use. It is asked only
    after ``ResourceAccessPolicy`` allowed the resource, and allowing a signature
    of a resource that was denied is not expressible — which is the point of
    having two.

    Wire concrete implementations via
    ``ToolCatalogBuilder(resource_access_policy=..., tool_access_policy=...)``.
    If no policy is configured every signature is offered (default behaviour).
    """

    async def is_allowed(
        self,
        resource: ResourceDef,
        signature_name: str,
        caller: RequestContext | None,
        model: str | None,
    ) -> bool: ...


@runtime_checkable
class ModelAccessPolicy(Protocol):
    """Controls which callers may invoke a given model.

    Checked by StepMachineWorkflowExecutor._bind_llm() before a BoundModelClient
    is created. Raises WorkflowConfigurationError when access is denied so the
    step machine terminates cleanly.

    Wire a concrete implementation via StepMachineWorkflowExecutor(model_access_policy=...).
    If no policy is configured all models are accessible (default behaviour).
    """

    async def is_allowed(
        self,
        model_name: str,
        caller: RequestContext,
    ) -> bool: ...


@runtime_checkable
class PlaybookAccessPolicy(Protocol):
    """Controls which callers may invoke a given playbook.

    Checked by PlaybookCatalogBuilder.build() — playbooks for which this
    returns False are excluded from the request-scoped catalog. ``model`` is
    the effective model of the request; ``caller`` may be None for internal
    invocations without a request context.

    Wire a concrete implementation via PlaybookCatalogBuilder(access_policy=...).
    If no policy is configured all playbooks are accessible (default behaviour).
    """

    async def is_allowed(
        self,
        playbook_name: str,
        caller: RequestContext | None,
        model: str | None,
    ) -> bool: ...
