# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Resolve, authorize, then instantiate resources requested by workflow steps.

Authorization precedes construction because constructors can open connections.
Denied access raises instead of silently omitting a required resource.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from nlght.adapters.outbound.stores.connections import StoreConnections
from nlght.adapters.outbound.tools.loader import ToolLoader
from nlght.core.errors.errors import ResourceAccessDeniedError, WorkflowConfigurationError
from nlght.core.runtime.resource import ResourceDef
from nlght.ports.outbound.resource_repository import ResourceRepository

if TYPE_CHECKING:
    from nlght.core.entry.context import RequestContext
    from nlght.core.tools.tool import ToolBase
    from nlght.ports.outbound.access_policy import ResourceAccessPolicy

logger = logging.getLogger(__name__)


class ResourceActivator:
    """Resolves a resource address, authorizes it, and only then builds it."""

    def __init__(
        self,
        *,
        resources: ResourceRepository,
        loader: ToolLoader,
        access_policy: ResourceAccessPolicy | None = None,
        store_connections: StoreConnections | None = None,
    ) -> None:
        self._resources = resources
        self._loader = loader
        self._access_policy = access_policy
        self._connections = store_connections

    async def resolve(self, *, kind: str, name: str) -> ResourceDef:
        """Find the single enabled resource at an address.

        Args:
            kind (str): Resource kind in the address.
            name (str): Resource name in the address.

        Returns:
            ResourceDef: Resource at the requested address.

        Raises:
            WorkflowConfigurationError: No resource exists or multiple rows
                violate the unique-address constraint from migration 0007.
        """
        found = [
            resource
            for resource in await self._resources.find_by_kind(kind)
            if resource.name == name
        ]
        if not found:
            raise WorkflowConfigurationError(
                f"No enabled '{kind}' resource named '{name}'."
            )
        if len(found) > 1:
            raise WorkflowConfigurationError(
                f"Resource address '{kind}/{name}' matches {len(found)} enabled "
                f"resources. The address must identify one; check that the "
                f"uq_resource_address constraint is present."
            )
        return found[0]

    async def activate(
        self,
        *,
        kind: str,
        name: str,
        caller: RequestContext | None,
        **runtime_deps: Any,  # noqa: ANN401 (forwarded into the tool constructor)
    ) -> ToolBase:
        """Authorize a resource and instantiate its runtime tool.

        Args:
            kind (str): Kind of the requested resource.
            name (str): Name of the requested resource.
            caller (RequestContext | None): Principal requesting access.
            **runtime_deps (Any): Dependencies forwarded to the tool loader.

        Returns:
            ToolBase: Authorized resource instance.

        Raises:
            WorkflowConfigurationError: The resource address is missing or
                ambiguous.
            ResourceAccessDeniedError: The policy denies the caller.

        Side Effects:
            Instantiates the resource after its access check.
        """
        resource = await self.resolve(kind=kind, name=name)
        await self.authorize(resource, caller)
        return self._loader.instantiate(
            kind=resource.kind,
            provider=resource.provider,
            name=resource.name,
            config=resource.config,
            resource_id=str(resource.resource_id),
            store_connections=self._connections,
            **runtime_deps,
        )

    async def authorize(
        self, resource: ResourceDef, caller: RequestContext | None
    ) -> None:
        """Check access to a resource using the resolved workflow context.

        Args:
            resource (ResourceDef): Resource to authorize.
            caller (RequestContext | None): Caller and resolved workflow scope.

        Raises:
            ResourceAccessDeniedError: The policy denies access or fails.

        Notes:
            A caller without a workflow does not match a workflow-scoped rule.
            Policy errors deny access rather than granting it.
        """
        if self._access_policy is None:
            return
        try:
            allowed = await self._access_policy.is_allowed(resource, caller, None)
        except Exception as exc:
            logger.warning(
                "resource.activation.policy_error | resource=%s error=%s: denying",
                resource.address, exc,
            )
            raise ResourceAccessDeniedError(
                address=resource.address,
                workflow=caller.workflow if caller else None,
                reason="the access policy could not be evaluated",
            ) from exc
        if not allowed:
            logger.info(
                "resource.activation.denied | resource=%s workflow=%s cid=%s",
                resource.address,
                caller.workflow if caller else None,
                caller.correlation_id if caller else None,
            )
            raise ResourceAccessDeniedError(
                address=resource.address,
                workflow=caller.workflow if caller else None,
            )
