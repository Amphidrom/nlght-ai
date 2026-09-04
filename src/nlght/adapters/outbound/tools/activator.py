# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""The one place a resource becomes an instance.

There were seven, all the same twenty lines copied: find the resources of a
kind, filter by name, take the first, hand it to the loader. None of them asked
a policy. So a workflow step could activate any resource in the database by
naming it in its own configuration, and the access rules that governed the model
tool catalog governed nothing here.

That mattered most where it was least visible. `data_index_writer` and
`knowledge_index_writer` declare no signatures, which was treated as their
protection: a model can never call them because they never enter a catalog. But
the thing that actually uses a writer is a step, and a step reached it without
passing anything at all.

So the resolution and the authorization become one operation, and the guard is
before the instantiation rather than after it:

    resolve the address  →  check the policy  →  instantiate

Not "instantiate, then check": constructing a store opens engines and clients,
and a denied caller should not cause connections to be made on their behalf.

**A denial here raises.** The tool catalog *filters* — a model is simply not
told about what it may not use — but a step that names a resource has stated a
requirement, and quietly continuing without it would run a workflow that is not
the workflow that was configured. A forbidden access is not a missing optional
capability.
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
        """The one enabled resource at this address.

        The address is unique (migration `0007`), so "the first row wins" is no
        longer a decision anybody is making by accident. A second row here would
        mean the constraint is missing, and that is worth an error rather than a
        silent pick.
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
        """The authorized instance, or an error explaining which half refused."""
        resource = await self.resolve(kind=kind, name=name)
        await self.authorize(resource, caller)
        return self._loader.instantiate(
            kind=resource.kind,
            provider=resource.provider,
            name=resource.name,
            config=resource.config,
            store_connections=self._connections,
            **runtime_deps,
        )

    async def authorize(
        self, resource: ResourceDef, caller: RequestContext | None
    ) -> None:
        """Raises unless this caller may use this resource.

        The policy sees the caller the executor established — including the
        workflow it actually resolved and is running, which is what a
        ``workflow`` condition compares against. A caller outside any workflow
        has none, and a workflow-scoped rule therefore denies it; that is
        deliberate, so that running outside a flow is not a way past a rule.

        A policy that raises denies. Whatever went wrong, the one thing that
        must not follow from a broken authorization check is authorization.
        """
        if self._access_policy is None:
            return
        try:
            allowed = await self._access_policy.is_allowed(resource, caller, None)
        except Exception as exc:  # noqa: BLE001 (a failed check is a denial)
            logger.warning(
                "resource.activation.policy_error | resource=%s error=%s — denying",
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
