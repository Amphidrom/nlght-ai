# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from nlght.adapters.outbound.stores.connections import StoreConnections
from nlght.adapters.outbound.tools.action_gate import ActionGate
from nlght.adapters.outbound.tools.action_policy import DefaultActionPolicy
from nlght.adapters.outbound.tools.contract import BoundToolContract
from nlght.adapters.outbound.tools.loader import ToolLoader
from nlght.core.access import tool_subject
from nlght.core.entry.context import RequestContext
from nlght.core.errors.errors import (
    AmbiguousToolNameError,
    ToolActionDeniedError,
    ToolActionSemanticsError,
    ToolArgumentsError,
)
from nlght.core.tools.action import ActionExecutionContext, ExecutionCapabilities
from nlght.ports.outbound.embedding_client import EmbeddingClient
from nlght.ports.outbound.os_runtime import OsRuntimeFactory
from nlght.ports.outbound.resource_repository import ResourceRepository
from nlght.ports.outbound.tool_catalog import ToolCatalog, ToolContract

if TYPE_CHECKING:
    from nlght.core.runtime.resource import ResourceDef
    from nlght.core.workspace.workspace import WorkspaceContext
    from nlght.ports.outbound.access_policy import (
        ResourceAccessPolicy,
        ToolAccessPolicy,
    )
    from nlght.ports.outbound.action_policy import ActionApprovalPort, ActionPolicy
    from nlght.ports.outbound.store_coordinator import StoreCoordinator

logger = logging.getLogger(__name__)


def _execution_capabilities(runtime: object | None) -> ExecutionCapabilities:
    if runtime is None:
        return ExecutionCapabilities()
    capability_fn = getattr(runtime, "security_capabilities", None)
    if capability_fn is None:
        return ExecutionCapabilities.unconfined()
    capabilities = capability_fn()
    return (
        capabilities
        if isinstance(capabilities, ExecutionCapabilities)
        else ExecutionCapabilities.unconfined()
    )


class AdapterToolCatalog(ToolCatalog):
    """Implements ToolCatalog — request-scoped Tool lookup.

    Request-scoped — built once per workflow invocation by the
    ``ToolCatalogBuilder`` and never mutated afterward.
    """

    def __init__(
        self,
        contracts: dict[str, ToolContract],
        *,
        action_gate: ActionGate | None = None,
        caller: RequestContext | None = None,
        session_id: str | None = None,
    ) -> None:
        self._contracts = contracts
        self._action_gate = action_gate or ActionGate(policy=DefaultActionPolicy())
        self._caller = caller
        self._session_id = session_id

    def get(self, name: str) -> ToolContract:
        if name not in self._contracts:
            raise KeyError(f"Tool '{name}' not in catalog. Available: {sorted(self._contracts)}")
        return self._contracts[name]

    def get_or_none(self, name: str) -> ToolContract | None:
        return self._contracts.get(name)

    def all(self) -> list[ToolContract]:
        return list(self._contracts.values())

    def names(self) -> list[str]:
        return list(self._contracts.keys())

    async def execute(
        self,
        tc: dict[str, Any],
        *,
        has_external_untrusted_input: bool = True,
    ) -> str:
        """Executes a tool call and returns the result as a string.

        Returns error strings instead of raising exceptions — timeouts
        must be handled by the caller via ``asyncio.wait_for``.
        """
        name = tc.get("name", "")
        inp  = tc.get("input") or {}
        if not isinstance(inp, dict):
            inp = {}
        contract = self.get_or_none(name)
        if contract is None:
            logger.warning("tool.catalog.execute.not_found | name=%s", name)
            return f"Tool '{name}' not available."
        context = self._execution_context(
            contract,
            has_external_untrusted_input=has_external_untrusted_input,
        )
        try:
            prepared = self._action_gate.prepare(contract, inp, context)
            result = await self._action_gate.execute(prepared)
            content = getattr(result, "content", None)
            return content if isinstance(content, str) else str(result)
        except (ToolActionDeniedError, ToolActionSemanticsError, ToolArgumentsError) as exc:
            logger.info(
                "tool.catalog.execute.denied | tool=%s reason=%s",
                context.tool_subject,
                getattr(exc, "reason", type(exc).__name__),
            )
            return str(exc)
        except Exception as exc:
            logger.warning("tool.catalog.execute.error | name=%s error=%s", name, exc)
            return f"Tool error: {exc}"

    def _execution_context(
        self,
        contract: ToolContract,
        *,
        has_external_untrusted_input: bool,
    ) -> ActionExecutionContext:
        caller = self._caller
        return ActionExecutionContext(
            request_id=caller.request_id if caller else "internal",
            correlation_id=caller.correlation_id if caller else "internal",
            principal=caller.principal if caller else None,
            workflow=caller.workflow if caller else None,
            session_id=self._session_id,
            resource_address=contract.resource_address,
            signature_name=contract.name,
            has_external_untrusted_input=has_external_untrusted_input,
        )

    def __len__(self) -> int:
        return len(self._contracts)

    def __bool__(self) -> bool:
        return bool(self._contracts)


class TemporaryToolCatalog:
    """Ephemeral Catalog wrapper for the duration of a single Step run.

    Wraps an existing ``ToolCatalog`` and adds extra Contracts without
    mutating the original catalog. Extra contracts override base contracts
    of the same name. Simply discard after the Step run — no cleanup
    required.
    """

    def __init__(
        self,
        base: ToolCatalog | None,
        *,
        extra: list[ToolContract] | None = None,
        action_gate: ActionGate | None = None,
        caller: RequestContext | None = None,
        session_id: str | None = None,
    ) -> None:
        self._base = base
        self._extra: dict[str, ToolContract] = {c.name: c for c in (extra or [])}
        if isinstance(base, AdapterToolCatalog):
            self._action_gate = action_gate or base._action_gate
            self._caller = caller or base._caller
            self._session_id = session_id if session_id is not None else base._session_id
        else:
            self._action_gate = action_gate or ActionGate(policy=DefaultActionPolicy())
            self._caller = caller
            self._session_id = session_id

    def get(self, name: str) -> ToolContract:
        if name in self._extra:
            return self._extra[name]
        if self._base is not None:
            return self._base.get(name)
        raise KeyError(f"Tool '{name}' not found.")

    def get_or_none(self, name: str) -> ToolContract | None:
        if name in self._extra:
            return self._extra[name]
        return self._base.get_or_none(name) if self._base is not None else None

    def all(self) -> list[ToolContract]:
        extra_names = set(self._extra)
        base = [c for c in (self._base.all() if self._base else []) if c.name not in extra_names]
        return base + list(self._extra.values())

    def names(self) -> list[str]:
        return [c.name for c in self.all()]

    async def execute(
        self,
        tc: dict[str, Any],
        *,
        has_external_untrusted_input: bool = True,
    ) -> str:
        name = tc.get("name", "")
        inp  = tc.get("input") or {}
        if not isinstance(inp, dict):
            inp = {}
        contract = self.get_or_none(name)
        if contract is None:
            logger.warning("tool.temp_catalog.execute.not_found | name=%s", name)
            return f"Tool '{name}' not available."
        caller = self._caller
        context = ActionExecutionContext(
            request_id=caller.request_id if caller else "internal",
            correlation_id=caller.correlation_id if caller else "internal",
            principal=caller.principal if caller else None,
            workflow=caller.workflow if caller else None,
            session_id=self._session_id,
            resource_address=contract.resource_address,
            signature_name=contract.name,
            has_external_untrusted_input=has_external_untrusted_input,
        )
        try:
            prepared = self._action_gate.prepare(contract, inp, context)
            result = await self._action_gate.execute(prepared)
            content = getattr(result, "content", None)
            return content if isinstance(content, str) else str(result)
        except (ToolActionDeniedError, ToolActionSemanticsError, ToolArgumentsError) as exc:
            logger.info(
                "tool.temp_catalog.execute.denied | tool=%s reason=%s",
                context.tool_subject,
                getattr(exc, "reason", type(exc).__name__),
            )
            return str(exc)
        except Exception as exc:
            logger.warning("tool.temp_catalog.execute.error | name=%s error=%s", name, exc)
            return f"Tool error: {exc}"


class ToolCatalogBuilder:
    """Builds a request-scoped ``AdapterToolCatalog`` from DB resources.

    Flow per invocation:
    1. ``ResourceRepository.list_enabled()`` — all active resources from the DB
    2. For each resource: is the kind known to the ``ToolLoader``?
    3. Yes → instantiate, read signatures, create ``BoundToolContract``s
    4. No → skip (not an error — the resource simply isn't a tool of this kind)

    Resources with no registered kind are silently ignored — the DB may
    contain resources for other subsystems.
    """

    def __init__(
        self,
        *,
        loader: ToolLoader,
        resource_repository: ResourceRepository,
        os_runtime: OsRuntimeFactory | None = None,
        resource_access_policy: ResourceAccessPolicy | None = None,
        tool_access_policy: ToolAccessPolicy | None = None,
        embedding_client: EmbeddingClient | None = None,
        store_connections: StoreConnections | None = None,
        action_policy: ActionPolicy | None = None,
        action_approval: ActionApprovalPort | None = None,
    ) -> None:
        self._loader = loader
        self._repo = resource_repository
        self._os_runtime: OsRuntimeFactory | None = os_runtime
        self._resource_access_policy = resource_access_policy
        self._tool_access_policy = tool_access_policy
        self._embedding_client = embedding_client
        # A catalog is rebuilt per run; without shared pools every rebuild would
        # open a fresh connection per store activation and abandon it.
        self._store_connections = store_connections
        self._action_gate = ActionGate(
            policy=action_policy or DefaultActionPolicy(),
            approval=action_approval,
        )

    async def build(
        self,
        model: str | None = None,
        caller: RequestContext | None = None,
        store_coordinator: StoreCoordinator | None = None,
        workspace: WorkspaceContext | None = None,
        session_id: str | None = None,
    ) -> AdapterToolCatalog:
        cid = caller.correlation_id if caller else None
        effective_runtime = (
            self._os_runtime.bind(cid) if self._os_runtime is not None and cid else self._os_runtime
        )

        resources = await self._repo.list_enabled()
        contracts: dict[str, ToolContract] = {}
        # Which resource contributed each visible name, so a collision can name
        # both sides of itself.
        offered_by: dict[str, str] = {}

        for resource in resources:
            if resource.kind not in self._loader:
                continue

            # May this caller use the resource at all? Nothing about its
            # individual operations can put back a resource denied here.
            if not await self._resource_allowed(resource, caller, model):
                continue

            # What this resource *would* offer, without building it. Signatures
            # are a classmethod, so the whole question — which of its operations
            # may this caller be offered — is answerable from the class.
            implementation = self._loader.implementation_for(
                kind=resource.kind, provider=resource.provider
            )
            if implementation is None:
                continue

            offered = [
                sig
                for sig in implementation.signatures()
                if await self._signature_allowed(resource, sig.name, caller, model)
            ]
            for sig in offered:
                if sig.action is None:
                    raise ToolActionSemanticsError(
                        tool_subject(resource.address, sig.name)
                    )
            if not offered:
                # Nothing of this resource would reach the model, so there is
                # nothing to build it for. This is also why a writer — which
                # declares no signatures at all — is never constructed here: it
                # is not a model tool, and the catalog is the model's view.
                logger.debug(
                    "tool.catalog.no_visible_operations | resource=%s", resource.address,
                )
                continue

            try:
                instance = self._loader.instantiate(
                    kind=resource.kind,
                    provider=resource.provider,
                    name=resource.name,
                    config=resource.config,
                    os_runtime=effective_runtime,
                    store_coordinator=store_coordinator,
                    workspace=workspace,
                    embedding_client=self._embedding_client,
                    store_connections=self._store_connections,
                )
                for sig in offered:
                    claimed = offered_by.get(sig.name)
                    if claimed is not None:
                        raise AmbiguousToolNameError(
                            operation=sig.name,
                            addresses=(claimed, resource.address),
                        )
                    offered_by[sig.name] = resource.address
                    contract = BoundToolContract(
                        name=sig.name,
                        description=sig.description,
                        parameters=sig.parameters,
                        instance=instance,
                        method_name=sig.method_name,
                        terminal=sig.terminal,
                        action=sig.action,
                        argument_binder=sig.argument_binder,
                        resource_address=resource.address,
                        resource_config=resource.config,
                        runtime_capabilities=_execution_capabilities(effective_runtime),
                    )
                    contracts[sig.name] = contract
                    logger.debug(
                        "tool.catalog.add | resource=%s operation=%s",
                        resource.address,
                        sig.name,
                    )

            except AmbiguousToolNameError:
                # Not a resource that failed to build — a catalog that cannot be
                # built. Skipping one resource here would silently pick the
                # other, which is the behaviour this replaces.
                raise
            except Exception as exc:
                logger.warning(
                    "tool.catalog.skip | kind=%s name=%s error=%s",
                    resource.kind,
                    resource.name,
                    exc,
                )

        logger.info(
            "tool.catalog.built | resources_checked=%d contracts=%d",
            len(resources),
            len(contracts),
        )
        return AdapterToolCatalog(
            contracts,
            action_gate=self._action_gate,
            caller=caller,
            session_id=session_id,
        )

    async def _resource_allowed(
        self,
        resource: ResourceDef,
        caller: RequestContext | None,
        model: str | None,
    ) -> bool:
        """Whether this caller may use the resource at all.

        A policy that raises denies. Whatever went wrong, the one thing that must
        not follow from a broken authorization check is authorization.
        """
        if self._resource_access_policy is None:
            return True
        try:
            allowed = await self._resource_access_policy.is_allowed(resource, caller, model)
        except Exception as exc:  # noqa: BLE001 (a failed check is a denial)
            logger.warning(
                "tool.catalog.policy_error | resource=%s error=%s — denying",
                resource.address, exc,
            )
            return False
        if not allowed:
            logger.info(
                "tool.catalog.resource_denied | resource=%s cid=%s",
                resource.address, caller.correlation_id if caller else None,
            )
        return allowed

    async def _signature_allowed(
        self,
        resource: ResourceDef,
        signature_name: str,
        caller: RequestContext | None,
        model: str | None,
    ) -> bool:
        """Whether this one operation of an allowed resource is offered."""
        if self._tool_access_policy is None:
            return True
        try:
            allowed = await self._tool_access_policy.is_allowed(
                resource, signature_name, caller, model
            )
        except Exception as exc:  # noqa: BLE001 (a failed check is a denial)
            logger.warning(
                "tool.catalog.policy_error | tool=%s error=%s — denying",
                tool_subject(resource.address, signature_name), exc,
            )
            return False
        if not allowed:
            logger.info(
                "tool.catalog.tool_denied | tool=%s cid=%s",
                tool_subject(resource.address, signature_name),
                caller.correlation_id if caller else None,
            )
        return allowed
