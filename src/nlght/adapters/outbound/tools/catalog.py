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
from nlght.core.model.trust import (
    PLATFORM_TEXT,
    UNANNOTATED,
    ContentExposure,
    ContentLedger,
    ContentOrigin,
)
from nlght.core.tools.action import ActionExecutionContext, ExecutionCapabilities
from nlght.core.tools.tool import ToolOutput
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


def _as_output(result: object) -> ToolOutput:
    """Convert a tool result to content carrying its origin label.

    Args:
        result (object): Raw result or an already labelled ``ToolOutput``.

    Returns:
        ToolOutput: Original labelled output, or unannotated text for a raw
        result.
    """
    if isinstance(result, ToolOutput):
        return result
    content = getattr(result, "content", None)
    return ToolOutput(content if isinstance(content, str) else str(result), UNANNOTATED)


async def _execute_through_gate(
    gate: ActionGate,
    contract: ToolContract,
    arguments: dict[str, Any],
    context: ActionExecutionContext,
    *,
    log_prefix: str,
) -> ToolOutput:
    """Execute one tool call and label the result or failure text.

    Args:
        gate (ActionGate): Authorization and execution gate.
        contract (ToolContract): Tool contract to invoke.
        arguments (dict[str, Any]): Proposed tool arguments.
        context (ActionExecutionContext): Caller and exposure context.
        log_prefix (str): Prefix for execution log events.

    Returns:
        ToolOutput: Labelled result. Platform-authored denial text is trusted.
        Argument and execution errors are unannotated because they may echo
        caller input or remote output.

    Side Effects:
        Runs the tool after authorization and logs failures.
    """
    try:
        prepared = gate.prepare(contract, arguments, context)
        return _as_output(await gate.execute(prepared))
    except (ToolActionDeniedError, ToolActionSemanticsError) as exc:
        logger.info(
            "%s.execute.denied | tool=%s reason=%s",
            log_prefix,
            context.tool_subject,
            getattr(exc, "reason", type(exc).__name__),
        )
        return ToolOutput(str(exc), PLATFORM_TEXT)
    except ToolArgumentsError as exc:
        logger.info(
            "%s.execute.denied | tool=%s reason=%s",
            log_prefix,
            context.tool_subject,
            type(exc).__name__,
        )
        return ToolOutput(str(exc), UNANNOTATED)
    except Exception as exc:
        logger.warning("%s.execute.error | name=%s error=%s", log_prefix, contract.name, exc)
        return ToolOutput(f"Tool error: {exc}", UNANNOTATED)


class AdapterToolCatalog(ToolCatalog):
    """Request-scoped implementation of the ToolCatalog port.

    ``ToolCatalogBuilder`` builds one catalog per workflow invocation. Its
    contracts do not change after construction.
    """

    def __init__(
        self,
        contracts: dict[str, ToolContract],
        *,
        action_gate: ActionGate | None = None,
        caller: RequestContext | None = None,
        session_id: str | None = None,
        ledger: ContentLedger | None = None,
    ) -> None:
        self._contracts = contracts
        self._action_gate = action_gate or ActionGate(policy=DefaultActionPolicy())
        self._caller = caller
        self._session_id = session_id
        self._ledger = ledger if ledger is not None else ContentLedger()

    @property
    def ledger(self) -> ContentLedger:
        return self._ledger

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

    async def execute_with_origin(
        self,
        tc: dict[str, Any],
        *,
        context_origin: ContentOrigin | None = None,
    ) -> ToolOutput:
        """Execute a tool call and retain its result's content origin.

        Args:
            tc (dict[str, Any]): Tool call containing ``name`` and ``input``.
            context_origin (ContentOrigin | None): Origin of the proposed call.
                Missing origin is treated as unannotated.

        Returns:
            ToolOutput: Result or error text with its content origin.

        Side Effects:
            Records call and result origins in the request ledger and invokes
            the selected tool. The caller enforces any overall timeout.
        """
        self._ledger.record(context_origin if context_origin is not None else UNANNOTATED)
        name = tc.get("name", "")
        inp  = tc.get("input") or {}
        if not isinstance(inp, dict):
            inp = {}
        contract = self.get_or_none(name)
        if contract is None:
            logger.warning("tool.catalog.execute.not_found | name=%s", name)
            # Echoes the proposed name, so it is not platform text alone.
            output = ToolOutput(f"Tool '{name}' not available.", UNANNOTATED)
        else:
            output = await _execute_through_gate(
                self._action_gate,
                contract,
                inp,
                self._execution_context(contract, exposure=self._ledger.exposure),
                log_prefix="tool.catalog",
            )
        self._ledger.record(output.origin)
        return output

    async def execute(
        self,
        tc: dict[str, Any],
        *,
        context_origin: ContentOrigin | None = None,
    ) -> str:
        """Execute a tool call and return only its text content.

        Args:
            tc (dict[str, Any]): Tool call containing ``name`` and ``input``.
            context_origin (ContentOrigin | None): Origin of the proposed call.

        Returns:
            str: Result or error text from ``execute_with_origin``.

        Side Effects:
            Records call and result origins and invokes the selected tool.
        """
        return (await self.execute_with_origin(tc, context_origin=context_origin)).content

    def _execution_context(
        self,
        contract: ToolContract,
        *,
        exposure: ContentExposure,
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
            exposure=exposure,
        )

    def __len__(self) -> int:
        return len(self._contracts)

    def __bool__(self) -> bool:
        return bool(self._contracts)


class TemporaryToolCatalog:
    """Ephemeral Catalog wrapper for the duration of a single Step run.

    Wraps an existing ``ToolCatalog`` and adds extra Contracts without
    mutating the original catalog. Extra contracts override base contracts
    of the same name. Discard the wrapper after the step run. No cleanup is
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
        # One ledger for the run: what a step's own tools return counts for the
        # base catalog's next decision, and the other way round.
        base_ledger = getattr(base, "ledger", None)
        self._ledger = base_ledger if isinstance(base_ledger, ContentLedger) else ContentLedger()

    @property
    def ledger(self) -> ContentLedger:
        return self._ledger

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

    async def execute_with_origin(
        self,
        tc: dict[str, Any],
        *,
        context_origin: ContentOrigin | None = None,
    ) -> ToolOutput:
        self._ledger.record(context_origin if context_origin is not None else UNANNOTATED)
        name = tc.get("name", "")
        inp  = tc.get("input") or {}
        if not isinstance(inp, dict):
            inp = {}
        contract = self.get_or_none(name)
        if contract is None:
            logger.warning("tool.temp_catalog.execute.not_found | name=%s", name)
            output = ToolOutput(f"Tool '{name}' not available.", UNANNOTATED)
        else:
            caller = self._caller
            context = ActionExecutionContext(
                request_id=caller.request_id if caller else "internal",
                correlation_id=caller.correlation_id if caller else "internal",
                principal=caller.principal if caller else None,
                workflow=caller.workflow if caller else None,
                session_id=self._session_id,
                resource_address=contract.resource_address,
                signature_name=contract.name,
                exposure=self._ledger.exposure,
            )
            output = await _execute_through_gate(
                self._action_gate, contract, inp, context, log_prefix="tool.temp_catalog",
            )
        self._ledger.record(output.origin)
        return output

    async def execute(
        self,
        tc: dict[str, Any],
        *,
        context_origin: ContentOrigin | None = None,
    ) -> str:
        return (await self.execute_with_origin(tc, context_origin=context_origin)).content


class ToolCatalogBuilder:
    """Builds a request-scoped ``AdapterToolCatalog`` from DB resources.

    The builder reads enabled resources, skips unknown kinds, and checks
    resource and signature policies before instantiation. It creates bound
    contracts only for visible signatures. Resources for other subsystems and
    resources with no visible signatures are not instantiated.
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
        action_authorization_enabled: bool = True,
        policy_timeout_seconds: float = 30.0,
        approval_timeout_seconds: float = 300.0,
    ) -> None:
        self._loader = loader
        self._repo = resource_repository
        self._os_runtime: OsRuntimeFactory | None = os_runtime
        self._resource_access_policy = resource_access_policy
        self._tool_access_policy = tool_access_policy
        self._embedding_client = embedding_client
        # A catalog is rebuilt per run. Without shared pools every rebuild would
        # open a fresh connection per store activation and abandon it.
        self._store_connections = store_connections
        self._action_gate = ActionGate(
            policy=action_policy or DefaultActionPolicy(),
            approval=action_approval,
            enabled=action_authorization_enabled,
            policy_timeout_seconds=policy_timeout_seconds,
            approval_timeout_seconds=approval_timeout_seconds,
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
            # are a classmethod, so the whole question, which of its operations
            # may this caller be offered, is answerable from the class.
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
                # nothing to build it for. This is also why a writer, which
                # declares no signatures at all, is never constructed here: it
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
                    resource_id=str(resource.resource_id),
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
                # This catalog cannot be built, even though the resource was found.
                # Skipping one resource here would silently pick the
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
        """Check whether the caller may use a resource.

        Args:
            resource (ResourceDef): Resource requested by the catalog.
            caller (RequestContext | None): Established caller identity.
            model (str | None): Model requesting the resource.

        Returns:
            bool: ``True`` if allowed or no policy is configured. Policy errors
            deny access and are logged.
        """
        if self._resource_access_policy is None:
            return True
        try:
            allowed = await self._resource_access_policy.is_allowed(resource, caller, model)
        except Exception as exc:
            logger.warning(
                "tool.catalog.policy_error | resource=%s error=%s: denying",
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
        """Check whether a resource operation is offered to the caller.

        Args:
            resource (ResourceDef): Resource containing the operation.
            signature_name (str): Operation name to authorize.
            caller (RequestContext | None): Established caller identity.
            model (str | None): Model requesting the operation.

        Returns:
            bool: ``True`` if allowed or no policy is configured. Policy errors
            deny access and are logged.
        """
        if self._tool_access_policy is None:
            return True
        try:
            allowed = await self._tool_access_policy.is_allowed(
                resource, signature_name, caller, model
            )
        except Exception as exc:
            logger.warning(
                "tool.catalog.policy_error | tool=%s error=%s: denying",
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
