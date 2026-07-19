# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from nlght.adapters.outbound.tools.contract import BoundToolContract
from nlght.adapters.outbound.tools.loader import ToolLoader
from nlght.ports.outbound.os_runtime import OsRuntimeFactory
from nlght.ports.outbound.resource_repository import ResourceRepository
from nlght.ports.outbound.tool_catalog import ToolCatalog, ToolContract

if TYPE_CHECKING:
    from nlght.core.entry.context import RequestContext
    from nlght.core.workspace.workspace import WorkspaceContext
    from nlght.ports.outbound.access_policy import ToolAccessPolicy
    from nlght.ports.outbound.store_coordinator import StoreCoordinator

logger = logging.getLogger(__name__)


class AdapterToolCatalog(ToolCatalog):
    """Implements ToolCatalog — request-scoped Tool lookup.

    Request-scoped — built once per workflow invocation by the
    ``ToolCatalogBuilder`` and never mutated afterward.
    """

    def __init__(self, contracts: dict[str, ToolContract]) -> None:
        self._contracts = contracts

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

    async def execute(self, tc: dict[str, Any]) -> str:
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
        try:
            result = await contract.execute(**inp)
            content = getattr(result, "content", None)
            return content if isinstance(content, str) else str(result)
        except Exception as exc:
            logger.warning("tool.catalog.execute.error | name=%s error=%s", name, exc)
            return f"Tool error: {exc}"

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
    ) -> None:
        self._base = base
        self._extra: dict[str, ToolContract] = {c.name: c for c in (extra or [])}

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

    async def execute(self, tc: dict[str, Any]) -> str:
        name = tc.get("name", "")
        inp  = tc.get("input") or {}
        if not isinstance(inp, dict):
            inp = {}
        contract = self.get_or_none(name)
        if contract is None:
            logger.warning("tool.temp_catalog.execute.not_found | name=%s", name)
            return f"Tool '{name}' not available."
        try:
            result = await contract.execute(**inp)
            content = getattr(result, "content", None)
            return content if isinstance(content, str) else str(result)
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
        access_policy: ToolAccessPolicy | None = None,
    ) -> None:
        self._loader = loader
        self._repo = resource_repository
        self._os_runtime: OsRuntimeFactory | None = os_runtime
        self._access_policy = access_policy

    async def build(
        self,
        model: str | None = None,
        caller: RequestContext | None = None,
        store_coordinator: StoreCoordinator | None = None,
        workspace: WorkspaceContext | None = None,
    ) -> AdapterToolCatalog:
        cid = caller.correlation_id if caller else None
        effective_runtime = (
            self._os_runtime.bind(cid) if self._os_runtime is not None and cid else self._os_runtime
        )

        resources = await self._repo.list_enabled()
        contracts: dict[str, ToolContract] = {}

        for resource in resources:
            if resource.kind not in self._loader:
                continue

            # optional access policy — model- and caller-based restrictions
            if self._access_policy is not None:
                try:
                    if not await self._access_policy.is_allowed(resource, caller, model):
                        logger.info(
                            "tool.catalog.access_denied | tool=%s cid=%s",
                            resource.name, caller.correlation_id if caller else None,
                        )
                        continue
                except Exception as exc:
                    logger.warning(
                        "tool.catalog.policy_error | tool=%s error=%s — denying",
                        resource.name, exc,
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
                )
                for sig in type(instance).signatures():
                    contract = BoundToolContract(
                        name=sig.name,
                        description=sig.description,
                        parameters=sig.parameters,
                        instance=instance,
                        method_name=sig.method_name,
                        terminal=sig.terminal,
                    )
                    contracts[sig.name] = contract
                    logger.debug(
                        "tool.catalog.add | tool=%s operation=%s",
                        resource.name,
                        sig.name,
                    )

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
        return AdapterToolCatalog(contracts)
