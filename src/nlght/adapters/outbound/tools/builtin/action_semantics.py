# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Reusable action declarations for built-in model tools."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath, PureWindowsPath
from urllib.parse import urlparse

from nlght.core.errors.errors import ToolArgumentsError
from nlght.core.tools.action import (
    ActionResolutionContext,
    BoundDestination,
    ConcreteAction,
    DataEgressClass,
    DestinationKind,
    ExecutionClass,
    JsonValue,
    ScopeClass,
    SideEffectClass,
    StaticActionSemantics,
)

READ_REQUEST = StaticActionSemantics(
    SideEffectClass.NONE,
    DataEgressClass.NONE,
    ScopeClass.REQUEST,
)
READ_SESSION = StaticActionSemantics(
    SideEffectClass.NONE,
    DataEgressClass.NONE,
    ScopeClass.SESSION,
)
READ_RUNTIME = StaticActionSemantics(
    SideEffectClass.NONE,
    DataEgressClass.NONE,
    ScopeClass.SANDBOX,
)
WRITE_RUNTIME = StaticActionSemantics(
    SideEffectClass.REVERSIBLE_WRITE,
    DataEgressClass.NONE,
    ScopeClass.SANDBOX,
)
#: A write that replaces whatever is at a caller-chosen path.
#:
#: Deliberately not `WRITE_RUNTIME`. Reversible means there is a way back, and
#: the three operations on that constant have one: git tracks a checkout and a
#: commit, and a created directory can be removed. Writing text over an existing
#: file has none — the previous content is gone the moment it lands, and the
#: class has to say so even though today's default policy happens to treat both
#: write classes the same (ADR-0069).
OVERWRITING_WRITE_RUNTIME = StaticActionSemantics(
    SideEffectClass.DESTRUCTIVE_WRITE,
    DataEgressClass.NONE,
    ScopeClass.SANDBOX,
)
DELETE_RUNTIME = StaticActionSemantics(
    SideEffectClass.DESTRUCTIVE_WRITE,
    DataEgressClass.NONE,
    ScopeClass.SANDBOX,
)
EXECUTE_RUNTIME = StaticActionSemantics(
    SideEffectClass.CODE_EXECUTION,
    DataEgressClass.NONE,
    ScopeClass.SANDBOX,
    ExecutionClass.CONFINED,
)
ARBITRARY_EXTERNAL_WRITE = StaticActionSemantics(
    SideEffectClass.EXTERNAL_WRITE,
    DataEgressClass.ARBITRARY_DESTINATION,
    ScopeClass.EXTERNAL,
)
ARBITRARY_EXTERNAL_READ_WRITE = StaticActionSemantics(
    SideEffectClass.REVERSIBLE_WRITE,
    DataEgressClass.ARBITRARY_DESTINATION,
    ScopeClass.EXTERNAL,
)


def _string_argument(arguments: Mapping[str, JsonValue], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ToolArgumentsError(f"tool argument '{name}' must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class ConfiguredNetworkReadSemantics:
    """Read through one endpoint fixed in operator-controlled resource config."""

    config_key: str = "endpoint"

    def resolve(
        self,
        arguments: Mapping[str, JsonValue],
        context: ActionResolutionContext,
    ) -> ConcreteAction:
        del arguments
        endpoint = str(context.resource_config.get(self.config_key, "")).strip()
        parsed = urlparse(endpoint)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ToolArgumentsError(
                f"resource config '{self.config_key}' is not a bound HTTP destination"
            )
        destination = BoundDestination.from_value(
            kind=DestinationKind.NETWORK,
            scope=ScopeClass.EXTERNAL,
            value=f"{parsed.scheme}://{parsed.hostname}:{parsed.port or (443 if parsed.scheme == 'https' else 80)}",
        )
        return ConcreteAction(
            side_effect=SideEffectClass.NONE,
            egress=DataEgressClass.BOUNDED_DESTINATION,
            scope=ScopeClass.EXTERNAL,
            capabilities=context.runtime_capabilities,
            destination=destination,
        )


@dataclass(frozen=True)
class ArbitraryNetworkReadSemantics:
    url_parameter: str = "url"

    def resolve(
        self,
        arguments: Mapping[str, JsonValue],
        context: ActionResolutionContext,
    ) -> ConcreteAction:
        url = _string_argument(arguments, self.url_parameter)
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            raise ToolArgumentsError(f"tool argument '{self.url_parameter}' must be an HTTP URL")
        destination = BoundDestination.from_value(
            kind=DestinationKind.NETWORK,
            scope=ScopeClass.EXTERNAL,
            value=f"{parsed.scheme}://{parsed.hostname}:{parsed.port or (443 if parsed.scheme == 'https' else 80)}",
        )
        return ConcreteAction(
            side_effect=SideEffectClass.NONE,
            egress=DataEgressClass.ARBITRARY_DESTINATION,
            scope=ScopeClass.EXTERNAL,
            capabilities=context.runtime_capabilities,
            destination=destination,
        )


@dataclass(frozen=True)
class ArbitraryNetworkCloneSemantics(ArbitraryNetworkReadSemantics):
    url_parameter: str = "repo_url"

    def resolve(
        self,
        arguments: Mapping[str, JsonValue],
        context: ActionResolutionContext,
    ) -> ConcreteAction:
        base = super().resolve(arguments, context)
        return ConcreteAction(
            side_effect=SideEffectClass.REVERSIBLE_WRITE,
            egress=base.egress,
            scope=base.scope,
            capabilities=base.capabilities,
            destination=base.destination,
        )


@dataclass(frozen=True)
class HttpProbeSemantics:
    def resolve(
        self,
        arguments: Mapping[str, JsonValue],
        context: ActionResolutionContext,
    ) -> ConcreteAction:
        base = ArbitraryNetworkReadSemantics().resolve(arguments, context)
        method_value = arguments.get("method", "GET")
        if not isinstance(method_value, str):
            raise ToolArgumentsError("tool argument 'method' must be a string")
        method = method_value.upper()
        if method not in {"GET", "HEAD", "OPTIONS", "POST", "PUT", "PATCH", "DELETE"}:
            raise ToolArgumentsError("tool argument 'method' is not a supported HTTP method")
        side_effect = SideEffectClass.NONE
        if method in {"POST", "PUT", "PATCH"}:
            side_effect = SideEffectClass.EXTERNAL_WRITE
        elif method == "DELETE":
            side_effect = SideEffectClass.DESTRUCTIVE_WRITE
        return ConcreteAction(
            side_effect=side_effect,
            egress=base.egress,
            scope=base.scope,
            capabilities=base.capabilities,
            destination=base.destination,
        )


@dataclass(frozen=True)
class CheckOnlySemantics:
    """A dry-run boolean selects read-only or reversible runtime mutation."""

    check_parameter: str = "check_only"

    def resolve(
        self,
        arguments: Mapping[str, JsonValue],
        context: ActionResolutionContext,
    ) -> ConcreteAction:
        semantics = READ_RUNTIME if arguments.get(self.check_parameter) is True else WRITE_RUNTIME
        return semantics.resolve(arguments, context)


ARBITRARY_NETWORK_READ = ArbitraryNetworkReadSemantics()
ARBITRARY_NETWORK_CLONE = ArbitraryNetworkCloneSemantics()
CONFIGURED_NETWORK_READ = ConfiguredNetworkReadSemantics()
HTTP_PROBE = HttpProbeSemantics()
CHECK_ONLY_WRITE = CheckOnlySemantics()


@dataclass(frozen=True)
class RelativePathBinder:
    """Bind selected tool parameters to repository-relative paths."""

    fields: tuple[str, ...] = ()
    list_fields: tuple[str, ...] = ()

    @staticmethod
    def _bind_one(value: JsonValue, name: str) -> str:
        if not isinstance(value, str):
            raise ToolArgumentsError(f"tool argument '{name}' must be a string")
        path = value.replace("\\", "/").strip()
        if (
            not path
            or "\0" in path
            or PurePosixPath(path).is_absolute()
            or PureWindowsPath(path).is_absolute()
        ):
            raise ToolArgumentsError(f"tool argument '{name}' must be repository-relative")
        parts = [part for part in path.split("/") if part not in ("", ".")]
        if any(part == ".." for part in parts):
            raise ToolArgumentsError(f"tool argument '{name}' escapes the repository scope")
        return "/".join(parts) or "."

    def bind(
        self,
        arguments: Mapping[str, JsonValue],
        context: ActionResolutionContext,
    ) -> Mapping[str, JsonValue]:
        del context
        bound = dict(arguments)
        for name in self.fields:
            if name in bound:
                bound[name] = self._bind_one(bound[name], name)
        for name in self.list_fields:
            value = bound.get(name)
            if value is None:
                continue
            if not isinstance(value, list):
                raise ToolArgumentsError(f"tool argument '{name}' must be an array")
            bound[name] = [self._bind_one(item, name) for item in value]
        return bound


@dataclass(frozen=True)
class HttpMethodBinder:
    def bind(
        self,
        arguments: Mapping[str, JsonValue],
        context: ActionResolutionContext,
    ) -> Mapping[str, JsonValue]:
        del context
        bound = dict(arguments)
        method = str(bound.get("method", "GET")).upper()
        if method not in {"GET", "HEAD", "OPTIONS", "POST", "PUT", "PATCH", "DELETE"}:
            raise ToolArgumentsError("tool argument 'method' is not a supported HTTP method")
        bound["method"] = method
        return bound


HTTP_METHOD_BINDER = HttpMethodBinder()
