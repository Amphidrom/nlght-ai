# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Define provider-neutral contracts for model-proposed actions."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol, TypeAlias, runtime_checkable

from nlght.core.model.trust import ContentExposure

if TYPE_CHECKING:
    from nlght.core.entry.context import PrincipalRef

JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


class ActionDecision(StrEnum):
    ALLOW = "allow"
    REQUIRE_CONFIRMATION = "require_confirmation"
    DENY = "deny"


class ActionDecisionReason(StrEnum):
    POLICY_ALLOWED = "policy_allowed"
    APPROVAL_REQUIRED = "approval_required"
    POLICY_DENIED = "policy_denied"
    UNKNOWN_ACTION_SEMANTICS = "unknown_action_semantics"
    ARBITRARY_EGRESS = "arbitrary_egress"
    HOST_SCOPE = "host_scope"
    UNBOUNDED_CODE_EXECUTION = "unbounded_code_execution"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    APPROVAL_UNAVAILABLE = "approval_unavailable"
    APPROVAL_REJECTED = "approval_rejected"
    APPROVAL_MISMATCH = "approval_mismatch"
    PARAMETER_SCOPE_VIOLATION = "parameter_scope_violation"
    INVALID_ARGUMENTS = "invalid_arguments"
    ACTION_RESOLUTION_ERROR = "action_resolution_error"
    POLICY_ERROR = "policy_error"
    APPROVAL_ERROR = "approval_error"
    POLICY_TIMEOUT = "policy_timeout"
    APPROVAL_TIMEOUT = "approval_timeout"
    #: Authorization is switched off for this deployment: nothing was asked,
    #: and the audit must not read as if a policy had allowed it (ADR-0073).
    AUTHORIZATION_DISABLED = "authorization_disabled"


class SideEffectClass(StrEnum):
    NONE = "none"
    REVERSIBLE_WRITE = "reversible_write"
    DESTRUCTIVE_WRITE = "destructive_write"
    EXTERNAL_WRITE = "external_write"
    PRIVILEGE_CHANGE = "privilege_change"
    CODE_EXECUTION = "code_execution"


class DataEgressClass(StrEnum):
    NONE = "none"
    BOUNDED_DESTINATION = "bounded_destination"
    EXTERNAL_DESTINATION = "external_destination"
    ARBITRARY_DESTINATION = "arbitrary_destination"


class ScopeClass(StrEnum):
    REQUEST = "request"
    SESSION = "session"
    WORKSPACE = "workspace"
    SANDBOX = "sandbox"
    EXTERNAL = "external"
    HOST = "host"


class ExecutionClass(StrEnum):
    NONE = "none"
    CONFINED = "confined"
    UNCONFINED = "unconfined"


class FilesystemCapability(StrEnum):
    NONE = "none"
    WORKSPACE_ONLY = "workspace_only"
    SANDBOX = "sandbox"
    HOST = "host"


class NetworkCapability(StrEnum):
    NONE = "none"
    BOUNDED = "bounded"
    ARBITRARY = "arbitrary"


class ProcessCapability(StrEnum):
    NONE = "none"
    SANDBOXED = "sandboxed"
    HOST = "host"


class SecretsCapability(StrEnum):
    NONE = "none"
    MAY_READ = "may_read"


class DestinationKind(StrEnum):
    FILESYSTEM = "filesystem"
    NETWORK = "network"
    GIT_REMOTE = "git_remote"
    SESSION = "session"
    RESOURCE = "resource"


class ApprovalDecision(StrEnum):
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass(frozen=True)
class ExecutionCapabilities:
    filesystem: FilesystemCapability = FilesystemCapability.NONE
    network: NetworkCapability = NetworkCapability.NONE
    process: ProcessCapability = ProcessCapability.NONE
    secrets: SecretsCapability = SecretsCapability.NONE

    @classmethod
    def unconfined(cls) -> ExecutionCapabilities:
        """Represent a runtime with no enforceable capability contract.

        Returns:
            ExecutionCapabilities: Capabilities at host scope with arbitrary network access.
        """
        return cls(
            filesystem=FilesystemCapability.HOST,
            network=NetworkCapability.ARBITRARY,
            process=ProcessCapability.HOST,
            secrets=SecretsCapability.MAY_READ,
        )


@dataclass(frozen=True)
class BoundDestination:
    """Where one bound action goes, in two forms with two different readers.

    ``identifier`` is a digest of the correlation value (for a network
    destination, its origin). It is safe for logs and stable across calls that
    reach the same place.

    ``target`` is the full destination exactly as the tool will use it: a
    complete URL, including its path, so a policy can rule on a path and not
    only on a host. It is for the policy and the approval adapter only and is
    never logged (ADR-0072).
    """

    kind: DestinationKind
    scope: ScopeClass
    identifier: str
    target: str = field(repr=False)

    @classmethod
    def from_value(
        cls,
        *,
        kind: DestinationKind,
        scope: ScopeClass,
        value: str,
        target: str | None = None,
    ) -> BoundDestination:
        """Bind a correlation value to a log-safe destination identifier.

        Args:
            kind (DestinationKind): Destination kind.
            scope (ScopeClass): Scope of the destination.
            value (str): Correlation value to bind.
            target (str | None): Optional full destination target.

        Returns:
            BoundDestination: A destination with a SHA-256 identifier and full target.
        """
        digest = hashlib.sha256(f"{kind.value}\0{value}".encode()).hexdigest()
        return cls(
            kind=kind,
            scope=scope,
            identifier=f"sha256:{digest}",
            target=value if target is None else target,
        )


@dataclass(frozen=True)
class ConcreteAction:
    side_effect: SideEffectClass
    egress: DataEgressClass
    scope: ScopeClass
    execution: ExecutionClass = ExecutionClass.NONE
    capabilities: ExecutionCapabilities = field(default_factory=ExecutionCapabilities)
    destination: BoundDestination | None = None


@dataclass(frozen=True)
class ActionResolutionContext:
    """Trusted facts available while resolving one signature invocation."""

    resource_address: str
    resource_config: Mapping[str, Any]
    runtime_capabilities: ExecutionCapabilities


@runtime_checkable
class ActionSemanticsResolver(Protocol):
    """Deterministically turns bound JSON arguments into one concrete action."""

    def resolve(
        self,
        arguments: Mapping[str, JsonValue],
        context: ActionResolutionContext,
    ) -> ConcreteAction: ...


@runtime_checkable
class ToolArgumentBinder(Protocol):
    """Constrain schema-valid model arguments into their effective JSON form."""

    def bind(
        self,
        arguments: Mapping[str, JsonValue],
        context: ActionResolutionContext,
    ) -> Mapping[str, JsonValue]: ...


@dataclass(frozen=True)
class StaticActionSemantics:
    """Action semantics that do not vary with individual arguments."""

    side_effect: SideEffectClass
    egress: DataEgressClass
    scope: ScopeClass
    execution: ExecutionClass = ExecutionClass.NONE

    def resolve(
        self,
        arguments: Mapping[str, JsonValue],
        context: ActionResolutionContext,
    ) -> ConcreteAction:
        del arguments
        scope = self.scope
        if scope == ScopeClass.SANDBOX:
            filesystem = context.runtime_capabilities.filesystem
            if filesystem == FilesystemCapability.HOST:
                scope = ScopeClass.HOST
            elif filesystem == FilesystemCapability.WORKSPACE_ONLY:
                scope = ScopeClass.WORKSPACE
        execution = self.execution
        if execution != ExecutionClass.NONE:
            process = context.runtime_capabilities.process
            execution = (
                ExecutionClass.CONFINED
                if process == ProcessCapability.SANDBOXED
                else ExecutionClass.UNCONFINED
            )
        return ConcreteAction(
            side_effect=self.side_effect,
            egress=self.egress,
            scope=scope,
            execution=execution,
            capabilities=context.runtime_capabilities,
        )


@dataclass(frozen=True)
class ActionExecutionContext:
    request_id: str
    correlation_id: str
    principal: PrincipalRef | None
    workflow: str | None
    session_id: str | None
    resource_address: str
    signature_name: str
    #: The untrusted sources the proposing model has seen (ADR-0074). A signal
    #: for the policy. The framework decides nothing on it.
    exposure: ContentExposure

    @property
    def tool_subject(self) -> str:
        return f"{self.resource_address}::{self.signature_name}"

    @property
    def has_external_untrusted_input(self) -> bool:
        """Derive the untrusted-input flag from recorded exposure.

        Returns:
            bool: Whether at least one external untrusted source was seen.
        """
        return self.exposure.has_external_untrusted_input


@dataclass(frozen=True)
class ActionDecisionResult:
    decision: ActionDecision
    reason: ActionDecisionReason


@dataclass(frozen=True)
class ActionReview:
    """One immutable bound action, as the policy and the approval adapter see it.

    Both receive the same review of the same snapshot the tool will execute,
    and ``action_fingerprint`` binds a decision to exactly that snapshot.

    ``effective_arguments_json`` and ``action.destination.target`` may contain
    sensitive values. They are for the policy and the approval adapter only:
    the framework neither copies them into decision logs nor forwards them to
    any model. The implementation decides which fields to send to a reviewing
    model (ADR-0072).
    """

    context: ActionExecutionContext
    action: ConcreteAction
    action_fingerprint: str
    effective_arguments_json: str

    @property
    def effective_arguments(self) -> dict[str, JsonValue]:
        """Isolate the bound argument snapshot from caller mutations.

        Returns:
            dict[str, JsonValue]: A new argument dictionary on every access.
        """
        parsed: dict[str, JsonValue] = json.loads(self.effective_arguments_json)
        return parsed


@dataclass(frozen=True)
class ApprovalResult:
    decision: ApprovalDecision
    action_fingerprint: str
    approval_id: str | None = None


def action_fingerprint(
    *,
    context: ActionExecutionContext,
    action: ConcreteAction,
    effective_arguments_json: str,
) -> str:
    """Bind approval to the exact tool action and trusted context.

    Args:
        context (ActionExecutionContext): Bound execution context and exposure.
        action (ConcreteAction): Concrete action being approved.
        effective_arguments_json (str): Canonical JSON of effective arguments.

    Returns:
        str: A hexadecimal SHA-256 fingerprint of the review snapshot.
    """
    material = {
        "request_id": context.request_id,
        "correlation_id": context.correlation_id,
        "principal_id": context.principal.id if context.principal else None,
        "workflow": context.workflow,
        "session_id": context.session_id,
        "resource_address": context.resource_address,
        "signature_name": context.signature_name,
        "exposure": [asdict(ref) for ref in context.exposure.untrusted_sources],
        "action": asdict(action),
        "effective_arguments": json.loads(effective_arguments_json),
    }
    canonical = json.dumps(
        material,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()
