# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

class AIRuntimeError(Exception):
    """Base exception for the AI Runtime platform."""


class ConfigurationError(AIRuntimeError):
    """Raised when configuration cannot be loaded or mapped."""


class UnsupportedProtocolError(AIRuntimeError):
    """Raised when no supported protocol could be detected."""


class SessionAccessDeniedError(AIRuntimeError):
    """A session was asked for by somebody it does not belong to.

    Worded identically whether the session belongs to another principal or does
    not exist. Distinguishing them would answer "is this session id real?" for
    anybody willing to ask, which is the enumeration the ownership check exists
    to prevent.
    """


class PrincipalConflictError(AIRuntimeError):
    """Two trusted sources established two different principals for one call.

    Not resolved by preferring one: both sources were configured because both
    are trusted, so a disagreement between them is a broken deployment and not
    a tie to break. Picking a winner would make which identity a request runs
    as depend on resolver order.
    """


class ResourceAddressAlreadyExists(AIRuntimeError):
    """Two resources cannot share ``<kind>/<name>``.

    The guarantee is the database constraint, not this class: a check before the
    insert races another writer, and an address that two rows can answer to
    cannot carry an access decision. This is what the constraint violation is
    translated into, so a caller sees a conflict it can report rather than a
    driver error it cannot.
    """

    def __init__(self, kind: str, name: str) -> None:
        self.kind = kind
        self.name = name
        self.address = f"{kind}/{name}"
        super().__init__(f"Resource address '{self.address}' already exists.")


class TriggerResolutionError(AIRuntimeError):
    """Raised when a trigger cannot be resolved from a request."""


class WorkflowNotFoundError(AIRuntimeError):
    """Raised when no enabled workflow matches the trigger operation."""


class PermanentError(AIRuntimeError):
    """A failure that will fail identically however often it is retried.

    The distinction the worker needs and could not previously make: a timeout, a
    refused connection and a busy backend are worth another attempt; a
    misconfigured step and a write the domain refuses are not. Retrying the
    second kind burns attempts on an outcome that cannot change — and where a
    model is involved it is worse than wasteful, because a nondeterministic
    answer can make the retry *succeed* and hide the fault instead of surfacing
    it.

    A type rather than a message. Classifying by matching text is a guess about
    wording that breaks the moment somebody rephrases an error, and it cannot
    distinguish two failures that read alike.
    """


class ResourceAccessDeniedError(PermanentError):
    """A caller asked for a resource the access policy does not grant it.

    Permanent by construction: the same workflow asking again gets the same
    answer, so retrying is not a recovery. A ``PermanentError`` so the durable
    execution path does not spend attempts on it.

    Raised rather than filtered because the caller *named* this resource. The
    tool catalog filters — a model is not told about what it may not use — but a
    step naming a resource has stated a requirement, and continuing without it
    would run something other than the workflow that was configured.
    """

    def __init__(
        self,
        address: str,
        workflow: str | None = None,
        reason: str | None = None,
    ) -> None:
        self.address = address
        self.workflow = workflow
        where = f"workflow '{workflow}'" if workflow else "a caller outside any workflow"
        because = f" — {reason}" if reason else ""
        super().__init__(
            f"Access to resource '{address}' is not granted to {where}{because}."
        )


class AmbiguousToolNameError(PermanentError):
    """Two resources this caller may use offer the same model-facing operation.

    A model calls an operation by its short name, so two resources offering
    `data-search` are two answers to one question. The catalog used to key by
    that name and let the second write over the first, which meant one of the
    two activations was unreachable and which one depended on the order rows
    came back in.

    Raised at build time rather than resolved, because there is no correct
    resolution available here. Renaming the operation would change an API a
    model was told about; picking one would be the silent behaviour under a
    different name. Which resource this workflow should see is a configuration
    decision, and the operator is the one who can make it — narrowing it with a
    `resource` or `tool` rule removes the collision, because a resource the
    caller may not use offers nothing to collide with.

    Permanent: the same configuration produces the same clash on every request.
    """

    def __init__(self, operation: str, addresses: tuple[str, ...]) -> None:
        self.operation = operation
        self.addresses = addresses
        named = " and ".join(f"'{address}'" for address in addresses)
        super().__init__(
            f"Operation '{operation}' is offered by {named}. A model calls it by "
            f"that one name, so only one resource can provide it to a given "
            f"workflow — restrict one of them with an access rule, or stop "
            f"enabling both."
        )


class WorkflowConfigurationError(PermanentError):
    """Raised when a workflow or step is misconfigured at runtime."""


class WorkflowExecutionError(AIRuntimeError):
    """Raised when a workflow terminates via the 'failed' exit verdict."""


class ToolNotFoundError(AIRuntimeError):
    """Raised when a tool name is not present in the ToolCatalog."""


class ToolExecutionError(AIRuntimeError):
    """Raised when a tool's execute() call fails at runtime."""


class ToolActionSemanticsError(PermanentError):
    """A model-callable tool did not declare resolvable action semantics."""

    def __init__(self, subject: str) -> None:
        self.subject = subject
        super().__init__(
            f"Tool '{subject}' has no action semantics and cannot enter an "
            "execution-capable catalog."
        )


class ToolArgumentsError(PermanentError):
    """A model-proposed argument set does not satisfy the tool signature."""


class ToolActionDeniedError(PermanentError):
    """The deterministic action boundary refused a concrete tool invocation."""

    def __init__(self, subject: str, reason: str) -> None:
        self.subject = subject
        self.reason = reason
        super().__init__(f"Action '{subject}' denied: {reason}.")
