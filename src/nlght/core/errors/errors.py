# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

class AIRuntimeError(Exception):
    """Base exception for the AI Runtime platform."""


class ConfigurationError(AIRuntimeError):
    """Raised when configuration cannot be loaded or mapped."""


class UnsupportedProtocolError(AIRuntimeError):
    """Raised when no supported protocol could be detected."""


class TriggerResolutionError(AIRuntimeError):
    """Raised when a trigger cannot be resolved from a request."""


class WorkflowNotFoundError(AIRuntimeError):
    """Raised when no enabled workflow matches the trigger operation."""


class WorkflowConfigurationError(AIRuntimeError):
    """Raised when a workflow or step is misconfigured at runtime."""


class WorkflowExecutionError(AIRuntimeError):
    """Raised when a workflow terminates via the 'failed' exit verdict."""


class ToolNotFoundError(AIRuntimeError):
    """Raised when a tool name is not present in the ToolCatalog."""


class ToolExecutionError(AIRuntimeError):
    """Raised when a tool's execute() call fails at runtime."""