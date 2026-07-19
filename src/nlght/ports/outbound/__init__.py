# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Outbound port protocols — interfaces implemented by infrastructure adapters.

Import individual ports directly for minimal coupling:

    from nlght.ports.outbound.model_client import ModelClient
    from nlght.ports.outbound.signal_emitter import SignalEmitter
"""

from nlght.ports.outbound.access_policy import (
    ModelAccessPolicy,
    PlaybookAccessPolicy,
    ToolAccessPolicy,
)
from nlght.ports.outbound.access_rule_repository import AccessRuleRepository
from nlght.ports.outbound.lexical_store import LexicalResult, LexicalStore
from nlght.ports.outbound.licensing import LicensingPort
from nlght.ports.outbound.metering import MeteringPort
from nlght.ports.outbound.model_client import ModelClient, ModelStreamEvent
from nlght.ports.outbound.model_gateway import ModelGateway
from nlght.ports.outbound.model_provider_backend import (
    ModelProviderBackend,
    NativeProxyCapable,
    RunningModelsCapable,
)
from nlght.ports.outbound.os_runtime import OsRuntime
from nlght.ports.outbound.playbook_catalog import PlaybookCatalog
from nlght.ports.outbound.protocol_detection import ProtocolDetector
from nlght.ports.outbound.resource_repository import ResourceRepository
from nlght.ports.outbound.session_backend import SessionBackend
from nlght.ports.outbound.session_key_resolver import SessionKeyResolver
from nlght.ports.outbound.signal_emitter import SignalEmitter
from nlght.ports.outbound.store_coordinator import StoreCoordinator, StoreCoordinatorFactory
from nlght.ports.outbound.tool_catalog import ToolCatalog, ToolContract
from nlght.ports.outbound.trigger_resolution import TriggerResolver
from nlght.ports.outbound.vector_store import VectorResult, VectorStore
from nlght.ports.outbound.workflow_executor import WorkflowExecutor
from nlght.ports.outbound.workflow_repository import WorkflowRepository
from nlght.ports.outbound.workspace_manager import WorkspaceManager

__all__ = [
    # Access policies
    "AccessRuleRepository",
    "ModelAccessPolicy",
    "PlaybookAccessPolicy",
    "ToolAccessPolicy",
    # Licensing
    "LicensingPort",
    # Infrastructure protocols
    "MeteringPort",
    "ModelClient",
    "ModelStreamEvent",
    "ModelGateway",
    "ModelProviderBackend",
    "NativeProxyCapable",
    "RunningModelsCapable",
    "OsRuntime",
    "ProtocolDetector",
    "ResourceRepository",
    "SessionBackend",
    "SessionKeyResolver",
    "SignalEmitter",
    "PlaybookCatalog",
    "LexicalStore",
    "LexicalResult",
    "StoreCoordinator",
    "StoreCoordinatorFactory",
    "ToolCatalog",
    "ToolContract",
    "VectorStore",
    "VectorResult",
    "TriggerResolver",
    "WorkflowExecutor",
    "WorkflowRepository",
    "WorkspaceManager",
]
