# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from typing import Any

from nlght.core.workflow.workflow import WorkflowInvocation


def serialize_invocation(invocation: WorkflowInvocation) -> dict[str, Any]:
    trigger = invocation.trigger
    return {
        "trigger": {
            "kind": trigger.kind.value,
            "protocol": trigger.protocol.value,
            "operation": trigger.operation,
            "payload": trigger.payload,
            "metadata": trigger.metadata,
            "context": {
                "correlation_id": trigger.context.correlation_id,
                "request_id": trigger.context.request_id,
                "received_at": trigger.context.received_at.isoformat(),
                "path": trigger.context.path,
                "method": trigger.context.method,
            },
        },
        "workflow": {
            "name": invocation.workflow.name,
            "workflow_id": str(invocation.workflow.workflow_id),
            "capabilities": invocation.workflow.capabilities,
        },
        "version": {
            "version": invocation.version.version,
            "status": invocation.version.status,
            "steps": [
                {
                    "name": s.name,
                    "type": s.type,
                    "position": s.position,
                    "is_start": s.is_start,
                    "is_terminal": s.is_terminal,
                    "transitions": s.transitions,
                }
                for s in invocation.version.steps
            ],
        },
    }
