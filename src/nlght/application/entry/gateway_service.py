# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from nlght.core.entry.context import RequestContext
from nlght.core.errors.errors import UnsupportedProtocolError, WorkflowNotFoundError
from nlght.core.workflow.workflow import WorkflowInvocation
from nlght.ports.outbound.principal_resolver import PrincipalResolver
from nlght.ports.outbound.protocol_detection import ProtocolDetector
from nlght.ports.outbound.resource_repository import ResourceRepository
from nlght.ports.outbound.trigger_resolution import TriggerResolver
from nlght.ports.outbound.workflow_repository import WorkflowRepository


class GatewayService:
    def __init__(
        self,
        protocol_detector: ProtocolDetector,
        trigger_resolver: TriggerResolver,
        workflow_repository: WorkflowRepository | None = None,
        resource_repository: ResourceRepository | None = None,
        principal_resolver: PrincipalResolver | None = None,
    ) -> None:
        self._protocol_detector = protocol_detector
        self._trigger_resolver = trigger_resolver
        self._workflow_repository = workflow_repository
        self._resource_repository = resource_repository
        self._principal_resolver = principal_resolver

    async def process(
        self,
        *,
        path: str,
        method: str,
        headers: dict[str, str],
        query_params: dict[str, str],
        raw_body: bytes,
        client_host: str | None,
        workflow_name: str | None = None,
        session_key: str | None = None,
    ) -> WorkflowInvocation:
        normalized_headers = {k.lower(): v for k, v in headers.items()}

        # Established here because this is the one place an inbound
        # `RequestContext` is built, so nothing downstream can be reached with a
        # principal that skipped the resolver. `None` where none is configured
        # or the trust conditions did not hold, which is a definite answer.
        principal = (
            await self._principal_resolver.resolve(
                path=path,
                method=method,
                headers=normalized_headers,
                query_params=query_params,
                client_host=client_host,
                raw_body=raw_body,
            )
            if self._principal_resolver is not None
            else None
        )

        context = RequestContext(
            correlation_id=session_key or str(uuid.uuid4()),
            request_id=str(uuid.uuid4()),
            received_at=datetime.now(UTC),
            path=path,
            method=method,
            headers=normalized_headers,
            query_params=query_params,
            client_host=client_host,
            principal=principal,
        )

        protocol = await self._protocol_detector.detect(
            path=path,
            method=method,
            headers=normalized_headers,
            raw_body=raw_body,
        )

        if protocol.kind.value == "unknown":
            raise UnsupportedProtocolError("Unsupported or unknown protocol.")

        trigger = await self._trigger_resolver.resolve(
            protocol=protocol,
            context=context,
            raw_body=raw_body,
            session_key=session_key,
        )

        if self._workflow_repository is None:
            raise WorkflowNotFoundError(
                f"No workflow repository configured — cannot route operation '{trigger.operation}'."
            )

        lookup_name = workflow_name if workflow_name is not None else trigger.operation
        workflow = await self._workflow_repository.find_by_name(lookup_name)
        if workflow is None or not workflow.enabled:
            raise WorkflowNotFoundError(
                f"No enabled workflow found for operation '{lookup_name}'."
            )

        version = await self._workflow_repository.find_active_version(workflow.workflow_id)
        if version is None:
            raise WorkflowNotFoundError(
                f"Workflow '{workflow.name}' has no active version."
            )

        return WorkflowInvocation(trigger=trigger, workflow=workflow, version=version)
