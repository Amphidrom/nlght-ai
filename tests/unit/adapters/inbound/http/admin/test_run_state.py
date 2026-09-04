# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""What a run is doing, which is not always what its execution is doing."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from nlght.adapters.inbound.http.admin.services import run_state
from nlght.core.entry.context import RequestContext
from nlght.core.execution import ExecutionRecord, ExecutionStatus, FanOutSummary
from nlght.core.protocol.protocol import ProtocolKind
from nlght.core.trigger.trigger import Trigger, TriggerKind


def _record(status: ExecutionStatus) -> ExecutionRecord:
    now = datetime(2026, 8, 27, tzinfo=UTC)
    trigger = Trigger(
        kind=TriggerKind.INBOUND_EVENT,
        protocol=ProtocolKind.GENERIC_JSON,
        operation="ingest",
        payload={},
        context=RequestContext(
            correlation_id="corr", request_id="req", received_at=now,
            path="/", method="POST", headers={}, query_params={}, client_host=None,
        ),
    )
    return ExecutionRecord(
        execution_id=uuid.uuid4(), workflow_id=uuid.uuid4(), workflow_version_id=uuid.uuid4(),
        trigger=trigger, idempotency_key="k", required_capabilities=(), priority=0,
        artifact_refs=(), source_config_revision=None, max_attempts=3, attempt_count=1,
        fencing_token=1, status=status, owner_worker_id=None, owner_boot_token=None,
        lease_expires_at=None, next_attempt_at=now, cancel_requested=False, progress={},
        result=None, diagnostics=None, created_at=now, updated_at=now,
        started_at=now, completed_at=now,
    )


def _summary(*, total: int, succeeded: int = 0, failed: int = 0,
             cancelled: int = 0, pending: int = 0) -> FanOutSummary:
    return FanOutSummary(
        parent_execution_id=uuid.uuid4(), total=total, pending=pending,
        succeeded=succeeded, failed=failed, cancelled=cancelled,
    )


def test_a_run_that_never_fanned_out_is_its_execution() -> None:
    for status in ExecutionStatus:
        assert run_state(_record(status), None).label == status.value


def test_a_distribution_that_handed_work_out_is_not_finished() -> None:
    # The reported bug: a knowledge ingestion read `succeeded` while most of its
    # children had not run. The execution really did succeed — it submits and
    # ends, because waiting would hold a worker for as long as the corpus takes.
    # The run had not.
    state = run_state(
        _record(ExecutionStatus.SUCCEEDED),
        _summary(total=151, succeeded=12, pending=139),
    )

    assert state.label == "in progress"
    assert state.badge == "running"
    assert "139 still to finish" in state.detail


def test_a_run_is_only_succeeded_once_every_child_is() -> None:
    state = run_state(
        _record(ExecutionStatus.SUCCEEDED), _summary(total=151, succeeded=151)
    )

    assert state.label == "succeeded"
    assert "all 151 children succeeded" in state.detail


def test_a_settled_run_with_failed_children_does_not_read_as_success() -> None:
    state = run_state(
        _record(ExecutionStatus.SUCCEEDED), _summary(total=10, succeeded=8, failed=2)
    )

    assert state.label == "children failed"
    assert state.badge == "failed"


def test_cancelled_children_are_reported_as_such() -> None:
    state = run_state(
        _record(ExecutionStatus.SUCCEEDED), _summary(total=10, succeeded=8, cancelled=2)
    )

    assert state.label == "partly cancelled"


def test_failed_children_outrank_cancelled_ones() -> None:
    # Something needs a person either way, and a failure is the louder of the two.
    state = run_state(
        _record(ExecutionStatus.SUCCEEDED),
        _summary(total=10, succeeded=6, failed=2, cancelled=2),
    )

    assert state.label == "children failed"


def test_a_distribution_that_failed_stays_failed_whatever_its_children_did() -> None:
    # It may have submitted some children before failing; the run still failed.
    state = run_state(
        _record(ExecutionStatus.FAILED), _summary(total=3, succeeded=3)
    )

    assert state.label == "failed"
    assert state.detail == "the distribution itself"


def test_children_still_running_beat_a_still_running_distribution() -> None:
    state = run_state(_record(ExecutionStatus.RUNNING), _summary(total=5, pending=5))

    assert state.label == "running"
