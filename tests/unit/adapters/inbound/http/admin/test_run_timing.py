# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""When a run finished, which for a fan-out is not when its execution did."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

from nlght.adapters.inbound.http.admin.services import run_timing
from nlght.core.entry.context import RequestContext
from nlght.core.execution import ExecutionRecord, ExecutionStatus, FanOutSummary
from nlght.core.protocol.protocol import ProtocolKind
from nlght.core.trigger.trigger import Trigger, TriggerKind

_T0 = datetime(2026, 8, 27, 10, 0, 0, tzinfo=UTC)


def _at(seconds: float) -> datetime:
    return _T0 + timedelta(seconds=seconds)


def _record(
    *,
    status: ExecutionStatus = ExecutionStatus.SUCCEEDED,
    started_at: datetime | None = _T0,
    completed_at: datetime | None = None,
) -> ExecutionRecord:
    trigger = Trigger(
        kind=TriggerKind.INBOUND_EVENT,
        protocol=ProtocolKind.GENERIC_JSON,
        operation="ingest",
        payload={},
        context=RequestContext(
            correlation_id="corr", request_id="req", received_at=_T0,
            path="/", method="POST", headers={}, query_params={}, client_host=None,
        ),
    )
    return ExecutionRecord(
        execution_id=uuid.uuid4(), workflow_id=uuid.uuid4(), workflow_version_id=uuid.uuid4(),
        trigger=trigger, idempotency_key="k", required_capabilities=(), priority=0,
        artifact_refs=(), source_config_revision=None, max_attempts=3, attempt_count=1,
        fencing_token=1, status=status, owner_worker_id=None, owner_boot_token=None,
        lease_expires_at=None, next_attempt_at=_T0, cancel_requested=False, progress={},
        result=None, diagnostics=None, created_at=_T0, updated_at=_T0,
        started_at=started_at, completed_at=completed_at,
    )


def _summary(
    *,
    total: int,
    succeeded: int = 0,
    pending: int = 0,
    first_started_at: datetime | None = None,
    last_completed_at: datetime | None = None,
    child_seconds: float = 0.0,
) -> FanOutSummary:
    return FanOutSummary(
        parent_execution_id=uuid.uuid4(), total=total, pending=pending,
        succeeded=succeeded, failed=0, cancelled=0,
        first_started_at=first_started_at,
        last_completed_at=last_completed_at,
        child_seconds=child_seconds,
    )


def test_a_run_that_never_fanned_out_is_its_own_execution() -> None:
    timing = run_timing(_record(completed_at=_at(30)), None)

    assert timing.finished_at == _at(30)
    assert timing.elapsed == timedelta(seconds=30)
    assert timing.running is False
    # There are no children, so there is no child time — not zero, which would
    # read as "the children took no time".
    assert timing.child_time is None


def test_a_fan_out_finishes_with_its_last_child_not_its_distribution() -> None:
    # The whole point. The distribution ended two seconds in, having handed the
    # corpus out; the ingest ran for another twenty minutes.
    timing = run_timing(
        _record(completed_at=_at(2)),
        _summary(
            total=100, succeeded=100,
            first_started_at=_at(3), last_completed_at=_at(1200),
            child_seconds=7200.0,
        ),
    )

    assert timing.finished_at == _at(1200)
    assert timing.elapsed == timedelta(seconds=1200)
    assert timing.running is False


def test_nothing_has_finished_while_a_child_is_still_pending() -> None:
    # A maximum completion time exists — 87 children have finished — but it is
    # not a finish, and reporting it as one is how a half-done run reads as over.
    timing = run_timing(
        _record(completed_at=_at(2)),
        _summary(
            total=100, succeeded=87, pending=13,
            first_started_at=_at(3), last_completed_at=_at(900),
            child_seconds=5400.0,
        ),
        now=_at(1000),
    )

    assert timing.finished_at is None
    assert timing.running is True
    # Still counting, against now rather than against the last child seen.
    assert timing.elapsed == timedelta(seconds=1000)


def test_child_time_is_the_work_spent_not_the_time_that_passed() -> None:
    timing = run_timing(
        _record(completed_at=_at(2)),
        _summary(
            total=100, succeeded=100,
            first_started_at=_at(3), last_completed_at=_at(600),
            child_seconds=4800.0,
        ),
    )

    assert timing.child_time == timedelta(seconds=4800)
    assert timing.elapsed == timedelta(seconds=600)
    # Eight times as much work as time — which is what the worker pool bought.
    assert timing.parallel_factor is not None
    assert round(timing.parallel_factor, 2) == 8.0


def test_a_single_child_reports_no_parallel_factor() -> None:
    # One child doing one thing is not parallelism worth mentioning.
    timing = run_timing(
        _record(completed_at=_at(2)),
        _summary(
            total=1, succeeded=1,
            first_started_at=_at(3), last_completed_at=_at(600),
            child_seconds=597.0,
        ),
    )

    assert timing.parallel_factor is None


def test_a_parent_outliving_every_child_still_decides_the_finish() -> None:
    # It can happen: the distribution's own step keeps going after the last
    # share was handed out. The run ends with whichever ended last.
    timing = run_timing(
        _record(completed_at=_at(900)),
        _summary(
            total=5, succeeded=5,
            first_started_at=_at(3), last_completed_at=_at(400),
            child_seconds=1500.0,
        ),
    )

    assert timing.finished_at == _at(900)


def test_a_run_not_yet_claimed_is_measured_from_submission() -> None:
    timing = run_timing(
        _record(status=ExecutionStatus.QUEUED, started_at=None), None, now=_at(45)
    )

    assert timing.started_at == _T0
    assert timing.elapsed == timedelta(seconds=45)
    assert timing.running is True


def test_timestamps_that_lost_their_offset_are_read_as_utc() -> None:
    # SQLite has nowhere to keep the offset, so the same column comes back naive.
    # Comparing that against an aware `now` raises rather than being merely
    # wrong, which would take the whole executions page down.
    naive = _T0.replace(tzinfo=None)
    timing = run_timing(
        _record(started_at=naive, completed_at=None), None, now=_at(60)
    )

    assert timing.elapsed == timedelta(seconds=60)
