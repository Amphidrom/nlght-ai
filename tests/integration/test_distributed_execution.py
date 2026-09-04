# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import uuid
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from nlght.adapters.outbound.hive_mind.simple import SimpleStoreCoordinatorFactory
from nlght.adapters.outbound.persistence.execution_repository import (
    SqlAlchemyExecutionArtifactStore,
    SqlAlchemyExecutionRepository,
)
from nlght.adapters.outbound.persistence.models import Workflow, WorkflowVersion
from nlght.core.entry.context import PrincipalRef, RequestContext
from nlght.core.errors.errors import SessionAccessDeniedError
from nlght.core.execution import ExecutionStatus, ExecutionSubmission, WorkerRegistration
from nlght.core.protocol.protocol import ProtocolKind
from nlght.core.session import SessionAccess
from nlght.core.trigger.trigger import Trigger, TriggerKind


def _trigger() -> Trigger:
    return Trigger(
        kind=TriggerKind.HTTP_REQUEST,
        protocol=ProtocolKind.GENERIC_JSON,
        operation="ingest",
        payload={"document": "fixture"},
        context=RequestContext(
            correlation_id="corr-1",
            request_id="req-1",
            received_at=datetime(2026, 8, 21, tzinfo=UTC),
            path="/ingest",
            method="POST",
            headers={},
            query_params={},
            client_host="127.0.0.1",
        ),
    )


async def _seed_workflow(sqlite_engine) -> tuple[uuid.UUID, uuid.UUID]:
    workflow_id = uuid.uuid4()
    version_id = uuid.uuid4()
    async with AsyncSession(sqlite_engine) as session:
        async with session.begin():
            session.add(
                Workflow(
                    workflow_id=workflow_id,
                    name=f"ingestion-{workflow_id}",
                    enabled=True,
                    capabilities=[],
                )
            )
            session.add(
                WorkflowVersion(
                    workflow_version_id=version_id,
                    workflow_id=workflow_id,
                    version=1,
                    status="active",
                )
            )
    return workflow_id, version_id


def _submission(
    workflow_id: uuid.UUID,
    version_id: uuid.UUID,
    key: str,
    *,
    capabilities: tuple[str, ...] = (),
    exclusive: bool = False,
) -> ExecutionSubmission:
    return ExecutionSubmission(
        workflow_id=workflow_id,
        workflow_version_id=version_id,
        trigger=_trigger(),
        idempotency_key=key,
        required_capabilities=capabilities,
        exclusive=exclusive,
    )


def _worker(*capabilities: str) -> WorkerRegistration:
    return WorkerRegistration(
        worker_id=uuid.uuid4(),
        boot_token=uuid.uuid4(),
        instance_name="test-worker",
        capabilities=capabilities,
    )


async def test_submit_is_idempotent_and_rejects_key_reuse(sqlite_engine) -> None:
    workflow_id, version_id = await _seed_workflow(sqlite_engine)
    repository = SqlAlchemyExecutionRepository(sqlite_engine)
    submission = _submission(workflow_id, version_id, "source:revision-1")

    first = await repository.submit(submission)
    second = await repository.submit(submission)

    assert second.execution_id == first.execution_id
    assert second.status is ExecutionStatus.QUEUED
    with pytest.raises(ValueError, match="different execution submission"):
        await repository.submit(
            _submission(workflow_id, version_id, "source:revision-1", capabilities=("parser:pdf",))
        )


async def test_workers_claim_disjoint_compatible_jobs(sqlite_engine) -> None:
    workflow_id, version_id = await _seed_workflow(sqlite_engine)
    repository = SqlAlchemyExecutionRepository(sqlite_engine)
    pdf = _worker("parser:pdf")
    markdown = _worker("parser:markdown")
    now = datetime.now(UTC)
    for worker in (pdf, markdown):
        await repository.register_worker(worker, expires_at=now + timedelta(minutes=1))
    pdf_job = await repository.submit(
        _submission(workflow_id, version_id, "pdf", capabilities=("parser:pdf",))
    )
    markdown_job = await repository.submit(
        _submission(workflow_id, version_id, "markdown", capabilities=("parser:markdown",))
    )

    claim_time = datetime.now(UTC) + timedelta(milliseconds=1)
    pdf_claim = await repository.claim(
        pdf,
        now=claim_time,
        lease_expires_at=claim_time + timedelta(seconds=30),
    )
    markdown_claim = await repository.claim(
        markdown,
        now=claim_time,
        lease_expires_at=claim_time + timedelta(seconds=30),
    )

    assert pdf_claim is not None and pdf_claim.execution.execution_id == pdf_job.execution_id
    assert markdown_claim is not None
    assert markdown_claim.execution.execution_id == markdown_job.execution_id
    assert await repository.claim(
        pdf,
        now=claim_time,
        lease_expires_at=claim_time + timedelta(seconds=30),
    ) is None


async def test_worker_without_capabilities_is_a_wildcard(sqlite_engine) -> None:
    # An empty-capability worker (the single-node gateway+worker default) must
    # claim a job even when the workflow declares a requirement — capabilities
    # are opt-in for heterogeneous pools, not a gate every job must pass.
    workflow_id, version_id = await _seed_workflow(sqlite_engine)
    repository = SqlAlchemyExecutionRepository(sqlite_engine)
    wildcard = _worker()  # no declared capabilities
    now = datetime.now(UTC)
    await repository.register_worker(wildcard, expires_at=now + timedelta(minutes=1))
    job = await repository.submit(
        _submission(workflow_id, version_id, "needs-knowledge", capabilities=("ingestion:knowledge",))
    )

    claim_time = datetime.now(UTC) + timedelta(milliseconds=1)
    claim = await repository.claim(
        wildcard, now=claim_time, lease_expires_at=claim_time + timedelta(seconds=30)
    )

    assert claim is not None
    assert claim.execution.execution_id == job.execution_id


async def test_worker_with_capabilities_still_skips_a_requirement_it_lacks(sqlite_engine) -> None:
    # A worker that *does* declare capabilities stays restricted to jobs whose
    # requirements it fully satisfies — only the empty set is a wildcard.
    workflow_id, version_id = await _seed_workflow(sqlite_engine)
    repository = SqlAlchemyExecutionRepository(sqlite_engine)
    pdf = _worker("parser:pdf")
    now = datetime.now(UTC)
    await repository.register_worker(pdf, expires_at=now + timedelta(minutes=1))
    await repository.submit(
        _submission(workflow_id, version_id, "needs-markdown", capabilities=("parser:markdown",))
    )

    claim_time = datetime.now(UTC) + timedelta(milliseconds=1)
    claim = await repository.claim(
        pdf, now=claim_time, lease_expires_at=claim_time + timedelta(seconds=30)
    )

    assert claim is None


async def test_blocking_workflow_runs_one_execution_at_a_time(sqlite_engine) -> None:
    # A blocking (exclusive) workflow serialises: while one execution is
    # leased/running, a second is not claimable; it becomes claimable only once
    # the first reaches a terminal state.
    workflow_id, version_id = await _seed_workflow(sqlite_engine)
    repository = SqlAlchemyExecutionRepository(sqlite_engine)
    worker = _worker()
    now = datetime.now(UTC)
    await repository.register_worker(worker, expires_at=now + timedelta(minutes=1))
    first = await repository.submit(_submission(workflow_id, version_id, "a", exclusive=True))
    second = await repository.submit(_submission(workflow_id, version_id, "b", exclusive=True))

    # After the submits, not before: `submit` stamps next_attempt_at with its own
    # clock, and an execution is only claimable once that moment has passed. A
    # claim time taken before the inserts is a race the test loses as soon as
    # they take longer than the millisecond it allowed for.
    claim_time = datetime.now(UTC) + timedelta(milliseconds=1)
    lease = claim_time + timedelta(seconds=30)
    first_claim = await repository.claim(worker, now=claim_time, lease_expires_at=lease)
    blocked = await repository.claim(worker, now=claim_time, lease_expires_at=lease)

    assert first_claim is not None and first_claim.execution.execution_id == first.execution_id
    assert first_claim.execution.exclusive is True
    assert blocked is None

    await repository.mark_running(first_claim, lease_expires_at=lease)
    await repository.succeed(first_claim, {"ok": True})

    second_claim = await repository.claim(worker, now=claim_time, lease_expires_at=lease)
    assert second_claim is not None
    assert second_claim.execution.execution_id == second.execution_id


async def test_non_blocking_workflow_allows_parallel_executions(sqlite_engine) -> None:
    # Without the exclusive flag, two executions of the same workflow run at once.
    workflow_id, version_id = await _seed_workflow(sqlite_engine)
    repository = SqlAlchemyExecutionRepository(sqlite_engine)
    worker = _worker()
    now = datetime.now(UTC)
    await repository.register_worker(worker, expires_at=now + timedelta(minutes=1))
    first = await repository.submit(_submission(workflow_id, version_id, "a"))
    second = await repository.submit(_submission(workflow_id, version_id, "b"))

    # See above: the claim time has to be later than what `submit` recorded.
    claim_time = datetime.now(UTC) + timedelta(milliseconds=1)
    lease = claim_time + timedelta(seconds=30)
    first_claim = await repository.claim(worker, now=claim_time, lease_expires_at=lease)
    second_claim = await repository.claim(worker, now=claim_time, lease_expires_at=lease)

    assert first_claim is not None and second_claim is not None
    assert {first_claim.execution.execution_id, second_claim.execution.execution_id} == {
        first.execution_id,
        second.execution_id,
    }


async def test_expired_lease_is_recovered_and_stale_owner_is_fenced(sqlite_engine) -> None:
    workflow_id, version_id = await _seed_workflow(sqlite_engine)
    repository = SqlAlchemyExecutionRepository(sqlite_engine)
    first_worker = _worker()
    second_worker = _worker()
    now = datetime.now(UTC)
    for worker in (first_worker, second_worker):
        await repository.register_worker(worker, expires_at=now + timedelta(minutes=1))
    await repository.submit(_submission(workflow_id, version_id, "recover"))

    claim_time = datetime.now(UTC) + timedelta(milliseconds=1)
    first = await repository.claim(
        first_worker,
        now=claim_time,
        lease_expires_at=claim_time + timedelta(seconds=1),
    )
    assert first is not None
    assert await repository.mark_running(
        first,
        lease_expires_at=claim_time + timedelta(seconds=1),
    )
    second = await repository.claim(
        second_worker,
        now=claim_time + timedelta(seconds=2),
        lease_expires_at=claim_time + timedelta(seconds=32),
    )

    assert second is not None
    assert second.execution.execution_id == first.execution.execution_id
    assert second.fencing_token > first.fencing_token
    assert not await repository.succeed(first, {"stale": True})
    assert await repository.mark_running(
        second,
        lease_expires_at=claim_time + timedelta(seconds=32),
    )
    assert await repository.succeed(second, {"ok": True})


async def test_cancel_fences_running_completion(sqlite_engine) -> None:
    workflow_id, version_id = await _seed_workflow(sqlite_engine)
    repository = SqlAlchemyExecutionRepository(sqlite_engine)
    worker = _worker()
    now = datetime.now(UTC)
    await repository.register_worker(worker, expires_at=now + timedelta(minutes=1))
    submitted = await repository.submit(_submission(workflow_id, version_id, "cancel"))
    claim_time = datetime.now(UTC) + timedelta(milliseconds=1)
    claim = await repository.claim(
        worker,
        now=claim_time,
        lease_expires_at=claim_time + timedelta(seconds=30),
    )
    assert claim is not None
    assert await repository.mark_running(
        claim,
        lease_expires_at=claim_time + timedelta(seconds=30),
    )

    cancelled = await repository.cancel(submitted.execution_id)

    assert cancelled is not None and cancelled.status is ExecutionStatus.CANCELLED
    assert cancelled.cancel_requested
    assert not await repository.succeed(claim, {"too_late": True})


async def test_expired_last_attempt_becomes_terminal_and_stopped_worker_cannot_claim(
    sqlite_engine,
) -> None:
    workflow_id, version_id = await _seed_workflow(sqlite_engine)
    repository = SqlAlchemyExecutionRepository(sqlite_engine)
    worker = _worker()
    now = datetime.now(UTC)
    await repository.register_worker(worker, expires_at=now + timedelta(minutes=1))
    submitted = await repository.submit(
        replace(_submission(workflow_id, version_id, "exhaust"), max_attempts=1)
    )
    claim_time = datetime.now(UTC) + timedelta(milliseconds=1)
    claim = await repository.claim(
        worker,
        now=claim_time,
        lease_expires_at=claim_time + timedelta(seconds=1),
    )
    assert claim is not None
    assert await repository.mark_running(
        claim,
        lease_expires_at=claim_time + timedelta(seconds=1),
    )

    assert await repository.claim(
        worker,
        now=claim_time + timedelta(seconds=2),
        lease_expires_at=claim_time + timedelta(seconds=32),
    ) is None
    terminal = await repository.get(submitted.execution_id)
    assert terminal is not None
    assert terminal.status is ExecutionStatus.FAILED
    assert terminal.diagnostics is not None
    assert terminal.diagnostics["type"] == "AttemptsExhausted"

    await repository.submit(_submission(workflow_id, version_id, "after-stop"))
    await repository.stop_worker(worker.worker_id, worker.boot_token)
    assert await repository.claim(
        worker,
        now=claim_time + timedelta(seconds=3),
        lease_expires_at=claim_time + timedelta(seconds=33),
    ) is None


async def test_execution_artifacts_are_durable_and_content_idempotent(sqlite_engine) -> None:
    workflow_id, version_id = await _seed_workflow(sqlite_engine)
    repository = SqlAlchemyExecutionRepository(sqlite_engine)
    store = SqlAlchemyExecutionArtifactStore(sqlite_engine)
    execution = await repository.submit(_submission(workflow_id, version_id, "artifact"))

    first = await store.put(
        execution_id=execution.execution_id,
        name="source.txt",
        content_type="text/plain",
        content=b"durable source",
    )
    second = await store.put(
        execution_id=execution.execution_id,
        name="source.txt",
        content_type="text/plain",
        content=b"durable source",
    )

    assert second.artifact_id == first.artifact_id
    loaded = await store.get(first.artifact_id)
    assert loaded is not None
    assert loaded.artifact_id == first.artifact_id
    assert loaded.checksum == first.checksum
    assert loaded.content == first.content
    with pytest.raises(ValueError, match="different content"):
        await store.put(
            execution_id=execution.execution_id,
            name="source.txt",
            content_type="text/plain",
            content=b"changed",
        )


# ---------------------------------------------------------------------------
# Observation — the read-only surface the admin UI is built on
# ---------------------------------------------------------------------------


async def test_listing_runs_hides_children_behind_their_parent(sqlite_engine) -> None:
    # A fanned-out corpus is one run with many children. Listing every execution
    # would bury the runs that matter under the shares of one of them.
    workflow_id, version_id = await _seed_workflow(sqlite_engine)
    repository = SqlAlchemyExecutionRepository(sqlite_engine)
    parent = await repository.submit(_submission(workflow_id, version_id, "parent"))
    for index in range(3):
        await repository.submit(
            replace(
                _submission(workflow_id, version_id, f"child-{index}"),
                parent_execution_id=parent.execution_id,
            )
        )

    runs = await repository.list_runs()

    assert [run.execution_id for run in runs] == [parent.execution_id]
    assert await repository.count_runs() == 1
    assert len(await repository.list_children(parent.execution_id)) == 3


async def test_listing_runs_filters_by_status_and_workflow(sqlite_engine) -> None:
    workflow_id, version_id = await _seed_workflow(sqlite_engine)
    other_workflow_id, other_version_id = await _seed_workflow(sqlite_engine)
    repository = SqlAlchemyExecutionRepository(sqlite_engine)
    queued = await repository.submit(_submission(workflow_id, version_id, "queued"))
    cancelled = await repository.submit(_submission(workflow_id, version_id, "cancelled"))
    await repository.cancel(cancelled.execution_id)
    await repository.submit(_submission(other_workflow_id, other_version_id, "other"))

    by_status = await repository.list_runs(status=ExecutionStatus.QUEUED)
    by_workflow = await repository.list_runs(workflow_id=other_workflow_id)

    assert queued.execution_id in {run.execution_id for run in by_status}
    assert cancelled.execution_id not in {run.execution_id for run in by_status}
    assert {run.workflow_id for run in by_workflow} == {other_workflow_id}


async def test_fan_out_summaries_answer_for_a_whole_page_at_once(sqlite_engine) -> None:
    # One query for the listing, not one per row.
    workflow_id, version_id = await _seed_workflow(sqlite_engine)
    repository = SqlAlchemyExecutionRepository(sqlite_engine)
    first = await repository.submit(_submission(workflow_id, version_id, "first"))
    second = await repository.submit(_submission(workflow_id, version_id, "second"))
    for parent, count in ((first, 2), (second, 1)):
        for index in range(count):
            await repository.submit(
                replace(
                    _submission(workflow_id, version_id, f"{parent.idempotency_key}-{index}"),
                    parent_execution_id=parent.execution_id,
                )
            )

    summaries = await repository.fan_out_summaries(
        (first.execution_id, second.execution_id)
    )

    assert summaries[first.execution_id].total == 2
    assert summaries[second.execution_id].total == 1
    assert summaries[first.execution_id].pending == 2


async def test_attempts_name_the_worker_that_held_each_try(sqlite_engine) -> None:
    # The record says where a run ended up. This says how it got there — without
    # it a run that succeeded on its second try is indistinguishable from one
    # that succeeded outright.
    workflow_id, version_id = await _seed_workflow(sqlite_engine)
    repository = SqlAlchemyExecutionRepository(sqlite_engine)
    record = await repository.submit(_submission(workflow_id, version_id, "retried"))
    worker = _worker()
    now = datetime.now(UTC)
    await repository.register_worker(worker, expires_at=now + timedelta(minutes=5))

    first = await repository.claim(worker, now=now, lease_expires_at=now + timedelta(minutes=1))
    assert first is not None
    await repository.fail(first, {"type": "RuntimeError", "message": "boom"}, retry_at=now)
    second = await repository.claim(worker, now=now, lease_expires_at=now + timedelta(minutes=1))
    assert second is not None
    # succeed() only accepts a running execution, which is what the worker loop
    # does between claiming and finishing.
    await repository.mark_running(second, lease_expires_at=now + timedelta(minutes=1))
    await repository.succeed(second, {"ok": True})

    attempts = await repository.attempts(record.execution_id)

    assert [attempt.attempt_number for attempt in attempts] == [1, 2]
    assert attempts[0].status == "failed"
    assert attempts[0].diagnostics is not None
    assert attempts[0].diagnostics["message"] == "boom"
    assert attempts[1].status == "succeeded"
    assert all(attempt.instance_name == "test-worker" for attempt in attempts)


async def test_a_worker_that_stopped_without_deregistering_reads_as_expired(
    sqlite_engine,
) -> None:
    # It stays 'active' in its own row until the lease runs out, and whatever it
    # held is only reclaimed then. Saying so is the point of the workers view.
    workflow_id, version_id = await _seed_workflow(sqlite_engine)
    repository = SqlAlchemyExecutionRepository(sqlite_engine)
    now = datetime.now(UTC)
    alive, dead = _worker("ingestion"), _worker()
    await repository.register_worker(alive, expires_at=now + timedelta(minutes=5))
    await repository.register_worker(dead, expires_at=now - timedelta(minutes=1))
    await repository.submit(_submission(workflow_id, version_id, "held"))
    # After the submit, for the reason above.
    claim_time = datetime.now(UTC) + timedelta(milliseconds=1)
    claim = await repository.claim(
        alive, now=claim_time, lease_expires_at=claim_time + timedelta(minutes=1)
    )
    assert claim is not None

    workers = {w.worker_id: w for w in await repository.list_workers(now=claim_time)}

    assert workers[alive.worker_id].expired is False
    assert workers[alive.worker_id].capabilities == ("ingestion",)
    assert workers[alive.worker_id].running == 1
    assert workers[dead.worker_id].expired is True
    assert workers[dead.worker_id].running == 0


async def test_status_counts_separate_in_flight_from_settled(sqlite_engine) -> None:
    workflow_id, version_id = await _seed_workflow(sqlite_engine)
    repository = SqlAlchemyExecutionRepository(sqlite_engine)
    await repository.submit(_submission(workflow_id, version_id, "waiting"))
    cancelled = await repository.submit(_submission(workflow_id, version_id, "stopped"))
    await repository.cancel(cancelled.execution_id)

    counts = await repository.status_counts()

    assert counts.in_flight == 1
    assert counts.of(ExecutionStatus.CANCELLED) == 1
    assert counts.total == 2


async def test_the_claim_asks_postgres_to_skip_rows_another_worker_holds(
    sqlite_engine,
) -> None:
    """`FOR UPDATE ... SKIP LOCKED` is the whole of how two workers share a queue.

    Without `SKIP LOCKED` a second worker blocks on the row the first is
    claiming instead of taking the next one, and a pool of workers degenerates
    into one worker with an audience. Nothing observable here would say so:
    SQLite serialises anyway, and it ignores `FOR UPDATE` entirely, so a claim
    that had lost the clause would still pass every other test in this file.

    So this asserts what PostgreSQL would receive rather than what SQLite does —
    the statements the repository really emits, compiled for the dialect they
    were written for. It fails the moment `skip_locked=True` goes missing.
    """
    from sqlalchemy import event
    from sqlalchemy.dialects import postgresql
    from sqlalchemy.sql import Select

    workflow_id, version_id = await _seed_workflow(sqlite_engine)
    repository = SqlAlchemyExecutionRepository(sqlite_engine)
    await repository.submit(_submission(workflow_id, version_id, "claimable"))
    worker = _worker()
    now = datetime.now(UTC) + timedelta(milliseconds=1)
    await repository.register_worker(worker, expires_at=now + timedelta(minutes=1))

    emitted: list[str] = []

    def _record(conn, clauseelement, multiparams, params, execution_options):  # noqa: ANN001, ANN202
        if isinstance(clauseelement, Select):
            emitted.append(
                str(clauseelement.compile(dialect=postgresql.dialect()))
            )

    event.listen(sqlite_engine.sync_engine, "before_execute", _record)
    try:
        claim = await repository.claim(
            worker, now=now, lease_expires_at=now + timedelta(minutes=1)
        )
    finally:
        event.remove(sqlite_engine.sync_engine, "before_execute", _record)

    assert claim is not None
    locking = [sql for sql in emitted if "FOR UPDATE" in sql]
    assert locking, "the claim took no row lock at all"
    assert all("SKIP LOCKED" in sql for sql in locking), (
        "a claim locked a row without SKIP LOCKED — a second worker would wait "
        "for it instead of claiming the next execution"
    )


# ---------------------------------------------------------------------------
# What a durable execution carries across the gateway → queue → worker boundary
# ---------------------------------------------------------------------------

async def test_the_principal_survives_the_queue_so_a_worker_can_open_the_session(
    sqlite_engine,
) -> None:
    """Ownership has to hold across the boundary, not only inside one process.

    The gateway establishes who is calling; the worker claims the job later, on
    another node, with none of the request left — no headers, no connection, no
    caller. If the principal does not travel with the execution, the session key
    still does, and the locator arrives without the identity. That is the exact
    inversion ownership exists to end, and it fails silently: the work runs,
    just as nobody.

    So the established principal is delegated through the queue, and the worker
    adopts it rather than re-authenticating a request that has long since
    returned.
    """
    workflow_id, version_id = await _seed_workflow(sqlite_engine)
    repository = SqlAlchemyExecutionRepository(sqlite_engine)
    worker = _worker()
    now = datetime.now(UTC)
    await repository.register_worker(worker, expires_at=now + timedelta(minutes=1))

    submitted = replace(
        _trigger(),
        session_key="session-of-alice",
        context=replace(_trigger().context, principal=PrincipalRef("alice")),
    )
    await repository.submit(
        ExecutionSubmission(
            workflow_id=workflow_id,
            workflow_version_id=version_id,
            trigger=submitted,
            idempotency_key="owned-job",
            required_capabilities=(),
        )
    )

    claim_time = datetime.now(UTC) + timedelta(milliseconds=1)
    claim = await repository.claim(
        worker, now=claim_time, lease_expires_at=claim_time + timedelta(seconds=30)
    )
    assert claim is not None
    claimed = claim.execution.trigger

    assert claimed.session_key == "session-of-alice"
    assert claimed.context.principal == PrincipalRef("alice")

    # And the point of carrying it: the worker can now open alice's session,
    # and nobody else can.
    access = SessionAccess(SimpleStoreCoordinatorFactory(), enforced=True)
    access.open("session-of-alice", PrincipalRef("alice"))       # created by the gateway
    assert access.open(claimed.session_key, claimed.context.principal) is not None
    with pytest.raises(SessionAccessDeniedError):
        access.open(claimed.session_key, PrincipalRef("bob"))


async def test_a_job_queued_before_principals_existed_arrives_as_nobody(
    sqlite_engine,
) -> None:
    """The fail direction, pinned.

    Rows written before this field existed have no principal, and inventing one
    for them is the only genuinely dangerous option. `None` means no identity
    was established — which under enforcement is a refusal, not a session opened
    for whoever claimed the job.
    """
    workflow_id, version_id = await _seed_workflow(sqlite_engine)
    repository = SqlAlchemyExecutionRepository(sqlite_engine)
    worker = _worker()
    now = datetime.now(UTC)
    await repository.register_worker(worker, expires_at=now + timedelta(minutes=1))
    await repository.submit(
        ExecutionSubmission(
            workflow_id=workflow_id,
            workflow_version_id=version_id,
            trigger=replace(_trigger(), session_key="legacy-session"),  # no principal
            idempotency_key="legacy-job",
            required_capabilities=(),
        )
    )

    claim_time = datetime.now(UTC) + timedelta(milliseconds=1)
    claim = await repository.claim(
        worker, now=claim_time, lease_expires_at=claim_time + timedelta(seconds=30)
    )
    assert claim is not None
    assert claim.execution.trigger.context.principal is None

    access = SessionAccess(SimpleStoreCoordinatorFactory(), enforced=True)
    with pytest.raises(SessionAccessDeniedError):
        access.open("legacy-session", claim.execution.trigger.context.principal)


async def test_the_queue_does_not_carry_a_workflow_name_for_a_policy_to_believe(
    sqlite_engine,
) -> None:
    """The other half, and it goes the opposite way on purpose.

    The principal comes from the caller and can only be delegated. The workflow
    is something the executor *knows* — it is running it — so a copy stored at
    submission time would be a second answer to a question with an
    authoritative one. For a fan-out child that copy would be the parent's name,
    and a `workflow` condition would then authorize the child against the wrong
    flow.

    The executor stamps `invocation.workflow.name` before any policy reads the
    context, so nothing is lost by not storing it — and something is gained:
    there is no stored value for a policy to believe.
    """
    workflow_id, version_id = await _seed_workflow(sqlite_engine)
    repository = SqlAlchemyExecutionRepository(sqlite_engine)
    worker = _worker()
    now = datetime.now(UTC)
    await repository.register_worker(worker, expires_at=now + timedelta(minutes=1))
    await repository.submit(
        ExecutionSubmission(
            workflow_id=workflow_id,
            workflow_version_id=version_id,
            trigger=replace(
                _trigger(),
                context=replace(_trigger().context, workflow="the-parent-flow"),
            ),
            idempotency_key="stamped-job",
            required_capabilities=(),
        )
    )

    claim_time = datetime.now(UTC) + timedelta(milliseconds=1)
    claim = await repository.claim(
        worker, now=claim_time, lease_expires_at=claim_time + timedelta(seconds=30)
    )
    assert claim is not None
    assert claim.execution.trigger.context.workflow is None
