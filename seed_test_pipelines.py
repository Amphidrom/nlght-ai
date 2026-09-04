#!/usr/bin/env python
"""Seed runnable example ingestion pipelines. Development helper, not application code.

Creates the resource activations and two active workflow versions needed to run
ingestion end to end, so a pipeline can be triggered without clicking eight
steps together in the admin UI.

    uv run python seed_test_pipelines.py --docs-root /path/to/docs

Idempotent: re-running replaces the seeded workflows and resources by name and
leaves everything else in the database alone. Run the migrations first, since
this only inserts rows:

    uv run nlght-ai migrate                # platform database
    uv run nlght-ai migrate-knowledge -x url=...   # knowledge database (separate)

Backend URLs default to localhost and can be overridden by environment
variables (QDRANT_URL, OPENSEARCH_URL, POSTGRES_URL) or the flags below.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from nlght.adapters.outbound.persistence.models import (
    Resource,
    Workflow,
    WorkflowStep,
    WorkflowVersion,
)

DATA_WORKFLOW = "ingest-data"
# Only seeded when the deployment distributes: the work flow one child runs over
# the share of documents its trigger names. Its steps are the serial pipeline's,
# built by the same function — a distributed deployment adds a flow in front of
# the work, it does not duplicate the work.
DATA_WORK_WORKFLOW = "ingest-data-document"
KNOWLEDGE_WORKFLOW = "ingest-knowledge"
KNOWLEDGE_WORK_WORKFLOW = "ingest-knowledge-document"
# Carries review decisions into the search index. Independent of the
# ingestion pipelines: it is triggered when a reviewer has worked a batch,
# not when a document arrives.
KNOWLEDGE_PUBLISH_WORKFLOW = "publish-knowledge"

DATA_STORE = "data-main"
DATA_WRITER = "data-main-writer"
KNOWLEDGE_STORE = "knowledge-main"
KNOWLEDGE_WRITER = "knowledge-main-writer"


_DEFAULT_CONFIG_PATHS = (Path(".config/platform.yaml"), Path(".config/platform.yml"))


def _config() -> dict[str, Any]:
    """Load platform.yaml the way the runtime does.

    NLGHT_CONFIG wins; otherwise .config/platform.yaml, then .yml — the same
    resolution nlght-ai itself uses, so the seed cannot read a different file
    than the runtime it is seeding for.
    """
    override = os.environ.get("NLGHT_CONFIG")
    candidates = (Path(override),) if override else _DEFAULT_CONFIG_PATHS
    for path in candidates:
        if path.exists():
            return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {}


def _configured_url(section: str) -> str | None:
    persistence = ((_config().get("integrations") or {}).get("persistence") or {})
    url = (persistence.get(section) or {}).get("url")
    return str(url) if url else None


def _database_url(explicit: str | None) -> str:
    """Resolve PostgreSQL from the flag, the environment, or platform.yaml."""
    if explicit:
        return explicit
    if os.environ.get("POSTGRES_URL"):
        return os.environ["POSTGRES_URL"]
    url = _configured_url("workflows")
    if url:
        return url
    raise SystemExit(
        "No platform database URL. Pass --database-url, set POSTGRES_URL, or configure "
        "integrations.persistence.workflows.url in .config/platform.yaml."
    )


def _knowledge_url(args: argparse.Namespace) -> str:
    """The knowledge database, which is separate from the platform's.

    Not resolvable from platform.yaml: it has no entry there. Every resource
    that uses the knowledge database carries its own connection, so the seed
    must be told which one to write into these activations (ADR-0033).
    """
    if args.knowledge_url:
        return str(args.knowledge_url)
    if os.environ.get("KNOWLEDGE_URL"):
        return os.environ["KNOWLEDGE_URL"]
    raise SystemExit(
        "No knowledge database URL. Pass --knowledge-url or set KNOWLEDGE_URL. "
        "There is no platform.yaml entry for it: the knowledge database is named "
        "by the resources that use it. It must not be the platform database."
    )


def _step(
    version_id: uuid.UUID,
    position: int,
    name: str,
    type_: str,
    config: dict[str, Any],
    *,
    is_start: bool = False,
    is_terminal: bool = False,
) -> WorkflowStep:
    return WorkflowStep(
        workflow_step_id=uuid.uuid4(),
        workflow_version_id=version_id,
        position=position,
        name=name,
        type=type_,
        enabled=True,
        config=config,
        transitions={},
        is_start=is_start,
        is_terminal=is_terminal,
        is_resume=False,
    )


def _chain(steps: list[WorkflowStep]) -> None:
    """Wire each step to the next on DEFAULT, and route the common side verdicts.

    Ingestion steps report 'empty', 'skipped', 'unchanged', and 'incomplete' as
    ordinary outcomes rather than failures, so each is routed forward too — an
    unchanged source must reach the terminal step, not stall the run.
    """
    side_verdicts = ("empty", "skipped", "unchanged", "incomplete")
    for current, following in zip(steps, steps[1:], strict=False):
        target = str(following.workflow_step_id)
        current.transitions = {
            "DEFAULT": target,
            **{verdict: target for verdict in side_verdicts},
        }


DATA_INCLUDE = ["**/*.md", "**/*.txt", "**/*.py"]
KNOWLEDGE_INCLUDE = ["**/*.md", "**/*.adoc", "**/*.txt"]


def _source_config(docs_root: str, source_id: str, include: list[str]) -> dict[str, Any]:
    """One source definition, shared by the acquiring run and by every child.

    A child acquires its own share from the same source, so both must be
    configured to look at the same place — the trigger, not the config, is what
    narrows a child to its documents.
    """
    return {
        "type": "filesystem",
        "source_id": source_id,
        "roots": [{"path": docs_root, "alias": source_id}],
        "include": list(include),
    }


def _document_steps(version_id: uuid.UUID) -> list[WorkflowStep]:
    """Process, embed, and write the documents this run was given.

    The serial tail of the data pipeline, and also the whole of the child
    workflow — the difference is only how many documents arrive.
    """
    return [
        _step(version_id, 2, "process", "ingestion.process", {
            "chunk_size": 2000, "chunk_overlap": 200,
        }),
        _step(version_id, 3, "embed", "ingestion.embed", {"identity": True}),
        _step(version_id, 4, "write", "ingestion.write", {"writer": DATA_WRITER}),
    ]


def _data_steps(version_id: uuid.UUID, docs_root: str, source_id: str) -> list[WorkflowStep]:
    """Acquire the source and index it.

    The whole data pipeline when a deployment does not distribute, and the work
    flow one child runs when it does — the same steps either way. A child differs
    only in what its trigger hands it: a share instead of the whole source.
    """
    acquire = _step(version_id, 1, "acquire", "ingestion.source",
                    _source_config(docs_root, source_id, DATA_INCLUDE), is_start=True)
    tail = _positioned(_document_steps(version_id), 2)
    done = _step(version_id, 2 + len(tail), "done", "done", {}, is_terminal=True)
    steps = [acquire, *tail, done]
    _chain(steps)
    _loop_on_more(tail, ("process", "embed", "write"))
    return steps


def _data_distribution_steps(
    version_id: uuid.UUID, docs_root: str, source_id: str, documents_per_child: int
) -> list[WorkflowStep]:
    """Acquire the source and hand it out. Three steps, and no work of its own.

    This flow exists only where the deployment is distributed. It never processes
    a document: it observes the source once and submits one child execution per
    share, which is why it has no processing steps to leave unused.
    """
    acquire = _step(version_id, 1, "acquire", "ingestion.source",
                    _source_config(docs_root, source_id, DATA_INCLUDE), is_start=True)
    distribute = _step(version_id, 2, "distribute", "ingestion.fan_out", {
        "workflow": DATA_WORK_WORKFLOW,
        "document_batch_size": documents_per_child,
    })
    done = _step(version_id, 3, "done", "done", {}, is_terminal=True)
    steps = [acquire, distribute, done]
    _chain(steps)
    return steps


def _knowledge_tail(
    version_id: uuid.UUID, source_id: str, provider: str, model: str
) -> list[WorkflowStep]:
    """Everything after acquisition: process, parse, extract, refine, persist.

    The serial tail of the knowledge pipeline, and also the whole of the child
    workflow — the difference is only how many documents arrive.
    """
    return [
        _step(version_id, 0, "process", "ingestion.process", {"chunk_size": 4000}),
        # One dispatching parser instead of a per-format chain: each document is
        # routed to the parser its extension names (md, adoc, html, pdf), and
        # anything else falls back to prose. The include above admits .md/.adoc/
        # .txt, so a markdown-only step would silently drop the AsciiDoc.
        _step(version_id, 0, "parse", "knowledge.parse_auto", {
            "tags": {"product": source_id, "version": "unversioned"},
        }),
        # The writer is what lets extraction record what it has already read, so
        # a second run over an unchanged corpus asks the model nothing at all.
        # Without it every run re-extracts everything and the assertions drift.
        _step(version_id, 0, "extract", "knowledge.extract", {
            "model_provider": provider, "model": model, "min_confidence": 0.3,
            "writer": KNOWLEDGE_WRITER,
        }),
        # Cheap structural passes first: every candidate they drop is one the
        # more expensive scoring steps never have to look at.
        _step(version_id, 0, "canonicalize", "knowledge.canonicalize", {}),
        _step(version_id, 0, "validate", "knowledge.validate", {
            "confidence_threshold_fact": 0.3, "confidence_threshold_rule": 0.5,
        }),
        _step(version_id, 0, "atomicity", "knowledge.atomicity", {}),
        _step(version_id, 0, "identity-merge", "knowledge.identity_merge", {}),
        _step(version_id, 0, "dedup", "knowledge.dedup", {}),
        _step(version_id, 0, "noise-filter", "knowledge.noise_filter", {}),
        _step(version_id, 0, "domain-classify", "knowledge.domain_classify", {}),
        _step(version_id, 0, "quality-score", "knowledge.quality_score", {}),
        _step(version_id, 0, "graph-quality", "knowledge.graph_quality", {}),
        _step(version_id, 0, "meta-enrichment", "knowledge.meta_enrichment", {}),
        # Thresholds rather than `all`: with everything quarantined a seeded
        # corpus is invisible to retrieval until somebody reviews it by hand,
        # which makes the pipeline impossible to try out. Above 0.6 goes
        # straight in; below it waits for a person, which is the interesting
        # half to look at anyway.
        _step(version_id, 0, "review-flag", "knowledge.review_flag", {
            "policy": "thresholds", "min_confidence": 0.6,
        }),
        _step(version_id, 0, "persist", "knowledge.persist", {"writer": KNOWLEDGE_WRITER}),
    ]


def _positioned(steps: list[WorkflowStep], start: int) -> list[WorkflowStep]:
    """Number a list of steps consecutively from ``start``."""
    for offset, step in enumerate(steps):
        step.position = start + offset
    return steps


def _knowledge_steps(
    version_id: uuid.UUID, docs_root: str, source_id: str, provider: str, model: str
) -> list[WorkflowStep]:
    """Acquire the source and refine it into reviewed claims.

    As with data: the whole pipeline when nothing is distributed, and the work
    flow one child runs when it is.
    """
    acquire = _step(version_id, 1, "acquire", "ingestion.source",
                    _source_config(docs_root, source_id, KNOWLEDGE_INCLUDE), is_start=True)
    tail = _positioned(_knowledge_tail(version_id, source_id, provider, model), 2)
    done = _step(version_id, 2 + len(tail), "done", "done", {}, is_terminal=True)
    steps = [acquire, *tail, done]
    _chain(steps)
    _loop_on_more(tail, ("process", "extract"))
    return steps


def _knowledge_distribution_steps(
    version_id: uuid.UUID, docs_root: str, source_id: str, documents_per_child: int
) -> list[WorkflowStep]:
    """Acquire the source and hand it out; the knowledge work happens in children."""
    acquire = _step(version_id, 1, "acquire", "ingestion.source",
                    _source_config(docs_root, source_id, KNOWLEDGE_INCLUDE), is_start=True)
    distribute = _step(version_id, 2, "distribute", "ingestion.fan_out", {
        "workflow": KNOWLEDGE_WORK_WORKFLOW,
        "document_batch_size": documents_per_child,
    })
    done = _step(version_id, 3, "done", "done", {}, is_terminal=True)
    steps = [acquire, distribute, done]
    _chain(steps)
    return steps


def _loop_on_more(steps: list[WorkflowStep], names: tuple[str, ...]) -> None:
    """Route each named step's ``more`` back to itself.

    A batching step reports ``more`` while its documents or units remain; routing
    it back is what finishes the set through ordinary workflow hops.
    """
    for step in steps:
        if step.name in names:
            step.transitions["more"] = str(step.workflow_step_id)


def _resources(qdrant_url: str, opensearch_url: str, knowledge_url: str) -> list[Resource]:
    """Activations for both pipelines.

    The knowledge store and writer carry their own ``pg_url``: the knowledge
    graph is a separate database, and the connection belongs to the activation
    rather than to platform configuration.
    """
    return [
        Resource(
            resource_id=uuid.uuid4(), name=DATA_STORE, kind="data_store",
            provider="qdrant+opensearch", enabled=True,
            config={
                # The two indexes are discovery; the corpus says which revision
                # is published and what it says, so the store reads it directly
                # (ADR-0062).
                "pg_url": knowledge_url,
                "qdrant_url": qdrant_url, "opensearch_url": opensearch_url,
                "collection": "data-main", "index": "data-main",
            },
        ),
        Resource(
            resource_id=uuid.uuid4(), name=DATA_WRITER, kind="data_index_writer",
            provider="qdrant+opensearch", enabled=True,
            config={
                # The writer owns its index state too, so it carries the
                # knowledge database connection alongside its backends.
                "pg_url": knowledge_url,
                "qdrant_url": qdrant_url, "opensearch_url": opensearch_url,
                "collection": "data-main", "index": "data-main",
            },
        ),
        Resource(
            resource_id=uuid.uuid4(), name=KNOWLEDGE_STORE, kind="knowledge_store",
            provider="postgres+opensearch", enabled=True,
            config={"os_url": opensearch_url, "pg_url": knowledge_url},
        ),
        Resource(
            resource_id=uuid.uuid4(), name=KNOWLEDGE_WRITER, kind="knowledge_index_writer",
            provider="postgres+opensearch", enabled=True,
            config={"os_url": opensearch_url, "pg_url": knowledge_url},
        ),
    ]


async def _replace_workflow(
    session: AsyncSession,
    name: str,
    steps_for: Callable[[uuid.UUID], list[WorkflowStep]],
    capabilities: list[str],
    concurrency: str = "non-blocking",
    max_hops: int | None = None,
) -> None:
    """Delete a previously seeded workflow of this name, then recreate it."""
    existing = (
        await session.execute(select(Workflow).where(Workflow.name == name))
    ).scalar_one_or_none()
    if existing is not None:
        await session.delete(existing)
        await session.flush()

    workflow = Workflow(
        workflow_id=uuid.uuid4(),
        name=name,
        description="Seeded example pipeline (seed_test_pipelines.py)",
        enabled=True,
        capabilities=capabilities,
        concurrency=concurrency,
        max_hops=max_hops,
    )
    session.add(workflow)
    await session.flush()

    version = WorkflowVersion(
        workflow_version_id=uuid.uuid4(),
        workflow_id=workflow.workflow_id,
        version=1,
        status="active",
    )
    session.add(version)
    await session.flush()

    for step in steps_for(version.workflow_version_id):
        session.add(step)


async def _delete_workflows(session: AsyncSession, *names: str) -> None:
    """Remove seeded workflows that this configuration does not use."""
    for name in names:
        existing = (
            await session.execute(select(Workflow).where(Workflow.name == name))
        ).scalar_one_or_none()
        if existing is not None:
            await session.delete(existing)
    await session.flush()


def _knowledge_publish_steps(version_id: uuid.UUID) -> list[WorkflowStep]:
    """Make approved knowledge findable and take back what was refused.

    One step and a terminal one. `more` routes back to itself, because a sweep
    works through the graph in bounded batches like every other batching step.
    """
    publish = _step(version_id, 1, "publish", "knowledge.publish",
                    {"writer": KNOWLEDGE_WRITER, "batch_size": 200}, is_start=True)
    done = _step(version_id, 2, "done", "done", {}, is_terminal=True)
    steps = [publish, done]
    _chain(steps)
    _loop_on_more(steps, ("publish",))
    # An empty graph is not a failure; it reaches the terminal step like a sweep
    # that published something.
    publish.transitions = {**publish.transitions, "empty": str(done.workflow_step_id)}
    return steps


async def seed(args: argparse.Namespace) -> None:
    engine = create_async_engine(_database_url(args.database_url))
    docs_root = str(Path(args.docs_root).expanduser().resolve())

    try:
        async with AsyncSession(engine) as session, session.begin():
            names = [DATA_STORE, DATA_WRITER, KNOWLEDGE_STORE, KNOWLEDGE_WRITER]
            await session.execute(delete(Resource).where(Resource.name.in_(names)))
            for resource in _resources(
                args.qdrant_url, args.opensearch_url, _knowledge_url(args)
            ):
                session.add(resource)

            # Both ingestion pipelines are blocking: a reindex must not run
            # concurrently with itself over the same corpus. In a distributed
            # deployment that still holds — the children are siblings of the work
            # flow, so a second corpus run waits for them.
            if args.distributed:
                await _replace_workflow(
                    session,
                    DATA_WORKFLOW,
                    lambda vid: _data_distribution_steps(
                        vid, docs_root, args.source_id, args.documents_per_child
                    ),
                    ["ingestion:data"],
                    concurrency="blocking",
                    max_hops=args.max_hops,
                )
                await _replace_workflow(
                    session,
                    DATA_WORK_WORKFLOW,
                    lambda vid: _data_steps(vid, docs_root, args.source_id),
                    ["ingestion:data"],
                    concurrency="non-blocking",
                    max_hops=args.max_hops,
                )
                await _replace_workflow(
                    session,
                    KNOWLEDGE_WORKFLOW,
                    lambda vid: _knowledge_distribution_steps(
                        vid, docs_root, args.source_id, args.documents_per_child
                    ),
                    ["ingestion:knowledge"],
                    concurrency="blocking",
                    max_hops=args.max_hops,
                )
                await _replace_workflow(
                    session,
                    KNOWLEDGE_WORK_WORKFLOW,
                    lambda vid: _knowledge_steps(
                        vid, docs_root, args.source_id, args.model_provider, args.model
                    ),
                    ["ingestion:knowledge"],
                    concurrency="non-blocking",
                    max_hops=args.max_hops,
                )
            else:
                await _replace_workflow(
                    session,
                    DATA_WORKFLOW,
                    lambda vid: _data_steps(vid, docs_root, args.source_id),
                    ["ingestion:data"],
                    concurrency="blocking",
                    max_hops=args.max_hops,
                )
                await _replace_workflow(
                    session,
                    KNOWLEDGE_WORKFLOW,
                    lambda vid: _knowledge_steps(
                        vid, docs_root, args.source_id, args.model_provider, args.model
                    ),
                    ["ingestion:knowledge"],
                    concurrency="blocking",
                    max_hops=args.max_hops,
                )
                # A work flow left over from a distributed seed would be a
                # pipeline nothing triggers.
                await _delete_workflows(session, DATA_WORK_WORKFLOW, KNOWLEDGE_WORK_WORKFLOW)

            # Publication is orthogonal to how ingestion is spread: it reads
            # PostgreSQL and writes the index, whichever shape produced the
            # assertions. Non-blocking, because it never conflicts with itself —
            # every write it makes is an upsert or a delete by identity.
            await _replace_workflow(
                session,
                KNOWLEDGE_PUBLISH_WORKFLOW,
                _knowledge_publish_steps,
                ["ingestion:knowledge"],
                concurrency="non-blocking",
                    max_hops=args.max_hops,
            )
    finally:
        await engine.dispose()

    print(f"seeded resources: {', '.join(names)}")
    if args.distributed:
        print(f"seeded workflows: {DATA_WORKFLOW} + {DATA_WORK_WORKFLOW}, "
              f"{KNOWLEDGE_WORKFLOW} + {KNOWLEDGE_WORK_WORKFLOW}")
        print(f"architecture:     distributed — {DATA_WORKFLOW} and {KNOWLEDGE_WORKFLOW} acquire "
              f"and hand out {args.documents_per_child} document(s) per child execution; "
              f"the work flows do the work")
    else:
        print(f"seeded workflows: {DATA_WORKFLOW}, {KNOWLEDGE_WORKFLOW}")
        print("architecture:     single flow — each pipeline acquires and works on one worker")
    print(f"max hops:         {'unlimited' if args.max_hops == 0 else args.max_hops} per run")
    print(f"source root:      {docs_root}")
    print(f"knowledge db:     {_knowledge_url(args)}")
    print()
    print("next: start a worker (uv run nlght-ai worker), then trigger")
    print(f"       '{DATA_WORKFLOW}' or '{KNOWLEDGE_WORKFLOW}' like any other workflow.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--docs-root", required=True, help="directory to ingest")
    parser.add_argument("--source-id", default="testdocs", help="stable source identity")
    parser.add_argument(
        "--database-url", default=None, help="platform PostgreSQL URL (async driver)"
    )
    parser.add_argument(
        "--knowledge-url", default=None,
        help="knowledge PostgreSQL URL — a separate database from the platform's",
    )
    parser.add_argument(
        "--qdrant-url", default=os.environ.get("QDRANT_URL", "http://localhost:6333")
    )
    parser.add_argument(
        "--opensearch-url", default=os.environ.get("OPENSEARCH_URL", "http://localhost:9200")
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help=(
            "seed a distributed architecture: each pipeline becomes a distribution flow that "
            "acquires and hands out shares, plus a work flow that any worker claims. Without "
            "it each pipeline is one flow that acquires and works"
        ),
    )
    parser.add_argument(
        "--documents-per-child",
        type=int,
        default=1,
        help="documents handed to one child execution (--distributed only)",
    )
    parser.add_argument(
        "--max-hops",
        type=int,
        default=100_000,
        help=(
            "steps one run of a seeded pipeline may take (0 = unlimited). A batching step "
            "spends one on each document or unit, so even a five-document child needs more "
            "than the runtime default of 20 — and a whole corpus needs thousands"
        ),
    )
    parser.add_argument("--model-provider", default="ollama", help="provider for knowledge.extract")
    parser.add_argument("--model", default="qwen3.8:27B", help="model for knowledge.extract")
    args = parser.parse_args()

    if not Path(args.docs_root).expanduser().exists():
        print(f"docs root does not exist: {args.docs_root}", file=sys.stderr)
        return 2

    asyncio.run(seed(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
