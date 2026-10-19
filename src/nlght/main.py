# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Any

import uvicorn
import yaml

from nlght.shared.logging import setup_logging

if TYPE_CHECKING:
    from fastapi import FastAPI

_DEFAULT_CONFIG_PATH = ".config/platform.yaml"


def _read_raw(config_path: str) -> dict[str, Any]:
    """Read platform configuration without starting an event loop.

    Args:
        config_path (str): Path to the platform YAML file.

    Returns:
        dict[str, Any]: Parsed configuration, or an empty mapping if absent.

    Side Effects:
        Reads the configuration file when it exists.
    """
    try:
        with open(config_path, encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except FileNotFoundError:
        return {}


def _gateway_binding(raw: dict[str, Any]) -> tuple[str, int]:
    for gw in raw.get("gateways", []):
        if gw.get("kind") == "http" and gw.get("enabled", True):
            cfg = gw.get("config") or {}
            return str(cfg.get("host", "0.0.0.0")), int(cfg.get("port", 8000))
    return "0.0.0.0", 8000


_LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")


def _warn_if_admin_on_public_binding(host: str) -> None:
    """Warn when the unauthenticated Admin UI is exposed beyond loopback.

    Args:
        host (str): Host address bound by the HTTP gateway.

    Side Effects:
        Emits a warning when NLGHT_ADMIN is enabled on a public binding.
    """
    admin_enabled = os.environ.get("NLGHT_ADMIN", "").lower() in ("1", "true")
    if admin_enabled and host not in _LOOPBACK_HOSTS:
        logging.getLogger(__name__).warning(
            "NLGHT_ADMIN is enabled on a non-loopback binding (%s). The Admin "
            "UI has NO authentication. Anyone who can reach this port can "
            "modify workflows, resources, and access policies. Restrict the "
            "binding to 127.0.0.1 or put an authenticating proxy in front.",
            host,
        )


def serve() -> None:
    config_path = os.environ.get("NLGHT_CONFIG", _DEFAULT_CONFIG_PATH)
    raw = _read_raw(config_path)

    log_levels: dict[str, str] = (raw.get("logging") or {}).get("level") or {}
    setup_logging(level_overrides=log_levels, app_name="nlght")

    if str((raw.get("execution") or {}).get("role", "gateway+worker")) == "worker":
        raise RuntimeError(
            "execution.role=worker does not expose a gateway; start it with 'nlght-ai worker'"
        )

    host, port = _gateway_binding(raw)
    reload = os.environ.get("NLGHT_RELOAD", "").lower() in ("1", "true")

    _warn_if_admin_on_public_binding(host)

    # Pin the resolved path so every worker/reload cycle picks up the same file.
    os.environ["NLGHT_CONFIG"] = config_path

    uvicorn.run(
        "nlght.main:create_app",
        host=host,
        port=port,
        reload=reload,
        factory=True,
    )


async def _run_worker(config_path: str) -> None:
    from nlght.bootstrap.wiring import build_container  # noqa: PLC0415

    container = await build_container(config_path=config_path)
    if container.execution_worker is None:
        for subsystem in reversed(container.subsystems):
            await subsystem.stop()
        raise RuntimeError(
            "worker command requires execution.role=worker or execution.role=gateway+worker "
            "and configured workflow persistence"
        )

    stopped = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, stopped.set)
        except NotImplementedError:  # pragma: no cover - Windows event-loop policy
            pass

    try:
        await stopped.wait()
    finally:
        for adapter in container.inbound_adapters:
            await adapter.stop()
        for subsystem in reversed(container.subsystems):
            await subsystem.stop()


def worker() -> None:
    config_path = os.environ.get("NLGHT_CONFIG", _DEFAULT_CONFIG_PATH)
    raw = _read_raw(config_path)
    log_levels: dict[str, str] = (raw.get("logging") or {}).get("level") or {}
    setup_logging(level_overrides=log_levels, app_name="nlght-worker")
    asyncio.run(_run_worker(config_path))


def create_app() -> FastAPI:
    # Lazy: `nlght-ai migrate` must not pay the FastAPI/container import cost.
    from nlght.bootstrap.fastapi_app import create_app as _create_app  # noqa: PLC0415

    config_path = os.environ.get("NLGHT_CONFIG", _DEFAULT_CONFIG_PATH)
    return _create_app(config_path=config_path)


def migrate(argv: list[str] | None = None) -> None:
    """Run platform database migrations with Alembic.

    Args:
        argv (list[str] | None): Alembic arguments. Defaults to ``upgrade head``.

    Side Effects:
        Sets NLGHT_CONFIG, runs Alembic against the platform migration chain,
        and exits with the subprocess status.

    Examples:
        ``nlght-ai migrate downgrade -1`` rolls back one revision.
        ``nlght-ai migrate current`` prints the current revision.
    """
    config_path = os.environ.get("NLGHT_CONFIG", _DEFAULT_CONFIG_PATH)
    os.environ["NLGHT_CONFIG"] = config_path

    # Load alembic.ini from the installed package.
    ini_path = files("nlght").joinpath("alembic.ini")

    args = list(argv) if argv else ["upgrade", "head"]

    result = subprocess.run([
        sys.executable, "-m", "alembic",
        "-c", str(ini_path),
        *args
    ])
    sys.exit(result.returncode)


def migrate_knowledge(argv: list[str] | None = None) -> None:
    """Run the knowledge database's separate Alembic migration chain.

    Args:
        argv (list[str] | None): Alembic arguments. Defaults to ``upgrade head``.

    Side Effects:
        Sets NLGHT_CONFIG, runs Alembic against the knowledge database, and
        exits with the subprocess status. Platform database revisions are
        unaffected.

    Examples:
        ``nlght-ai migrate-knowledge downgrade -1`` rolls back one revision.
        ``nlght-ai migrate-knowledge current`` prints the current revision.
    """
    config_path = os.environ.get("NLGHT_CONFIG", _DEFAULT_CONFIG_PATH)
    os.environ["NLGHT_CONFIG"] = config_path

    ini_path = files("nlght").joinpath("knowledge_alembic.ini")
    args = list(argv) if argv else ["upgrade", "head"]

    result = subprocess.run([
        sys.executable, "-m", "alembic",
        "-c", str(ini_path),
        *args
    ])
    sys.exit(result.returncode)


_USAGE = """usage: nlght-ai <command> [args...]

commands:
  serve               start the runtime platform
  worker              start only the distributed execution worker
  migrate [args...]   run platform database migrations (defaults to 'upgrade head')
  migrate-knowledge [args...]
                      run knowledge graph database migrations (separate database)
  knowledge-report -x url=... [--baseline FILE] [--compare FILE] [--expect FILE]
                      a picture of the corpus, for holding a rerun against the
                      run before it; exits non-zero when the corpus moved
"""


def _read_expectations(path: str) -> dict[str, object]:
    """Read a corpus expectation file as YAML or JSON.

    Args:
        path (str): Path to the expectation file.

    Returns:
        dict[str, object]: Parsed expectation mapping.

    Side Effects:
        Reads the file from disk.
    """

    text = Path(path).read_text(encoding="utf-8")
    if path.endswith((".yaml", ".yml")):
        return dict(yaml.safe_load(text) or {})
    return dict(json.loads(text))


async def _run_knowledge_report(
    url: str,
    baseline: str | None,
    compare_to: str | None,
    expect: str | None = None,
) -> int:
    from sqlalchemy.ext.asyncio import create_async_engine  # noqa: PLC0415

    from nlght.adapters.outbound.persistence.knowledge_repository import (  # noqa: PLC0415
        SqlAlchemyKnowledgeRepository,
    )
    from nlght.core.knowledge import (  # noqa: PLC0415
        LineageReport,
        check_expectations,
        compare,
        parse_expectations,
    )

    engine = create_async_engine(url)
    try:
        report = await SqlAlchemyKnowledgeRepository(engine).lineage_report()
    finally:
        await engine.dispose()

    print(json.dumps(report.as_summary(), indent=2, sort_keys=True))

    if baseline:
        Path(baseline).write_text(
            json.dumps(report.as_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"\nbaseline written to {baseline}", file=sys.stderr)

    if compare_to:
        previous = LineageReport.from_dict(
            json.loads(Path(compare_to).read_text(encoding="utf-8"))
        )
        result = compare(previous, report)
        print(f"\n{result}", file=sys.stderr)

        if expect:
            # What the corpus said this run should do, held against what it did.
            # A comparison says what moved. Only this says whether that was the
            # movement anybody wanted. "Two revisions" is as true of a threshold
            # moving to 60 as of one moving to 6000.
            outcome = check_expectations(
                previous, report, result, parse_expectations(_read_expectations(expect))
            )
            print(f"\n{outcome}", file=sys.stderr)
            if not outcome.met:
                return 1
        # The exit code is the answer: a rerun that moved the corpus fails, so
        # this can sit in a script without anybody having to read the output.
        return 0 if result.unchanged else 1
    return 0


def knowledge_report(argv: list[str] | None = None) -> None:
    """Report corpus state and compare it with a saved baseline.

    Args:
        argv (list[str] | None): ``-x url=...`` and optional ``--baseline``,
            ``--compare``, or ``--expect`` arguments.

    Raises:
        SystemExit: Always exits with the report status. Invalid arguments,
            unmet expectations, or a changed corpus produce a nonzero status.

    Side Effects:
        Reads the knowledge database. May write a baseline file. Prints the
        report and comparison.

    Examples:
        ``nlght-ai knowledge-report -x url=... --baseline before.json``
        ``nlght-ai knowledge-report -x url=... --compare before.json``
    """
    args = list(argv or [])
    url = ""
    baseline: str | None = None
    compare_to: str | None = None
    expect: str | None = None
    index = 0
    while index < len(args):
        item = args[index]
        if item == "-x" and index + 1 < len(args) and args[index + 1].startswith("url="):
            url = args[index + 1][4:]
            index += 2
            continue
        if item == "--baseline" and index + 1 < len(args):
            baseline = args[index + 1]
            index += 2
            continue
        if item == "--compare" and index + 1 < len(args):
            compare_to = args[index + 1]
            index += 2
            continue
        if item == "--expect" and index + 1 < len(args):
            expect = args[index + 1]
            index += 2
            continue
        raise SystemExit(f"knowledge-report: unexpected argument '{item}'")

    if expect and not compare_to:
        # An expectation is about a *change*, so it needs both pictures. Checking
        # one against nothing would report every case as unevaluable, which reads
        # like a broken pipeline rather than a missing argument.
        raise SystemExit("knowledge-report: --expect needs --compare BASELINE")

    if not url:
        raise SystemExit(
            "knowledge-report needs the knowledge database: -x url=postgresql+asyncpg://..."
        )
    raise SystemExit(asyncio.run(_run_knowledge_report(url, baseline, compare_to, expect)))


def cli() -> None:
    """Dispatch the ``nlght-ai`` command from process arguments.

    Side Effects:
        Runs the selected command or prints usage and exits for invalid input.
    """
    args = sys.argv[1:]
    if not args:
        print(_USAGE, end="", file=sys.stderr)
        raise SystemExit(2)

    command, rest = args[0], args[1:]
    if command == "serve":
        serve()
    elif command == "worker":
        worker()
    elif command == "migrate":
        migrate(rest)
    elif command == "migrate-knowledge":
        migrate_knowledge(rest)
    elif command == "knowledge-report":
        knowledge_report(rest)
    else:
        print(f"nlght-ai: unknown command '{command}'\n{_USAGE}", end="", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    serve()
