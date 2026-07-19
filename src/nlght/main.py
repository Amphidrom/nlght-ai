# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import logging
import os
import subprocess
import sys
from importlib.resources import files
from typing import TYPE_CHECKING, Any

import uvicorn
import yaml

from nlght.shared.logging import setup_logging

if TYPE_CHECKING:
    from fastapi import FastAPI

_DEFAULT_CONFIG_PATH = ".config/platform.yaml"


def _read_raw(config_path: str) -> dict[str, Any]:
    """Read the platform YAML synchronously — no event loop needed."""
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
    """RLS-009: the Admin UI has no authentication — warn loudly when it is
    enabled on a binding that is reachable beyond loopback."""
    admin_enabled = os.environ.get("NLGHT_ADMIN", "").lower() in ("1", "true")
    if admin_enabled and host not in _LOOPBACK_HOSTS:
        logging.getLogger(__name__).warning(
            "NLGHT_ADMIN is enabled on a non-loopback binding (%s). The Admin "
            "UI has NO authentication — anyone who can reach this port can "
            "modify workflows, resources, and access policies. Restrict the "
            "binding to 127.0.0.1 or put an authenticating proxy in front.",
            host,
        )


def serve() -> None:
    config_path = os.environ.get("NLGHT_CONFIG", _DEFAULT_CONFIG_PATH)
    raw = _read_raw(config_path)

    log_levels: dict[str, str] = (raw.get("logging") or {}).get("level") or {}
    setup_logging(level_overrides=log_levels, app_name="nlght")

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


def create_app() -> FastAPI:
    # Lazy: `nlght-ai migrate` must not pay the FastAPI/container import cost.
    from nlght.bootstrap.fastapi_app import create_app as _create_app  # noqa: PLC0415

    config_path = os.environ.get("NLGHT_CONFIG", _DEFAULT_CONFIG_PATH)
    return _create_app(config_path=config_path)


def migrate(argv: list[str] | None = None) -> None:
    """Run database migrations: nlght-ai migrate [alembic-command [args...]]

    Resolves NLGHT_CONFIG (or the default path) before invoking alembic,
    so the migration environment picks up the correct platform.yaml without
    manual environment setup.

    Defaults to ``upgrade head`` when called with no arguments:

        nlght-ai migrate                   # → alembic upgrade head
        nlght-ai migrate downgrade -1      # → alembic downgrade -1
        nlght-ai migrate current           # → alembic current
    """
    config_path = os.environ.get("NLGHT_CONFIG", _DEFAULT_CONFIG_PATH)
    os.environ["NLGHT_CONFIG"] = config_path

    # alembic.ini aus dem installierten Package laden
    ini_path = files("nlght").joinpath("alembic.ini")

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
  migrate [args...]   run database migrations (defaults to 'upgrade head')
"""


def cli() -> None:
    """CLI entry point: nlght-ai <command> [args...]"""
    args = sys.argv[1:]
    if not args:
        print(_USAGE, end="", file=sys.stderr)
        raise SystemExit(2)

    command, rest = args[0], args[1:]
    if command == "serve":
        serve()
    elif command == "migrate":
        migrate(rest)
    else:
        print(f"nlght-ai: unknown command '{command}'\n{_USAGE}", end="", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    serve()
