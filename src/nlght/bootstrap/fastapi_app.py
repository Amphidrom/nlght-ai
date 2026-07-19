# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from sqlalchemy import text

from nlght.bootstrap.wiring import build_container

_DEFAULT_CONFIG_PATH = ".config/platform.yaml"
logger = logging.getLogger(__name__)


def create_app(
    config_path: str | None = None,
) -> FastAPI:
    """Create the FastAPI application.

    Protocol adapters are wired up inside the lifespan after the container is built,
    so that route registration happens in the same async context as subsystem startup.
    """
    _config_path = config_path or os.environ.get("NLGHT_CONFIG", _DEFAULT_CONFIG_PATH)
    admin_enabled = os.environ.get("NLGHT_ADMIN", "").lower() in ("1", "true")

    @asynccontextmanager
    async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
        container = await build_container(config_path=_config_path)
        app.state.container = container
        for adapter in container.http_protocol_adapters:
            app.include_router(adapter.build_router())
            logger.info("Registered protocol adapter: %s", type(adapter).__name__)
        if admin_enabled:
            from nlght.adapters.inbound.http.admin.router import (  # noqa: PLC0415 (lazy import: optional extra or deliberate startup-cost/cycle avoidance)
                admin_router,  # noqa: PLC0415 (lazy import: optional extra or deliberate startup-cost/cycle avoidance)
            )
            app.include_router(admin_router, prefix="/admin")
            logger.info("Admin UI mounted at /admin")
        metering = getattr(container, "metering", None)
        if metering is not None and hasattr(metering, "make_asgi_app"):
            app.mount(metering.endpoint, metering.make_asgi_app())
            logger.info("Metrics scrape endpoint mounted at %s", metering.endpoint)
        try:
            yield
        finally:
            for adapter in getattr(container, "inbound_adapters", []):
                await adapter.stop()
            for subsystem in reversed(getattr(container, "subsystems", [])):
                await subsystem.stop()

    app = FastAPI(
        title="nlght Runtime Platform",
        version="0.1.0",
        lifespan=_lifespan,
    )

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz(response: Response) -> dict[str, str]:
        """Reports whether the app can actually serve traffic, not just that the process is up.

        Only checks the database — the one dependency required for every
        request path. Model providers are deliberately not pinged here: a
        readiness probe that calls out to external LLM APIs would make pod
        health depend on third-party latency/availability instead of this
        process's own state.
        """
        engine = getattr(app.state.container, "engine", None)
        if engine is None:
            return {"status": "ok", "database": "not configured"}
        try:
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
        except Exception:
            response.status_code = 503
            return {"status": "unavailable", "database": "unreachable"}
        return {"status": "ok", "database": "ok"}

    return app
