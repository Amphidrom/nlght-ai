# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from nlght.adapters.inbound.http.response import serialize_invocation
from nlght.core.errors.errors import AIRuntimeError, WorkflowNotFoundError

if TYPE_CHECKING:
    from nlght.ports.outbound.session_key_resolver import SessionKeyResolver


class GenericJsonHttpProtocolAdapter:
    """Catch-all route for generic JSON requests. Register last to avoid shadowing."""

    def __init__(self, session_key_resolver: SessionKeyResolver | None = None) -> None:
        self._session_key_resolver = session_key_resolver

    def build_router(self) -> APIRouter:
        router = APIRouter()

        @router.api_route(
            "/{full_path:path}",
            methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        )
        async def generic_ingress(full_path: str, request: Request) -> JSONResponse:
            _ = full_path
            container = request.app.state.container
            try:
                raw_body = await request.body()
                session_key = await self._session_key_resolver.resolve(
                    path=request.url.path,
                    method=request.method,
                    headers={k.lower(): v for k, v in request.headers.items()},
                    query_params=dict(request.query_params),
                    raw_body=raw_body,
                ) if self._session_key_resolver else None
                invocation = await container.gateway_service.process(
                    path=request.url.path,
                    method=request.method,
                    headers=dict(request.headers),
                    query_params=dict(request.query_params),
                    raw_body=raw_body,
                    client_host=request.client.host if request.client else None,
                    session_key=session_key,
                )
                return JSONResponse(serialize_invocation(invocation))
            except WorkflowNotFoundError as exc:
                raise HTTPException(status_code=404, detail=str(exc)) from exc
            except AIRuntimeError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

        return router
