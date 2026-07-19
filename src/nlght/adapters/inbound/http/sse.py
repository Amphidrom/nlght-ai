# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any

from fastapi.responses import StreamingResponse


def sse_response(generator: AsyncGenerator[str, None]) -> StreamingResponse:
    """Wrap an async string generator into a proper SSE StreamingResponse."""
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",       # disable nginx buffering
            "Connection": "keep-alive",
        },
    )


async def sse_event(data: object, event: str | None = None) -> str:
    """Format a single SSE event frame."""
    lines = []
    if event:
        lines.append(f"event: {event}")
    lines.append(f"data: {json.dumps(data)}")
    return "\n".join(lines) + "\n\n"


async def invocation_stream(payload: dict[str, Any]) -> AsyncGenerator[str, None]:
    """Placeholder streaming generator.

    Yields the resolved invocation as a single SSE data event followed by
    the OpenAI-style [DONE] sentinel.  Replace this generator with a real
    workflow-executor stream once the execution layer is implemented.
    """
    yield await sse_event(payload)
    yield "data: [DONE]\n\n"
