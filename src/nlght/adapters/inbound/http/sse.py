# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any

from fastapi.responses import StreamingResponse


def sse_response(generator: AsyncGenerator[str, None]) -> StreamingResponse:
    """Wrap generated text events in a streaming HTTP response.

    Args:
        generator (AsyncGenerator[str, None]): Async source of SSE frames.

    Returns:
        StreamingResponse: Response with SSE media type and no buffering.
    """
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
    """Encode one JSON value as an SSE event frame.

    Args:
        data (object): JSON-serializable event payload.
        event (str | None): Optional SSE event name.

    Returns:
        str: Frame with optional event line and JSON data line.

    Raises:
        TypeError: The payload cannot be serialized as JSON.
    """
    lines = []
    if event:
        lines.append(f"event: {event}")
    lines.append(f"data: {json.dumps(data)}")
    return "\n".join(lines) + "\n\n"


async def invocation_stream(payload: dict[str, Any]) -> AsyncGenerator[str, None]:
    """Yield an invocation payload followed by the terminal SSE marker.

    Args:
        payload (dict[str, Any]): JSON-serializable invocation payload.

    Yields:
        str: One data event, then the ``[DONE]`` marker.
    """
    yield await sse_event(payload)
    yield "data: [DONE]\n\n"
