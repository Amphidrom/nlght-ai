# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import json

from fastapi.responses import StreamingResponse

from nlght.adapters.inbound.http.sse import invocation_stream, sse_event, sse_response


async def test_sse_event_formats_data_only_frame() -> None:
    frame = await sse_event({"ok": True})

    assert frame == f"data: {json.dumps({'ok': True})}\n\n"


async def test_sse_event_includes_event_name() -> None:
    frame = await sse_event({"token": "hi"}, event="message")

    assert frame == 'event: message\ndata: {"token": "hi"}\n\n'


def test_sse_response_sets_streaming_headers() -> None:
    async def _gen():
        yield "data: one\n\n"

    response = sse_response(_gen())

    assert isinstance(response, StreamingResponse)
    assert response.media_type == "text/event-stream"
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["x-accel-buffering"] == "no"
    assert response.headers["connection"] == "keep-alive"


async def test_invocation_stream_yields_payload_then_done() -> None:
    chunks = [chunk async for chunk in invocation_stream({"id": "inv-1"})]

    assert chunks == ['data: {"id": "inv-1"}\n\n', "data: [DONE]\n\n"]
