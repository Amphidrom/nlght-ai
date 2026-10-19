# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Encode, decode, and reassemble transport-neutral signal frames.

The assembler retains bounded partial messages until completion or eviction.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass

from nlght.core.signals.signal import Signal

# Reassembly buffers held at once, and fragments per buffer. Both cap what a
# truncated or malicious sender can make a receiver retain. The oldest buffer is
# dropped first, which costs at most one incomplete signal of a live stream.
_MAX_PENDING_MESSAGES = 256
_MAX_FRAGMENTS = 4096

@dataclass(frozen=True)
class StreamFrame:
    """One decoded frame: a signal (possibly a fragment of one) or a close."""

    origin: uuid.UUID
    execution_id: uuid.UUID
    signal: Signal | None
    """The signal, or ``None`` for an end-of-stream frame."""
    message_id: int = 0
    index: int = 0
    count: int = 1

    @property
    def is_close(self) -> bool:
        return self.signal is None

    @property
    def is_fragment(self) -> bool:
        return self.count > 1


def encode_signal(
    origin: uuid.UUID,
    execution_id: uuid.UUID,
    signal: Signal,
    *,
    message_id: int,
    budget: int,
) -> list[str]:
    """Encode a signal as frames that fit the transport byte budget.

    Args:
        origin (uuid.UUID): Identity of the publishing node.
        execution_id (uuid.UUID): Execution that emitted the signal.
        signal (Signal): Signal to encode.
        message_id (int): Positive ID shared by fragments of this signal.
        budget (int): Maximum encoded bytes per frame.

    Returns:
        list[str]: One frame, or ordered fragments for a larger signal.

    Raises:
        ValueError: A fragmented signal has an invalid message ID, cannot fit
            the frame budget, or exceeds the receiver's fragment limit.
    """
    single = _encode(
        {
            "o": origin.hex,
            "e": execution_id.hex,
            "t": "s",
            "r": signal.role,
            "k": signal.kind,
            "c": signal.content,
        }
    )
    if len(single.encode()) <= budget:
        return [single]

    if type(message_id) is not int or message_id < 1:
        raise ValueError("fragmented signals require a positive message ID")
    chunks = _split(signal.content, origin, execution_id, signal, message_id, budget)
    total = len(chunks)
    return [
        _encode(
            {
                "o": origin.hex,
                "e": execution_id.hex,
                "t": "s",
                "r": signal.role,
                "k": signal.kind,
                "c": chunk,
                "m": message_id,
                "i": index,
                "n": total,
            }
        )
        for index, chunk in enumerate(chunks)
    ]


def encode_close(origin: uuid.UUID, execution_id: uuid.UUID) -> str:
    """Encode an end-of-stream frame for one execution.

    Args:
        origin (uuid.UUID): Identity of the publishing node.
        execution_id (uuid.UUID): Execution whose stream ended.

    Returns:
        str: Encoded close frame.
    """
    return _encode({"o": origin.hex, "e": execution_id.hex, "t": "c"})


def decode_frame(payload: str) -> StreamFrame | None:
    """Decode a supported stream frame and reject malformed metadata.

    Args:
        payload (str): JSON frame received from the transport.

    Returns:
        StreamFrame | None: Decoded frame, or None for unreadable and unsupported input.
    """
    try:
        raw = json.loads(payload)
    except (TypeError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    try:
        origin = uuid.UUID(hex=str(raw["o"]))
        execution_id = uuid.UUID(hex=str(raw["e"]))
        kind = str(raw["t"])
    except (KeyError, ValueError, AttributeError):
        return None

    if kind == "c":
        return StreamFrame(origin=origin, execution_id=execution_id, signal=None)
    if kind != "s":
        return None
    try:
        signal = Signal(
            role=str(raw["r"]),
            content=str(raw["c"]),
            kind=str(raw["k"]),
        )
    except KeyError:
        return None
    message_id = raw.get("m", 0)
    index = raw.get("i", 0)
    count = raw.get("n", 1)
    if any(type(value) is not int for value in (message_id, index, count)):
        return None
    if (
        count < 1
        or count > _MAX_FRAGMENTS
        or index < 0
        or index >= count
        or (count > 1 and message_id < 1)
        or (count == 1 and message_id < 0)
    ):
        return None
    return StreamFrame(
        origin=origin,
        execution_id=execution_id,
        signal=signal,
        message_id=message_id,
        index=index,
        count=count,
    )


class StreamFrameAssembler:
    """Rejoins fragmented signals. Passes whole ones straight through.

    Kept separate from the transport so every adapter reassembles identically.
    Buffers are bounded (``_MAX_PENDING_MESSAGES``) and evicted oldest-first, so a
    sender that dies mid-signal cannot grow a receiver's memory.
    """

    def __init__(self) -> None:
        self._pending: dict[tuple[str, str, int], dict[int, str]] = {}

    def push(self, frame: StreamFrame) -> StreamFrame | None:
        """Accept a frame and reassemble a complete signal when possible.

        Args:
            frame (StreamFrame): Decoded signal fragment, whole signal, or close frame.

        Returns:
            StreamFrame | None: Complete frame when deliverable, otherwise None.

        Side Effects:
            May add or evict a bounded partial-message buffer.
        """
        if not frame.is_fragment or frame.signal is None:
            return frame

        key = (frame.origin.hex, frame.execution_id.hex, frame.message_id)
        parts = self._pending.get(key)
        if parts is None:
            if len(self._pending) >= _MAX_PENDING_MESSAGES:
                self._pending.pop(next(iter(self._pending)), None)
            parts = {}
            self._pending[key] = parts
        if len(parts) >= _MAX_FRAGMENTS:
            self._pending.pop(key, None)
            return None
        parts[frame.index] = frame.signal.content

        if len(parts) < frame.count:
            return None
        self._pending.pop(key, None)
        content = "".join(parts[index] for index in sorted(parts))
        return StreamFrame(
            origin=frame.origin,
            execution_id=frame.execution_id,
            signal=Signal(role=frame.signal.role, content=content, kind=frame.signal.kind),
        )

    def forget(self, execution_id: uuid.UUID) -> None:
        """Discard partial messages for a completed execution.

        Args:
            execution_id (uuid.UUID): Execution whose pending fragments are removed.

        Side Effects:
            Removes matching reassembly buffers.
        """
        for key in [key for key in self._pending if key[1] == execution_id.hex]:
            del self._pending[key]


def _split(
    content: str,
    origin: uuid.UUID,
    execution_id: uuid.UUID,
    signal: Signal,
    message_id: int,
    budget: int,
) -> list[str]:
    """Split signal text by actual encoded byte size.

    Args:
        content (str): Signal text to fragment.
        origin (uuid.UUID): Identity of the publishing node.
        execution_id (uuid.UUID): Execution that emitted the signal.
        signal (Signal): Signal whose role and kind enter each frame.
        message_id (int): Positive ID shared by the fragments.
        budget (int): Maximum encoded bytes per frame.

    Returns:
        list[str]: Ordered content slices that fit in encoded frames.

    Raises:
        ValueError: The budget is too small or too many fragments are needed.
    """
    overhead = len(
        _encode(
            {
                "o": origin.hex,
                "e": execution_id.hex,
                "t": "s",
                "r": signal.role,
                "k": signal.kind,
                "c": "",
                "m": message_id,
                "i": _MAX_FRAGMENTS - 1,
                "n": _MAX_FRAGMENTS,
            }
        ).encode()
    )
    room = budget - overhead
    if room < 1:
        raise ValueError("frame budget cannot hold a fragment header")

    chunks: list[str] = []
    rest = content
    while rest:
        if _encoded_size(rest[0]) > room:
            raise ValueError("frame budget cannot hold one encoded character")
        # `room` characters can never encode to fewer than `room` bytes, so it is a
        # safe starting guess. Shrink until the encoded slice fits.
        take = min(len(rest), room)
        while take > 1 and _encoded_size(rest[:take]) > room:
            # Scale by how far over budget we are, and always make progress.
            scaled = take * room // _encoded_size(rest[:take])
            take = max(1, min(take - 1, scaled))
        chunks.append(rest[:take])
        if len(chunks) > _MAX_FRAGMENTS:
            raise ValueError("signal needs more fragments than the receiver accepts")
        rest = rest[take:]
    return chunks


def _encoded_size(text: str) -> int:
    """Measure text inside a JSON-encoded frame.

    Args:
        text (str): Content to measure.

    Returns:
        int: Encoded byte count excluding the surrounding JSON quotes.
    """
    return len(_encode(text).encode()) - 2


def _encode(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)
