# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Wire format for an execution stream carried between nodes.

Transport-neutral on purpose: it turns a ``Signal`` into self-describing text
frames and back, and knows nothing about PostgreSQL, Redis, or any other
carrier. A transport adapter supplies only the byte budget of one message and
the delivery mechanism; this module owns framing, fragmentation, and reassembly,
so a second transport reuses it instead of reinventing it.

A frame is a small JSON object:

``o``  origin — the publishing broker's node id, so a transport that echoes a
       message back to its sender can drop its own frames.
``e``  execution id the frame belongs to.
``t``  ``"s"`` a signal, or ``"c"`` end-of-stream for that execution.
``r``  signal role, ``k`` signal kind, ``c`` signal content (signal frames only).
``m``  message id, ``i`` fragment index, ``n`` fragment count — present only when
       one signal did not fit in a single message and was split.

Fragmentation exists because a carrier bounds one message (PostgreSQL caps a
``NOTIFY`` payload at 8000 bytes). Content is split so that each *encoded* frame
fits the budget; the receiver rejoins the pieces in index order. Fragments of one
signal are published back-to-back on a single connection, so they arrive in
order; a reassembly that never completes is bounded and discarded (see
``StreamFrameAssembler``).
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass

from nlght.core.signals.signal import Signal

# Reassembly buffers held at once, and fragments per buffer. Both cap what a
# truncated or malicious sender can make a receiver retain; the oldest buffer is
# dropped first, which costs at most one incomplete signal of a live stream.
_MAX_PENDING_MESSAGES = 256
_MAX_FRAGMENTS = 4096

# Smallest content slice we will cut. Guards the splitting loop against pathological
# escaping (a character can encode to six bytes) demanding an empty slice.
_MIN_FRAGMENT_CHARS = 8


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
    """Encode one signal as the frames needed to carry it within ``budget`` bytes.

    Returns a single frame whenever the signal fits, which is the ordinary case —
    a token delta is far below any transport's message limit.
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
    """Encode the end-of-stream frame for one execution."""
    return _encode({"o": origin.hex, "e": execution_id.hex, "t": "c"})


def decode_frame(payload: str) -> StreamFrame | None:
    """Decode one frame, or ``None`` if it is not one we understand.

    A transport carries whatever anyone put on it; an unreadable frame is
    dropped rather than allowed to break a live stream.
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
    return StreamFrame(
        origin=origin,
        execution_id=execution_id,
        signal=signal,
        message_id=int(raw.get("m", 0)),
        index=int(raw.get("i", 0)),
        count=int(raw.get("n", 1)),
    )


class StreamFrameAssembler:
    """Rejoins fragmented signals; passes whole ones straight through.

    Kept separate from the transport so every adapter reassembles identically.
    Buffers are bounded (``_MAX_PENDING_MESSAGES``) and evicted oldest-first, so a
    sender that dies mid-signal cannot grow a receiver's memory.
    """

    def __init__(self) -> None:
        self._pending: dict[tuple[str, str, int], dict[int, str]] = {}

    def push(self, frame: StreamFrame) -> StreamFrame | None:
        """Feed one frame; return the complete frame once it can be delivered."""
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
        """Drop any half-assembled signal of a stream that has ended."""
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
    """Cut ``content`` so every resulting *encoded* frame fits ``budget``.

    Measures the real encoding rather than assuming a ratio: JSON escaping
    expands some characters up to sixfold, and a fixed estimate would either
    overflow the budget or waste most of it.
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
                "i": 0,
                "n": 999_999,
            }
        ).encode()
    )
    room = max(budget - overhead, _MIN_FRAGMENT_CHARS)

    chunks: list[str] = []
    rest = content
    while rest:
        # `room` characters can never encode to fewer than `room` bytes, so it is a
        # safe starting guess; shrink until the encoded slice fits.
        take = min(len(rest), room)
        while take > 1 and _encoded_size(rest[:take]) > room:
            # Scale by how far over budget we are, and always make progress.
            scaled = take * room // _encoded_size(rest[:take])
            take = max(1, min(take - 1, scaled))
        chunks.append(rest[:take])
        rest = rest[take:]
    return chunks


def _encoded_size(text: str) -> int:
    """Bytes ``text`` occupies inside a frame — its JSON encoding without the quotes."""
    return len(_encode(text).encode()) - 2


def _encode(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)
