# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import uuid

from nlght.adapters.outbound.signals.execution_stream_wire import (
    StreamFrameAssembler,
    decode_frame,
    encode_close,
    encode_signal,
)
from nlght.core.signals.signal import Signal

_BUDGET = 7800


def _sig(content: str, kind: str = "token") -> Signal:
    return Signal(role="assistant", content=content, kind=kind)


def _roundtrip(signal: Signal, budget: int = _BUDGET) -> Signal | None:
    """Encode, decode, and reassemble a signal the way a receiving node does."""
    origin, execution_id = uuid.uuid4(), uuid.uuid4()
    assembler = StreamFrameAssembler()
    complete = None
    for payload in encode_signal(
        origin, execution_id, signal, message_id=7, budget=budget
    ):
        assert len(payload.encode()) <= budget
        frame = decode_frame(payload)
        assert frame is not None
        complete = assembler.push(frame)
    return None if complete is None else complete.signal


def test_a_small_signal_travels_as_one_frame() -> None:
    frames = encode_signal(
        uuid.uuid4(), uuid.uuid4(), _sig("hello"), message_id=1, budget=_BUDGET
    )

    assert len(frames) == 1
    assert _roundtrip(_sig("hello")) == _sig("hello")


def test_frames_carry_origin_and_execution() -> None:
    origin, execution_id = uuid.uuid4(), uuid.uuid4()

    frame = decode_frame(
        encode_signal(origin, execution_id, _sig("x"), message_id=1, budget=_BUDGET)[0]
    )

    assert frame is not None
    assert frame.origin == origin
    assert frame.execution_id == execution_id
    assert frame.is_close is False


def test_a_close_frame_decodes_as_end_of_stream() -> None:
    origin, execution_id = uuid.uuid4(), uuid.uuid4()

    frame = decode_frame(encode_close(origin, execution_id))

    assert frame is not None
    assert frame.is_close is True
    assert frame.execution_id == execution_id


def test_an_oversized_signal_is_fragmented_and_rejoined() -> None:
    content = "".join(f"chunk-{index:05d} " for index in range(4000))

    assert len(content.encode()) > _BUDGET
    assert _roundtrip(_sig(content, kind="result")) == _sig(content, kind="result")


def test_fragments_respect_the_budget_even_when_escaping_expands_content() -> None:
    # A control character escapes to six bytes (\u0007) on the wire — the worst
    # case, which a naive character-count split would overflow.
    content = "\u0007" * 4000

    assert _roundtrip(_sig(content)) == _sig(content)


def test_a_tiny_budget_still_produces_deliverable_frames() -> None:
    assert _roundtrip(_sig("abcdefghijklmnopqrstuvwxyz"), budget=200) == _sig(
        "abcdefghijklmnopqrstuvwxyz"
    )


def test_an_incomplete_fragment_set_yields_nothing() -> None:
    origin, execution_id = uuid.uuid4(), uuid.uuid4()
    payloads = encode_signal(
        origin, execution_id, _sig("x" * 40_000), message_id=3, budget=_BUDGET
    )
    assembler = StreamFrameAssembler()

    for payload in payloads[:-1]:
        frame = decode_frame(payload)
        assert frame is not None
        assert assembler.push(frame) is None


def test_forget_drops_a_half_assembled_signal_of_that_execution() -> None:
    origin, execution_id = uuid.uuid4(), uuid.uuid4()
    payloads = encode_signal(
        origin, execution_id, _sig("x" * 40_000), message_id=3, budget=_BUDGET
    )
    assembler = StreamFrameAssembler()
    for payload in payloads[:-1]:
        frame = decode_frame(payload)
        assert frame is not None
        assembler.push(frame)

    assembler.forget(execution_id)

    last = decode_frame(payloads[-1])
    assert last is not None
    # The buffer is gone, so the final fragment alone cannot complete the signal.
    assert assembler.push(last) is None


def test_unreadable_payloads_are_dropped_rather_than_raising() -> None:
    assert decode_frame("not json") is None
    assert decode_frame("[]") is None
    assert decode_frame('{"o":"nope","e":"nope","t":"s"}') is None
    unknown_kind = f'{{"o":"{uuid.uuid4().hex}","e":"{uuid.uuid4().hex}","t":"?"}}'
    assert decode_frame(unknown_kind) is None
