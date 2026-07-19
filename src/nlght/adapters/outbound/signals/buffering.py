# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from nlght.core.signals.signal import Signal
from nlght.ports.outbound.signal_emitter import SignalEmitter


class BufferingSignalEmitter(SignalEmitter):
    """Collects signals in memory.

    Used by ``StepMachineWorkflowExecutor.execute()`` to gather all step output
    before assembling the non-streaming response dict.
    """

    def __init__(self) -> None:
        self._signals: list[Signal] = []

    async def emit(self, signal: Signal) -> None:
        self._signals.append(signal)

    def collected(self) -> list[Signal]:
        return list(self._signals)
