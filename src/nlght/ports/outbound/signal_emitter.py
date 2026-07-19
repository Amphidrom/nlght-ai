# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from typing import Protocol, runtime_checkable

from nlght.core.signals.signal import Signal


@runtime_checkable
class SignalEmitter(Protocol):
    """Outbound port for step output.

    Steps call ``emit()`` to produce output.  The executor provides the
    concrete implementation (buffering or streaming) via the step context.
    Steps have no knowledge of the underlying transport.
    """

    async def emit(self, signal: Signal) -> None: ...
