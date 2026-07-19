# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Signal emitter adapters — buffering (capture) and streaming (SSE queue)."""

from nlght.adapters.outbound.signals.buffering import BufferingSignalEmitter
from nlght.adapters.outbound.signals.streaming import QueuedSignalEmitter

__all__ = [
    "BufferingSignalEmitter",
    "QueuedSignalEmitter",
]
