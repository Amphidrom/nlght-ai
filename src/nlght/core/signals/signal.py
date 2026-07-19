# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Signal:
    """A unit of output emitted by a workflow step.

    Steps emit signals without knowing about the transport layer.
    Serialisation to SSE, JSON, or any other format is the adapter's concern.

    kind values:
      "token"  — incremental content fragment (e.g. an LLM output token)
      "result" — a complete, final result payload
      "status" — non-user-facing progress / diagnostic message
      "done"   — signals end of output for this invocation
    """

    role: str
    content: str
    kind: str = "token"
