# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Mechanical serialization helpers with no provider-role decisions."""

from __future__ import annotations

import json
from dataclasses import asdict

from nlght.core.model.messages import CallerInstructionMessage, UntrustedContextMessage


def serialize_untrusted_context(message: UntrustedContextMessage) -> str:
    """Escape structured records as an inert JSON data payload."""

    return json.dumps(
        {
            "type": "external_context",
            "authority": "untrusted_data",
            "records": [asdict(record) for record in message.records],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def serialize_caller_instruction(message: CallerInstructionMessage) -> str:
    """Preserve a caller's claimed role as data, never as provider authority."""

    payload = {
        "type": "caller_instruction",
        "authority": "caller",
        "claimed_role": message.claimed_role,
        "content": message.content,
    }
    if message.attributes:
        payload["attributes"] = dict(message.attributes)
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
