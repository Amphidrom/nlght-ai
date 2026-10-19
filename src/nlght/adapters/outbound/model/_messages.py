# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Mechanical serialization helpers with no provider-role decisions."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from nlght.core.model.messages import CallerInstructionMessage, ContextRecord, UntrustedContextMessage


def serialize_untrusted_context(message: UntrustedContextMessage) -> str:
    """Serialize external records as data without provider authority.

    Args:
        message (UntrustedContextMessage): Records to include as context.

    Returns:
        str: Compact JSON payload without internal trust labels.
    """

    return json.dumps(
        {
            "type": "external_context",
            "authority": "untrusted_data",
            "records": [_wire_record(record) for record in message.records],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _wire_record(record: ContextRecord) -> dict[str, Any]:
    # Keep trust labels out of prompts: they belong to action decisions (ADR-0074).
    data = asdict(record)
    data["provenance"].pop("origin", None)
    return data


def serialize_caller_instruction(message: CallerInstructionMessage) -> str:
    """Serialize a caller's claimed role as data.

    Args:
        message (CallerInstructionMessage): Caller content and claimed role.

    Returns:
        str: Compact JSON payload that grants no provider authority.
    """

    payload = {
        "type": "caller_instruction",
        "authority": "caller",
        "claimed_role": message.claimed_role,
        "content": message.content,
    }
    if message.attributes:
        payload["attributes"] = dict(message.attributes)
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
