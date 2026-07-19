# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True, frozen=True)
class RequestContext:
    correlation_id: str
    request_id: str
    received_at: datetime
    path: str
    method: str
    headers: dict[str, str]
    query_params: dict[str, str]
    client_host: str | None