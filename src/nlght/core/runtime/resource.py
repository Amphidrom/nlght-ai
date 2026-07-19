# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True, frozen=True)
class ResourceDef:
    resource_id: uuid.UUID
    name: str
    kind: str
    provider: str
    config: dict[str, Any]
    enabled: bool = True
