# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import uuid
from typing import Protocol

from nlght.core.runtime.resource import ResourceDef


class ResourceRepository(Protocol):
    async def find_by_id(self, resource_id: uuid.UUID) -> ResourceDef | None: ...

    async def find_by_kind(self, kind: str) -> list[ResourceDef]: ...

    async def list_enabled(self) -> list[ResourceDef]: ...
