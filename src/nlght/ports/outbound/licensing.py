# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class LicensingPort(Protocol):
    def allows(self, capability: str) -> bool: ...
    def limit(self, capability: str, default: int) -> int: ...
    def context_metadata(self) -> dict[str, Any]: ...
