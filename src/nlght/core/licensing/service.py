# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from typing import Any

from nlght.core.licensing.license import License


class LicenseService:
    def __init__(self, license: License) -> None:
        self._license = license

    def allows(self, capability: str) -> bool:
        return self._license.allows(capability)

    def limit(self, capability: str, default: int) -> int:
        return self._license.limit(capability, default)

    def context_metadata(self) -> dict[str, Any]:
        return {
            "customer_id": self._license.customer_id,
            "tier": self._license.tier.name.lower(),
        }