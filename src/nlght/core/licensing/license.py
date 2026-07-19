# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import enum
from dataclasses import dataclass, field


class Tier(enum.IntEnum):
    FREE       = 0
    COMMERCIAL = 1
    OEM        = 2
    ENTERPRISE = 3

    @classmethod
    def from_str(cls, value: str) -> Tier:
        try:
            return cls[value.upper()]
        except KeyError as err:
            raise ValueError(f"Unknown tier: {value!r}") from err


@dataclass(frozen=True)
class License:
    customer_id: str
    tier: Tier
    entitlements: frozenset[str] = field(default_factory=frozenset)
    limits: dict[str, int] = field(default_factory=dict)

    def allows(self, capability: str) -> bool:
        return capability in self.entitlements

    def limit(self, capability: str, default: int) -> int:
        return self.limits.get(capability, default)