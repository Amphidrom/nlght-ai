# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from typing import Protocol

from nlght.core.config.snapshot import PlatformConfigSnapshot


class ConfigurationSource(Protocol):
    async def load(self) -> PlatformConfigSnapshot:
        raise NotImplementedError