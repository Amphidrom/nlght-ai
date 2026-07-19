# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from nlght.core.config.snapshot import PlatformConfigSnapshot
from nlght.ports.outbound.configuration_source import ConfigurationSource


class ConfigurationService:
    def __init__(self, source: ConfigurationSource) -> None:
        self._source = source

    async def load_snapshot(self) -> PlatformConfigSnapshot:
        return await self._source.load()
