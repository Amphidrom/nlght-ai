# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

import asyncio

from nlght.application.config.configuration_service import ConfigurationService
from nlght.core.config.snapshot import PlatformConfigSnapshot


class StubConfigurationSource:
    def __init__(self, snapshot: PlatformConfigSnapshot) -> None:
        self.snapshot = snapshot

    async def load(self) -> PlatformConfigSnapshot:
        return self.snapshot


def test_configuration_service_delegates_to_source() -> None:
    snapshot = PlatformConfigSnapshot()
    service = ConfigurationService(StubConfigurationSource(snapshot))

    loaded = asyncio.run(service.load_snapshot())

    assert loaded is snapshot
