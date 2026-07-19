# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Hive-Mind store coordinator adapters — full persistence and in-memory variants."""

from nlght.adapters.outbound.hive_mind.coordinator import (
    HiveMindStoreCoordinator,
    HiveMindStoreCoordinatorFactory,
)
from nlght.adapters.outbound.hive_mind.simple import (
    SimpleStoreCoordinator,
    SimpleStoreCoordinatorFactory,
)

__all__ = [
    "HiveMindStoreCoordinator",
    "HiveMindStoreCoordinatorFactory",
    "SimpleStoreCoordinator",
    "SimpleStoreCoordinatorFactory",
]
