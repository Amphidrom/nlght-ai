# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from typing import Any, Protocol


class RuntimeSubsystemFactory(Protocol):
    def kind(self) -> str:
        raise NotImplementedError

    async def build(self, config: dict[str, Any]) -> object:
        raise NotImplementedError


class SubsystemLifecycle(Protocol):
    async def start(self) -> None:
        raise NotImplementedError

    async def stop(self) -> None:
        raise NotImplementedError
