# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import pytest

from nlght.ports.outbound.runtime_subsystem import RuntimeSubsystemFactory, SubsystemLifecycle


async def test_runtime_subsystem_protocol_default_methods_raise() -> None:
    class _Factory(RuntimeSubsystemFactory):
        pass

    class _Lifecycle(SubsystemLifecycle):
        pass

    factory = _Factory()
    lifecycle = _Lifecycle()

    with pytest.raises(NotImplementedError):
        factory.kind()
    with pytest.raises(NotImplementedError):
        await factory.build({})
    with pytest.raises(NotImplementedError):
        await lifecycle.start()
    with pytest.raises(NotImplementedError):
        await lifecycle.stop()
