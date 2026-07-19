# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""OS runtime adapters — local host execution and Docker container execution."""

from nlght.adapters.outbound.os_runtime.docker import DockerOsRuntime, DockerOsRuntimeFactory
from nlght.adapters.outbound.os_runtime.local import LocalOsRuntime

__all__ = [
    "DockerOsRuntime",
    "DockerOsRuntimeFactory",
    "LocalOsRuntime",
]
