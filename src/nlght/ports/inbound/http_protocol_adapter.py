# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from typing import Protocol

from fastapi import APIRouter


class HttpProtocolAdapter(Protocol):
    def build_router(self) -> APIRouter: ...
