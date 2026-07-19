# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from typing import Protocol

from nlght.core.access.rule import AccessRule


class AccessRuleRepository(Protocol):
    """Read access to the stored access-policy rules.

    Administrative CRUD lives in the admin service; the runtime only ever
    needs the enabled rule set.
    """

    async def list_enabled(self) -> list[AccessRule]: ...
