# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True, frozen=True)
class ResourceDef:
    """One configured resource: what it is, where it is, and how to build it.

    Two identities, and they answer different questions.

    ``resource_id`` is the persistent identity of a row, stable across
    every rename. ``address`` is how everything else addresses it: unique,
    human-readable, and the thing an operator writes in a policy rule. Nothing
    resolves a resource by ``resource_id`` today. Every caller asks for a kind
    and a name.

    ``provider`` is deliberately *not* part of the address. It selects which
    registered implementation instantiates this resource, which is a different
    question from which resource this is. Two rows differing only in provider
    would otherwise be two resources at one address, and the address would stop
    being one.
    """

    resource_id: uuid.UUID
    name: str
    kind: str
    provider: str
    config: dict[str, Any]
    enabled: bool = True

    @property
    def address(self) -> str:
        """Build the canonical runtime resource locator.

        Returns:
            str: Address in kind/name form.
        """
        return f"{self.kind}/{self.name}"
