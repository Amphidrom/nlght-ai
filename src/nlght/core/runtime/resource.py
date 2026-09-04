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

    ``resource_id`` is the persistent identity — what a row *is*, stable across
    every rename. ``address`` is how everything else addresses it: unique,
    human-readable, and the thing an operator writes in a policy rule. Nothing
    resolves a resource by ``resource_id`` today; every caller asks for a kind
    and a name.

    ``provider`` is deliberately *not* part of the address. It selects which
    registered implementation instantiates this resource, which is a different
    question from which resource this is — two rows differing only in provider
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
        """The unique runtime locator, ``<kind>/<name>``.

        Defined once, here, rather than formatted at each call site. The
        activator resolves by it, an access rule names it, the admin surface
        reports a conflict with it and the logs print it — four places that have
        to agree on one string, and would agree until one of them didn't.
        """
        return f"{self.kind}/{self.name}"
