# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from typing import Protocol

from nlght.core.entry.context import PrincipalRef


class PrincipalResolver(Protocol):
    """Establishes who is making a call, from a source configured as trusted.

    The question it answers is narrow and worth stating precisely:

        which principal was **established** for this call, on the basis of the
        configured trust source

    and not:

        which user id did the request claim

    That distinction is the whole reason this port is separate from
    `SessionKeyResolver`. A session key is a *locator*: reading it out of a
    header, a query parameter or a body is fine, because knowing where something
    lives grants nothing. A principal is *authorization-relevant*, so an
    implementation may only use a source whose trust model is explicit —
    a gateway the deployment declared trustworthy, a validated token, a
    certificate, a key resolved against a registry.

    An implementation that reads an identifier out of an arbitrary request and
    returns it would satisfy this signature and violate its contract. The
    signature cannot prevent that; the port says it so a reviewer can.

    `None` means no principal was established. It never means "anonymous": a
    caller with no principal owns nothing, which is a different statement from a
    caller who is somebody called nobody.
    """

    async def resolve(
        self,
        *,
        path: str,
        method: str,
        headers: dict[str, str],
        query_params: dict[str, str],
        client_host: str | None,
        raw_body: bytes,
    ) -> PrincipalRef | None: ...
