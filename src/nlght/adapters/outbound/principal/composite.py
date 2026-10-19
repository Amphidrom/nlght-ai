# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Several trusted sources, and what to do when they disagree."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from nlght.core.entry.context import PrincipalRef
from nlght.core.errors.errors import PrincipalConflictError
from nlght.ports.outbound.principal_resolver import PrincipalResolver

logger = logging.getLogger(__name__)


class CompositePrincipalResolver:
    """Asks every configured resolver and requires them to agree.

    **Not first-wins.** Each resolver is there because the deployment trusts its
    source, so two of them establishing two different principals is a broken
    deployment. For example, a client certificate and API key identifying
    different principals conflict. Preferring one would make the identity
    assigned to a request depend on resolver order, which is the
    kind of thing that is discovered during an incident.

    Agreement is fine and expected: two sources establishing the same principal
    is corroboration.

    A resolver that establishes nothing simply abstains. Nothing established by
    anybody is `None`: no identity has been established, which differs from a denial.
    """

    def __init__(self, resolvers: Sequence[PrincipalResolver]) -> None:
        self._resolvers = tuple(resolvers)

    async def resolve(
        self,
        *,
        path: str,
        method: str,
        headers: dict[str, str],
        query_params: dict[str, str],
        client_host: str | None,
        raw_body: bytes,
    ) -> PrincipalRef | None:
        established: dict[str, PrincipalRef] = {}
        for resolver in self._resolvers:
            found = await resolver.resolve(
                path=path, method=method, headers=headers,
                query_params=query_params, client_host=client_host, raw_body=raw_body,
            )
            if found is not None:
                established[found.id] = found
        if not established:
            return None
        if len(established) > 1:
            # The ids are named because a deployment cannot be repaired without
            # knowing which two sources disagreed. Nothing else about the call is.
            raise PrincipalConflictError(
                f"trusted sources established different principals for one call: "
                f"{', '.join(sorted(established))}. Both were configured as trusted, "
                f"so this is a deployment fault rather than a tie to break."
            )
        return next(iter(established.values()))
