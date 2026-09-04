# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""A principal asserted by an upstream the deployment has declared trustworthy.

The smallest resolver that can honestly exist today, and the one the platform's
own documented posture already assumes: it is meant to run behind a trusted
boundary. This makes that assumption a configured statement instead of an
unspoken one.

The difference from reading a header is the whole point:

    any request + X-Principal-Id            →  a claim
    a request from the configured upstream  →  an assertion
      + X-Principal-Id

So a trust condition is **required**. A resolver constructed without one would
accept an identity from anybody, which is the gap this exists to close, and it
refuses to be built rather than doing that quietly.
"""

from __future__ import annotations

import fnmatch
import hmac
import logging

from nlght.core.entry.context import PrincipalRef
from nlght.core.errors.errors import ConfigurationError

logger = logging.getLogger(__name__)


class TrustedGatewayPrincipalResolver:
    """Reads an assertion from an upstream proven to be the configured one.

    Two ways to prove it, and at least one is required:

        trusted_hosts    the immediate peer must match one of these patterns
        shared_secret    the upstream must present a secret only it knows

    Both may be given, and then both must hold: a network position *and* a
    secret is what a deployment behind a gateway on a shared network needs.
    """

    def __init__(
        self,
        *,
        header: str,
        trusted_hosts: tuple[str, ...] = (),
        secret_header: str = "",
        secret: str = "",
    ) -> None:
        if not header.strip():
            raise ConfigurationError("principal resolver needs the header carrying the assertion")
        if not trusted_hosts and not (secret_header and secret):
            raise ConfigurationError(
                "a trusted-gateway principal resolver needs a trust condition: "
                "'trusted_hosts', or 'secret_header' with 'secret'. Without one it "
                "would accept an identity asserted by any caller, which is the gap "
                "it exists to close."
            )
        self._header = header.strip().lower()
        self._hosts = trusted_hosts
        self._secret_header = secret_header.strip().lower()
        self._secret = secret

    async def resolve(
        self,
        *,
        path: str,  # noqa: ARG002
        method: str,  # noqa: ARG002
        headers: dict[str, str],
        query_params: dict[str, str],  # noqa: ARG002
        client_host: str | None,
        raw_body: bytes,  # noqa: ARG002
    ) -> PrincipalRef | None:
        if not self._trusted(headers, client_host):
            return None
        asserted = str(headers.get(self._header, "")).strip()
        return PrincipalRef(id=asserted) if asserted else None

    def _trusted(self, headers: dict[str, str], client_host: str | None) -> bool:
        """Whether this call really came from the upstream that was configured.

        Every configured condition must hold. A deployment that names both a
        host range and a secret is saying the assertion counts only from that
        position *and* with that proof, and honouring only one of them would
        quietly weaken what it asked for.
        """
        if self._hosts:
            if client_host is None:
                return False
            if not any(fnmatch.fnmatch(client_host, pattern) for pattern in self._hosts):
                logger.warning(
                    "principal.untrusted_origin | host=%s — assertion ignored", client_host,
                )
                return False
        if self._secret_header:
            supplied = str(headers.get(self._secret_header, ""))
            # Constant time: a secret compared with `==` leaks its prefix to
            # anybody willing to time the answer.
            if not supplied or not hmac.compare_digest(supplied, self._secret):
                logger.warning("principal.untrusted_upstream | the shared secret did not match")
                return False
        return True
