# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True, frozen=True)
class PrincipalRef:
    """Who was *established* to be making this call.

    Deliberately one field. Session ownership needs a stable reference and
    nothing else, and a principal that also carried roles, tenants or scopes
    would tempt every consumer to make its own authorization decision from
    whatever looked relevant.

    It says nothing about *how* the identity was established — a gateway
    assertion, an API key, an OIDC token and a client certificate all produce
    the same reference. That is what lets the authentication source change
    later without ownership being rebuilt (ADR-0060).

    The absence of one is expressed by `RequestContext.principal is None`, which
    means "not established" and never "anonymous user". A caller with no
    principal is not a principal named nobody.
    """

    id: str

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("a principal reference needs an id")


@dataclass(slots=True, frozen=True)
class RequestContext:
    """Inbound request identity and metadata, carried through the invocation.

    ``workflow`` is filled in by the workflow executor once the invocation's
    workflow is known, so access policies can restrict a subject to specific
    flows through a ``workflow`` condition. It is ``None`` for a context that
    has not entered a workflow yet.
    """

    correlation_id: str
    request_id: str
    received_at: datetime
    path: str
    method: str
    headers: dict[str, str]
    query_params: dict[str, str]
    client_host: str | None
    workflow: str | None = None
    #: Who was established to be calling, where a `PrincipalResolver` could say.
    #:
    #: `None` means no identity was established — either none is configured, or
    #: the configured trust conditions were not met. It is authorization-relevant
    #: in exactly one direction: a `None` principal can own nothing. Nothing may
    #: read a header and treat it as this.
    principal: PrincipalRef | None = None
