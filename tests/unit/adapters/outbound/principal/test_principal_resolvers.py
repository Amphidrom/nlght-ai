# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Establishing a principal, and refusing to pretend one was established.

The whole value of this boundary is one distinction:

    any request + X-Principal-Id            a claim
    a request from the configured upstream  an assertion
      + X-Principal-Id

Most of these tests are about the first case not becoming the second by
accident — which is exactly what would happen if the resolver simply read the
header.
"""

from __future__ import annotations

import pytest

from nlght.adapters.outbound.principal import (
    CompositePrincipalResolver,
    TrustedGatewayPrincipalResolver,
)
from nlght.core.entry.context import PrincipalRef
from nlght.core.errors.errors import ConfigurationError, PrincipalConflictError


async def _resolve(resolver, *, headers=None, client_host=None):  # noqa: ANN001, ANN202
    return await resolver.resolve(
        path="/v1/chat/completions", method="POST",
        headers=headers or {}, query_params={},
        client_host=client_host, raw_body=b"",
    )


def _gateway(**kwargs):  # noqa: ANN003, ANN202
    kwargs.setdefault("header", "x-principal-id")
    return TrustedGatewayPrincipalResolver(**kwargs)


# The trust condition is not optional
# ---------------------------------------------------------------------------

def test_a_resolver_without_a_trust_condition_refuses_to_exist() -> None:
    """The gap this closes, closed at construction rather than at runtime.

    A resolver that reads the header from anybody is the thing that must not be
    built, so it is not buildable — rather than built and then relied upon to
    behave.
    """
    with pytest.raises(ConfigurationError, match="needs a trust condition"):
        TrustedGatewayPrincipalResolver(header="x-principal-id")


def test_a_resolver_without_a_header_refuses_to_exist() -> None:
    with pytest.raises(ConfigurationError, match="header carrying the assertion"):
        TrustedGatewayPrincipalResolver(header=" ", trusted_hosts=("10.0.0.1",))


async def test_an_assertion_from_the_trusted_host_establishes_a_principal() -> None:
    resolver = _gateway(trusted_hosts=("10.0.0.*",))

    found = await _resolve(
        resolver, headers={"x-principal-id": "alice"}, client_host="10.0.0.7",
    )

    assert found == PrincipalRef(id="alice")


async def test_the_same_assertion_from_anywhere_else_establishes_nothing() -> None:
    # The header is identical. Only the origin differs, and that is the whole
    # difference between a claim and an assertion.
    resolver = _gateway(trusted_hosts=("10.0.0.*",))

    assert await _resolve(
        resolver, headers={"x-principal-id": "alice"}, client_host="203.0.113.9",
    ) is None


async def test_an_unknown_origin_establishes_nothing() -> None:
    resolver = _gateway(trusted_hosts=("10.0.0.*",))

    assert await _resolve(resolver, headers={"x-principal-id": "alice"}) is None


async def test_a_shared_secret_can_stand_in_for_a_host_range() -> None:
    resolver = _gateway(secret_header="x-gateway-secret", secret="s3cret")

    assert await _resolve(resolver, headers={
        "x-principal-id": "alice", "x-gateway-secret": "s3cret",
    }) == PrincipalRef(id="alice")
    assert await _resolve(resolver, headers={
        "x-principal-id": "alice", "x-gateway-secret": "wrong",
    }) is None
    assert await _resolve(resolver, headers={"x-principal-id": "alice"}) is None


async def test_both_conditions_must_hold_when_both_are_configured() -> None:
    """A deployment that names both is asking for both.

    Honouring either alone would quietly weaken what it configured — and it
    would do so in the direction that accepts more.
    """
    resolver = _gateway(
        trusted_hosts=("10.0.0.*",), secret_header="x-gateway-secret", secret="s3cret",
    )
    asserted = {"x-principal-id": "alice", "x-gateway-secret": "s3cret"}

    assert await _resolve(resolver, headers=asserted, client_host="10.0.0.7") is not None
    assert await _resolve(resolver, headers=asserted, client_host="203.0.113.9") is None
    assert await _resolve(
        resolver, headers={"x-principal-id": "alice"}, client_host="10.0.0.7",
    ) is None


async def test_a_trusted_upstream_that_asserts_nothing_establishes_nothing() -> None:
    # Trusted and silent is not the same as trusted and anonymous. There is
    # simply no identity here.
    resolver = _gateway(trusted_hosts=("10.0.0.*",))

    assert await _resolve(resolver, client_host="10.0.0.7") is None


# Several sources, and what disagreement means
# ---------------------------------------------------------------------------

class _Fixed:
    def __init__(self, principal: PrincipalRef | None) -> None:
        self._principal = principal

    async def resolve(self, **_kwargs: object) -> PrincipalRef | None:
        return self._principal


async def test_agreement_between_two_sources_is_corroboration() -> None:
    resolver = CompositePrincipalResolver([
        _Fixed(PrincipalRef("alice")), _Fixed(PrincipalRef("alice")),
    ])

    assert await _resolve(resolver) == PrincipalRef("alice")


async def test_disagreement_is_an_error_and_not_a_tie_to_break() -> None:
    """Never first-wins.

    Each resolver is configured because its source is trusted, so two of them
    establishing two different principals is a broken deployment — a certificate
    for one identity carrying a key for another. Preferring one would make the
    identity a request runs as depend on the order somebody listed them in.
    """
    resolver = CompositePrincipalResolver([
        _Fixed(PrincipalRef("alice")), _Fixed(PrincipalRef("bob")),
    ])

    with pytest.raises(PrincipalConflictError, match="alice, bob"):
        await _resolve(resolver)


async def test_a_source_that_establishes_nothing_simply_abstains() -> None:
    resolver = CompositePrincipalResolver([_Fixed(None), _Fixed(PrincipalRef("alice"))])

    assert await _resolve(resolver) == PrincipalRef("alice")


async def test_nothing_established_by_anybody_is_no_identity() -> None:
    resolver = CompositePrincipalResolver([_Fixed(None), _Fixed(None)])

    assert await _resolve(resolver) is None


# The reference itself
# ---------------------------------------------------------------------------

def test_a_principal_reference_needs_an_id() -> None:
    with pytest.raises(ValueError, match="needs an id"):
        PrincipalRef(id="  ")


def test_a_principal_carries_nothing_but_its_identity() -> None:
    # Roles, tenants and scopes are deliberately absent: a principal that
    # carried them would tempt every consumer to make its own authorization
    # decision from whatever looked relevant.
    assert {field.name for field in PrincipalRef.__dataclass_fields__.values()} == {"id"}
