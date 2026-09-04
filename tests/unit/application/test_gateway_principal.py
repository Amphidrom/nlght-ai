# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""The principal reaches the request context, and nothing else can put one there."""

from __future__ import annotations

import dataclasses

import pytest

from nlght.core.entry.context import PrincipalRef, RequestContext


def test_a_context_has_no_principal_unless_one_was_established() -> None:
    context = RequestContext(
        correlation_id="c", request_id="r",
        received_at=__import__("datetime").datetime.now(__import__("datetime").UTC),
        path="/", method="POST", headers={"x-principal-id": "alice"},
        query_params={}, client_host="203.0.113.9",
    )

    assert context.principal is None, "a header is not an identity"


def test_the_principal_survives_the_replace_that_fan_out_uses() -> None:
    """Child executions inherit the caller, and this is why.

    `fan_out` derives a child trigger with
    `dataclasses.replace(parent_context, correlation_id=…, request_id=…)`, so
    every field it does not name is carried. A principal added to the context is
    therefore propagated without the fan-out knowing it exists — and the
    workflow name is still overwritten downstream by the executor with the
    workflow actually being run.
    """
    import datetime

    parent = RequestContext(
        correlation_id="c", request_id="r",
        received_at=datetime.datetime.now(datetime.UTC),
        path="/", method="POST", headers={}, query_params={}, client_host="10.0.0.7",
        workflow="parent-flow", principal=PrincipalRef("alice"),
    )

    child = dataclasses.replace(
        parent, correlation_id="c:fanout:0", request_id="r:fanout:0",
    )

    assert child.principal == PrincipalRef("alice")
    assert child.workflow == "parent-flow", "the executor restamps this, not the fan-out"


@pytest.mark.parametrize("configured", [{}, {"type": ""}])
def test_no_configuration_means_no_principal_is_ever_established(configured: dict) -> None:
    from nlght.bootstrap.wiring import _build_principal_resolver

    assert _build_principal_resolver(configured) is None


def test_an_unknown_resolver_type_is_refused_rather_than_ignored() -> None:
    # A misspelled type that silently disabled identity would be the worst
    # possible failure: configured, ineffective, and undetectable.
    from nlght.bootstrap.wiring import _build_principal_resolver
    from nlght.core.errors.errors import ConfigurationError

    with pytest.raises(ConfigurationError, match="unknown principal resolver type"):
        _build_principal_resolver({"type": "trusted_gatway"})


def test_the_factory_builds_the_configured_shapes() -> None:
    from nlght.adapters.outbound.principal import (
        CompositePrincipalResolver,
        TrustedGatewayPrincipalResolver,
    )
    from nlght.bootstrap.wiring import _build_principal_resolver

    single = _build_principal_resolver({
        "type": "trusted_gateway", "header": "x-principal-id",
        "trusted_hosts": ["10.0.0.*"],
    })
    assert isinstance(single, TrustedGatewayPrincipalResolver)

    composed = _build_principal_resolver({
        "type": "composite",
        "resolvers": [
            {"type": "trusted_gateway", "header": "x-principal-id",
             "trusted_hosts": ["10.0.0.*"]},
            {"type": "trusted_gateway", "header": "x-id",
             "secret_header": "x-s", "secret": "v"},
        ],
    })
    assert isinstance(composed, CompositePrincipalResolver)


def test_a_configured_resolver_without_a_trust_condition_fails_at_startup() -> None:
    """Loudly, at boot, rather than by accepting everyone once traffic arrives."""
    from nlght.bootstrap.wiring import _build_principal_resolver
    from nlght.core.errors.errors import ConfigurationError

    with pytest.raises(ConfigurationError, match="needs a trust condition"):
        _build_principal_resolver({"type": "trusted_gateway", "header": "x-principal-id"})
