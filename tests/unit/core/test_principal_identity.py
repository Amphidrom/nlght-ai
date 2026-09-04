# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""The shape of `PrincipalRef` is part of the authorization architecture.

Normally a test against an exact field list is noise. Here the single field *is*
the decision (ADR-0060): a principal that also carried roles, scopes or tenants
would put an authorization input on an object every step and tool already holds,
and the first `if "admin" in ctx.principal.roles` would be a local authorization
decision made outside any policy.

Authentication may of course produce roles. They belong to a separate,
authorization-facing type that only a policy consumes — not to the reference
that identifies who is calling.
"""

from __future__ import annotations

import dataclasses

from nlght.core.entry.context import PrincipalRef


def test_principal_ref_remains_identity_only() -> None:
    assert tuple(PrincipalRef.__dataclass_fields__) == ("id",)


def test_principal_ref_is_frozen() -> None:
    # A mutable reference could be re-pointed after an ownership check.
    assert dataclasses.fields(PrincipalRef)
    principal = PrincipalRef(id="user-1")
    try:
        principal.id = "user-2"  # type: ignore[misc]
    except dataclasses.FrozenInstanceError:
        return
    raise AssertionError("PrincipalRef must be frozen")
