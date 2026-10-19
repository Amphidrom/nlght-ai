# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Keep PrincipalRef limited to caller identity.

Roles and scopes belong to policy inputs, because carrying them on the
principal lets steps make local authorization decisions (ADR-0060)."""

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
