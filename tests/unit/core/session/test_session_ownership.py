# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Enforce session ownership across every access route.

Knowing a session key alone must not grant access to its state or workspace.
"""

from __future__ import annotations

import pytest

from nlght.adapters.outbound.hive_mind.simple import SimpleStoreCoordinatorFactory
from nlght.core.entry.context import PrincipalRef
from nlght.core.errors.errors import SessionAccessDeniedError
from nlght.core.session import SessionAccess

ALICE = PrincipalRef("alice")
BOB = PrincipalRef("bob")


def _access(**kwargs) -> SessionAccess:
    kwargs.setdefault("enforced", True)
    return SessionAccess(SimpleStoreCoordinatorFactory(), **kwargs)


# The owner, and everybody else

def test_creating_a_session_makes_the_caller_its_owner() -> None:
    access = _access()

    access.open("s1", ALICE)

    assert access.exists("s1", ALICE) is True


def test_the_owner_resumes_their_own_session() -> None:
    access = _access()
    first = access.open("s1", ALICE)

    again = access.open("s1", ALICE)

    assert again is first, "the same session, not a new one"


def test_somebody_else_who_knows_the_key_is_refused() -> None:
    access = _access()
    access.open("s1", ALICE)

    with pytest.raises(SessionAccessDeniedError):
        access.open("s1", BOB)


def test_a_cached_session_is_still_authorized() -> None:
    """Enforce ownership even when a session factory returns a cache hit."""
    access = _access()
    access.open("s1", ALICE)
    assert access.open("s1", ALICE) is not None, "warm in the cache"

    with pytest.raises(SessionAccessDeniedError):
        access.open("s1", BOB)


def test_no_principal_is_refused_rather_than_treated_as_legacy() -> None:
    # If `None` fell back to unrestricted access, the entire boundary would be
    # bypassable by omitting a header.
    access = _access()
    access.open("s1", ALICE)

    with pytest.raises(SessionAccessDeniedError):
        access.open("s1", None)


def test_saving_is_authorized_on_its_own() -> None:
    # Not trusted from the earlier open: holding a coordinator is not evidence
    # of anything by the time something writes with it.
    access = _access()
    coordinator = access.open("s1", ALICE)

    access.save("s1", coordinator, ALICE)
    with pytest.raises(SessionAccessDeniedError):
        access.save("s1", coordinator, BOB)
    with pytest.raises(SessionAccessDeniedError):
        access.save("s1", coordinator, None)


# What a refusal is allowed to reveal

def test_a_refusal_names_neither_the_owner_nor_the_session_contents() -> None:
    """Avoid disclosing owner or session state in an authorization refusal."""
    access = _access()
    access.open("s1", ALICE)

    with pytest.raises(SessionAccessDeniedError) as denied:
        access.open("s1", BOB)

    assert "alice" not in str(denied.value)


def test_every_stranger_is_refused_in_the_same_words() -> None:
    # A message that varies with who is asking, or with which key, is a
    # differential an attacker reads: the keys that answer differently are the
    # keys worth attacking.
    access = _access()
    access.open("s1", ALICE)
    access.open("s2", ALICE)

    with pytest.raises(SessionAccessDeniedError) as bob_on_s1:
        access.open("s1", BOB)
    with pytest.raises(SessionAccessDeniedError) as carol_on_s1:
        access.open("s1", PrincipalRef("carol"))

    assert str(bob_on_s1.value) == str(carol_on_s1.value)


def test_exists_answers_about_the_caller_and_not_about_the_key() -> None:
    access = _access()
    access.open("s1", ALICE)

    assert access.exists("s1", ALICE) is True
    assert access.exists("s1", BOB) is False, "not an oracle for somebody else's session"
    assert access.exists("s1", None) is False


# Sessions that predate ownership

def test_an_unowned_session_is_not_adopted_by_whoever_asks_next() -> None:
    """Deny adoption of a legacy session with the same detail for each caller."""
    factory = SimpleStoreCoordinatorFactory()
    factory.get_or_create("legacy")  # written the old way, with no owner recorded
    access = SessionAccess(factory, enforced=True)

    with pytest.raises(SessionAccessDeniedError) as alice_denied:
        access.open("legacy", ALICE)
    with pytest.raises(SessionAccessDeniedError) as bob_denied:
        access.open("legacy", BOB)
    assert str(alice_denied.value) == str(bob_denied.value)
    assert str(alice_denied.value) == "no session 'legacy' for this principal"

    owned_access = SessionAccess(SimpleStoreCoordinatorFactory(), enforced=True)
    owned_access.open("legacy", ALICE)
    with pytest.raises(SessionAccessDeniedError) as foreign_denied:
        owned_access.open("legacy", BOB)
    assert str(bob_denied.value) == str(foreign_denied.value)


def test_save_requires_a_previously_claimed_owner() -> None:
    """Deny saves to unknown and ownerless sessions without creating ownership."""
    factory = SimpleStoreCoordinatorFactory()
    legacy = factory.get_or_create("legacy")
    access = SessionAccess(factory, enforced=True)

    with pytest.raises(SessionAccessDeniedError) as unknown_denied:
        access.save("unknown", legacy, ALICE)
    with pytest.raises(SessionAccessDeniedError) as legacy_denied:
        access.save("legacy", legacy, ALICE)

    assert str(unknown_denied.value) == "no session 'unknown' for this principal"
    assert str(legacy_denied.value) == "no session 'legacy' for this principal"
    assert factory.exists("unknown") is False
    assert factory.owner_of("legacy") is None


# A deployment that establishes no identity

def test_without_identity_configured_the_platform_behaves_as_before() -> None:
    """Enforcement is a property of the configuration, never of a request.

    A deployment with no `principal:` block has said it runs inside a trusted
    boundary. That is a statement an operator made; it is not a fallback a
    request can trigger by leaving a header out.
    """
    access = _access(enforced=False)

    assert access.open("s1", None) is not None
    assert access.open("s1", ALICE) is not None


def test_an_ephemeral_session_belongs_to_nobody_and_needs_no_owner() -> None:
    # No session key was asked for, so there is nothing to own. It is never
    # persisted, so neither the caller nor another principal can reach it again.
    access = _access()

    assert access.ephemeral("cid-1") is not None
