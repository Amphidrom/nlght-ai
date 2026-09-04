# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Who may open a session, written around the attack rather than the happy path.

Knowing a `session_key` used to be the whole of the authorization. These tests
are mostly one shape: somebody who knows a key and is not its owner gets nothing,
by every route — including the route through a factory that keeps its sessions in
memory and never consults a backend.
"""

from __future__ import annotations

import pytest

from nlght.adapters.outbound.hive_mind.simple import SimpleStoreCoordinatorFactory
from nlght.core.entry.context import PrincipalRef
from nlght.core.errors.errors import SessionAccessDeniedError
from nlght.core.session import SessionAccess

ALICE = PrincipalRef("alice")
BOB = PrincipalRef("bob")


def _access(**kwargs) -> SessionAccess:  # noqa: ANN003
    kwargs.setdefault("enforced", True)
    return SessionAccess(SimpleStoreCoordinatorFactory(), **kwargs)


# The owner, and everybody else
# ---------------------------------------------------------------------------

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
    """The path a check at the storage layer would miss entirely.

    `SimpleStoreCoordinatorFactory` keeps sessions in memory and returns a cache
    hit without touching any backend. A check placed at `SessionBackend.load()`
    would never run for the second request — which is exactly the request
    somebody with a stolen key makes.
    """
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
# ---------------------------------------------------------------------------

def test_a_refusal_names_neither_the_owner_nor_the_session_contents() -> None:
    """Otherwise the refusal itself is the disclosure.

    "belongs to alice" tells a stranger that alice exists, that she uses this
    system, and which key of hers is live — three things the caller was not
    entitled to and did not have a moment earlier.
    """
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
# ---------------------------------------------------------------------------

def test_an_unowned_session_is_not_adopted_by_whoever_asks_next() -> None:
    """First-caller-wins would be an account takeover performed by a migration.

    A session written before ownership existed has no owner. That is not the
    same as being unclaimed, and no ordinary request may become its owner.
    """
    factory = SimpleStoreCoordinatorFactory()
    factory.get_or_create("legacy")  # written the old way, with no owner recorded
    access = SessionAccess(factory, enforced=True)

    with pytest.raises(SessionAccessDeniedError, match="no recorded owner"):
        access.open("legacy", ALICE)
    with pytest.raises(SessionAccessDeniedError, match="no recorded owner"):
        access.open("legacy", BOB)


# A deployment that establishes no identity
# ---------------------------------------------------------------------------

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
    # persisted, so nobody reaches it again — including its caller.
    access = _access()

    assert access.ephemeral("cid-1") is not None
