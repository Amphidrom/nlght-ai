# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""The one boundary a session is reached through.

Knowing a `session_key` used to be the whole of the authorization. It is a
*locator*: it says where something lives, which is not a statement about who may
open it.

    session_id        locator
    ownership         authorization

**The boundary sits around the factory, not under it.** One of the two coordinator
factories keeps its sessions in memory and never reaches a backend on a cache
hit, so a check placed at `SessionBackend.load()` would be bypassed entirely by
the second request for a session — which is precisely the request an attacker
with a session id makes. Every normal path goes through here instead.
"""

from __future__ import annotations

import logging

from nlght.core.entry.context import PrincipalRef
from nlght.core.errors.errors import SessionAccessDeniedError
from nlght.ports.outbound.store_coordinator import (
    StoreCoordinator,
    StoreCoordinatorFactory,
)

logger = logging.getLogger(__name__)


class SessionAccess:
    """Opens a session for a principal, or refuses to.

    `enforced` says whether this deployment establishes identity at all. It is a
    property of the configuration — a `principal:` block is present or it is not
    — and never of an individual request. That distinction is the whole point:

        not enforced   no identity is established anywhere; sessions are
                       unowned and behave as they always have. The deployment
                       has said it runs inside a trusted boundary.

        enforced       every session is owned, and a request that establishes no
                       principal is refused. `None` is not a compatibility mode,
                       because a per-request fallback would leave exactly the
                       hole this exists to close, reachable by omitting a header.
    """

    def __init__(
        self, factory: StoreCoordinatorFactory, *, enforced: bool
    ) -> None:
        self._factory = factory
        self._enforced = enforced

    def open(
        self, session_key: str, principal: PrincipalRef | None
    ) -> StoreCoordinator:
        """The session behind this key, if it belongs to this principal.

        Creation and resumption are told apart before anything is decided,
        because they are different questions with different answers:

            create   this principal becomes the owner
            resume   the recorded owner must be this principal

        `get_or_create` answers both at once, which is why the distinction is
        made here rather than read out of it afterwards.
        """
        if not self._enforced:
            return self._factory.get_or_create(session_key)

        if principal is None:
            raise SessionAccessDeniedError(
                "no principal was established for this call, so no session may be "
                "opened. A session belongs to somebody."
            )

        owner = self._factory.owner_of(session_key)
        if owner is None and self._factory.exists(session_key):
            # It exists and has no owner: written before ownership, or by a path
            # that records none. Whoever asks next does not get to become its
            # owner — that would be an account takeover performed by a
            # migration (ADR-0061).
            logger.warning(
                "session.unowned | key=%s — refused; an unowned session is not "
                "an unclaimed one", session_key,
            )
            raise SessionAccessDeniedError(
                f"session '{session_key}' has no recorded owner and cannot be "
                f"adopted. It predates ownership; recovering it is a privileged "
                f"operation, not a request."
            )

        if owner is not None and owner != principal.id:
            # Deliberately the same message as a session that does not exist:
            # telling the two apart turns this into an oracle for which session
            # ids are real.
            logger.warning(
                "session.denied | key=%s principal=%s — not the owner",
                session_key, principal.id,
            )
            raise SessionAccessDeniedError(f"no session '{session_key}' for this principal")

        coordinator = self._factory.get_or_create(session_key)
        if owner is None:
            self._factory.claim_ownership(session_key, principal.id)
            logger.info(
                "session.created | key=%s owner=%s", session_key, principal.id,
            )
        return coordinator

    def ephemeral(self, correlation_id: str) -> StoreCoordinator:
        """A session for one call, belonging to nobody and outliving nothing.

        A request with no session key asks for no session, so there is nothing to
        own and nothing to authorize. It is never persisted, so it cannot be
        reached again by anybody — including the caller who caused it.
        """
        return self._factory.get_or_create(f"__ephemeral__{correlation_id}")

    def exists(self, session_key: str, principal: PrincipalRef | None) -> bool:
        """Whether *this principal* has a session under this key.

        Never whether the key is in use. An honest answer about somebody else's
        session is an oracle: ask about enough ids and the ones that come back
        `True` are the ones worth attacking.
        """
        if not self._enforced:
            return self._factory.exists(session_key)
        if principal is None:
            return False
        return self._factory.owner_of(session_key) == principal.id

    def save(
        self, session_key: str, coordinator: StoreCoordinator, principal: PrincipalRef | None
    ) -> None:
        """Persist a session this principal owns.

        Checked again rather than trusted from the earlier `open`: a save is its
        own operation, and a caller holding a coordinator is not evidence of
        anything by the time it writes.
        """
        if self._enforced:
            if principal is None:
                raise SessionAccessDeniedError("no principal established; nothing may be saved")
            owner = self._factory.owner_of(session_key)
            if owner is None and self._factory.exists(session_key):
                # Same refusal as `open`, for the same reason: writing into an
                # unowned session would claim it by the back door, and a write
                # is a worse way to acquire one than a read.
                raise SessionAccessDeniedError(
                    f"session '{session_key}' has no recorded owner and cannot be "
                    f"written to."
                )
            if owner is not None and owner != principal.id:
                raise SessionAccessDeniedError(
                    f"no session '{session_key}' for this principal"
                )
        self._factory.save(session_key, coordinator)
