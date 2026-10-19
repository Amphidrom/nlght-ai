# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Authorize session creation, resumption, and persistence."""

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
    """Gate session access with deployment-level principal enforcement.

    When enforcement is active, every session has an owner and requests without
    an established principal are refused. Disabling enforcement is a deployment
    configuration choice, never a fallback chosen by an individual request.
    """

    def __init__(
        self, factory: StoreCoordinatorFactory, *, enforced: bool
    ) -> None:
        self._factory = factory
        self._enforced = enforced

    def open(
        self, session_key: str, principal: PrincipalRef | None
    ) -> StoreCoordinator:
        """Open or create a session for its authorized principal.

        Args:
            session_key (str): Requested session identity.
            principal (PrincipalRef | None): Identity established by the deployment, when available.

        Returns:
            StoreCoordinator: Coordinator for the accessible session.

        Raises:
            SessionAccessDeniedError: Enforcement is active and ownership cannot be established.

        Side Effects:
            Reads or creates a session and may claim ownership.
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
            # owner. That would be an account takeover performed by a
            # migration (ADR-0061).
            logger.warning(
                "session.unowned | key=%s: refused; an unowned session is not "
                "an unclaimed one", session_key,
            )
            raise SessionAccessDeniedError(f"no session '{session_key}' for this principal")

        if owner is not None and owner != principal.id:
            # Deliberately the same message as a session that does not exist:
            # telling the two apart turns this into an oracle for which session
            # ids are real.
            logger.warning(
                "session.denied | key=%s principal=%s: not the owner",
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
        """Obtain a coordinator for a call without a named session.

        Args:
            correlation_id (str): Identity of the current call.

        Returns:
            StoreCoordinator: Coordinator under a synthetic ephemeral key.

        Side Effects:
            May create or reuse factory-held state for that key. The executor
            does not save it as a named session.
        """
        return self._factory.get_or_create(f"__ephemeral__{correlation_id}")

    def exists(self, session_key: str, principal: PrincipalRef | None) -> bool:
        """Check whether a session is visible to the established principal.

        Args:
            session_key (str): Session identity to check.
            principal (PrincipalRef | None): Identity established by the deployment, when available.

        Returns:
            bool: True only when the session exists and is accessible.

        Side Effects:
            Reads session ownership or existence.
        """
        if not self._enforced:
            return self._factory.exists(session_key)
        if principal is None:
            return False
        return self._factory.owner_of(session_key) == principal.id

    def save(
        self, session_key: str, coordinator: StoreCoordinator, principal: PrincipalRef | None
    ) -> None:
        """Persist a session after rechecking ownership.

        Args:
            session_key (str): Session identity to save.
            coordinator (StoreCoordinator): Session state to persist.
            principal (PrincipalRef | None): Identity established by the deployment, when available.

        Raises:
            SessionAccessDeniedError: No principal is established, the session is
                unowned, or another principal owns it.

        Side Effects:
            Persists the coordinator when authorized.
        """
        if self._enforced:
            if principal is None:
                raise SessionAccessDeniedError("no principal established; nothing may be saved")
            owner = self._factory.owner_of(session_key)
            if owner != principal.id:
                # Require a prior authorized open: a save to an unknown key
                # must not create an unowned session by bypassing ownership.
                raise SessionAccessDeniedError(
                    f"no session '{session_key}' for this principal"
                )
        self._factory.save(session_key, coordinator)
