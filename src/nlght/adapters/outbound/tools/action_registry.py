# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Registry for the deployment's execution-time action authorization (ADR-0071)."""

from __future__ import annotations

import logging

from nlght.core.errors.errors import ConfigurationError
from nlght.ports.outbound.action_policy import ActionApprovalPort, ActionPolicy

logger = logging.getLogger(__name__)


class ActionAuthorizationRegistry:
    """Holds the one ``ActionPolicy`` and the one ``ActionApprovalPort`` a deployment sets.

    Instances are registered, not classes: an approval adapter carries its own
    client and configuration. Call ``register_*()`` before ``serve()``. The
    container reads the registry once when it is built.

    Each slot takes one value. A second registration raises instead of
    replacing the first, because a later import silently swapping a security
    control is exactly what nobody would notice.

    Nothing registered keeps the built-in behaviour: ``DefaultActionPolicy``,
    and every ``REQUIRE_CONFIRMATION`` is denied.
    """

    def __init__(self) -> None:
        self._policy: ActionPolicy | None = None
        self._approval: ActionApprovalPort | None = None

    def register_policy(self, policy: ActionPolicy) -> None:
        if not isinstance(policy, ActionPolicy):
            raise TypeError(f"{type(policy).__name__} does not implement ActionPolicy")
        if self._policy is not None:
            raise ConfigurationError(
                f"an ActionPolicy is already registered ({type(self._policy).__name__}); "
                "only one is supported"
            )
        self._policy = policy
        logger.debug("action_policy.registered | type=%s", type(policy).__name__)

    def register_approval(self, approval: ActionApprovalPort) -> None:
        if not isinstance(approval, ActionApprovalPort):
            raise TypeError(f"{type(approval).__name__} does not implement ActionApprovalPort")
        if self._approval is not None:
            raise ConfigurationError(
                f"an ActionApprovalPort is already registered ({type(self._approval).__name__}); "
                "only one is supported"
            )
        self._approval = approval
        logger.debug("action_approval.registered | type=%s", type(approval).__name__)

    def policy(self) -> ActionPolicy | None:
        return self._policy

    def approval(self) -> ActionApprovalPort | None:
        return self._approval


# Register application policies before server startup.
#
# Register before server start:
#
#   from nlght.adapters.outbound.tools.action_registry import action_authorization_registry
#
#   action_authorization_registry.register_policy(MyActionPolicy())
#   action_authorization_registry.register_approval(MyApprovalAdapter(...))
action_authorization_registry: ActionAuthorizationRegistry = ActionAuthorizationRegistry()
