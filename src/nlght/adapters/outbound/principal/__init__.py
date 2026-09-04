# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Establishing who is calling, from sources a deployment has declared trusted."""

from nlght.adapters.outbound.principal.composite import CompositePrincipalResolver
from nlght.adapters.outbound.principal.gateway import TrustedGatewayPrincipalResolver

__all__ = ["CompositePrincipalResolver", "TrustedGatewayPrincipalResolver"]
