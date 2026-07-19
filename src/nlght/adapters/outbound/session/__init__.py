# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Session key resolver adapters — extract session identity from inbound requests."""

from nlght.adapters.outbound.session.body import BodyParameterSessionKeyResolver
from nlght.adapters.outbound.session.composite import CompositeSessionKeyResolver
from nlght.adapters.outbound.session.header import HeaderSessionKeyResolver
from nlght.adapters.outbound.session.query import QueryParamSessionKeyResolver

__all__ = [
    "BodyParameterSessionKeyResolver",
    "CompositeSessionKeyResolver",
    "HeaderSessionKeyResolver",
    "QueryParamSessionKeyResolver",
]
