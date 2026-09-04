# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Turning ranked findings into the text a caller may spend.

One step, between retrieval and whatever writes a prompt, and deliberately the
dullest one: no expansion, no re-ranking, no judgement about which of two
overlapping findings is the better one. See `passage` for why each of those is
somebody else's decision.
"""

from nlght.core.context.passage import (
    ContextBudget,
    ContextPassage,
    ContextSelection,
    build_context,
    passages_from,
    select,
)

__all__ = [
    "ContextBudget",
    "ContextPassage",
    "ContextSelection",
    "build_context",
    "passages_from",
    "select",
]
