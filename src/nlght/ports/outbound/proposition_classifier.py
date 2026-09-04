# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""The one judgement about meaning this pipeline delegates.

Everything else in the identity design is decided by rules over strings and
stored state, deliberately: rules can be read, argued with, and tested. This is
the one place where that stops working. Whether a rule's body changed is a fact
about a language — "must" becoming "must not" is a different rule and "must"
becoming "shall" is the same one — and no rule over characters separates those
without knowing what the words mean.

The alternative was a word list. It would need `not`, `never`, `unless`,
`disabled`, `may`, `shall` and their equivalents in every language a customer
writes documentation in, and it would be wrong for `must` in one direction and
right in the other from the very first pair. That list is never finished and
nobody can maintain it.

So the question is asked once, narrowly, with three answers and a way to say "I
cannot tell". A port rather than a call into a model client, so what happens when
the answer is doubt stays a decision this codebase makes.
"""

from __future__ import annotations

from typing import Protocol


class PropositionClassifier(Protocol):
    """Asks whether two wordings of one claim say the same thing."""

    async def classify(self, old: str, new: str) -> str:
        """`rewrite`, `normative`, or `ambiguous` when it cannot tell.

        Only ever asked about the undecidable band — same quantities, different
        words — so an implementation may assume the two differ and that no
        threshold, version or count moved between them.

        `ambiguous` is a real answer and not a failure. It resolves to a review,
        which is the conservative direction: a review nobody needed costs a
        minute, and an approval nobody gave costs trust. An implementation that
        cannot reach its backend should say `ambiguous` rather than guess.
        """
        ...
