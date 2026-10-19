# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Define the delegated wording-change judgement boundary."""

from __future__ import annotations

from typing import Protocol


class PropositionClassifier(Protocol):
    """Asks whether two wordings of one claim say the same thing."""

    async def classify(self, old: str, new: str) -> str:
        """Classify a changed claim body that rules cannot settle.

        Args:
            old (str): Previous claim wording.
            new (str): Current wording with unchanged numeric quantities.

        Returns:
            str: `rewrite`, `normative`, or `ambiguous` for uncertainty.

        Side Effects:
            May call a model provider. Backend failure yields ambiguity.
        """
        ...
