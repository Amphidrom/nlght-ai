# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Trusted system-instruction construction.

This module deliberately has no dependency on ``MentalModel``, retrieval,
session stores, workflow messages, or prompt slots. A value that can reach this
builder is deployment-authored instruction text; all runtime knowledge is built
separately by :mod:`nlght.core.hive_mind.model_context`.
"""

from __future__ import annotations

DEFAULT_CONSTRAINTS: list[str] = [
    "Never use placeholders like [value], [insert here], or similar if it wasn't asked explicitly.",
    "Never invent, estimate, or approximate values — if unsure, say so.",
]


class SystemPromptBuilder:
    """Build only deployment-trusted instructions."""

    def build(
        self,
        ego: str,
        constraints: list[str] | None = None,
    ) -> str:
        """Return trusted instructions; no knowledge input exists by design."""

        selected_constraints = DEFAULT_CONSTRAINTS if constraints is None else constraints
        sections = [ego.strip()]
        if selected_constraints:
            sections.append(self._constraints_block(selected_constraints))
        return "\n\n".join(section for section in sections if section)

    @staticmethod
    def _constraints_block(constraints: list[str]) -> str:
        lines = ["Always follow these constraints:"]
        lines.extend(f"- {constraint}" for constraint in constraints)
        return "\n".join(lines)
