# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Construct trusted deployment-authored system instructions."""

from __future__ import annotations

DEFAULT_CONSTRAINTS: list[str] = [
    "Never use placeholders like [value], [insert here], or similar if it wasn't asked explicitly.",
    "Never invent, estimate, or approximate values. If unsure, say so.",
]


class SystemPromptBuilder:
    """Build only deployment-trusted instructions."""

    def build(
        self,
        ego: str,
        constraints: list[str] | None = None,
    ) -> str:
        """Render trusted instructions without retrieved knowledge.

        Args:
            ego (str): Trusted system identity text.
            constraints (list[str] | None): Optional trusted instruction constraints.

        Returns:
            str: System-instruction text.
        """

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
