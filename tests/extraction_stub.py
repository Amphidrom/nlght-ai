# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Helpers for standing in for the extraction model.

Test-only, and deliberately not in `src`. Nothing in the platform needs to parse
its own prompt back apart; this exists because a *stub* has to behave like a
model that quotes correctly, and three test modules were each carrying their own
copy of the same eight lines.
"""

from __future__ import annotations


def quote_from(prompt: str) -> str:
    """Extract a quoted sentence from the prompt's source text.

The stub reads the first nonempty line after TEXT: and replaces double quotes
    for its JSON response.

    Args:
        prompt (str): Extraction prompt containing a TEXT section.

    Returns:
        str: A source line suitable for a scripted quotation, or an empty string."""
    body = prompt.split("TEXT:", 1)[-1]
    for line in body.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped.replace('"', "'")
    return ""
