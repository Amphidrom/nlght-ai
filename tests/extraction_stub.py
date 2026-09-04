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
    """A sentence the prompt actually contains, for a stub to quote back.

    A candidate's wording has to occur in the unit it was extracted from
    (ADR-0048), so a stub answering one canned sentence for every unit models a
    model that misquotes — and the pipeline is right to reject it. Substituting
    `{quote}` in a scripted response keeps the stub honest about the one thing
    the extraction boundary now verifies.

    The extraction prompt ends with the unit under `TEXT:`, so the first
    non-empty line after it is a sentence the source really has. Quotes are
    flattened because the response the stub returns is JSON.
    """
    body = prompt.split("TEXT:", 1)[-1]
    for line in body.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped.replace('"', "'")
    return ""
