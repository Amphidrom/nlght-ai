# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Verify retrieval metadata admits textual terms only."""

from nlght.core.retrieval import as_fields


def test_as_fields_excludes_nontext_sequence_members() -> None:
    """Exclude numeric, byte, and empty metadata from searchable terms."""
    fields = as_fields({
        "title": "Guide",
        "tags": ["alpha", 42, b"beta", "", "gamma"],
        "bytes": b"raw",
        "numbers": [1, 2],
    })

    assert fields == {"title": ("Guide",), "tags": ("alpha", "gamma")}
