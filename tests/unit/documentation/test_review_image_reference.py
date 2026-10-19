# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Keep the knowledge review page aligned with the distributed image.

The source is absent from the public snapshot, so the page must not direct
readers to build it locally."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).parents[3]
REFERENCE = ROOT / "docs/documentation/ingestion/knowledge-review.mdx"
EXCLUDE = ROOT / "scripts/release/public-exclude.txt"


def test_the_review_source_is_kept_out_of_the_public_snapshot() -> None:
    # The page tells a reader to pull an image because the source is not there
    # to build. If `review` ever returns to the snapshot that sentence is wrong.
    stripped = [
        line.strip()
        for line in EXCLUDE.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    assert "review" in stripped


def test_the_page_does_not_offer_a_build_a_reader_cannot_run() -> None:
    text = REFERENCE.read_text(encoding="utf-8")

    # `review/Dockerfile` may be named because it defines the image build, and
    # somebody with the development repository can use it. What it must not be
    # is the instruction inside a shell block a public reader is meant to paste.
    for block in re.findall(r"```bash\n(.*?)```", text, re.S):
        assert "docker build" not in block, (
            "the page pastes a docker build for a path that is not in the public repository"
        )
