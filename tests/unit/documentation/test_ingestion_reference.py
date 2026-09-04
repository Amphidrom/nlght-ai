# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import re
import uuid
from pathlib import Path

from seed_test_pipelines import _knowledge_tail

from nlght.adapters.outbound.workflow.registry import step_registry

REFERENCE = Path(__file__).parents[3] / "docs/documentation/ingestion/step-reference.mdx"
START = "{/* knowledge-refinement-order:start */}"
END = "{/* knowledge-refinement-order:end */}"


def test_canonical_knowledge_refinement_order_contains_only_registered_steps() -> None:
    text = REFERENCE.read_text(encoding="utf-8")
    assert text.count(START) == 1
    assert text.count(END) == 1
    order = text.split(START, 1)[1].split(END, 1)[0]
    named_steps = re.findall(r"`(knowledge\.[a-z_]+)`", order)

    assert named_steps
    assert len(named_steps) == len(set(named_steps))
    assert set(named_steps) <= set(step_registry._registry)

    seeded_types = [
        step.type
        for step in _knowledge_tail(uuid.UUID(int=0), "docs", "provider", "model")
    ]
    seeded_refinement = seeded_types[
        seeded_types.index("knowledge.canonicalize"):
        seeded_types.index("knowledge.review_flag")
    ]
    assert named_steps == seeded_refinement
