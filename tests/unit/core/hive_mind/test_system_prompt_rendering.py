# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Render model context deterministically from a reduced view.

The renderer must preserve supplied wording without adding knowledge or
mutating its inputs.
"""

from __future__ import annotations

from datetime import UTC, datetime

from nlght.core.hive_mind.model_context import ModelContextBuilder, PromptInputPolicy
from nlght.core.hive_mind.models import (
    Level,
    MentalElement,
    MentalModel,
    Presentation,
    Representation,
    Retention,
)
from nlght.core.hive_mind.relevance import reduce_to_budget


def _element(element_id: str, text: str, section: str, order: int,
             kind: str = "memory") -> MentalElement:
    return MentalElement(
        element_id=element_id,
        kind=kind,
        representations=(
            Representation(level=Level.FULL, text=text, cost=len(text) // 4 or 1),
            Representation(level=Level.OMIT, text="", cost=0),
        ),
        presentation=Presentation(section, order=order),
        retention=Retention.USEFUL,
    )


def _render(elements, *, ego: str = "Ego.", budget: int = 10_000) -> str:
    del ego
    records = ModelContextBuilder()._build_records(
        mental_model=MentalModel(turn_id="t", built_at=datetime.now(UTC)),
        policy=PromptInputPolicy(),
        view=reduce_to_budget(elements, budget=budget),
        slots=None,
    )
    lines: list[str] = []
    seen_sections: set[str] = set()
    for record in records:
        if record.section and record.section not in seen_sections:
            lines.append(record.section)
            seen_sections.add(record.section)
        lines.append(record.content)
    return "\n".join(lines)


def test_rendering_twice_gives_byte_identical_output() -> None:
    elements = [_element("a", "  - one", "First:", 10),
                _element("b", "  - two", "Second:", 20)]

    assert _render(elements) == _render(elements)


def test_rendering_does_not_mutate_what_it_was_given() -> None:
    elements = [_element("a", "  - one", "First:", 10)]
    view = reduce_to_budget(elements, budget=100)
    before = (view.kept, view.forgotten, view.cost, elements[0].representations)

    ModelContextBuilder()._build_records(
        mental_model=MentalModel(turn_id="t", built_at=datetime.now(UTC)),
        policy=PromptInputPolicy(), view=view, slots=None,
    )

    assert (view.kept, view.forgotten, view.cost, elements[0].representations) == before


def test_the_text_of_a_representation_is_never_altered() -> None:
    # Whatever the adapter wrote is what appears. A renderer that trimmed or
    # rewrapped would make a quotation stop matching its source.
    odd = "  - text with  double  spaces and a trailing tab\t"
    prompt = _render([_element("a", odd, "Section:", 10)])

    assert odd in prompt


def test_grouping_follows_presentation_and_nothing_else() -> None:
    """Group rendered elements by presentation metadata rather than kind."""
    prompt = _render([
        _element("a", "  - from memory", "Shared:", 10, kind="result"),
        _element("b", "  - from a corpus", "Shared:", 10, kind="passage"),
        _element("c", "  - elsewhere", "Other:", 20, kind="result"),
    ])

    assert prompt.index("Shared:") < prompt.index("Other:")
    assert prompt.count("Shared:") == 1, "one heading, both elements under it"
    assert "  - from memory" in prompt and "  - from a corpus" in prompt


def test_sections_appear_in_their_stated_order_whatever_order_they_arrive_in() -> None:
    forwards = _render([
        _element("a", "  - one", "First:", 10),
        _element("b", "  - two", "Second:", 20),
    ])
    backwards = _render([
        _element("b", "  - two", "Second:", 20),
        _element("a", "  - one", "First:", 10),
    ])

    assert forwards.index("First:") < forwards.index("Second:")
    assert backwards.index("First:") < backwards.index("Second:")


def test_a_multi_line_entry_is_kept_byte_identical() -> None:
    # Context content stays opaque. Provider serialization escapes it later.
    prompt = _render([_element("a", "  - first line\nsecond line", "Section:", 10)])

    assert "  - first line\nsecond line" in prompt


def test_an_omitted_element_leaves_no_trace() -> None:
    prompt = _render(
        [_element("a", "  - kept", "Section:", 10),
         _element("b", "  - dropped", "Section:", 10)],
        budget=1,
    )

    assert "dropped" not in prompt


def test_an_empty_view_renders_the_framing_alone() -> None:
    prompt = _render([], ego="You are precise.")

    assert prompt == ""
