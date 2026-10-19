# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Describe stable section anchors emitted by each knowledge parser.

Parsing supplies anchor evidence without resolving slots or assigning IDs."""

from __future__ import annotations

import pytest

from nlght.adapters.outbound.workflow.steps.knowledge.parsers import (
    _UnitBuilder,
    parse_asciidoc,
    parse_html,
    parse_markdown,
    parse_pdf,
    parse_prose,
)
from nlght.core.knowledge import AUTHORED_ANCHOR, DERIVED_ANCHOR, NO_ANCHOR


def _units(parse, text):
    builder = _UnitBuilder("s", "doc", {}, "rev")
    parse(text, builder)
    builder.flush()
    return builder.units


ADOC = """[[howto.actuator]]
= Actuator

Spring Boot includes the Spring Boot Actuator.

[[howto.actuator.customizing-sanitization]]
== Customizing Sanitization

Define a SanitizingFunction bean.
"""


def test_asciidoc_keeps_the_anchor_its_author_wrote() -> None:
    """Preserve authored AsciiDoc anchors across section parsing."""
    units = _units(parse_asciidoc, ADOC)

    assert [unit.anchor for unit in units] == [
        "howto.actuator",
        "howto.actuator.customizing-sanitization",
    ]
    assert all(unit.anchor_strength == AUTHORED_ANCHOR for unit in units)


def test_an_authored_anchor_is_not_normalised() -> None:
    # Taken exactly as written. Lower-casing or slugifying it would be inventing
    # a second identifier for something the source already named.
    units = _units(parse_asciidoc, "[[Howto.Actuator_v2]]\n= Actuator\n\nText.\n")

    assert units[0].anchor == "Howto.Actuator_v2"


def test_an_asciidoc_section_without_an_anchor_falls_back(
) -> None:
    units = _units(parse_asciidoc, "= Actuator\n\nText.\n")

    assert units[0].anchor_strength == DERIVED_ANCHOR


def test_html_takes_the_heading_id_when_there_is_one() -> None:
    units = _units(
        parse_html,
        '<h1 id="expenses">Expenses</h1><p>Above CHF 500, approval is required.</p>',
    )

    assert units[0].anchor == "expenses"
    assert units[0].anchor_strength == AUTHORED_ANCHOR


def test_html_invents_nothing_for_a_heading_without_an_id() -> None:
    """The temptation this exists to refuse.

    A generated identifier would look authored and behave like a position: it
    would move whenever the heading moved, and a slot built on it would be new
    on every run. The honest answer is the heading path, marked as derived.
    """
    units = _units(parse_html, "<h1>Expenses</h1><p>Above CHF 500, approval is required.</p>")

    assert units[0].anchor_strength == DERIVED_ANCHOR
    assert units[0].anchor == "Expenses"


MARKDOWN = """# Expenses

Above CHF 500, approval is required.

## Review

The limit is reviewed yearly.
"""


def test_markdown_derives_a_heading_path_and_says_it_is_derived() -> None:
    units = _units(parse_markdown, MARKDOWN)

    assert [unit.anchor for unit in units] == ["Expenses", "Expenses > Review"]
    assert all(unit.anchor_strength == DERIVED_ANCHOR for unit in units)


def test_a_derived_anchor_is_a_path_and_not_just_the_last_heading() -> None:
    # Two sections called "Overview" under different parents are two sections.
    # An anchor that was only the last heading would merge them.
    units = _units(
        parse_markdown,
        "# Billing\n\n## Overview\n\nA.\n\n# Shipping\n\n## Overview\n\nB.\n",
    )

    assert [unit.anchor for unit in units if unit.content in {"A.", "B."}] == [
        "Billing > Overview",
        "Shipping > Overview",
    ]


def test_renaming_a_heading_moves_a_derived_anchor() -> None:
    """Change a derived anchor when its heading is renamed."""
    before = _units(parse_markdown, "# Expenses\n\nAbove CHF 500.\n")
    after = _units(parse_markdown, "# Expense policy\n\nAbove CHF 500.\n")

    assert before[0].anchor != after[0].anchor
    assert before[0].content == after[0].content


def test_prose_has_no_anchor_and_says_so() -> None:
    units = _units(parse_prose, "Above CHF 500, approval is required.\n\nReviewed yearly.\n")

    assert units
    assert all(unit.anchor_strength == NO_ANCHOR for unit in units)
    assert all(unit.anchor == "" for unit in units)


def test_a_unit_without_an_anchor_may_not_carry_a_persisted_slot() -> None:
    """Deny persistent slots to units without a resolvable anchor."""
    prose = _units(parse_prose, "Above CHF 500, approval is required.\n")
    adoc = _units(parse_asciidoc, ADOC)

    assert prose[0].stable_slot_anchor is False
    assert adoc[0].stable_slot_anchor is True


@pytest.mark.parametrize(
    ("parser", "text", "strength", "stable"),
    [
        (parse_asciidoc, "[[a]]\n= T\n\nBody.\n", AUTHORED_ANCHOR, True),
        (parse_asciidoc, "= T\n\nBody.\n", DERIVED_ANCHOR, True),
        (parse_html, '<h1 id="a">T</h1><p>Body.</p>', AUTHORED_ANCHOR, True),
        (parse_html, "<h1>T</h1><p>Body.</p>", DERIVED_ANCHOR, True),
        (parse_markdown, "# T\n\nBody.\n", DERIVED_ANCHOR, True),
        (parse_prose, "Body.\n", NO_ANCHOR, False),
    ],
)
def test_the_capability_matrix(parser, text: str, strength: str, stable: bool) -> None:
    """What each format offers, asserted in one place rather than inferred.

    The registry will be handed `anchor`, `anchor_strength` and `document_id`
    and decide from those. It must never decide by looking at a file extension:
    an HTML page with ids and one without are different cases in the same
    format, and only the unit knows which it is.
    """
    unit = _units(parser, text)[0]

    assert unit.anchor_strength == strength
    assert unit.stable_slot_anchor is stable


def test_pdf_offers_nothing_either() -> None:
    pytest.importorskip("pdfminer.high_level")
    builder = _UnitBuilder("s", "doc", {}, "rev")

    with pytest.raises(Exception):  # noqa: B017 (not a PDF. The point is the contract)
        parse_pdf(b"not a pdf", builder)
