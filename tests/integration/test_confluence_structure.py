# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Keep Confluence source markup available to knowledge parsers.

Acquisition retains the source form while search text remains a separate
representation. Parsing uses the source form to preserve document structure.
"""

from __future__ import annotations

import pytest

from nlght.adapters.outbound.ingestion.confluence_source import (
    ConfluenceDocumentSource,
    ConfluenceResponse,
    ConfluenceSourceSettings,
)
from nlght.adapters.outbound.workflow.steps.knowledge.parsers import (
    _UnitBuilder,
    parse_html,
)
from nlght.core.knowledge import AUTHORED_ANCHOR

pytestmark = pytest.mark.integration

PAGE = (
    '<h1 id="expenses">Expenses</h1>'
    "<p>Expenses above CHF 500 require approval.</p>"
    '<h2 id="threshold-review">Threshold review</h2>'
    "<p>The limit is reviewed yearly.</p>"
)


class _Transport:
    def __init__(self, responses: list[ConfluenceResponse]) -> None:
        self._responses = list(responses)

    async def search(self, *, base_url, params, authorization, timeout_seconds):
        return self._responses.pop(0)


def _source() -> ConfluenceDocumentSource:
    transport = _Transport(
        [
            ConfluenceResponse(
                200,
                {},
                {
                    "results": [
                        {
                            "id": "10",
                            "title": "Expenses",
                            "status": "current",
                            "body": {"storage": {"value": PAGE}},
                            "version": {"number": 3},
                            "_links": {"webui": "/spaces/ENG/pages/10"},
                        }
                    ]
                },
            )
        ]
    )
    return ConfluenceDocumentSource(
        ConfluenceSourceSettings(
            source_id="wiki",
            base_url="https://docs.example/wiki",
            space_keys=("ENG",),
            user_email="a@example.com",
            api_token="t",
            page_size=10,
            max_pages=1,
            timeout_seconds=3.0,
            retry_base_seconds=0.0,
            retry_max_seconds=0.0,
        ),
        transport,
    )


async def test_a_confluence_page_produces_knowledge_units() -> None:
    """The defect, measured. It produced zero."""
    snapshot = await _source().acquire()
    document = snapshot.documents[0]

    builder = _UnitBuilder("wiki", document.document_id, {}, document.source_revision_id)
    parse_html(document.source_form.decode(), builder)
    builder.flush()

    assert builder.units, "a page with two sections produced nothing"
    assert len(builder.units) == 2


async def test_a_heading_id_reaches_the_unit_that_needs_it() -> None:
    """The half a units-only fix would have quietly dropped.

    A heading's `id` is authored, stable across a rename, and exactly what slot
    identity resolves against. Recovering the units without it would mean doing
    this again.
    """
    snapshot = await _source().acquire()
    document = snapshot.documents[0]

    builder = _UnitBuilder("wiki", document.document_id, {}, document.source_revision_id)
    parse_html(document.source_form.decode(), builder)
    builder.flush()

    assert [unit.anchor for unit in builder.units] == ["expenses", "threshold-review"]
    # A field, not a metadata key: authored, and marked as such, so the registry
    # never has to guess how much an anchor is worth.
    assert all(unit.anchor_strength == AUTHORED_ANCHOR for unit in builder.units)
    assert all(unit.stable_slot_anchor for unit in builder.units)


async def test_the_search_text_is_left_exactly_as_it_was() -> None:
    """Keep search text unchanged while retaining the source form separately."""
    snapshot = await _source().acquire()
    document = snapshot.documents[0]

    assert document.content.decode() == (
        "Expenses\nExpenses above CHF 500 require approval.\n"
        "Threshold review\nThe limit is reviewed yearly."
    )
    assert b"<h1" in document.source_form
