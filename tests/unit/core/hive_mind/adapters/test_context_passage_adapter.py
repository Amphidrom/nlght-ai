# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""ContextPassage → MentalElement — the last retrieval-specific code there is.

After this adapter a passage is an element like any other and is weighed against a
remembered fact by the same rules. Everything retrieval knows about itself stops
here.
"""

from __future__ import annotations

from nlght.core.context.elements import SECTION_RETRIEVED, passage_elements
from nlght.core.context.passage import ContextPassage
from nlght.core.hive_mind.models import Level, Retention
from nlght.core.retrieval import DOCUMENT, LEXICAL, VECTOR, Provenance

from .conftest import assert_element_contract, assert_no_invented_short_form


def _passage(text: str, rank: int = 1, provenance: Provenance | None = None) -> ContextPassage:
    return ContextPassage(
        text=text, carrier=DOCUMENT, carrier_id=f"d{rank}",
        found_by=(LEXICAL,), rank=rank,
        provenance=provenance or Provenance(),
    )


def test_the_observed_wording_is_carried_unchanged() -> None:
    # ADR-0048 all the way through: the sentence a citation quotes is the one the
    # source said. The adapter may add a bullet for the prompt; it may not touch
    # the words.
    element = passage_elements([_passage("The value is limited to 100 characters.")])[0]

    assert "The value is limited to 100 characters." in element.at(Level.FULL).text


def test_the_fusion_rank_becomes_relevance() -> None:
    first, second, third = passage_elements([
        _passage("a", rank=1), _passage("b", rank=2), _passage("c", rank=3),
    ])

    assert first.relevance > second.relevance > third.relevance


def test_the_rank_never_becomes_retention() -> None:
    """The two questions the design keeps apart.

    Rank says how well this matched the question among what was found. Retention
    says what losing it would cost. A passage that ranked first is well matched,
    not important — and if rank raised retention it would start outranking things
    that genuinely must not be lost.
    """
    first, last = passage_elements([_passage("a", rank=1), _passage("b", rank=2)])

    assert first.retention is last.retention is Retention.USEFUL


def test_the_support_provenance_survives_whole() -> None:
    from nlght.core.knowledge import SupportingEvidence

    trail = Provenance(
        assertion_id="a1",
        processing_revision_id="r1",
        support=(SupportingEvidence(
            document_id="doc-a", observed_document_revision="rev-3",
            document_path="policies/expenses.adoc", slot_id="slot-1",
        ),),
    )

    element = passage_elements([_passage("the claim", provenance=trail)])[0]

    assert element.provenance is trail
    assert element.provenance.support[0].document_path == "policies/expenses.adoc"


def test_a_passage_belongs_to_its_own_section() -> None:
    element = passage_elements([_passage("x")])[0]

    assert element.presentation.section == SECTION_RETRIEVED.section


def test_two_stores_finding_one_passage_stay_one_element() -> None:
    # The fusion already decided that; the adapter must not undo it by making an
    # element per finder.
    passage = ContextPassage(
        text="found twice", carrier=DOCUMENT, carrier_id="d1",
        found_by=(LEXICAL, VECTOR), rank=1,
    )

    assert len(passage_elements([passage])) == 1


def test_the_element_contract_holds() -> None:
    for element in passage_elements([_passage("a", rank=1), _passage("b", rank=2)]):
        assert_element_contract(element)
        assert_no_invented_short_form(element)
