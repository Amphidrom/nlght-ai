# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""The wording is observed, never assembled — and that holds for every kind.

    observed_text  is authoritative for the wording
    structure      is never authoritative for the wording

Structure may enrich a claim. It may not restate one. A claim is quoted back to
a person — in a review queue, in a citation, in a diff — and quoting it from the
fields shows them a sentence their source does not contain.

Parametrised across `fact`, `rule`, `decision` and `pattern` on purpose. The
defect was easiest to see on a fact, where `subject`/`predicate`/`object` cannot
even attempt a sentence and the old code joined them into "requires spring_boot
java_17". But it was never a fact problem: `rule` and `decision` carry prose in
their fields, and reassembling *from those* is the same substitution, only
harder to notice because the result reads well.
"""

from __future__ import annotations

import pytest

from nlght.adapters.outbound.workflow.steps.knowledge.persist import _assertion_text
from nlght.core.knowledge import ExtractedItem

#: One case per kind: the sentence a source really contains, and structured
#: fields whose values are deliberately *not* that sentence.
_CASES = {
    "fact": (
        "Spring Boot requires Java 17.",
        {"subject": "spring_boot", "predicate": "requires", "object": "java_17"},
    ),
    "rule": (
        "If auditing is enabled, a retention period must be configured.",
        {
            "subject": "auditing",
            "rule_property": "retention_requirement",
            "rule_text": "a retention period must be configured",
        },
    ),
    "decision": (
        "We will use PostgreSQL for persistence.",
        {
            "subject": "persistence",
            "decision_type": "datastore_choice",
            "decision": "PostgreSQL",
            "effect": "used for persistence",
        },
    ),
    "pattern": (
        "Health statuses are exported as metrics so a dashboard can chart them.",
        {"pattern_name": "health_as_metrics", "description": "export statuses as metrics"},
    ),
}

_KINDS = sorted(_CASES)


def _item(kind: str, *, shuffled: bool = False) -> ExtractedItem:
    wording, content = _CASES[kind]
    if shuffled:
        content = dict(reversed(list(content.items())))
    return ExtractedItem(
        kind=kind, type="t", content=content, confidence=0.9, observed_text=wording
    )


@pytest.mark.parametrize("kind", _KINDS)
def test_the_wording_survives_exactly(kind: str) -> None:
    """In, then out, character for character."""
    wording, _ = _CASES[kind]

    assert _assertion_text(_item(kind)) == wording


@pytest.mark.parametrize("kind", _KINDS)
def test_the_wording_does_not_depend_on_how_the_fields_are_ordered(kind: str) -> None:
    """Structure is enrichment, so rearranging it changes nothing about the claim.

    A model emits its JSON keys in whatever order it likes, and a text derived
    from `content.values()` moved with that order. Two runs over one unedited
    sentence then quoted it two different ways.
    """
    assert _assertion_text(_item(kind)) == _assertion_text(_item(kind, shuffled=True))


@pytest.mark.parametrize("kind", _KINDS)
def test_no_wording_is_no_wording_rather_than_an_assembled_one(kind: str) -> None:
    """The substitution this exists to prevent.

    `_assertion_text` used to pick the first non-empty of
    `rule_text`/`decision`/`description`/`object` and otherwise join every value.
    For a candidate with no observed wording that produced a sentence out of the
    fields — well-formed, quotable, and in no document. Empty is the honest
    answer, and the persist step counts it.
    """
    _, content = _CASES[kind]
    item = ExtractedItem(kind=kind, type="t", content=content, confidence=0.9)

    assert _assertion_text(item) == ""


@pytest.mark.parametrize("kind", _KINDS)
def test_wording_that_is_not_in_the_source_is_not_observed(kind: str) -> None:
    """Verified, not trusted.

    A field the model fills in is another model-generated string. One stored
    under the name "observed" while holding a paraphrase is worse than no
    wording at all, because everything downstream quotes it as if a person could
    find it in the document.
    """
    wording, content = _CASES[kind]
    source = f"Some heading\n\n{wording}\n\nAnd another sentence.\n"

    assert _item(kind).verified_against(source)
    # A paraphrase of the same claim is not the same wording.
    reworded = ExtractedItem(
        kind=kind, type="t", content=content, confidence=0.9,
        observed_text=wording.replace(".", " in production."),
    )
    assert not reworded.verified_against(source)


@pytest.mark.parametrize("kind", _KINDS)
def test_rewrapping_a_line_is_not_a_different_wording(kind: str) -> None:
    # A parser may reflow; that changes no word. Folding whitespace is the only
    # tolerance — anything fuzzier would admit the paraphrase above.
    wording, _ = _CASES[kind]
    head, _, tail = wording.partition(" ")

    assert _item(kind).verified_against(f"{head}\n   {tail}")


def test_a_fact_is_the_case_that_cannot_be_faked() -> None:
    """Named on its own because it is where the rule is unarguable.

    `subject`/`predicate`/`object` are not a sentence in any order. The old join
    produced "spring_boot requires java_17", which is not what the document says
    and never was — so for a fact there is no version of "derive it from the
    fields" that is merely imperfect.
    """
    item = _item("fact")

    assert _assertion_text(item) == "Spring Boot requires Java 17."
    assert _assertion_text(item) not in {
        " ".join(item.content.values()),
        " ".join(sorted(item.content.values())),
    }


# ---------------------------------------------------------------------------
# The rule, as a property of the code rather than of one call
# ---------------------------------------------------------------------------

def test_the_citation_text_is_read_and_not_assembled() -> None:
    """Rule 3, as a property of the function rather than of one call.

    `_assertion_text` is what a reviewer, a citation and a diff all quote. It
    used to pick the first non-empty of four fields and otherwise join every
    value, and each half looked reasonable on its own — what made it a defect is
    that a reader cannot tell an assembled sentence from an observed one.

    Asserted over the source because the fallback is what would come back: a
    later "a candidate with no wording should still show something" would
    reintroduce exactly this, and would pass every behavioural test above by
    firing only where they do not look.
    """
    import inspect  # noqa: PLC0415

    from nlght.adapters.outbound.workflow.steps.knowledge.persist import (  # noqa: PLC0415
        _assertion_text,
    )

    body = inspect.getsource(_assertion_text)
    code = body.split(chr(34) * 3)[-1]

    assert "observed_text" in body
    # One value, read out. No join, and no walk over the structured fields.
    assert ".join(" not in code
    assert "content" not in code


def test_the_only_field_assembled_from_values_is_the_search_bag() -> None:
    """Rules 3 and 5 together: a bag is allowed, and a bag is not a claim.

    Two places build a string out of a mapping's values — the writer's fallback
    and the enrichment step — and both are the BM25 bag. Named here rather than
    exempted silently, so a third has to be argued for instead of appearing.
    """
    import inspect  # noqa: PLC0415

    from nlght.adapters.outbound.stores import knowledge_writer  # noqa: PLC0415
    from nlght.adapters.outbound.workflow.steps.knowledge import (  # noqa: PLC0415
        persist,
        refine,
    )

    def assembling(module: object) -> list[tuple[str, str]]:
        """Every line that builds a string out of a mapping, with what encloses it.

        The enclosing name is the point: a line reads the same whether it makes a
        search bag or a citation, and only where it sits says which.
        """
        found, enclosing = [], ""
        for line in inspect.getsource(module).splitlines():
            stripped = line.strip()
            if stripped.startswith("def "):
                enclosing = stripped[4:].split("(")[0]
            if ".join(" in stripped and (
                "values()" in stripped or "for name in sorted(" in stripped
            ):
                found.append((enclosing, stripped))
        return found

    # The step that turns candidates into writes assembles no text at all.
    assert [
        line for _, line in assembling(persist) if "text" in line.lower()
    ] == []

    for module in (refine, knowledge_writer):
        for enclosing, line in assembling(module):
            is_bag = "search_bag" in enclosing or "text_all" in line or "text =" in line
            measures_only = "len(" in line
            assert is_bag or measures_only, (
                f"{module.__name__}.{enclosing} assembles a string from fields "
                f"without saying it is a search bag: {line}"
            )
