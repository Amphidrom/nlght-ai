# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Provenance survives every parser, as a contract rather than as a field list.

This exists because the same fault happened three times, and the guard written
after the second one did not catch the third.

    `document_id` was added and `knowledge.parse_auto` did not carry it
    `processing_revision_id` was added and `knowledge.parse_auto` did not carry it
    `document_path` was added and `knowledge.parse_auto` did not carry it

The first two produced seventeen assertions with no lineage at all. They were
fixed, `KnowledgeUnit` was given *fields* rather than a metadata dict — "a field
cannot be forgotten the way a dict key can" — and a regression test was written.
That test names the two fields that had failed:

    assert (lineage.document_id, lineage.document_revision) == ("doc-1", "rev-1")

So it was a test about `document_id` and `processing_revision_id`, not a test about
provenance. When a third field arrived it was green while half the corpus was
unnameable, and a live `--expect` run found it instead.

What follows is written so a **fourth** field cannot repeat this. Nothing here
names a provenance field: the contract is read off `KnowledgeUnit.provenance`, so
adding one turns these red until every parser carries it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from nlght.core.knowledge import KnowledgeUnit

_SRC = Path(__file__).resolve().parents[6] / "src" / "nlght"

#: The contract itself, read from the type rather than written down here. A field
#: added to `provenance` appears in this set on the next run, and every assertion
#: below starts failing until the pipeline carries it.
PROVENANCE = tuple(
    KnowledgeUnit(
        unit_ordinal="o", source_id="s", content="c",
        document_id="d", processing_revision_id="r", document_path="p",
    ).provenance
)


def _constructions(name: str) -> list[tuple[str, set[str]]]:
    """Every place in the source that builds `name`, with the keywords it passes."""
    found: list[tuple[str, set[str]]] = []
    for path in _SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            callee = node.func
            called = getattr(callee, "id", None) or getattr(callee, "attr", None)
            if called != name:
                continue
            found.append((
                path.relative_to(_SRC).as_posix(),
                {kw.arg for kw in node.keywords if kw.arg},
            ))
    return found


def test_the_contract_is_not_empty() -> None:
    # If `provenance` ever loses its fields this file would pass by testing
    # nothing, which is the failure mode it exists to prevent.
    assert len(PROVENANCE) >= 3


@pytest.mark.parametrize("field", PROVENANCE)
def test_every_unit_is_built_with_every_provenance_field(field: str) -> None:
    """No construction site may omit one.

    Parametrised over the contract, so a new field is checked at every site the
    moment it is added — which is precisely what did not happen for
    `document_path`, where one constructor was updated and the other was not.
    """
    sites = _constructions("KnowledgeUnit")

    assert sites, "no KnowledgeUnit construction found; this guard has gone blind"
    missing = [where for where, keywords in sites if field not in keywords]

    assert missing == [], (
        f"{field!r} is provenance and these build a unit without it: {missing}. "
        f"A parser that omits one leaves its claims unattributable, and the last "
        f"three times this happened nothing failed until a live run."
    )


def test_every_parser_reaches_a_builder_that_is_given_the_document() -> None:
    """The six parser steps share two constructors, and both must be complete.

    `knowledge.parse` builds units directly; the five in `parsers.py` — markdown,
    asciidoc, html, pdf and auto — all go through `_UnitBuilder`. So checking the
    two constructors covers all six, and covers a seventh parser written later
    that reuses either.
    """
    builders = _constructions("_UnitBuilder")

    assert builders, "no _UnitBuilder construction found; this guard has gone blind"
    # Positional or keyword, the builder has to be handed the document's own
    # path — it cannot derive one, and a builder that is not given it silently
    # produces units nothing can name.
    for where, _ in builders:
        assert "parsers.py" in where


@pytest.mark.parametrize("step", ["knowledge.parse", "knowledge.parse_auto"])
async def test_provenance_survives_into_the_lineage(step: str) -> None:
    """And it is carried, not merely constructed.

    The structural halves above say the pipeline *passes* every field. This says
    the value arrives — through the parser, through extraction's evidence dict,
    into the write the repository stores. Every provenance value has to be
    findable there; nothing is matched by name, so a field routed to the wrong
    attribute fails too.

    Two steps rather than six because the other four need their own formats; the
    constructor checks above are what cover those.
    """
    from nlght.adapters.outbound.workflow.registry import step_registry
    from nlght.adapters.outbound.workflow.steps.knowledge.persist import KnowledgePersistStep

    from .test_knowledge_extract_skip import _TEXT, _ctx, _document, _Llm, _step

    ctx = _ctx(_Llm())
    ctx.metadata["ingestion.processed"] = (_document(_TEXT),)
    await step_registry._registry[step](config={}).run(ctx)  # noqa: SLF001

    units = ctx.metadata["knowledge.units"]
    assert units, f"{step} produced no units"
    for unit in units:
        blank = [name for name, value in unit.provenance.items() if not str(value).strip()]
        assert blank == [], f"{step} left {blank} empty on a unit"

    await _step(None).run(ctx)
    item = ctx.metadata["knowledge.extracted"][0]

    # The extract boundary: evidence carries every provenance key.
    for name in PROVENANCE:
        assert str(item.evidence.get(name, "")).strip(), (
            f"{step}: evidence lost {name!r} between the unit and the candidate"
        )

    lineage, gap = KnowledgePersistStep._lineage(item, "run-1", "ver-1")  # noqa: SLF001

    assert gap is None, f"{step} left a candidate without {gap}"
    assert lineage is not None
    # Every value, wherever it was routed. Not by attribute name, so a field
    # wired to the wrong column is caught as surely as one that was dropped.
    carried = {
        str(getattr(lineage, name))
        for name in ("document_id", "document_revision", "document_path")
    }
    for name, value in units[0].provenance.items():
        assert str(value) in carried, (
            f"{step}: {name}={value!r} never reached the lineage write"
        )
