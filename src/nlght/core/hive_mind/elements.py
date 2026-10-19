# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Convert remembered state into budgetable mental elements."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

from nlght.core.hive_mind.models import (
    AtomType,
    Level,
    MentalElement,
    MentalModel,
    Presentation,
    Representation,
    Retention,
    estimate_tokens,
)
from nlght.core.model.trust import UNANNOTATED, ContentOrigin

#: The headings, and the order they appear in. Taken verbatim from what the
#: builder produced before, because this slice removes a coupling and must not
#: also move the furniture: whether the order is right is a separate question,
#: asked once the prompt is not also being restructured.
SECTION_CONVERSATION = Presentation("Recent conversation:", order=10)
SECTION_TASK_CONTEXT = Presentation("Active task context:", order=20)
SECTION_USER_FACTS = Presentation("Known facts about the user:", order=30)
SECTION_TASK_RESULTS = Presentation("Task results:", order=40)
SECTION_PLANNING = Presentation("Planning context:", order=50)

#: Which working-memory atoms read as context rather than as planning. A
#: judgement about what an atom *is*, which is why it lives here and no longer in
#: the renderer.
READABLE_ATOM_TYPES = frozenset(
    {AtomType.CONTEXT, AtomType.RESULT, AtomType.EVAL, AtomType.ERROR}
)

#: The tag that makes a session result a fact about the person.
USER_FACT_TAG = "user_fact"


def _atom_representations(
    text: str, *, key: str, kind: str
) -> tuple[Representation, ...]:
    """Offer full and derived compact forms for an atom.

    Args:
        text (str): Full atom text.
        key (str): Stable key used in compact form.
        kind (str): Kind label for the atom.

    Returns:
        tuple[Representation, ...]: Full and compact representations, ordered by completeness.
    """
    full = Representation(level=Level.FULL, text=text, cost=estimate_tokens(text))
    omit = Representation(level=Level.OMIT, text="", cost=0)
    if not key:
        return (full, omit)

    label = f"[{kind}] " if kind else ""
    brief = f"  - {label}candidate:{key}"
    short = Representation(
        level=Level.COMPACT, text=brief, cost=estimate_tokens(brief), derived=True,
    )
    if short.cost >= full.cost:
        # Not a shortening. Offering it would let a reduction step spend a move
        # and save nothing, and `MentalElement` would refuse the element
        # outright, which is the guarantee working as intended.
        return (full, omit)
    return (full, short, omit)


def _spoken(
    element_id: str,
    kind: str,
    text: str,
    *,
    presentation: Presentation,
    retention: Retention,
    relevance: float,
    payload: object = None,
    provenance: object = None,
) -> MentalElement:
    """Wrap text that supports full display or omission.

    Args:
        element_id (str): Stable element identity.
        kind (str): Kind of spoken element.
        text (str): Text available for full display.
        presentation (Presentation): Display policy for the element.
        retention (Retention): Retention policy for budget reduction.
        relevance (float): Current relevance score.
        payload (object): Original content behind the element.
        provenance (object): Source information for the content.

    Returns:
        MentalElement: Mental element with full display and omission options.
    """
    return MentalElement(
        element_id=element_id,
        kind=kind,
        representations=(
            Representation(level=Level.FULL, text=text, cost=estimate_tokens(text)),
            Representation(level=Level.OMIT, text="", cost=0),
        ),
        presentation=presentation,
        retention=retention,
        relevance=relevance,
        payload=payload,
        provenance=provenance,
    )


def remembered_origin(mental_model: MentalModel) -> ContentOrigin:
    """Combine origins carried by remembered model content.

    Args:
        mental_model (MentalModel): MentalModel with turns, atoms, and results.

    Returns:
        ContentOrigin: Combined ContentOrigin.
    """
    stamped = [
        *(turn.origin for turn in mental_model.recent_turns),
        *(scored.result.origin for scored in mental_model.known_results),
        *(scored.atom.origin for scored in mental_model.active_atoms),
    ]
    origins = [origin if origin is not None else UNANNOTATED for origin in stamped]
    return ContentOrigin.combine(*origins) if origins else ContentOrigin()


def turn_elements(mental_model: MentalModel) -> list[MentalElement]:
    """Convert recent conversation turns to mental elements.

    Args:
        mental_model (MentalModel): MentalModel conversation state.

    Returns:
        list[MentalElement]: Turn elements with origin and cost.
    """
    elements: list[MentalElement] = []
    for turn in mental_model.recent_turns[-3:]:
        entry = f"  [{turn.turn_nr}]"
        if turn.user_input:
            entry += f" user: {turn.user_input}"
        if turn.result_summary:
            entry += f" -> {turn.result_summary}"
        elements.append(_spoken(
            f"turn:{turn.turn_nr}",
            "turn",
            entry,
            presentation=SECTION_CONVERSATION,
            # A conversation is the frame a question is asked in. Losing the
            # middle of it silently is worse than saying less about the corpus.
            retention=Retention.IMPORTANT,
            relevance=float(turn.turn_nr),
            payload=turn,
        ))
    return elements


def _atom_element(
    element_id: str,
    text: str,
    *,
    key: str,
    kind: str,
    presentation: Presentation,
    relevance: float,
    payload: object,
) -> MentalElement:
    """Convert one working atom to a budgetable element.

    Args:
        element_id (str): Stable element identity.
        text (str): Text of the working atom.
        key (str): Atom key for compact display.
        kind (str): Kind label for the atom.
        presentation (Presentation): Display policy for the element.
        relevance (float): Current relevance score.
        payload (object): Original atom content.

    Returns:
        MentalElement: Mental element offering atom representations.
    """
    return MentalElement(
        element_id=element_id,
        kind="atom",
        representations=_atom_representations(text, key=key, kind=kind),
        presentation=presentation,
        retention=Retention.USEFUL,
        relevance=relevance,
        payload=payload,
    )


def context_atom_elements(
    mental_model: MentalModel,
    *,
    tags: Sequence[str] | None = None,
    resolve_references: Callable[[list[str]], list[str]] | None = None,
) -> list[MentalElement]:
    """Select working atoms that read as context.

    Args:
        mental_model (MentalModel): Working memory to select from.
        tags (Sequence[str] | None): Optional tag filter.
        resolve_references (Callable[[list[str]], list[str]] | None): Optional resolver
            for referenced labels.

    Returns:
        list[MentalElement]: Working atoms selected for context display.
    """
    pool = mental_model.active_atoms if tags else mental_model.top_atoms(10)
    readable = [
        scored for scored in pool
        if scored.atom.atom_type in READABLE_ATOM_TYPES
        and _matches(scored.atom.tags, tags)
    ]
    entries = [
        f"  - {render_atom(s.atom.atom_type, s.atom.content, s.atom.tags)}"
        for s in readable
    ]
    scores = [s.score.total for s in readable]
    if resolve_references is not None and entries:
        resolved = resolve_references(entries)
        if resolved:
            entries = resolved

    elements = [
        _atom_element(
            f"atom:context:{position}",
            entry,
            key=readable[position].atom.key if position < len(readable) else "",
            kind=readable[position].atom.kind if position < len(readable) else "",
            presentation=SECTION_TASK_CONTEXT,
            relevance=scores[position] if position < len(scores) else (
                scores[-1] if scores else 0.0
            ),
            payload=readable[position].atom if position < len(readable) else None,
        )
        for position, entry in enumerate(entries)
    ]

    return elements


def planning_atom_elements(mental_model: MentalModel) -> list[MentalElement]:
    """Select working atoms that read as planning.

    Args:
        mental_model (MentalModel): MentalModel working memory.

    Returns:
        list[MentalElement]: Planning atom elements.
    """
    planning = [
        scored for scored in mental_model.top_atoms(10)
        if scored.atom.atom_type not in READABLE_ATOM_TYPES
    ]
    return [
        _atom_element(
            f"atom:planning:{position}",
            f"  - {render_atom(s.atom.atom_type, s.atom.content, s.atom.tags)}",
            key=s.atom.key,
            kind=s.atom.kind,
            presentation=SECTION_PLANNING,
            relevance=s.score.total,
            payload=s.atom,
        )
        for position, s in enumerate(planning)
    ]




def result_elements(mental_model: MentalModel) -> list[MentalElement]:
    """Convert session results into budgetable elements.

    Args:
        mental_model (MentalModel): MentalModel session results.

    Returns:
        list[MentalElement]: Result elements with origin and retention.
    """
    elements: list[MentalElement] = []
    for position, scored in enumerate(mental_model.known_results):
        if USER_FACT_TAG not in (scored.result.tags or []):
            continue
        elements.append(_spoken(
            f"result:fact:{position}",
            "result",
            f"  - {scored.result.content}",
            presentation=SECTION_USER_FACTS,
            retention=Retention.IMPORTANT,
            relevance=scored.score.total,
            payload=scored.result,
        ))
    for position, scored in enumerate(mental_model.top_results(10)):
        if USER_FACT_TAG in (scored.result.tags or []):
            continue
        elements.append(_spoken(
            f"result:task:{position}",
            "result",
            f"  - {scored.result.content}",
            presentation=SECTION_TASK_RESULTS,
            retention=Retention.USEFUL,
            relevance=scored.score.total,
            payload=scored.result,
        ))
    return elements


def render_atom(atom_type: str, content: str, tags: list[str]) -> str:
    """Render an atom line with its kind tag once.

    Args:
        atom_type (str): Kind label for the atom.
        content (str): Atom text to show.
        tags (list[str]): Tags attached to the atom.

    Returns:
        str: Visible atom text with its kind tag once.
    """
    prefix = f"[{atom_type}]"
    kind_tag = next(
        (
            str(tag).strip()
            for tag in (tags or [])
            if str(tag).strip().startswith("kind:")
        ),
        "",
    )
    if kind_tag:
        prefix += f" [{kind_tag}]"
    clean = re.sub(r"^\[(kind:[^\]]+)\]\s*\|?\s*", "", str(content)).strip()
    return f"{prefix} {clean}"


def _matches(atom_tags: list[str], required: Sequence[str] | None) -> bool:
    if not required:
        return True
    if not atom_tags:
        return False
    have = {str(tag).strip() for tag in atom_tags if str(tag).strip()}
    want = {str(tag).strip() for tag in required if str(tag).strip()}
    return True if not want else bool(have & want)
