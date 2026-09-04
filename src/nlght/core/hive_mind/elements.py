# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Turning what the session remembers into elements a budget can weigh.

This is where domain knowledge is allowed to live, and it is the only place. An
adapter knows that a session result tagged `user_fact` is a fact about the person
rather than a task outcome, that a `SPEC` atom is planning rather than context,
and that a turn summary is three fields that read as one line. It says so once,
as a `Presentation`, and the renderer downstream never asks again.

That division is the whole point (ADR-0054). Before this, the prompt builder made
those judgements inline:

    "user_fact" in sr.result.tags        →  a different heading
    atom.atom_type in _READABLE_ATOM_TYPES  →  a different heading
    known_results                        →  the only thing a budget could drop

so both the format and the budget policy were decided by which store a thing came
out of. Two elements identical in what they say and where they belong now produce
the same prompt whatever they were made of, and that is asserted directly.

**Everything that changes content, count or meaning happens here**, before an
element exists — reference resolution included. Once an element has a cost, that
cost has to be the truth about what rendering it will produce, or the budget
guarantee is a fiction.
"""

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
    """What an atom can be said as, fullest first.

    The first honest `COMPACT` in the platform, and honest is the whole
    difficulty. It is not a truncation of the full text — that ends mid-claim and
    would read as a quotation of something the source never finished saying. It
    is built from fields the atom *has*: the stable short name of what it is
    about, and what sort of thing it is.

        FULL     [SPEC] candidate:ssl-keystore-path | the keystore lives at …
        COMPACT  [SPEC] [fact] candidate:ssl-keystore-path
        OMIT

    So it is the same information at a lower resolution — the subject without the
    substance — rather than a new interpretation of it. It is marked `derived`,
    because it is not a wording any source produced.

    **No compact form where there is nothing to build one from**, and none where
    it would not actually be cheaper: an atom whose content is already shorter
    than its own name offers `FULL` and `OMIT` only, and the reduction copes
    because levels are optional. Inventing one to keep the shapes uniform is how
    a compact representation stops being trustworthy.
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
        # and save nothing — and `MentalElement` would refuse the element
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
    """One element that can be said in full or not at all.

    No `COMPACT` level: none of these sorts has a shorter form anybody has
    written. Offering one would mean inventing text, which is the one thing the
    reduction is not allowed to do and an adapter should not do on its behalf
    either. A sort that gains a genuine short form gains a level here, and the
    reduction starts using it without changing.
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


def turn_elements(mental_model: MentalModel) -> list[MentalElement]:
    """The recent conversation, as the prompt has always shown it.

    The last three turns, and the slice is kept: it is what the corpus has been
    saying, and changing how much conversation a model sees is a prompt decision,
    not a consequence of removing a coupling.
    """
    elements: list[MentalElement] = []
    for turn in mental_model.recent_turns[-3:]:
        entry = f"  [{turn.turn_nr}]"
        if turn.user_input:
            entry += f" user: {turn.user_input}"
        if turn.result_summary:
            entry += f" → {turn.result_summary}"
        elements.append(_spoken(
            f"turn:{turn.turn_nr}",
            "turn",
            entry,
            presentation=SECTION_CONVERSATION,
            # A conversation is the frame a question is asked in; losing the
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
    """One atom, offering every form it honestly has."""
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
    """Working memory, split into what reads as context and what reads as planning.

    The split is a judgement about the atoms and is made here. The renderer sees
    two sections and no reason for them.

    `resolve_references` runs **before** any element exists, and that placement is
    the point rather than a convenience: it may return a different number of
    entries with different text, so applying it afterwards would make every cost
    already computed a lie and the budget guarantee with it. Where it changes the
    count, entries beyond the pool inherit the last score — a positional rule,
    stated because it is a rule and not a derivation. Nothing supplies a resolver
    today.
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
    """Working memory that reads as planning rather than as context.

    The complement of the readable types, and the same judgement seen from the
    other side. Its own adapter because the policy offers the two separately —
    `include_working_memory` and `include_delta` — and a caller may want one
    without the other.
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
    """Session results, separated into facts about the person and task outcomes.

    A fact about the person outlives the task that produced it, so it is held
    harder — a statement about the information, not about the store it came from.
    The old prompt drew the same line and drew it by reading a tag in the
    renderer.
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
    """One working atom as a line, with its kind tag kept visible exactly once.

    The content may already carry a `[kind:…]` prefix from whoever wrote it, so
    it is stripped and re-added rather than duplicated. Presentation embedded in
    stored content is not something this slice fixes; it is something this slice
    must not lose.
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
