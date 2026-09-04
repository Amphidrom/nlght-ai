# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""What survives a budget, and in which form.

The reduction is defined by what it refuses to know. A directive, a session
result and a retrieved passage are different things and stay different things —
but nothing here may ask which it is holding, because a rule that branches on
where information came from gives a corpus one policy per store and no way to
weigh a memory against a passage.

So most of these tests pin an absence: no type in the rule, no invented text, no
level that went back up, no mandatory element quietly dropped to make a number
work.
"""

from __future__ import annotations

import pytest

from nlght.core.hive_mind.models import (
    Level,
    MentalElement,
    Representation,
    Retention,
)
from nlght.core.hive_mind.relevance import reduce_to_budget


def _element(
    element_id: str,
    *,
    kind: str = "memory",
    retention: Retention = Retention.USEFUL,
    relevance: float = 0.5,
    full: int = 10,
    compact: int | None = None,
    provenance: object = None,
) -> MentalElement:
    representations = [
        Representation(level=Level.FULL, text=f"{element_id} in full", cost=full)
    ]
    if compact is not None:
        representations.append(
            Representation(
                level=Level.COMPACT, text=f"{element_id} briefly",
                cost=compact, derived=True,
            )
        )
    representations.append(Representation(level=Level.OMIT, text="", cost=0))
    return MentalElement(
        element_id=element_id,
        kind=kind,
        representations=tuple(representations),
        retention=retention,
        relevance=relevance,
        provenance=provenance,
    )


def _levels(view) -> dict[str, Level]:  # noqa: ANN001
    said = {element.element_id: representation.level for element, representation in view.kept}
    return {**{item.element_id: Level.OMIT for item in view.forgotten}, **said}


# 1-2. More budget is never less; less budget is never more
# ---------------------------------------------------------------------------

def test_more_budget_never_yields_a_less_complete_view() -> None:
    """The guarantee that makes two budgets comparable at all.

    It holds by construction rather than by arrangement: a reduction is a prefix
    of one fixed order of giving things up, so a larger budget applies a prefix
    of the steps a smaller one applies.
    """
    elements = [
        _element("a", retention=Retention.IMPORTANT, relevance=0.9, full=30, compact=10),
        _element("b", retention=Retention.USEFUL, relevance=0.5, full=30, compact=10),
        _element("c", retention=Retention.DISPENSABLE, relevance=0.1, full=30, compact=10),
    ]
    order = [Level.FULL, Level.COMPACT, Level.OMIT]

    previous = None
    for budget in range(0, 100, 5):
        current = _levels(reduce_to_budget(elements, budget=budget))
        if previous is not None:
            for element_id, level in current.items():
                assert order.index(level) <= order.index(previous[element_id]), (
                    f"{element_id} got less complete as the budget grew"
                )
        previous = current


def test_information_only_decreases_as_the_budget_shrinks() -> None:
    elements = [
        _element("a", retention=Retention.IMPORTANT, relevance=0.9, full=20, compact=5),
        _element("b", retention=Retention.USEFUL, relevance=0.4, full=20, compact=5),
    ]

    costs = [reduce_to_budget(elements, budget=b).cost for b in range(60, -1, -5)]

    assert costs == sorted(costs, reverse=True), "cost rose while the budget fell"


# 3-5. What goes first, what is shortened, what stays
# ---------------------------------------------------------------------------

def test_the_dispensable_is_forgotten_before_the_important() -> None:
    elements = [
        _element("keep", retention=Retention.IMPORTANT, relevance=0.2, full=20),
        _element("drop", retention=Retention.DISPENSABLE, relevance=0.9, full=20),
    ]

    view = reduce_to_budget(elements, budget=20)

    assert [item.element_id for item in view.forgotten] == ["drop"]
    # And note the relevance is the wrong way round on purpose: relevance orders
    # within a band and never across one. A dispensable thing that matches the
    # question well is still dispensable.


def test_an_important_element_is_compacted_before_it_disappears() -> None:
    elements = [
        _element("a", retention=Retention.IMPORTANT, relevance=0.5, full=40, compact=8),
        _element("b", retention=Retention.IMPORTANT, relevance=0.9, full=40, compact=8),
    ]

    view = reduce_to_budget(elements, budget=20)

    assert _levels(view) == {"a": Level.COMPACT, "b": Level.COMPACT}
    assert view.forgotten == (), "shortened, not lost"


def test_a_mandatory_element_survives() -> None:
    elements = [
        _element("rule", retention=Retention.MANDATORY, relevance=0.0, full=50, compact=12),
        _element("chat", retention=Retention.USEFUL, relevance=0.9, full=50),
    ]

    view = reduce_to_budget(elements, budget=12)

    assert _levels(view) == {"rule": Level.COMPACT, "chat": Level.OMIT}


def test_a_mandatory_element_too_large_is_reported_not_dropped() -> None:
    """Over budget is said out loud rather than made to go away.

    Dropping what was declared mandatory to make a number fit would be the
    reduction overruling the caller about what may be lost — and it would do it
    silently, which is the part that makes it indefensible.
    """
    elements = [_element("rule", retention=Retention.MANDATORY, full=500)]

    view = reduce_to_budget(elements, budget=10)

    assert view.over_budget is True
    assert _levels(view) == {"rule": Level.FULL}
    assert view.cost == 500


# 6. Provenance outlives the representation
# ---------------------------------------------------------------------------

def test_provenance_survives_being_compacted() -> None:
    trail = object()
    elements = [
        _element("p", retention=Retention.IMPORTANT, full=40, compact=5, provenance=trail),
    ]

    view = reduce_to_budget(elements, budget=5)

    element, representation = view.kept[0]
    assert representation.level is Level.COMPACT
    assert element.provenance is trail
    assert representation.derived is True, "a compacted form is not a verbatim quote"


def test_a_full_representation_is_not_marked_derived() -> None:
    # The distinction a citation rests on: what the source said, versus what was
    # made of it to save room (ADR-0053).
    element = _element("p", full=10)

    assert element.at(Level.FULL).derived is False


# 7. The type never decides
# ---------------------------------------------------------------------------

def test_the_kind_changes_nothing() -> None:
    """The invariant the whole design exists for.

    Two elements identical in everything a reduction may look at, differing only
    in what sort of thing they are, must reduce identically. If this ever fails,
    storage types are deciding policy again.
    """
    def _run(kinds: tuple[str, str]):  # noqa: ANN202
        return _levels(reduce_to_budget(
            [
                _element("x", kind=kinds[0], retention=Retention.USEFUL,
                         relevance=0.5, full=30, compact=6),
                _element("y", kind=kinds[1], retention=Retention.USEFUL,
                         relevance=0.4, full=30, compact=6),
            ],
            budget=12,
        ))

    assert _run(("memory", "passage")) == _run(("passage", "memory"))
    assert _run(("directive", "turn")) == _run(("memory", "passage"))


def test_relevance_orders_within_a_band_and_not_across_one() -> None:
    elements = [
        _element("low_but_important", retention=Retention.IMPORTANT, relevance=0.01, full=10),
        _element("high_but_useful", retention=Retention.USEFUL, relevance=0.99, full=10),
    ]

    view = reduce_to_budget(elements, budget=10)

    assert [item.element_id for item in view.forgotten] == ["high_but_useful"]


# 8-10. Any mixture of inputs, one engine
# ---------------------------------------------------------------------------

def test_retrieval_only_works_with_no_memory_at_all() -> None:
    passages = [
        _element(f"p{i}", kind="passage", retention=Retention.USEFUL,
                 relevance=1.0 - i / 10, full=10)
        for i in range(5)
    ]

    view = reduce_to_budget(passages, budget=25)

    assert len(view.kept) == 2
    assert view.cost <= 25


def test_memory_only_works_with_no_retrieval_at_all() -> None:
    memories = [
        _element(f"m{i}", kind="memory", retention=Retention.USEFUL,
                 relevance=1.0 - i / 10, full=10)
        for i in range(5)
    ]

    view = reduce_to_budget(memories, budget=25)

    assert len(view.kept) == 2


def test_both_together_are_decided_in_one_pass() -> None:
    """Not a retrieval budget beside a memory budget — one decision.

    A passage that matters less than a memory loses to it, and the other way
    round, because they are being weighed rather than allocated.
    """
    mixed = [
        _element("passage-strong", kind="passage", retention=Retention.USEFUL,
                 relevance=0.9, full=10),
        _element("memory-weak", kind="memory", retention=Retention.USEFUL,
                 relevance=0.1, full=10),
        _element("memory-strong", kind="memory", retention=Retention.USEFUL,
                 relevance=0.8, full=10),
        _element("passage-weak", kind="passage", retention=Retention.USEFUL,
                 relevance=0.2, full=10),
    ]

    view = reduce_to_budget(mixed, budget=20)

    assert {element.element_id for element, _ in view.kept} == {
        "passage-strong", "memory-strong",
    }


def test_an_empty_model_is_not_an_error() -> None:
    view = reduce_to_budget([], budget=0)

    assert (view.kept, view.forgotten, view.cost, view.over_budget) == ((), (), 0, False)


# The types refuse what they cannot mean
# ---------------------------------------------------------------------------

def test_an_element_must_offer_something() -> None:
    with pytest.raises(ValueError, match="offers no representation"):
        MentalElement(element_id="e", kind="memory", representations=())


def test_one_level_cannot_be_offered_twice() -> None:
    with pytest.raises(ValueError, match="offers a level twice"):
        MentalElement(
            element_id="e", kind="memory",
            representations=(
                Representation(level=Level.FULL, text="a", cost=1),
                Representation(level=Level.FULL, text="b", cost=2),
            ),
        )


def test_saying_nothing_cannot_cost_something() -> None:
    with pytest.raises(ValueError, match="costs nothing to say"):
        Representation(level=Level.OMIT, text="", cost=3)


def test_a_negative_budget_is_refused() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        reduce_to_budget([], budget=-1)


def test_a_shorter_form_that_costs_more_is_refused() -> None:
    """The condition the monotonicity guarantee actually rests on.

    A reduction is a prefix of one fixed order of giving things up, and that only
    bounds the total if every step makes it smaller. A "compact" form costing
    more than the full one would let a step *raise* the total, and "more budget
    is never less complete" would hold by luck of the data rather than by
    construction.
    """
    with pytest.raises(ValueError, match="is not a shorter form"):
        MentalElement(
            element_id="e", kind="memory",
            representations=(
                Representation(level=Level.FULL, text="short", cost=5),
                Representation(level=Level.COMPACT, text="somehow longer", cost=9),
            ),
        )


def test_an_element_need_not_offer_every_level() -> None:
    """Representation levels are optional, and that is load-bearing.

    Requiring `COMPACT` would force every adapter to invent one — by truncating,
    or by pulling out a sentence and hoping it carries the claim. The contract is
    that adapters *produce* representations and the engine only chooses among
    them; an adapter with nothing honest to offer must be able to offer nothing.

    The cost ordering therefore constrains only the levels an element actually
    has, and a reduction step for a level it does not offer is skipped.
    """
    only_whole = MentalElement(
        element_id="p", kind="passage",
        representations=(
            Representation(level=Level.FULL, text="the passage, verbatim", cost=40),
            Representation(level=Level.OMIT, text="", cost=0),
        ),
        retention=Retention.USEFUL,
    )

    assert only_whole.levels == (Level.FULL, Level.OMIT)
    assert reduce_to_budget([only_whole], budget=40).kept[0][1].level is Level.FULL
    # No compact step to take, so it goes straight from said to unsaid.
    assert reduce_to_budget([only_whole], budget=39).forgotten == (only_whole,)


def test_an_element_may_offer_only_one_level() -> None:
    # Nothing is required to be omittable either — a mandatory element with a
    # single representation is a thing an adapter may legitimately produce.
    fixed = MentalElement(
        element_id="rule", kind="directive",
        representations=(Representation(level=Level.FULL, text="always", cost=5),),
        retention=Retention.MANDATORY,
    )

    view = reduce_to_budget([fixed], budget=0)

    assert view.over_budget is True
    assert view.kept[0][1].text == "always"


# The competition, which is where the interesting bugs are
# ---------------------------------------------------------------------------

def _competing() -> list[MentalElement]:
    """One element per retention band, each with a real short form.

    A single element under pressure exercises almost nothing: it is kept or it
    is not. What has to hold is the *ordering* between elements that want the
    same room, which is why every case below runs all four together.
    """
    return [
        _element("mandatory", retention=Retention.MANDATORY, relevance=0.5,
                 full=100, compact=50),
        _element("important", retention=Retention.IMPORTANT, relevance=0.5,
                 full=90, compact=30),
        _element("useful", retention=Retention.USEFUL, relevance=0.5,
                 full=70, compact=25),
        _element("dispensable", retention=Retention.DISPENSABLE, relevance=0.5,
                 full=60, compact=20),
    ]


@pytest.mark.parametrize(
    ("budget", "expected"),
    [
        # Everything fits.
        (320, {"mandatory": Level.FULL, "important": Level.FULL,
               "useful": Level.FULL, "dispensable": Level.FULL}),
        # The cheapest thing to give up is the dispensable one's detail.
        (300, {"mandatory": Level.FULL, "important": Level.FULL,
               "useful": Level.FULL, "dispensable": Level.COMPACT}),
        # Then the dispensable one entirely.
        (270, {"mandatory": Level.FULL, "important": Level.FULL,
               "useful": Level.FULL, "dispensable": Level.OMIT}),
        # Then the merely useful is said briefly, then not at all.
        (230, {"mandatory": Level.FULL, "important": Level.FULL,
               "useful": Level.COMPACT, "dispensable": Level.OMIT}),
        (200, {"mandatory": Level.FULL, "important": Level.FULL,
               "useful": Level.OMIT, "dispensable": Level.OMIT}),
        # Only then is the important one shortened — never dropped before the
        # useful one has already gone.
        (150, {"mandatory": Level.FULL, "important": Level.COMPACT,
               "useful": Level.OMIT, "dispensable": Level.OMIT}),
        (100, {"mandatory": Level.FULL, "important": Level.OMIT,
               "useful": Level.OMIT, "dispensable": Level.OMIT}),
        # And last of all the mandatory one is shortened. It is never omitted.
        (50, {"mandatory": Level.COMPACT, "important": Level.OMIT,
              "useful": Level.OMIT, "dispensable": Level.OMIT}),
        (0, {"mandatory": Level.COMPACT, "important": Level.OMIT,
             "useful": Level.OMIT, "dispensable": Level.OMIT}),
    ],
)
def test_the_order_information_is_given_up_in(budget: int, expected: dict) -> None:
    assert _levels(reduce_to_budget(_competing(), budget=budget)) == expected


def test_every_budget_from_full_to_nothing_is_monotone() -> None:
    """Metamorphic rather than exemplary: the property over the whole range.

    Two runs at neighbouring budgets are compared against each other for every
    step down from "everything fits" to zero. A single example can pass while the
    rule holds only near the values somebody happened to pick.
    """
    elements = _competing()
    order = [Level.FULL, Level.COMPACT, Level.OMIT]

    previous = None
    for budget in range(320, -1, -1):
        current = _levels(reduce_to_budget(elements, budget=budget))
        if previous is not None:
            for name, level in current.items():
                assert order.index(level) >= order.index(previous[name]), (
                    f"{name} became fuller as the budget shrank, at {budget}"
                )
        previous = current


def test_the_view_costs_no_more_than_the_budget_whenever_that_is_possible() -> None:
    # The one exception is a mandatory set that cannot fit, which is reported
    # rather than forced — asserted separately above.
    elements = _competing()
    floor = elements[0].at(Level.COMPACT).cost

    for budget in range(floor, 321):
        view = reduce_to_budget(elements, budget=budget)
        assert view.cost <= budget, f"overspent at budget {budget}"
        assert view.over_budget is False


def test_the_same_input_always_reduces_the_same_way() -> None:
    # Determinism is what makes a prompt reproducible and a golden test
    # meaningful. Ties break on the element id, so nothing depends on dict or
    # set iteration order.
    elements = _competing()

    first = _levels(reduce_to_budget(elements, budget=175))
    for _ in range(5):
        assert _levels(reduce_to_budget(list(elements), budget=175)) == first
