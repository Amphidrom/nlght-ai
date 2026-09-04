# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Whether two complete propositions say the same thing.

A judgement and nothing else. It produces no canonical form, stores nothing, and
returns no merged proposition — because a "canonical" anything would be a data
model chosen by the side that was only supposed to compare:

    yes         the same proposition, said differently
    no          a judge looked and said they are different
    ambiguous   a judge looked and could not tell
    unjudged    nobody was asked

The last three all mean *do not merge*, and they are still four states rather
than two. Storing "nobody was asked" as `no` would let the record claim later
that something decided these were different claims, when the truth is that no
classifier was configured. An audit trail that cannot tell those apart is worse
than none, because it reads as evidence.

Only askable now, and that ordering was the point. Two claims can be compared
only once both can be held completely (`Proposition`), and comparing what a
lossy representation had kept would have been comparing the loss.

**Two things are decidable here and the rest is not.** Identical fields are the
same claim; a moved quantity is a different one — Java 17 against Java 21, where
no wording matters. Everything else needs to know what words mean.

That includes a case that looks structural and is not. A role *swap* and a role
*rename* are indistinguishable over the structure:

    {sender: Alice, recipient: Bob}  vs  {sender: Bob, recipient: Alice}
    {sender: Alice, recipient: Bob}  vs  {actor: Alice,  target: Bob}

Both admit a bijection of field names carrying one to the other, so any rule over
the shape answers both the same way — and they must end opposite: the first is a
different claim, the second is one claim named twice. Only knowing that `sender`
and `recipient` are different roles while `sender` and `actor` are the same one
separates them, and nothing in a structure says that.

So the undecidable band is delegated, as it was for `semantic_change`
(ADR-0045) — and the conservative direction is **not** the same. There, doubt
means a review: a review nobody needed costs a minute. Here, doubt means no
merge: two claims wrongly joined lose one of them, and nothing afterwards shows
which. The safe direction follows the harm, not a habit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from nlght.core.knowledge.knowledge_proposition import Proposition
from nlght.core.knowledge.proposition import material_content, normalise

SAME = "yes"
DIFFERENT = "no"
AMBIGUOUS = "ambiguous"
UNJUDGED = "unjudged"

#: The only verdict that permits joining two claims. Everything else — a judge
#: saying no, a judge unsure, or no judge at all — leaves them apart, and the
#: three are kept distinct because they mean different things to a reader.
_MERGEABLE = frozenset({SAME})


def may_merge(verdict: str) -> bool:
    """Whether this verdict permits treating the two as one claim."""
    return verdict in _MERGEABLE


@dataclass(slots=True, frozen=True)
class AssertionObservation:
    """One sighting of an assertion, as the equivalence judgement receives it.

    The input contract, and there is exactly one for every kind. A knowledge
    model with four of these would end up with four slightly different notions of
    truth, and "is this the same assertion" has to mean one thing.

        kind             which identity namespace this belongs to
        observed_text    what the source says, verbatim and quotable
        representation   which assertion inside it is being judged

    **Both are needed, and neither is the other.** `observed_text` is
    authoritative for the wording (ADR-0048) — never reconstructed from the
    structure. But it is not an identity, because one sentence carries several
    assertions:

        "When shutdown is enabled, the application waits for requests
         and rejects new requests."

            waits_for(application, active_requests)
            rejects(application, new_requests)

    Two assertions, one `observed_text`. A judgement made on the sentence alone
    could not tell them apart and would merge them; one made on the structure
    alone could not tell what an invented field name meant. So both go, and
    `representation` says which assertion within the sentence is the subject of
    the question.

    `representation` is whatever shape the kind already has — a
    `Proposition`'s fields for a fact, the structured payload for the
    others. Deliberately not one universal shape: `Proposition` is a fact's
    representation, and making it everyone's would invent an abstraction no
    invariant asks for.
    """

    kind: str
    observed_text: str
    representation: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.kind.strip():
            raise ValueError("an observation must say what kind it is")
        if not self.representation:
            raise ValueError("an observation needs a representation to judge")

    @classmethod
    def of(
        cls, proposition: Proposition, *, observed_text: str = ""
    ) -> AssertionObservation:
        """A fact's observation, from the representation facts already have."""
        from nlght.core.knowledge.knowledge import FACT  # noqa: PLC0415

        return cls(
            kind=FACT,
            observed_text=observed_text,
            representation=dict(proposition.fields),
        )

    def as_dict(self) -> dict[str, Any]:
        return {str(key): value for key, value in self.representation.items()}


def _values(value: Any) -> list[str]:  # noqa: ANN401 (it holds whatever was extracted)
    """Every leaf a proposition carries, wherever it sits in the structure."""
    if isinstance(value, Mapping):
        return [leaf for item in value.values() for leaf in _values(item)]
    if isinstance(value, (list, tuple)):
        return [leaf for item in value for leaf in _values(item)]
    return [str(value)] if value is not None else []


def _quantities(claim: AssertionObservation) -> frozenset[str]:
    """The quantities the claim asserts, from anywhere inside it.

    A threshold, a version, a port or a count. Nested, because a claim may carry
    one three levels down and a comparison that only looked at the top would call
    two different claims the same.

    Read from the representation and not from `observed_text`: the sentence may
    mention a number that belongs to a different claim standing beside this one.
    """
    return frozenset(
        quantity
        for value in _values(claim.representation)
        for quantity in material_content(value)
    )


def _shape(claim: AssertionObservation) -> dict[str, str]:
    return {name: normalise(str(value)) for name, value in claim.representation.items()}


def settled(a: AssertionObservation, b: AssertionObservation) -> str | None:
    """The answer where one can be had without knowing what the words mean.

    `None` for the band worth spending a judgement on: the same quantities said
    with different field names or different words.

    A different kind is decided here and is `no` — an answer, not an error.
    `kind` is part of the identity namespace, so a `rule` does not continue a
    `fact` however alike they read, and settling it here means the model is never
    asked a question whose answer is already known.
    """
    if a.kind != b.kind:
        return DIFFERENT
    if _shape(a) == _shape(b):
        return SAME
    if _quantities(a) != _quantities(b):
        # A different threshold, version or count is a different claim about the
        # world, whatever the sentence around it does.
        return DIFFERENT
    return None


def equivalent(
    a: AssertionObservation, b: AssertionObservation, *, verdict: str | None = None
) -> str:
    """Whether these two are one proposition, given what a judgement made of them.

    `verdict` is the answer for the undecidable band, or `None` when none was
    asked for — and those are different answers. Doubt a judge *expresses* is
    `ambiguous`; nobody having been asked is `unjudged`. Both leave the claims
    apart, and recording either as `no` would let the trail claim a decision that
    was never taken.
    """
    decided = settled(a, b)
    if decided is not None:
        return decided
    if verdict in (SAME, DIFFERENT, AMBIGUOUS):
        return verdict
    return UNJUDGED


@dataclass(slots=True, frozen=True)
class Judge:
    """What produced a verdict, in enough detail to revisit it.

    A verdict without its judge cannot be reconsidered when the judge changes —
    and it will change, because the undecidable band is answered by a model whose
    version moves. "The corpus says these are one claim" is worth very different
    amounts depending on which model said so and under which prompt.
    """

    classifier: str
    model: str
    version: str

    def __str__(self) -> str:
        return f"{self.classifier}/{self.model}/{self.version}"


class EquivalenceJudge(Protocol):
    """Whatever can answer the undecidable band, and say who answered.

    A protocol rather than a callable, because a verdict without its judge cannot
    be revisited when the judge changes — and it will change. "The corpus says
    these are one claim" is worth very different amounts depending on which model
    said so under which prompt, and the trail has to carry that.
    """

    @property
    def judge(self) -> Judge: ...

    async def classify(
        self, a: AssertionObservation, b: AssertionObservation
    ) -> str: ...


def continuation(assessments: Sequence[tuple[str, str]]) -> str | None:
    """The one assertion these judgements permit continuing, or none.

    ``assessments`` pairs a candidate assertion with the verdict reached about
    it. Unique or nothing, exactly like every rung of the matching ladder: two
    candidates judged the same claim is a corpus that cannot say which one this
    continues, and picking the first is how a wrong continuation arrives that no
    report will ever show.

    Only `yes` joins. A judge saying no, a judge unsure, and no judge at all all
    leave the claims apart — and they stay three different records, because the
    reader of an audit needs to tell "decided different" from "never asked".
    """
    joinable = {assertion for assertion, verdict in assessments if may_merge(verdict)}
    return next(iter(joinable)) if len(joinable) == 1 else None


@dataclass(slots=True, frozen=True)
class PropositionEquivalence:
    """One decision about one pair, and who made it.

    A decision about a **pair**, which is why it is a record of its own rather
    than a field on either proposition: `equivalent(A, B)` says nothing about A.
    Hanging it on A would make a statement about a relationship look like a
    property of a thing, and the next reader would use it as one.

    The pair is stored in a canonical order, because "X is equivalent to Y" is
    the same decision as "Y is equivalent to X" and storing both would let a
    corpus hold two answers to one question. That says nothing about whether
    `transfers(A, B)` equals `transfers(B, A)` — the claims are directed and the
    *equivalence relation between claims* is symmetric, and confusing the two is
    exactly the mistake this slice keeps stepping around.
    """

    left: str
    right: str
    verdict: str
    #: `None` where nothing decided, which is what `unjudged` means.
    judge: Judge | None = None
    decided_at: datetime | None = None

    @classmethod
    def between(
        cls,
        a: Proposition,
        b: Proposition,
        verdict: str,
        *,
        judge: Judge | None = None,
        decided_at: datetime | None = None,
    ) -> PropositionEquivalence:
        left, right = sorted((a.fingerprint, b.fingerprint))
        return cls(
            left=left, right=right, verdict=verdict, judge=judge, decided_at=decided_at
        )

    @property
    def may_merge(self) -> bool:
        return may_merge(self.verdict)
