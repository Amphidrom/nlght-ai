# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Decide whether two complete same-kind claims are equivalent."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from nlght.core.knowledge.knowledge import FACT
from nlght.core.knowledge.knowledge_proposition import Proposition
from nlght.core.knowledge.proposition import material_content, normalise

SAME = "yes"
DIFFERENT = "no"
AMBIGUOUS = "ambiguous"
UNJUDGED = "unjudged"

#: Keep denial, uncertainty, and absence of a judge distinct in the audit.
#: Only an affirmative verdict permits a join.
_MERGEABLE = frozenset({SAME})


def may_merge(verdict: str) -> bool:
    """Permit continuation only for an affirmative verdict.

    Args:
        verdict (str): Equivalence verdict.

    Returns:
        bool: Whether the verdict permits a merge.
    """
    return verdict in _MERGEABLE


@dataclass(slots=True, frozen=True)
class AssertionObservation:
    """Hold one same-kind comparison input for an equivalence judgement.

    Observed text preserves the source wording. Representation identifies one
    extracted claim within that wording and uses Proposition fields for every
    built-in kind. A sentence can contain several distinct claims.
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
        """Construct a fact observation from a proposition.

        Args:
            proposition (Proposition): Fact proposition to observe.
            observed_text (str): Optional wording from the source.

        Returns:
            AssertionObservation: AssertionObservation in the fact namespace.
        """
        return cls(
            kind=FACT,
            observed_text=observed_text,
            representation=dict(proposition.fields),
        )

    def as_dict(self) -> dict[str, Any]:
        return {str(key): value for key, value in self.representation.items()}


def _values(value: Any) -> list[str]:  # noqa: ANN401 (it holds whatever was extracted)
    """Collect leaf values from nested claim structure.

    Args:
        value (Any): Scalar, mapping, list, or tuple value.

    Returns:
        list[str]: String form of each non-null leaf.
    """
    if isinstance(value, Mapping):
        return [leaf for item in value.values() for leaf in _values(item)]
    if isinstance(value, (list, tuple)):
        return [leaf for item in value for leaf in _values(item)]
    return [str(value)] if value is not None else []


def _quantities(claim: AssertionObservation) -> frozenset[str]:
    """Collect quantity-bearing tokens from one claim.

    Args:
        claim (AssertionObservation): Assertion observation and its nested representation.

    Returns:
        frozenset[str]: Distinct normalized tokens containing digits.
    """
    return frozenset(
        quantity
        for value in _values(claim.representation)
        for quantity in material_content(value)
    )


def _shape(claim: AssertionObservation) -> dict[str, str]:
    return {name: normalise(str(value)) for name, value in claim.representation.items()}


def settled(a: AssertionObservation, b: AssertionObservation) -> str | None:
    """Decide exact, cross-kind, or quantity changes without a judge.

    Args:
        a (AssertionObservation): First assertion observation.
        b (AssertionObservation): Second assertion observation.

    Returns:
        str | None: Yes, no, or None for the undecidable band.
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
    """Combine deterministic comparison with an optional judge verdict.

    Args:
        a (AssertionObservation): First assertion observation.
        b (AssertionObservation): Second assertion observation.
        verdict (str | None): Optional judge verdict for unresolved cases.

    Returns:
        str: Yes, no, ambiguous, or unjudged result.
    """
    decided = settled(a, b)
    if decided is not None:
        return decided
    if verdict in (SAME, DIFFERENT, AMBIGUOUS):
        return verdict
    return UNJUDGED


@dataclass(slots=True, frozen=True)
class Judge:
    """Identify the classifier, model, and version behind a verdict.

    This provenance lets an audit distinguish judgements made under different
    model or prompt versions.
    """

    classifier: str
    model: str
    version: str

    def __str__(self) -> str:
        return f"{self.classifier}/{self.model}/{self.version}"


class EquivalenceJudge(Protocol):
    """Classify undecidable pairs and identify the judge for the audit trail."""

    @property
    def judge(self) -> Judge: ...

    async def classify(
        self, a: AssertionObservation, b: AssertionObservation
    ) -> str: ...


def continuation(assessments: Sequence[tuple[str, str]]) -> str | None:
    """Select one uniquely affirmed assertion.

    Args:
        assessments (Sequence[tuple[str, str]]): Candidate assertion IDs paired with verdicts.

    Returns:
        str | None: One assertion ID or None when no unique yes exists.
    """
    joinable = {assertion for assertion, verdict in assessments if may_merge(verdict)}
    return next(iter(joinable)) if len(joinable) == 1 else None


@dataclass(slots=True, frozen=True)
class PropositionEquivalence:
    """Record a pairwise verdict and its judge in canonical pair order.

    Canonical pair order prevents duplicate verdict records. It does not change
    the direction of either claim being compared.
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
