# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Project complete claims into built-in legacy payload shapes."""

from __future__ import annotations

from typing import Any

from nlght.core.knowledge.knowledge import (
    DECISION,
    FACT,
    PATTERN,
    RULE,
    DecisionPayload,
    FactPayload,
    KnowledgePayload,
    PatternPayload,
    RulePayload,
)
from nlght.core.knowledge.knowledge_proposition import Proposition
from nlght.core.knowledge.projection import NotRepresentable, Represented

LEGACY_FACT = "legacy fact payload"

#: What each kind's row can hold: the fields it requires, then the ones it may
#: also carry. A proposition with anything beyond these has no row, and not a
#: shortened one, because a row that drops a field reads as a claim and is not
#: the one that was made.
_SHAPES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    FACT: (("subject", "predicate", "object"), ()),
    RULE: (("rule_text",), ("subject", "rule_property")),
    PATTERN: (("pattern_name", "description"), ()),
    DECISION: (("decision", "effect"), ("subject", "decision_type")),
}

_TYPES: dict[str, type[Any]] = {
    FACT: FactPayload,
    RULE: RulePayload,
    PATTERN: PatternPayload,
    DECISION: DecisionPayload,
}


def required_fields(kind: str) -> tuple[str, ...]:
    """List fields required by a built-in legacy payload.

    Args:
        kind (str): Knowledge kind.

    Returns:
        tuple[str, ...]: Required field names.
    """
    required, _ = _SHAPES.get(kind, ((), ()))
    return required


def project(
    kind: str, proposition: Proposition
) -> Represented[KnowledgePayload] | NotRepresentable:
    """Project a claim without silently discarding extra fields.

    Args:
        kind (str): Knowledge kind whose legacy shape is requested.
        proposition (Proposition): Complete structured claim to project.

    Returns:
        Represented[KnowledgePayload] | NotRepresentable: Represented payload or a
            reason the projection would lose information.
    """
    shape = _SHAPES.get(kind)
    if shape is None:
        return NotRepresentable(
            shape=f"legacy {kind} payload",
            reason=f"'{kind}' has no legacy row",
            available=tuple(sorted(proposition.fields)),
        )

    required, optional = shape
    fields = dict(proposition.fields)
    known = set(required) | set(optional)
    extra = sorted(name for name in fields if name not in known)
    if extra:
        return NotRepresentable(
            shape=f"legacy {kind} payload",
            reason=(
                f"the claim carries {', '.join(extra)}, which the legacy payload has "
                f"no column for"
            ),
            available=tuple(sorted(fields)),
        )

    missing = [name for name in required if not str(fields.get(name, "")).strip()]
    if missing:
        return NotRepresentable(
            shape=f"legacy {kind} payload",
            reason=f"the claim has no {' or '.join(missing)}",
            available=tuple(sorted(fields)),
        )

    values = {name: str(fields[name]) for name in required}
    values.update(
        {name: str(fields[name]) for name in optional if str(fields.get(name, "")).strip()}
    )
    try:
        return Represented(_TYPES[kind](**values))
    except ValueError as exc:
        # Preserve the payload's validation failure, such as a rule carrying half its canonical pair,
        # for instance. Its refusal is the answer to "is a row possible", not an
        # error to raise past a caller that asked exactly that.
        return NotRepresentable(
            shape=f"legacy {kind} payload",
            reason=str(exc),
            available=tuple(sorted(fields)),
        )


def legacy_payload(kind: str, proposition: Proposition) -> KnowledgePayload | None:
    """Return a lossless legacy payload when one fits.

    Args:
        kind (str): Knowledge kind whose legacy shape is requested.
        proposition (Proposition): Complete structured claim to project.

    Returns:
        KnowledgePayload | None: Lossless legacy payload, or None when no such projection exists.
    """
    derived = project(kind, proposition)
    return derived.value if isinstance(derived, Represented) else None


def legacy_fact_payload(
    proposition: Proposition,
) -> Represented[FactPayload] | NotRepresentable:
    """Project a fact into its documented legacy row.

    Args:
        proposition (Proposition): Fact proposition.

    Returns:
        Represented[FactPayload] | NotRepresentable: Represented fact payload or refusal.
    """
    derived = project(FACT, proposition)
    if isinstance(derived, NotRepresentable):
        return derived
    if not isinstance(derived.value, FactPayload):
        raise TypeError(
            f"fact projection produced {type(derived.value).__name__}"
        )
    return Represented(derived.value)
