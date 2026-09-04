# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""The kind-specific payloads, derived from the proposition.

`Proposition` is the single business truth for every claim. `FactPayload`,
`RulePayload`, `PatternPayload` and `DecisionPayload` — the rows four tables were
built around — are produced *from* it, and only while that can be done without
losing anything.

    Proposition  --derive-->  the kind's payload

Never:

    Proposition  --write-->  A
    the payload  --write-->  B

Two writes can disagree, and nothing afterwards says which is right. One
derivation cannot: if the payload exists it is what the proposition says, and if
the proposition says more than a payload can hold there is no payload rather
than a shortened one.

That held for facts across several slices while the other three kinds built
their payload directly — which made the payload the truth for those kinds and
the proposition an extra. All four are projections of one thing now, and the
tables are what they had been becoming anyway: consumer shapes, not knowledge.

This module names fields on purpose, and it is the only place in core that may.
They are the legacy payloads' shapes, named where those payloads are built — not
privileged forms of a proposition, which is why `projection.py` still knows no
shapes at all.
"""

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
#: also carry. A proposition with anything beyond these has no row — and not a
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
    """What a kind must carry to say anything at all.

    Read from the same table the projection uses, so "what a rule needs" is
    stated once. A fact is absent on purpose: what it needs is an entity key —
    a predicate and something standing in it — which `entity_key` already
    decides and this must not restate.
    """
    required, _ = _SHAPES.get(kind, ((), ()))
    return required


def project(
    kind: str, proposition: Proposition
) -> Represented[KnowledgePayload] | NotRepresentable:
    """The legacy row this proposition makes, or a refusal saying why it makes none."""
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
        # The payload's own validation — a rule carrying half its canonical pair,
        # for instance. Its refusal is the answer to "is a row possible", not an
        # error to raise past a caller that asked exactly that.
        return NotRepresentable(
            shape=f"legacy {kind} payload",
            reason=str(exc),
            available=tuple(sorted(fields)),
        )


def legacy_payload(kind: str, proposition: Proposition) -> KnowledgePayload | None:
    """The row, or nothing where the projection would lose something."""
    derived = project(kind, proposition)
    return derived.value if isinstance(derived, Represented) else None


def legacy_fact_payload(
    proposition: Proposition,
) -> Represented[FactPayload] | NotRepresentable:
    """The fact row. Kept as a name because the fact case is the documented one."""
    derived = project(FACT, proposition)
    if isinstance(derived, NotRepresentable):
        return derived
    if not isinstance(derived.value, FactPayload):
        raise TypeError(
            f"fact projection produced {type(derived.value).__name__}"
        )
    return Represented(derived.value)
