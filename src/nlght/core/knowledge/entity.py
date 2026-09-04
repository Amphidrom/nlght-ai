# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""What an assertion *is*, and which of its simultaneous cases this one is.

Three levels, kept apart because they have different lifetimes:

    entity    which business quantity is this?
    variant   for which simultaneously valid case?
    revision  how has that case changed over time?

**Identity is a schema decision, not a similarity decision.** An assertion's id
does not identify what a claim currently says — it identifies the business
question whose value may change. `CHF 500 → 550` is one rule at two states;
`approval_threshold → prohibition_threshold` is two rules that must never
supersede one another. So the entity key is assembled from schema fields and
looked up deterministically, and the similarity matcher is left with entity
*linking* — which existing entity a new extraction probably corresponds to —
never with deciding what the database considers identity.

The dividing rule, applied per kind below:

    into the entity key go exactly those dimensions whose change means we are
    talking about a different business quantity — not those whose change merely
    describes a new state of the same quantity.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from nlght.core.ingestion.document import stable_digest
from nlght.core.knowledge.knowledge import (
    DECISION,
    FACT,
    PATTERN,
    RULE,
    canonical_term,
)

ENTITY_KEY_VERSION = "1"
"""Which scheme produced a key.

Stored beside the key rather than folded into it. Folding it in would make an
old key simply differ from a new one, which is silent: nothing could tell a key
of the previous scheme from one that names a different quantity. Kept apart, a
migration to a richer proposition can read the old keys, decide what each means,
and rewrite them deliberately.
"""


ENTITY_KEY_FIELDS: dict[str, tuple[str, ...]] = {
    # Thresholds, conditions and scope are the value. Five hundred is what the
    # rule currently says, not what it is.
    RULE: ("subject", "rule_property"),
    # Reversing a decision is a new state of that decision; what was chosen
    # before is history, not a separate decision.
    DECISION: ("subject", "decision_type"),
    # Already had the shape: a short name beside a mutable description.
    PATTERN: ("pattern_name",),
}
"""Kinds whose identity is one fixed pair of fields.

`fact` is absent because it has no such pair: its identity is the whole
proposition. See `entity_key`.
"""


class EntityKeyUnavailable(ValueError):
    """An assertion that cannot say which business quantity it is.

    Not an error in the corpus: assertions written before the identifying
    fields existed have none, and a caller decides what to do about that —
    usually leave them alone until they are extracted again.
    """


class EntityKeyVersionMismatch(ValueError):
    """One business question already recorded under a different key scheme.

    Both silent readings are wrong. Matching it would treat a key the old scheme
    produced as if the new one had produced it, and the two do not mean the same
    thing — that is what the version is for. Ignoring it quietly creates a second
    assertion for one question, and the corpus then holds the same quantity twice
    with nobody the wiser.

    So it stops. Migrating what the previous scheme recorded is a deliberate act,
    and this is what makes somebody perform it.
    """


@dataclass(slots=True, frozen=True)
class EntityKey:
    """A key together with the scheme that produced it.

    One value rather than two, so a key cannot be stored without its version and
    then be misread by whatever the scheme becomes next.
    """

    version: str
    key: str

    def __str__(self) -> str:
        return f"{self.version}:{self.key}"


def entity_key(kind: str, **fields: object) -> EntityKey:
    """The key for one assertion.

    For `rule`, `decision` and `pattern` that is one fixed pair of fields, and
    everything else passed in is the value and is ignored here on purpose — a
    caller may hand over a whole payload without knowing which half is which.

    For `fact` it is the **whole proposition**: predicate and every role. Not a
    projection over some of them.

    An earlier version tried to split a fact's roles into identifying and
    changeable ones, so that *OpenAI's CEO* could be one thing whose value moves
    from Alice to Bob. Doing that needs to know which relations are functional
    and which of their arguments carry identity — domain knowledge no platform
    has without an ontology, and neither a hardcoded list nor a customer writing
    one before ingesting anything is an answer.

    So the platform does not decide it. Two propositions are the same assertion
    when they say the same thing; *CEO(OpenAI, Alice)* and *CEO(OpenAI, Bob)* say
    different things and are two assertions, with a supersession recorded between
    them. That is also what §7.4 said from the start: a rewording keeps an
    identity, a material change makes a new assertion.
    """
    if kind == FACT:
        return _fact_key(**fields)

    identifying = ENTITY_KEY_FIELDS.get(kind)
    if identifying is None:
        raise EntityKeyUnavailable(f"unknown knowledge kind '{kind}'")
    missing = [name for name in identifying if not str(fields.get(name) or "").strip()]
    if missing:
        raise EntityKeyUnavailable(
            f"a '{kind}' needs {', '.join(identifying)} to name a business quantity; "
            f"missing {', '.join(missing)}"
        )
    return EntityKey(
        version=ENTITY_KEY_VERSION,
        key=stable_digest(
            kind, *(canonical_term(str(fields[name])) for name in identifying)
        ),
    )


def _fact_key(**fields: object) -> EntityKey:
    predicate = str(fields.get("predicate") or "").strip()
    if not predicate:
        raise EntityKeyUnavailable("a 'fact' needs a predicate to name what it claims")
    roles = {
        name: str(value)
        for name, value in fields.items()
        if name != "predicate" and str(value).strip()
    }
    if not roles:
        raise EntityKeyUnavailable(
            f"a 'fact' about '{predicate}' names nothing standing in that relation"
        )
    # Sorted, so the key describes the proposition rather than the order its
    # roles arrived in.
    return EntityKey(
        version=ENTITY_KEY_VERSION,
        key=stable_digest(
            FACT,
            canonical_term(predicate),
            *(
                f"{name}={canonical_term(value)}"
                for name, value in sorted(roles.items())
            ),
        ),
    )


def entity_key_or_none(kind: str, **fields: object) -> EntityKey | None:
    """The key, or nothing when this assertion cannot yet provide one."""
    try:
        return entity_key(kind, **fields)
    except EntityKeyUnavailable:
        return None


def _scope(values: object) -> frozenset[str]:
    """A scope as a comparable set, with absence meaning one default case."""
    if values is None:
        return frozenset()
    if isinstance(values, str):
        return frozenset({canonical_term(values)})
    if isinstance(values, Iterable):
        return frozenset(canonical_term(str(item)) for item in values if str(item).strip())
    return frozenset({canonical_term(str(values))})


EXACT = "exact"
WIDENED = "widened"
NARROWED = "narrowed"
AMBIGUOUS = "ambiguous"
NEW = "new"


@dataclass(slots=True, frozen=True)
class VariantMatch:
    """Which case of an entity a scope names, and how that was decided.

    ``resolution`` is carried out rather than swallowed so a caller can log or
    review an ``ambiguous`` outcome — it is the case where the corpus grew a
    variant that a person might have meant as a continuation.
    """

    variant_id: str
    resolution: str

    @property
    def is_continuation(self) -> bool:
        return self.resolution in (EXACT, WIDENED, NARROWED)


def resolve_variant(
    entity: object,
    scope: object,
    known: Mapping[str, object],
) -> VariantMatch:
    """Which case of this entity a scope names — an existing one, or a new one.

    The id is a **surrogate**, minted once and never derived from the scope.
    That is forced by the behaviour it has to have: widening "employees" to
    "employees and contractors" continues the variant it replaced, and a key
    derived from the scope would change with the scope and retire exactly the
    variant it was meant to continue.

    Resolution is a ladder, not a score:

        exact match
        → a single unambiguous widening or narrowing
        → otherwise a new variant

    Deliberately not "the largest overlap wins". Overlapping scopes that hold at
    the same time are ordinary — employees and managers, engineering and
    everyone — and picking the biggest intersection between them is a heuristic
    deciding identity, which is the thing the surrogate id exists to prevent. On
    a tie, or on an overlap that is neither containment nor equality, no lineage
    is asserted: a new variant is opened and the outcome says it was ambiguous.
    That errs towards a corpus holding one case too many, which a person can
    merge, rather than towards two cases silently becoming one, which nobody
    sees.
    """
    wanted = _scope(scope)
    scopes = {candidate: _scope(value) for candidate, value in known.items()}

    exact = [candidate for candidate, value in scopes.items() if value == wanted]
    if len(exact) == 1:
        return VariantMatch(exact[0], EXACT)
    if exact:
        # Two variants already claim this scope. Something upstream is wrong and
        # guessing between them would make it permanent.
        return VariantMatch(_mint(), AMBIGUOUS)

    widenings = [c for c, value in scopes.items() if value and value < wanted]
    narrowings = [c for c, value in scopes.items() if value and wanted < value]
    if len(widenings) == 1 and not narrowings:
        return VariantMatch(widenings[0], WIDENED)
    if len(narrowings) == 1 and not widenings:
        return VariantMatch(narrowings[0], NARROWED)
    if widenings or narrowings:
        return VariantMatch(_mint(), AMBIGUOUS)

    # Overlapping without containment: partly the same people, partly not.
    # Neither a continuation nor plainly separate, so nothing is claimed.
    if any(value & wanted for value in scopes.values()):
        return VariantMatch(_mint(), AMBIGUOUS)
    return VariantMatch(_mint(), NEW)


def _mint() -> str:
    return f"var_{uuid.uuid4().hex[:16]}"


def new_assertion_id() -> str:
    """A surrogate for one assertion, minted when it first appears.

    Opaque and never computed from content, so a rewording changes what the
    assertion says without changing which assertion it is — and a faulty
    matcher cannot retroactively decide what the database considers identity.
    """
    return f"a_{uuid.uuid4().hex[:16]}"
