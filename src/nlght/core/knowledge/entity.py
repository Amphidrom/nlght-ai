# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Separate claim identity, simultaneous variants, and revisions."""

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
    # Reversing a decision is a new state of that decision. What was chosen
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
    """Signal missing identity fields without inventing a key.

    Older assertions can lack those fields until they are extracted again.
    """


class EntityKeyVersionMismatch(ValueError):
    """Stop when one business question has incompatible key schemes.

    A migration must reconcile old keys before a new scheme can match them.
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
    """Derive a versioned key from the kind-specific identity fields.

    Args:
        kind (str): Kind of knowledge to identify.
        **fields (object): Kind-specific identity fields used to derive the key.

    Returns:
        EntityKey: EntityKey for the assertion.

    Raises:
        EntityKeyUnavailable: The kind or required identity fields are missing.
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
    """Resolve an entity key without raising for absent fields.

    Args:
        kind (str): Kind of knowledge to identify.
        **fields (object): Kind-specific identity fields used to derive the key.

    Returns:
        EntityKey | None: EntityKey or None when unavailable.
    """
    try:
        return entity_key(kind, **fields)
    except EntityKeyUnavailable:
        return None


def _scope(values: object) -> frozenset[str]:
    """Normalize variant scope for set comparison.

    Args:
        values (object): Optional scalar or iterable scope values.

    Returns:
        frozenset[str]: Set of canonical scope terms.
    """
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
    """Carry a matched variant ID and its resolution for audit or review."""

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
    """Match a scope to an existing or newly minted variant.

    Args:
        entity (object): Entity being resolved.
        scope (object): Scope of the arriving observation.
        known (Mapping[str, object]): Known variants keyed by identity.

    Returns:
        VariantMatch: VariantMatch with resolution and surrogate ID.

    Side Effects:
        May generate a random surrogate ID.
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
    """Mint a content-independent assertion identifier.

    Returns:
        str: New opaque assertion ID.

    Side Effects:
        Reads random UUID state.
    """
    return f"a_{uuid.uuid4().hex[:16]}"
