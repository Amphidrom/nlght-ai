# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""What was extracted, held whole and left uninterpreted.

The one thing this answers: *can we keep, completely, what the extraction
produced?* Nothing else. It does not say what a proposition is made of, which
field names a claim should have, which fields carry identity, whether arguments
are ordered, or whether two differently shaped claims say the same thing.

    **Representation may preserve meaning. It may not invent any.**

The restraint is not fastidiousness; it is the same defect three times over. The
entity key followed the model's *word* choice, and a corpus nobody edited grew
assertions (ADR-0043). `predicate + roles` would have followed its *label*
choice, moving the defect rather than removing it. `as_triple` made the legacy
payload's shape the privileged one inside core, deciding a representation while
the working entry recorded the question as open. Each was one small helpful
assumption.

So there is no `predicate` field here, and no `arguments` list. Both would be a
model: one says a proposition has a distinguished relation name, the other that
its parts are ordered. Either may turn out true, and neither is knowable from the
shape today's extraction happens to emit.

Unstable field names are not a defect at this level. A model saying `recipient`
on one pass and `target` on the next has described one claim two ways, and
noticing that is *equivalence* — a separate question, asked of two complete
structures, and only askable once both can be held completely. This is what makes
it askable.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

FINGERPRINT_VERSION = "1"
"""Which scheme produced a fingerprint, stored beside it and never folded in.

Normalisation and hashing will change. Folded into the value, an old fingerprint
would merely *differ* from a new one and nothing could tell that from two
different claims — the same reason `entity_key_version` is a column of its own.
"""


PropositionValue = (
    str | int | float | bool | None | Mapping[str, Any] | Sequence[Any]
)
"""Whatever the extraction produced: a scalar, a nested map, or an ordered list.

Wide on purpose. Narrowing it would be deciding what a claim is allowed to say.
"""


def _freeze(value: Any) -> Any:  # noqa: ANN401 (it holds whatever was extracted)
    """Make a value immutable without changing what it says.

    Lists become tuples and maps become plain dicts, so a proposition cannot be
    edited underneath its holder and a round trip returns the same thing. A
    structural change and not a semantic one: order inside a sequence is kept
    exactly, because losing it would lose something the extraction expressed.
    """
    if isinstance(value, Mapping):
        return {str(key): _freeze(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:  # noqa: ANN401
    """The same structure in JSON's vocabulary, for storing and sending."""
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(slots=True, frozen=True)
class Proposition:
    """One extracted claim of any kind, exactly as it arrived.

    A property of every knowledge assertion, not of facts. It was fact-only for
    several slices and the asymmetry kept producing special paths: a `rule` had
    no structured form to compare, so equivalence could not judge one; the
    revision could name a fact's structure and nothing else's; and every new
    surface had to ask which kind it was holding.

    Nothing about `fields` was ever fact-shaped — that is why generalising it
    costs nothing and invents nothing. There is no `subject`, no `predicate`, no
    roles; a rule's fields are a rule's, a decision's are a decision's, and the
    platform reads none of them as meaning anything.

    Equality is over the fields and their values. The *order* of field names
    carries no meaning — two extractions that named the same parts in a different
    order described the same structure, and treating that as a difference would
    make a claim depend on how a model happened to emit its JSON. Order *inside*
    a list does carry meaning, because expressing it was a choice the extraction
    made.
    """

    fields: Mapping[str, PropositionValue]

    def __post_init__(self) -> None:
        if not self.fields:
            # The single judgement representation is allowed: that there is
            # something to hold. A claim that says nothing is not a claim.
            raise ValueError("a proposition needs at least one field")
        object.__setattr__(self, "fields", _freeze(self.fields))

    @property
    def fingerprint(self) -> str:
        """A handle for referring to this exact structure. **Not an identity.**

        Two propositions with the same fingerprint are byte-identical after
        normalisation; two with different ones may still be the same claim, which
        is the whole subject of `equivalence`. It exists so a decision about a
        *pair* can name its sides without copying them.
        """
        from nlght.core.ingestion.document import stable_digest  # noqa: PLC0415

        return stable_digest(
            *(f"{key}={_thaw(value)!r}" for key, value in sorted(self.fields.items()))
        )

    def as_dict(self) -> dict[str, Any]:
        return {str(key): _thaw(value) for key, value in self.fields.items()}

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> Proposition:
        return cls(fields=dict(values))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Proposition):
            return NotImplemented
        return dict(self.fields) == dict(other.fields)

    def __hash__(self) -> int:
        return hash(tuple(sorted((key, str(value)) for key, value in self.fields.items())))
