# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Which assertion a freshly extracted proposition continues, if any.

The rule this exists to enforce:

    a model-generated description is not an identity

A live run showed what happens without it. An unedited sentence — "Each
SanitizingFunction is called in order until a function changes the value" — came
back from the model as `sanitizing_function_execution/ordered_until_changed` on
one pass and `sanitizing_functions/processing_order` on the next. Identity was
looked up by those fields, so one claim became two, and the approval stayed with
the one nobody would read again.

The entity key is therefore a **strong signal and not a precondition**. It says
"certainly the same claim" when it matches and nothing at all when it does not.

Deterministic throughout: no embeddings, no model. That is a deliberate limit
rather than a stage of construction — the ladder should first be shown to work,
and the case that breaks it is the argument for adding a semantic signal.

**Resemblance may raise doubt and may never settle it.** That is not caution, it
is what the measurements say. Lexical similarity runs backwards to what identity
needs:

    "Spring requires Java 17" ↔ "Spring requires Java 21"        0.96
    "Spring requires Java 17" ↔ "Spring requires Maven"          0.82
    "Expenses above CHF 500 require approval."
        ↔ "Approval is required for expenses above CHF 500."      0.51

The two claims that must stay apart resemble each other most; the paraphrase that
should join resembles least. No threshold does the right thing — a version bump
differs in two characters and a rewording rebuilds the sentence. So similarity
only ever produces *ambiguous* or *new* here, and every continuation is decided
by exact agreement on the wording or the key.

That is also the concrete argument for the one delegated judgement in
`semantics.py` (ADR-0045): the CHF paraphrase is a real continuation this cannot
see, and the fix is not a lower threshold — one low enough to catch it merges
Java 17 with Java 21 long before. A model asked the narrow question answers it
correctly; nothing over characters can.

**The decision rule is not "highest score wins".** A sum-and-maximum would pair
every new claim with whatever it least dislikes, which is how "Spring requires
Java 17" quietly becomes "Spring requires Java 21" under an old approval — the
two differ in two characters and resemble each other almost perfectly.

    exact agreement, and only one thing it can be  → continue that claim
    more than one thing it plausibly is            → ambiguous
    otherwise                                      → a new claim

Ambiguity is not a failure to be resolved by picking one. It is the honest
answer, and it is recorded so a better signal can be applied later to exactly
the cases that needed one.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from nlght.core.knowledge.lineage import AMBIGUOUS_FROM

#: How the continuation was decided. Recorded rather than discarded: which rung
#: answered says whether the corpus is held together by exact agreement or by
#: resemblance, and those are worth very different amounts of trust.
SLOT_TEXT = "slot_text"
SLOT_KEY = "slot_key"
DOCUMENT_TEXT = "document_text"
KEY = "key"
AMBIGUOUS = "ambiguous"
NEW = "new"


def normalise(text: str) -> str:
    """The wording, with what no proposition depends on removed.

    Case, whitespace and trailing punctuation. Nothing else: a normalisation
    that dropped words would start deciding which differences are material,
    which is the judgement this module is built to make explicitly.
    """
    return " ".join(str(text or "").split()).strip(" .;,:").lower()


def similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, normalise(left), normalise(right)).ratio()


def material_content(text: str) -> frozenset[str]:
    """The quantities a claim asserts, apart from the words carrying them.

    An entity key names the *question* an assertion answers and not the answer.
    A rule's key is its `subject` and `rule_property` — "what is the approval
    threshold for expenses" — while the threshold itself lives in the body. So a
    key that agrees says the two claims are about the same thing and nothing
    about whether they say the same thing about it, and confirming on it alone
    lets a changed rule be served under the approval given to the old one.

    Similarity cannot separate the two cases; it inverts them. Measured on the
    rule this was built for:

        "Expenses above CHF 500 require approval."
            ↔ "Expenses over CHF 500 must be signed off."     0.63  same claim
            ↔ "Expenses above CHF 550 require approval."      0.97  changed claim

    A rewording rebuilds the sentence around the quantity; a material change
    leaves the sentence and moves the quantity. So the quantities are compared
    directly, and the words are left to the rungs that require them to agree
    exactly.

    Deliberately narrow. It catches what a threshold, a version, a port or a
    count asserts, which is where materiality lives in technical and policy
    prose. It does not catch a change expressed purely in words — "must" into
    "must not" reads as a rewording here — and that class needs a semantic
    signal, the same one the paraphrase case needs.
    """
    return frozenset(
        token.strip(".,;:()[]{}\"'").lower()
        for token in str(text or "").split()
        if any(character.isdigit() for character in token)
    )


@dataclass(slots=True, frozen=True)
class Candidate:
    """An assertion already in the corpus, as the matcher sees it."""

    assertion_id: str
    entity_key: str
    #: The scheme the key was minted under. Compared with the key and never
    #: apart from it: two schemes can produce one string for different claims,
    #: which is the whole reason the version is a column of its own.
    entity_key_version: str
    #: The current wording — the latest revision's, not the first.
    text: str
    document_id: str
    #: What kind of claim it is. Required in practice: the ladder matches nothing
    #: without it, because a candidate that cannot say which identity namespace
    #: it belongs to may not be continued into.
    #:
    #: It became load-bearing when the wording became the source's own sentence
    #: (ADR-0048). Before that the text was the kind's own body — `rule_text` for
    #: a rule, `object` for a fact — so two claims read out of one sentence had
    #: different text and the text rungs could never confuse them. Now they have
    #: *identical* text, and the strongest rung matched a rule onto a fact's
    #: assertion; the graph's kind guard then refused the write and failed the
    #: run, which is the guard working and the ladder having answered wrongly.
    kind: str = ""
    #: Every slot this assertion has been seen in. A claim can stand in more
    #: than one place, and arriving in any of them is arriving where it stood.
    slot_ids: frozenset[str] = field(default_factory=frozenset)


@dataclass(slots=True, frozen=True)
class PropositionMatch:
    """Which assertion this proposition is, and how that was decided."""

    assertion_id: str | None
    resolution: str

    @property
    def is_continuation(self) -> bool:
        return self.assertion_id is not None


def _unique(candidates: Sequence[Candidate]) -> Candidate | None:
    """The one candidate, or nothing.

    Every rung requires a unique answer. Two candidates mean the rung cannot
    decide, and choosing between them is the heuristic that would quietly define
    identity — the thing the surrogate id exists to prevent.
    """
    return candidates[0] if len(candidates) == 1 else None


def match_proposition(
    *,
    entity_key: str,
    entity_key_version: str,
    text: str,
    slot_id: str | None,
    document_id: str,
    candidates: Iterable[Candidate],
    kind: str,
) -> PropositionMatch:
    """Which assertion this proposition continues.

    A ladder, strongest signal first, each rung requiring a unique answer:

        the same slot, saying the same thing
        → the same slot, under the same entity key and scheme
        → the same document, saying the same thing   (a claim that moved)
        → the same entity key anywhere               (a claim found elsewhere too)
        → otherwise ambiguous when several claims plausibly fit, else new

    The order is not arbitrary. Slot before document, because a slot is where
    the diff runs and a match inside it needs no recovery. Text before key,
    because the key is what the model chose and the text is what the source
    said — when they disagree, the source wins.

    And every rung that confirms requires exact agreement. Nothing is continued on
    resemblance, because the wordings that must stay apart resemble each other
    more closely than the ones that belong together.
    """
    # Never across kinds, and a candidate that cannot say what kind it is does
    # not match at all. A `rule` does not continue a `fact`, whatever they have
    # in common — `kind` is part of the identity namespace, so continuing one as
    # the other produces a write the graph is right to refuse.
    #
    # Not a new rule; one that used to hold by accident. The text rungs compared
    # kind-specific bodies, which differed. Since the wording is the source's own
    # sentence (ADR-0048), one sentence read as both a rule and a fact gives two
    # claims with *identical* text and the strongest rung fused them.
    #
    # An unknown kind is refused rather than treated as a wildcard, and the
    # difference matters: a wildcard would let exactly the fused match back in
    # through a candidate that happened to carry no kind, which is a hole in the
    # wall this rule is. Failing to continue an old assertion costs a duplicate
    # somebody can merge; merging two identity namespaces costs a claim, and
    # nothing afterwards shows which.
    known = [c for c in candidates if c.kind == kind]
    wording = normalise(text)

    in_slot = [
        candidate
        for candidate in known
        if slot_id is not None and slot_id in candidate.slot_ids
    ]

    same_wording = [c for c in in_slot if normalise(c.text) == wording]
    if (found := _unique(same_wording)) is not None:
        return PropositionMatch(found.assertion_id, SLOT_TEXT)

    scheme = (entity_key_version, entity_key)
    same_key_here = [c for c in in_slot if (c.entity_key_version, c.entity_key) == scheme]
    if (found := _unique(same_key_here)) is not None:
        return PropositionMatch(found.assertion_id, SLOT_KEY)

    # The slot did not answer. A section that was renamed or reordered carries
    # its claims to a new slot, and the wording is what recognises them there.
    in_document = [c for c in known if c.document_id == document_id]
    moved = [c for c in in_document if normalise(c.text) == wording]
    if (found := _unique(moved)) is not None:
        return PropositionMatch(found.assertion_id, DOCUMENT_TEXT)

    # The same claim asserted by another document is one claim reinforced, not
    # two. This is the only rung that leaves the document, and it needs the key
    # to be certain, because wording alone across documents would merge every
    # piece of boilerplate in the corpus.
    same_key = [c for c in known if (c.entity_key_version, c.entity_key) == scheme]
    if (found := _unique(same_key)) is not None:
        return PropositionMatch(found.assertion_id, KEY)
    if same_key:
        return PropositionMatch(None, AMBIGUOUS)

    # Nothing agreed exactly, and resemblance is all that is left. It may raise
    # doubt and it may never settle one — see the module docstring for the
    # measurements that decided it. Consulted only inside the slot: across a
    # document it would pair claims that merely share a subject.
    plausible = [
        candidate
        for candidate in in_slot
        if similarity(candidate.text, text) >= AMBIGUOUS_FROM
    ]
    if len(plausible) > 1:
        # Several things it could be. Picking the best one here is exactly how a
        # corpus acquires a wrong continuation that no report will ever show, and
        # naming the doubt is what lets a better signal be tried on precisely
        # these cases later.
        return PropositionMatch(None, AMBIGUOUS)
    return PropositionMatch(None, NEW)
