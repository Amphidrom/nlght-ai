# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Knowledge extraction pipeline types.

Mirrors the prototype's staged shape: a parser turns a document into
``KnowledgeUnit``s — the exact text handed to the model — extraction turns those
into ``ExtractedItem``s, and persistence turns those into graph assertions.

Identity is derived from the assertion's own content, so the same claim
extracted from two documents reinforces one assertion instead of creating two.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from nlght.core.ingestion.document import stable_digest
from nlght.core.knowledge.entity import entity_key_or_none
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
from nlght.core.knowledge.legacy import legacy_payload, required_fields

#: How much an anchor is worth, which is not the same question as whether there
#: is one.
#:
#:     authored   the source wrote this id — `[[section]]`, `id="section"`.
#:                Survives a rename, a reorder, an insertion above it, a rewrite.
#:     derived    computed from the heading path. Finds the section again while
#:                nobody renames a heading, and is a *signal* rather than an
#:                identity — which is why the resolver has fallback rungs.
#:     none       no headings at all. Nothing to resolve to, so no slot.
AUTHORED_ANCHOR = "authored"
DERIVED_ANCHOR = "derived"
NO_ANCHOR = "none"


@dataclass(slots=True, frozen=True)
class KnowledgeUnit:
    """One coherent block of text, as handed to the extraction model.

    Where it came from is carried in **fields, not in the metadata dict.** It
    was a convention before — every parser was expected to put `document_id`
    and `processing_revision_id` into `metadata` — and the second parser duly put in half
    of it. Every candidate then reached the graph and none reached the lineage,
    because an assertion with no document revision cannot answer what document D
    at revision R asserted, which is the question the incremental diff is built
    on. Seventeen assertions, no history, and nothing said why.

    A field cannot be forgotten the way a dict key can, and the constructor
    refuses a unit that does not know which document it came from.
    """

    #: Where this section sits in the document, and nothing more. It was called
    #: `unit_id`, and the name was doing damage: it read like an identity and was
    #: used as one, so a paragraph inserted at the top renumbered every section
    #: below it and every claim they held was retracted and recreated. Slot
    #: identity is resolved from what a section can be found by; this survives as
    #: ordering and as a trace back to the batch a claim came from.
    unit_ordinal: str
    source_id: str
    content: str
    document_id: str = ""
    processing_revision_id: str = ""
    #: What the source calls this document — a path, a page title, a URL.
    #:
    #: **Provenance, and nothing else.** It is carried so a later reader can be
    #: told which file a claim came from; it is never part of a slot's or an
    #: assertion's identity. A document that is renamed is the same document, so
    #: a path that moves must produce no new assertion and no revision.
    #:
    #: A field rather than a metadata key, for the reason `document_id` is one:
    #: one parser put it in `metadata` and the other did not, so half the corpus
    #: silently had no path at all — which is exactly how this was discovered.
    document_path: str = ""
    #: What a later run can find this section by, and how much that is worth.
    #: Fields rather than metadata keys for the same reason the provenance is:
    #: a field cannot be forgotten the way a dict key can, and the last time a
    #: parser forgot one, every candidate reached the graph and none reached the
    #: lineage.
    anchor: str = ""
    anchor_strength: str = NO_ANCHOR
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.unit_ordinal.strip() or not self.source_id.strip():
            raise ValueError("knowledge unit identity must not be empty")
        if not self.content.strip():
            raise ValueError("knowledge unit content must not be empty")
        if not self.document_id.strip():
            raise ValueError("a knowledge unit must know which document it came from")

    @property
    def stable_slot_anchor(self) -> bool:
        """Whether a persisted slot id may be minted for this unit at all.

        The rule the slot design is measured against: an id may exist only where
        a later run can resolve its way back to it. A section nothing can find
        again would get a slot that is new on every run, and retire and recreate
        everything it holds each time — worse than no slot, because it looks
        like stability and provides none.

        So a format with no headings gets no slot, permanently, and the document
        stays its smallest dependable diff boundary. That is a property of the
        format rather than a gap to close later.
        """
        return self.anchor_strength != NO_ANCHOR

    @property
    def provenance(self) -> dict[str, str]:
        """What an assertion's evidence needs in order to be diffable later."""
        return {
            "document_id": self.document_id,
            "processing_revision_id": self.processing_revision_id,
            "document_path": self.document_path,
        }


def fold_whitespace(text: str) -> str:
    """The same words with their spacing normalised, and nothing else changed.

    The one tolerance the observed wording is allowed. A parser may rewrap a
    line and that changes no word; anything beyond it — a different word, an
    added clause, a full stop — is a different sentence and is treated as one.

    Deliberately weaker than `normalise`, which folds case and punctuation to
    decide *continuity*. This decides what was **observed**, so it may not fold
    away a difference a reader would see.
    """
    return " ".join(text.split())


@dataclass(slots=True, frozen=True)
class ExtractedItem:
    """One candidate assertion, before validation and review classification."""

    kind: str
    type: str
    content: dict[str, Any]
    confidence: float
    #: The sentence the source states this claim in, exactly as it stands there.
    #:
    #: Authoritative for the wording, and the structured `content` never is. A
    #: claim is quoted back to a reader — in a review queue, in a citation, in a
    #: diff — and quoting it from the fields means showing somebody a sentence
    #: their source does not contain. For a `fact` the fields cannot even attempt
    #: it: `subject`/`predicate`/`object` are not a sentence, and joining them
    #: produced "requires spring_boot java_17".
    #:
    #: Verified rather than trusted (`verified_against`). A field the model fills
    #: in is another model-generated string, and one stored under the name
    #: "observed" while holding a paraphrase is worse than no wording at all.
    observed_text: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in _PAYLOAD_BUILDERS:
            raise ValueError(
                f"unknown knowledge kind '{self.kind}'; expected one of "
                f"{sorted(_PAYLOAD_BUILDERS)}"
            )
        if not self.type.strip():
            raise ValueError("extracted item type must not be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("extracted confidence must be within [0.0, 1.0]")

    def verified_against(self, source: str) -> bool:
        """Whether the wording claimed as observed really occurs in the source.

        Exact, after folding whitespace — a parser may rewrap a line, and that
        changes no word. Deliberately not fuzzy: a fuzzy check admits paraphrase,
        which is the single thing this rejects. "Spring Boot requires Java 17"
        and "Java 17 is required by Spring Boot" are the same claim and not the
        same wording, and only one of them is in the document.
        """
        if not self.observed_text.strip():
            return False
        return fold_whitespace(self.observed_text) in fold_whitespace(source)

    @property
    def proposition(self) -> Proposition:
        """What this candidate says, structured — for every kind.

        The complete extracted interpretation, kept exactly as it arrived. A
        fact's fields are a fact's, a rule's are a rule's, and the platform reads
        none of them as meaning anything: `fields` has no privileged roles, which
        is why one type serves all four without becoming an ontology.

        It was fact-only, and the asymmetry produced a special path everywhere it
        touched — equivalence could not judge a rule, a revision could name a
        fact's structure and nothing else's, and every surface had to ask which
        kind it was holding.
        """
        return Proposition(dict(self.content))

    def payload(self) -> KnowledgePayload | None:
        """The kind's legacy row, *derived* from the proposition, or nothing.

        A projection and never a second source of truth. The proposition is what
        the claim says; a payload is one consumer's shape for it, present only
        while that shape can hold the whole claim. "Alice transfers CHF 500 to
        Bob" produces none — a row saying "Alice transfers CHF 500" is
        well-formed, reads as true, and is not what the source said.

        For a fact this has been a derivation for several slices. The other three
        built their payload directly, which made them the truth for their kind
        and the proposition an extra; now all four are projections of one thing.
        """
        return legacy_payload(self.kind, self.proposition)

    def well_formed(self) -> None:
        """Raise if this candidate says nothing its kind can hold.

        The gate extraction rejects on, and deliberately not the same question as
        "does it make a payload". It used to be: a fact whose claim had four
        parts could not form the three-column payload, so a model that read the
        sentence correctly had its answer discarded as malformed and the corpus
        was left holding only the claims that happened to fit an old table.

        What is left for a fact is whether it can name what it claims — which is
        exactly the question `entity_key` already answers, and asking it here is
        the point rather than a convenience. A fact that cannot form a key can
        never be *placed*: no key means no lineage, no lineage means no revision,
        and a revision is where a proposition lives. Accepting one produced a
        graph row with no payload, no proposition and no revision — a node that
        says nothing, which is a worse outcome than the rejection it replaced.

        So the gate is the same rule the placement will apply, asked at the
        boundary where a candidate can still be counted and reported. It stood
        for one slice as "at least two fields", which was an approximation of
        this and let `{subject, object}` through — two parts, nothing naming the
        relation between them.
        """
        fields = self.proposition.as_dict()
        if self.kind == FACT:
            # A fact must name what it claims and something standing in it —
            # exactly what an entity key requires, asked here where a candidate
            # can still be counted rather than at the placement.
            if entity_key_or_none(self.kind, **fields) is None:
                raise ValueError(
                    f"a fact must name what it claims and something standing in it; "
                    f"got {sorted(fields)}"
                )
            return
        # The other kinds are identified by a canonical pair a corpus written
        # before it existed does not have, so an entity key cannot be the test
        # for them. What they must carry is the body a citation would quote.
        missing = [
            name
            for name in required_fields(self.kind)
            if not str(fields.get(name, "")).strip()
        ]
        if missing:
            raise ValueError(
                f"a '{self.kind}' needs {', '.join(missing)}; got {sorted(fields)}"
            )

    @property
    def fingerprint(self) -> str:
        """A hash of what this candidate says. **Not the identity of a claim.**

        It was called `identity`, and the name was doing damage: it addressed the
        graph node and it decided which assertion a sighting belonged to, and
        only the first of those is a job a content hash can hold. A claim whose
        wording moved hashed differently, so the corpus opened a second assertion
        for it and left the approval on the first (ADR-0043).

        Which assertion a sighting continues is now resolved from its slot and
        its wording. What remains here are the two things a content hash is
        actually good for: the address of the graph node — the same claim found
        in two places reinforces one node rather than duplicating it — and the
        fingerprint of a revision, where "the same content means the same state"
        is exactly the question.
        """
        canonical = json.dumps(
            {key: self.content[key] for key in sorted(self.content)},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).lower()
        return stable_digest(self.kind, self.type.lower(), canonical)


def _fact(content: dict[str, Any]) -> FactPayload:
    return FactPayload(
        subject=str(content["subject"]),
        predicate=str(content["predicate"]),
        object=str(content["object"]),
    )


def _rule(content: dict[str, Any]) -> RulePayload:
    return RulePayload(rule_text=str(content["rule_text"]))


def _pattern(content: dict[str, Any]) -> PatternPayload:
    return PatternPayload(
        pattern_name=str(content["pattern_name"]),
        description=str(content["description"]),
    )


def _decision(content: dict[str, Any]) -> DecisionPayload:
    return DecisionPayload(
        decision=str(content["decision"]),
        effect=str(content["effect"]),
    )


_PAYLOAD_BUILDERS: dict[str, Callable[[dict[str, Any]], KnowledgePayload]] = {
    FACT: _fact,
    RULE: _rule,
    PATTERN: _pattern,
    DECISION: _decision,
}


SEGMENTATION_VERSION = "1"
"""Bumped when `split_into_units` changes where it cuts.

Part of the extraction version, because segmentation decides what the model is
shown. A document's text can be byte-identical while the blocks carved out of it
are different, and the model answers a different question — so a change here has
to re-extract the corpus exactly as a prompt or a model change does.
"""


def split_into_units(
    text: str,
    *,
    source_id: str,
    document_id: str,
    processing_revision_id: str = "",
    document_path: str = "",
    metadata: dict[str, Any] | None = None,
    tags: dict[str, Any] | None = None,
) -> tuple[KnowledgeUnit, ...]:
    """Split prose into blocks on blank lines and ALL-CAPS section headings.

    The prototype's rule: a model reasons better over one coherent block than
    over a whole document, and blank lines are the most reliable structural
    signal across plain text, Markdown, and converted PDFs.
    """
    units: list[KnowledgeUnit] = []
    buffer: list[str] = []

    def flush() -> None:
        content = "\n".join(buffer).strip()
        buffer.clear()
        if content:
            units.append(
                KnowledgeUnit(
                    unit_ordinal=f"{source_id}:{document_id}:{len(units)}",
                    source_id=source_id,
                    content=content,
                    document_id=document_id,
                    processing_revision_id=processing_revision_id,
                    document_path=document_path,
                    metadata=dict(metadata or {}),
                    tags=dict(tags or {}),
                )
            )

    for raw in text.splitlines():
        line = raw.strip()
        if not line or (line.isupper() and len(line) > 3):
            flush()
            continue
        buffer.append(line)
    flush()

    if not units and text.strip():
        units.append(
            KnowledgeUnit(
                unit_ordinal=f"{source_id}:{document_id}:0",
                source_id=source_id,
                content=text.strip(),
                document_id=document_id,
                processing_revision_id=processing_revision_id,
                document_path=document_path,
                metadata=dict(metadata or {}),
                tags=dict(tags or {}),
            )
        )
    return tuple(units)
