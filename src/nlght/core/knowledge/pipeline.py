# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Define units and candidates for the knowledge extraction pipeline."""

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
#:     authored   the source wrote this id: `[[section]]`, `id="section"`.
#:                Survives a rename, a reorder, an insertion above it, a rewrite.
#:     derived    computed from the heading path. Finds the section again while
#:                nobody renames a heading, and is a *signal* rather than an
#:                identity. The resolver therefore has fallback rungs.
#:     none       no headings at all. Nothing to resolve to, so no slot.
AUTHORED_ANCHOR = "authored"
DERIVED_ANCHOR = "derived"
NO_ANCHOR = "none"


@dataclass(slots=True, frozen=True)
class KnowledgeUnit:
    """One coherent block of text, as handed to the extraction model.

    Where it came from is carried in **fields, not in the metadata dict.** It
    was a convention before. Every parser was expected to put `document_id`
    and `processing_revision_id` into `metadata`, and the second parser put in half
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
    #: identity is resolved from what a section can be found by. This survives as
    #: ordering and as a trace back to the batch a claim came from.
    unit_ordinal: str
    source_id: str
    content: str
    document_id: str = ""
    processing_revision_id: str = ""
    #: What the source calls this document: a path, a page title, or a URL.
    #:
    #: **Provenance, and nothing else.** It is carried so a later reader can be
    #: told which file a claim came from. It is never part of a slot's or an
    #: assertion's identity. A document that is renamed is the same document, so
    #: a path that moves must produce no new assertion and no revision.
    #:
    #: A field rather than a metadata key, for the reason `document_id` is one:
    #: one parser put it in `metadata` and the other did not, so half the corpus
    #: silently had no path at all. That exposed the missing field.
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
        """Permit a slot only when a later run can resolve its anchor.

        Returns:
            bool: Whether the unit can have a persistent slot.
        """
        return self.anchor_strength != NO_ANCHOR

    @property
    def provenance(self) -> dict[str, str]:
        """Expose evidence needed for later document comparisons.

        Returns:
            dict[str, str]: Document ID, processing revision ID, and path.
        """
        return {
            "document_id": self.document_id,
            "processing_revision_id": self.processing_revision_id,
            "document_path": self.document_path,
        }


def fold_whitespace(text: str) -> str:
    """Normalize spacing without changing words.

    Args:
        text (str): Source text.

    Returns:
        str: Whitespace-folded text.
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
    #: claim is quoted back to a reader in a review queue, citation, or diff.
    #: Quoting it from the fields means showing somebody a sentence
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
        """Verify extracted wording against the source after folding spacing.

        Args:
            source (str): Source text against which this candidate is verified.

        Returns:
            bool: Whether the exact wording occurs in the source.
        """
        if not self.observed_text.strip():
            return False
        return fold_whitespace(self.observed_text) in fold_whitespace(source)

    @property
    def proposition(self) -> Proposition:
        """Preserve all extracted fields as one structured claim.

        Returns:
            Proposition: Proposition for any built-in knowledge kind.
        """
        return Proposition(dict(self.content))

    def payload(self) -> KnowledgePayload | None:
        """Project a claim into its lossless legacy payload when possible.

        Returns:
            KnowledgePayload | None: Built-in payload or None if projection loses content.
        """
        return legacy_payload(self.kind, self.proposition)

    def well_formed(self) -> None:
        """Reject candidates that cannot express or place a claim.

        Side Effects:
            None. Raises for malformed or unplaceable claims.
        """
        fields = self.proposition.as_dict()
        if self.kind == FACT:
            # A fact must name what it claims and something standing in it:
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
        """Address exact claim content without defining assertion identity.

        Returns:
            str: Stable content fingerprint.
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
are different, and the model answers a different question. A change here has
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
    """Split prose on blank lines and all-capital headings.

    Args:
        text (str): Source text to segment.
        source_id (str): Ingestion source identity.
        document_id (str): Document identity.
        processing_revision_id (str): Processing revision of the document.
        document_path (str): Source document path.
        metadata (dict[str, Any] | None): Optional document metadata.
        tags (dict[str, Any] | None): Optional document tags.

    Returns:
        tuple[KnowledgeUnit, ...]: Ordered KnowledgeUnit values.
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
