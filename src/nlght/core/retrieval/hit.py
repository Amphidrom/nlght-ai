# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Represent store-native retrieval hits and their provenance."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from nlght.core.knowledge.knowledge import SupportingEvidence
from nlght.core.model.trust import UNANNOTATED, ContentOrigin

#: Which system answered.
LEXICAL = "lexical"
VECTOR = "vector"
KNOWLEDGE = "knowledge"

SOURCES = (LEXICAL, VECTOR, KNOWLEDGE)

#: What size of thing a hit is. Kept beside the source because they are not the
#: same question: a store is where an answer came from, a carrier is what was
#: returned, and a second lexical index over chunks would break the assumption
#: that one implies the other.
DOCUMENT = "document"
CHUNK = "chunk"
ASSERTION = "assertion"

CARRIERS = (DOCUMENT, CHUNK, ASSERTION)


@dataclass(slots=True, frozen=True)
class Provenance:
    """Where a hit came from, completely enough to cite it and to relate it.

    `document_id` is present on a chunk as well as on a document, which is what
    later lets context building see that they belong together. It does not
    merge. Two hits sharing a document are related, and relatedness is not
    identity: a document, a chunk of it and an assertion read out of it are
    three findings about one file, and collapsing them here would throw away the
    fact that three independent systems each found something.
    """

    document_id: str = ""
    chunk_id: str = ""
    assertion_id: str = ""
    #: Keep the source's revision ID for citation. It does *not*
    #: change when only the pipeline changes.
    source_revision_id: str = ""
    #: Keep the processed revision separate from the source revision because
    #: enrichment and chunking can change without source edits. Chunk offsets
    #: and the currency check use this processed revision.
    processing_revision_id: str = ""
    #: The state of a claim, which is not a document revision at all (ADR-0051).
    #: Only an assertion has one.
    knowledge_revision_id: str = ""
    #: Where a chunk sits in its document's processed text. `end_offset` is
    #: exclusive, so `content[start_offset:end_offset]` is the span. That
    #: is how a chunk is answered: cut from the stored revision, never taken
    #: from the payload's copy of it (ADR-0062).
    #:
    #: Zero on a document and on an assertion, which name no span. A document
    #: resolves to the whole revision rather than to an invented range.
    position: int = 0
    start_offset: int = 0
    end_offset: int = 0
    path: str = ""
    source_name: str = ""
    external_id: str = ""
    #: Keep all current claim supports rather than assigning one document as a
    #: claim's home. A second document can keep the claim alive when the first
    #: retracts it. Assertion `document_id` therefore stays empty (ADR-0044).
    support: tuple[SupportingEvidence, ...] = ()
    #: The source this finding came from and whether the store's operator
    #: trusts it (ADR-0074). Unset is unannotated, and unannotated is untrusted.
    origin: ContentOrigin = UNANNOTATED

    @property
    def state_revision(self) -> str:
        """Resolve the revision that identifies this finding's state.

        Returns:
            str: Processing revision when present, otherwise knowledge revision.
        """
        return self.processing_revision_id or self.knowledge_revision_id


@dataclass(slots=True, frozen=True)
class RetrievalHit:
    """One result, as the store that produced it saw it."""

    source: str
    carrier: str
    #: Identity *within its carrier space*. A chunk id and a document id may
    #: coincide as strings and still be different things, so nothing keys on
    #: this alone. Use `key` to identify a hit.
    carrier_id: str
    content: str
    provenance: Provenance = field(default_factory=Provenance)
    #: What the store said, in the store's own units. Never compared with
    #: another store's, and kept so a ranking can be explained and audited.
    raw_score: float = 0.0
    #: Where the store placed it, 1-based. This is the comparable quantity.
    source_rank: int = 0
    #: The structured terms the pipeline indexed alongside it.
    fields: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.source not in SOURCES:
            raise ValueError(f"unknown retrieval source '{self.source}'")
        if self.carrier not in CARRIERS:
            raise ValueError(f"unknown retrieval carrier '{self.carrier}'")

    @property
    def key(self) -> tuple[str, str]:
        """Identify a hit across stores without collapsing carriers.

        Returns:
            tuple[str, str]: Carrier-and-ID identity pair.
        """
        return (self.carrier, self.carrier_id)

    def terms(self, name: str) -> tuple[str, ...]:
        return tuple(self.fields.get(name, ()))


def as_fields(metadata: Mapping[str, object]) -> dict[str, tuple[str, ...]]:
    """Extract textual search terms from store metadata.

    Args:
        metadata (Mapping[str, object]): Mapping of arbitrary metadata values.

    Returns:
        dict[str, tuple[str, ...]]: Nonempty strings and textual sequence members keyed by field.
    """
    fields: dict[str, tuple[str, ...]] = {}
    for name, value in metadata.items():
        if isinstance(value, str):
            if value.strip():
                fields[name] = (value,)
        elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
            terms = tuple(item for item in value if isinstance(item, str) and item.strip())
            if terms:
                fields[name] = terms
    return fields
