# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""One thing a store returned, with its native shape intact.

A retrieval hit is deliberately *not* a normalised passage. The three stores
answer at three different sizes and they are kept that way:

    lexical    a document    the whole indexed file
    vector     a chunk       the part of a file that matched
    knowledge  an assertion  a reviewed claim

Turning those into one shape here would be answering a question that belongs to
context building — *which text do I actually give the model* — several steps too
early, and it would do it with no view of a token budget or of what else was
retrieved. So search says "these three systems found these things", and nothing
more.

`raw_score` is carried and never compared across stores. BM25 lands near 7,
cosine below 1, and a confidence is a probability; the numbers are not on one
scale and no amount of normalisation makes them one. What *is* comparable is
`source_rank`: this store put this hit above that one. Fusion works on that.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from nlght.core.knowledge.knowledge import SupportingEvidence

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
    later lets context building see that they belong together — *see*, not
    merge. Two hits sharing a document are related, and relatedness is not
    identity: a document, a chunk of it and an assertion read out of it are
    three findings about one file, and collapsing them here would throw away the
    fact that three independent systems each found something.
    """

    document_id: str = ""
    chunk_id: str = ""
    assertion_id: str = ""
    #: The source's own fassung of the document — what the source called this
    #: version of it. Provenance for a citation, and the thing that does *not*
    #: change when only the pipeline changes.
    source_revision_id: str = ""
    #: The processed fassung: source content plus the classifier, enricher and
    #: chunking that produced this text. Chunk offsets index it, and it is what
    #: the currency check compares against the published revision.
    processing_revision_id: str = ""
    #: The state of a claim, which is not a document revision at all (ADR-0051).
    #: Only an assertion has one.
    knowledge_revision_id: str = ""
    #: Where a chunk sits in its document's processed text. `end_offset` is
    #: exclusive, so `content[start_offset:end_offset]` is the span — and that
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
    #: Where a claim was observed — and only a claim has this.
    #:
    #: A document and a chunk *are* their document, so the fields above say
    #: everything there is to say about where they came from. An assertion is
    #: not in a document: it is supported by however many sightings the corpus
    #: holds, and a second document asserting it is what keeps it alive when the
    #: first stops (ADR-0044). Writing one of them into `document_id` would make
    #: whichever source was read first look like the claim's home, which is the
    #: one thing this must not do — so it is a list, and `document_id` stays
    #: empty for an assertion.
    support: tuple[SupportingEvidence, ...] = ()

    @property
    def state_revision(self) -> str:
        """Which state this finding is on, whatever kind of thing it is.

        A derivation, not a fourth field, and it exists for exactly one question:
        two findings are the same finding only if they are on the same state. A
        document and a chunk are on a processing revision; a claim is on a
        knowledge revision; the two are not comparable and never share a value.

        The stored fields keep one meaning each — that is the whole point of
        splitting them — and this reads whichever of them the carrier actually
        has. Answering by carrier at each call site instead would put carrier
        knowledge in every consumer and give the same answer until one of them
        drifted.
        """
        return self.processing_revision_id or self.knowledge_revision_id


@dataclass(slots=True, frozen=True)
class RetrievalHit:
    """One result, as the store that produced it saw it."""

    source: str
    carrier: str
    #: Identity *within its carrier space*. A chunk id and a document id may
    #: coincide as strings and still be different things, so nothing keys on
    #: this alone — `key` is what identifies a hit.
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
        """What makes this hit *this* hit, across every store.

        The carrier and its id together. Two stores returning the same chunk are
        one finding seen twice — that is evidence, and fusion counts it. A
        document and a chunk of it are two findings, and no shared
        `document_id` makes them one.
        """
        return (self.carrier, self.carrier_id)

    def terms(self, name: str) -> tuple[str, ...]:
        return tuple(self.fields.get(name, ()))


def as_fields(metadata: Mapping[str, object]) -> dict[str, tuple[str, ...]]:
    """A store's metadata as terms, dropping what has none.

    Numbers, timestamps and nested objects are not terms, and stringifying them
    would put a revision id into the same space as a symbol name.
    """
    fields: dict[str, tuple[str, ...]] = {}
    for name, value in metadata.items():
        if isinstance(value, str):
            if value.strip():
                fields[name] = (value,)
        elif isinstance(value, Sequence):
            terms = tuple(str(item) for item in value if str(item).strip())
            if terms:
                fields[name] = terms
    return fields
