# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""One name per revision, because there were three revisions and one name.

`revision_id` meant three different things depending on where you read it, and
nothing in the schema or the types said so:

    the source's fassung      what the source called this version of the file
    the processed fassung     that content plus the classifier, enricher and
                              chunking that produced the text — chunk offsets
                              index *this* one
    a claim's state           the knowledge revision a graph node represents
                              (ADR-0051), not a document revision at all

`ingestion_index_state` already knew the first two were different: it stores
`content_hash` (the source's) beside `enricher_hash` (the processed one). It
then stored a third column called `revision_id` holding the processed one again.
Meanwhile `Provenance.revision_id` carried whichever of the three the carrier
happened to have, and `passages_from` grouped every carrier by that one field —
so for an assertion the deduplication of findings ran over a knowledge revision
while for a document it ran over a processing revision, silently.

That is the shape of bug that does not announce itself. Every read is plausible,
every value is a hex string, and the first symptom is a citation pointing at a
revision that never said it.

So each identity gets its own name and keeps it everywhere — column, domain
type, index payload and provenance:

    source_revision_id
    processing_revision_id
    knowledge_revision_id

This migration renames the data side. `knowledge_revision_id` needs no column
change: inside the knowledge tables `revision_id` already means the knowledge
revision and nothing else, so it is correct there and is left alone. The name
only becomes ambiguous where the two domains meet, which is `Provenance`, and
that is a type rather than a table.

`current_revision_id` becomes `current_processing_revision_id` for the same
reason: it is the published *processed* fassung, the one a retrieval hit is
checked against, and a hit carries a processing revision to compare it with.

A rename rather than new columns beside the old. There is no production corpus
to keep compatible, and a compatibility column here would recreate exactly the
ambiguity the rename exists to remove — two names, both populated, drifting the
first time one write path forgets one.

Revision ID: 0021
Revises: 0020
"""

from __future__ import annotations

from alembic import op

revision: str = "0021"
down_revision: str | None = "0020"
branch_labels: str | None = None
depends_on: str | None = None

_RENAMES = (
    ("ingestion_document_revisions", "revision_id", "processing_revision_id"),
    ("ingestion_documents", "current_revision_id", "current_processing_revision_id"),
    ("ingestion_snapshot_observations", "revision_id", "processing_revision_id"),
    ("ingestion_index_state", "revision_id", "processing_revision_id"),
)


def upgrade() -> None:
    for table, old, new in _RENAMES:
        op.alter_column(table, old, new_column_name=new)


def downgrade() -> None:
    for table, old, new in _RENAMES:
        op.alter_column(table, new, new_column_name=old)
