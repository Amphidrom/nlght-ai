# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""The corpus stores the text it is a corpus of.

`ingestion_document_revisions` recorded everything about a revision except the
revision. Identity, path, provenance, processing verdict — and then a pointer,
`artifact_ref`, in place of the text. The full processed text existed in exactly
one place: the OpenSearch payload. So the search projection was the content
store, and every read of a document's wording was a read of an index.

That is the wrong way round in three ways at once. A projection is rebuilt,
re-mapped and dropped as an operational act, and nothing about doing so is
supposed to destroy a corpus. A projection is not addressable — a chunk's
offsets index the processed text, and the only way to honour them was to hope
the payload still held the same string. And a projection is not one text: a
document lives in OpenSearch as one field while its chunks live in Qdrant as
many, so "what does this revision say" had two answers of different shapes and
no authority between them.

The pointer was not an alternative. `artifact_ref` names source, document and
the source's own revision, which is the right provenance and the wrong read
path: a refetch returns whatever the source holds *now*, and for a source that
transforms at acquisition it does not even return the same representation. A
Confluence page is acquired with its markup stripped, chunk offsets are computed
on the stripped text, and the source can only ever hand back the markup. An
offset into one is meaningless in the other.

So the text moves in beside its identity:

    content = exactly ProcessedDocument.text
            = the string chunking ran over
            = the string chunk offsets index into

Nullable, with one meaning. A document that was observed and has no usable text
— binary, or undecodable — still belongs in the snapshot, because the tombstone
rule is "absent from a complete snapshot" and omitting it would declare it
deleted. It has nothing to store. `NULL` says that and nothing else, which is
why nothing is backfilled: a backfilled `NULL` would also mean "written before
this migration", and the two would never be separable again.

Existing rows therefore keep `NULL` and are not content-authoritative. There is
no production corpus, and inventing content for a revision is the one thing a
content authority must never do. Reindexing rebuilds them from their sources.

`TEXT`, not an object store and not a large-object handle. The corpus is source
documents; PostgreSQL holds those comfortably, and a second storage system is a
decision to take against measured sizes rather than in advance.

Revision ID: 0020
Revises: 0019
"""

from __future__ import annotations

import logging

import sqlalchemy as sa
from alembic import op

revision: str = "0020"
down_revision: str | None = "0019"
branch_labels: str | None = None
depends_on: str | None = None

logger = logging.getLogger("alembic.runtime.migration")


def upgrade() -> None:
    op.add_column(
        "ingestion_document_revisions",
        sa.Column("content", sa.Text(), nullable=True),
    )

    waiting = op.get_bind().execute(
        sa.text("SELECT count(*) FROM ingestion_document_revisions")
    ).scalar_one()
    if waiting:
        logger.warning(
            "ingestion.migration.0020 | revisions without stored content=%s — they "
            "carry no authoritative text until their sources are ingested again; "
            "not backfilled, because neither the index payload nor a refetch is "
            "the text the offsets were computed on",
            waiting,
        )


def downgrade() -> None:
    op.drop_column("ingestion_document_revisions", "content")
