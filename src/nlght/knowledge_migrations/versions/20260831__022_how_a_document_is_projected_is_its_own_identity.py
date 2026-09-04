# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""What the reindex decision could not see.

`_is_current` skipped a document when three things agreed: the source hash, the
enricher hash, and the index generation. Those cover the document and which
incarnation of an index holds it. Not one of them changes when:

    the embedding model is swapped
    its dimension changes
    the distance metric changes
    the payload schema gains a field

So changing an embedding model reindexed **nothing**. Every document read as
already current, the run reported success, and the collection went on holding
vectors produced by a model the queries were no longer embedded with — which
does not fail, it just quietly answers worse, and there is no error anywhere to
find. The generation could not catch it either: it rotates when a collection has
to be *created*, and swapping a model creates nothing.

The missing value is not a property of the document at all, which is why no
document-derived hash could ever have carried it. It is a property of the
projection: how this document is turned into index entries. So it gets its own
column beside the ones that describe the document.

    content_hash            the source's fassung
    enricher_hash           the processed fassung
    projection_fingerprint  payload schema, embedding provider, model,
                            dimension, distance metric
    index_generation        which incarnation of the index holds it

The immediate reason it exists now: chunk payloads gain `position`,
`start_offset` and `end_offset`, without which a chunk hit names no span of its
document and nothing can be resolved against the stored revision. That is a
payload schema change, and every already-indexed document has to be written
again for it — which is precisely what the three older values cannot express.

The alternative was to fold the payload version into `processing_revision_id`,
where the chunking parameters already sit. It would have worked and it would
have been a lie: that value identifies the processed *document*, and putting an
embedding model inside it makes two documents with identical text differ because
of infrastructure. One value, one meaning.

Existing rows default to the empty string, which matches no fingerprint any
writer computes, so the next run rewrites them. That is the correct outcome —
they were indexed under a projection nobody recorded, and there is no way to
find out which.

Revision ID: 0022
Revises: 0021
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0022"
down_revision: str | None = "0021"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.add_column(
        "ingestion_index_state",
        sa.Column(
            "projection_fingerprint",
            sa.Text(),
            nullable=False,
            server_default="",
        ),
    )


def downgrade() -> None:
    op.drop_column("ingestion_index_state", "projection_fingerprint")
