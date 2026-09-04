# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Evidence records what the document was called when it was read.

`document_id` is a hash. It answers "which document" for a diff and answers
nothing for a person, so a report could say a claim moved and never which file —
and an expectation file could not name `timeouts.adoc` at all.

The obvious place to look it up was `ingestion_documents`, and it is the wrong
one: a live corpus had that table empty, because the knowledge flow does not
write the ingestion record. Joining the two would tie two lifecycles together to
borrow a string, which is a decision of its own and not one to take by accident.

So the path travels with the sighting that saw it, the way the document revision
already does.

**Provenance, and never identity.** A renamed document is the same document: the
path may change without opening an assertion or writing a revision, and nothing
resolves a slot by it. What it is for is naming — in a report, in a citation, in
an expectation — and it is stored exactly as observed rather than reconstructed,
so two sightings of one document under two names each keep the name they saw.

Nullable, and nothing is backfilled. Evidence written before this has no path and
cannot be given one: what the file was called at that moment was not recorded.

Revision ID: 0015
Revises: 0014
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision: str = "0015"
down_revision: str | None = "0014"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    op.add_column(
        "knowledge_evidence",
        sa.Column("document_path", sa.Text(), nullable=False, server_default=""),
    )


def downgrade() -> None:
    op.drop_column("knowledge_evidence", "document_path")
