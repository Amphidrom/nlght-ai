# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""`resource` becomes a subject a rule can name.

The table has allowed three subject types since it was created: tool, model,
playbook. Two of those name what they authorize. The third did not — a rule with
`subject_type = 'tool'` was evaluated against `resource.name`, so it authorized
an entire configured resource: every operation it offers, and its use inside a
workflow step where no tool call happens at all. The name said one signature and
the check said one resource.

Splitting them needs a fourth value, and the CHECK constraint has to admit it
before any rule can carry it.

**Existing `tool` rows are not migrated, and this is deliberate.** An old rule
holds a bare name — `shell` — which under the new meaning matches no signature
(`shell` offers nine, all called `shell_*`) and under `resource` would need an
address whose kind the row does not record. Guessing `local/shell` from a bare
`shell` is exactly the kind of inference that produces a rule nobody wrote. With
no production deployment to preserve, seeds, fixtures and docs are moved to the
new meaning instead, and any row that predates this is left to be re-stated by
whoever wrote it.

The consequence is stated rather than hidden: after this migration, an existing
`tool` rule matches a signature name and almost certainly matches nothing. Under
allowlist semantics a rule that matches no subject restricts nothing, so such a
row is inert rather than dangerous — but it is also not doing what its author
intended, and it should be rewritten.

Revision ID: 0008
Revises: 0007
"""

from __future__ import annotations

import logging

import sqlalchemy as sa
from alembic import context, op

revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | None = None
depends_on: str | None = None

logger = logging.getLogger("alembic.runtime.migration")

_OLD = "subject_type IN ('tool', 'model', 'playbook')"
_NEW = "subject_type IN ('resource', 'tool', 'model', 'playbook')"


def upgrade() -> None:
    # Offline (`--sql`) there is no connection to count rows on; `execute()`
    # returns None. The warning is diagnostic only — the DDL below is the same
    # either way — so it is simply not available when rendering a script.
    if not context.is_offline_mode():
        _warn_about_stale_tool_rules()

    with op.batch_alter_table("access_policies") as batch:
        batch.drop_constraint("ck_access_policies_subject_type", type_="check")
        batch.create_check_constraint("ck_access_policies_subject_type", _NEW)


def _warn_about_stale_tool_rules() -> None:
    stale = op.get_bind().execute(
        sa.text("SELECT count(*) FROM access_policies WHERE subject_type = 'tool'")
    ).scalar_one()
    if stale:
        logger.warning(
            "access.migration.0008 | tool rules written under the old meaning=%s — "
            "they were compared against a resource name and are now compared "
            "against '<kind>/<name>::<signature>'. They match nothing and "
            "restrict nothing; rewrite them as 'resource' or 'tool' rules.",
            stale,
        )


def downgrade() -> None:
    op.execute(sa.text("DELETE FROM access_policies WHERE subject_type = 'resource'"))
    with op.batch_alter_table("access_policies") as batch:
        batch.drop_constraint("ck_access_policies_subject_type", type_="check")
        batch.create_check_constraint("ck_access_policies_subject_type", _OLD)
