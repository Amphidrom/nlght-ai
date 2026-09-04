# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""A resource is at one address, and only one resource is.

Every consumer resolves a resource the same way — ask for a kind, filter by
name, take the first row:

    found = [r for r in await repo.find_by_kind(kind) if r.name == name]
    resource = found[0]

Seven call sites, all of them that shape. So `(kind, name)` has been the runtime
address of a resource for as long as there have been resources, and nothing has
ever guaranteed it identifies one. Two enabled rows with the same kind and name
are accepted today, and which of them a workflow gets is decided by whatever
order PostgreSQL returns them in.

That is bad enough while it only decides configuration. It becomes something
else once an access rule names an address: "only workflow XY may use
data_store/data-main" is a statement about a locator, and a locator that two
rows can answer to cannot carry an authorization decision. The rule would be
satisfied, and the resource actually activated would be whichever row sorted
first.

So the address becomes unique before anything is authorized by it.

`provider` stays out of the constraint. It selects which registered
implementation instantiates a resource — `(kind, provider)` is the *registry*
key — and that is a different question from which resource this is. Including it
would permit two rows at one address that differ only in how they are built,
which is exactly the ambiguity being removed.

`resource_id` remains the primary key and the persistent identity. A resource
that is renamed is the same resource; a resource at a given address may over
time be a different one.

No data is migrated. A deployment holding a duplicate address has a
configuration question only its operator can answer — which of the two did you
mean — and this migration will refuse to apply until they have answered it. That
is the right failure: the alternative is choosing one on their behalf, silently,
which is the behaviour being ended.

Revision ID: 0007
Revises: 0006
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import context, op

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    # `--sql` renders this migration without a database, so there is nothing to
    # ask about existing rows: offline, `execute()` returns None. Skipping the
    # pre-flight check costs the friendly message, not the guarantee — the
    # constraint below is what actually enforces uniqueness, and PostgreSQL
    # refuses to create it while duplicates exist. An operator running the
    # generated script still cannot end up with an ambiguous address.
    if not context.is_offline_mode():
        _refuse_duplicate_addresses()

    op.create_unique_constraint("uq_resource_address", "resources", ["kind", "name"])


def _refuse_duplicate_addresses() -> None:
    duplicates = op.get_bind().execute(
        sa.text(
            "SELECT kind, name, count(*) AS n FROM resources "
            "GROUP BY kind, name HAVING count(*) > 1"
        )
    ).all()
    if duplicates:
        listed = ", ".join(f"{row[0]}/{row[1]} ({row[2]}x)" for row in duplicates)
        raise RuntimeError(
            "resources hold duplicate addresses and cannot be made unique "
            f"automatically: {listed}. Each address must identify one resource "
            "before it can carry an access decision — rename or remove the "
            "extras, then run this migration again."
        )


def downgrade() -> None:
    op.drop_constraint("uq_resource_address", "resources", type_="unique")
