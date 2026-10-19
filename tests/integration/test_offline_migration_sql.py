# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Render every migration to SQL without a database connection.

This catches migrations that require query results during Alembic's offline
mode, which operators use to review SQL before applying it.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_CONFIG = """\
integrations:
  persistence:
    workflows:
      backend: postgres
      url: "postgresql+asyncpg://user:pass@localhost:5432/nlght"
"""


def _render_offline_sql(tmp_path: Path) -> str:
    config = tmp_path / "platform.yaml"
    config.write_text(_CONFIG)
    # The console script is `nlght.main:cli`. Importing the module as __main__
    # would start the server instead. Driven in a subprocess because offline
    # rendering is what the operator actually runs, argv and all.
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from nlght.main import cli; "
            "sys.argv = ['nlght-ai', 'migrate', 'upgrade', 'head', '--sql']; cli()",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "NLGHT_CONFIG": str(config)},
        check=False,
    )
    assert completed.returncode == 0, (
        "offline migration SQL generation failed. A migration reads the "
        f"database before emitting DDL:\n{completed.stderr[-3000:]}"
    )
    return completed.stdout


def test_the_whole_migration_chain_renders_without_a_database(tmp_path: Path) -> None:
    sql = _render_offline_sql(tmp_path)

    # Every revision has to appear, or the chain stopped early without failing.
    for revision in ("0001", "0002", "0003", "0004", "0005", "0006", "0007", "0008"):
        assert f"'{revision}'" in sql, f"revision {revision} missing from offline SQL"


def test_offline_sql_still_emits_the_constraints_the_checks_guard(tmp_path: Path) -> None:
    """Skipping a pre-flight check offline must not skip the DDL it guards."""

    sql = _render_offline_sql(tmp_path)

    assert "uq_resource_address" in sql, (
        "0007 skipped its duplicate check offline and dropped the constraint too"
    )
    assert "subject_type IN ('resource', 'tool', 'model', 'playbook')" in sql, (
        "0008 skipped its warning offline and dropped the widened check too"
    )
