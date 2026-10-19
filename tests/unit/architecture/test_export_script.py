# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Regression checks for the Structurizr export script."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

_EXPORT_SCRIPT = Path(__file__).resolve().parents[3] / "docs" / "architecture" / "export.sh"


def test_invalid_export_restores_previous_workspace(tmp_path: Path) -> None:
    """Keep the tracked JSON and remove the snapshot after invalid output."""
    architecture = tmp_path / "docs" / "architecture"
    architecture.mkdir(parents=True)
    workspace = architecture / "workspace.json"
    previous = '{"model":{"ok":true},"views":{"one":[]}}\n'
    workspace.write_text(previous, encoding="utf-8")

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    docker = fake_bin / "docker"
    docker.write_text(
        '#!/bin/sh\n'
        'if [ "$1" = info ]; then exit 0; fi\n'
        "printf '%s\\n' '{invalid-json' > workspace.json\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
    env["TMPDIR"] = str(snapshots)
    result = subprocess.run(
        ["bash", str(_EXPORT_SCRIPT)],
        cwd=architecture,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert workspace.read_text(encoding="utf-8") == previous
    assert "restored the previous workspace.json" in result.stderr
    assert list(snapshots.iterdir()) == []
