# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from nlght.adapters.outbound.workspace.local import LocalWorkspaceManager, _slugify
from nlght.core.workspace.workspace import WorkspaceStrategy


def _workspace_root() -> Path:
    root = Path.cwd() / "test-output" / f"workspace-{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_slugify_replaces_unsafe_chars_and_limits_length() -> None:
    assert _slugify("tenant/acme:prod") == "tenant_acme_prod"
    assert _slugify("") == "session"
    assert _slugify("x" * 100) == "x" * 64


async def test_session_workspace_is_reused_then_uncached_on_release() -> None:
    root = _workspace_root()
    manager = LocalWorkspaceManager(root)

    try:
        first = await manager.get_or_create("tenant/acme")
        second = await manager.get_or_create("tenant/acme")

        assert first is second
        assert first.strategy == WorkspaceStrategy.SESSION
        assert first.session_key == "tenant/acme"
        assert first.root_path.exists()

        await manager.release(first)
        third = await manager.get_or_create("tenant/acme")

        assert third is not first
        assert third.root_path == first.root_path
        assert third.root_path.exists()
    finally:
        shutil.rmtree(root, ignore_errors=True)


async def test_ephemeral_workspace_is_deleted_on_release(monkeypatch) -> None:
    root = _workspace_root()
    manager = LocalWorkspaceManager(root)
    deleted_paths: list[Path] = []

    def _fake_delete_dir(path: Path) -> None:
        deleted_paths.append(path)

    try:
        monkeypatch.setattr(manager, "_delete_dir", _fake_delete_dir)
        workspace = await manager.get_or_create(None)
        workspace_root = workspace.root_path
        (workspace_root / "artifact.txt").write_text("data", encoding="utf-8")

        assert workspace.strategy == WorkspaceStrategy.EPHEMERAL
        assert workspace_root.exists()

        await manager.release(workspace)

        assert deleted_paths == [workspace_root]
    finally:
        shutil.rmtree(root, ignore_errors=True)
