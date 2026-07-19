# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import asyncio
import logging
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path

from nlght.core.workspace.workspace import WorkspaceContext, WorkspaceStrategy
from nlght.ports.outbound.workspace_manager import WorkspaceManager

logger = logging.getLogger(__name__)

# Filesystem-safe slug: alphanumeric + hyphen only, max 64 characters
_SLUG_SAFE = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")


def _slugify(key: str, max_len: int = 64) -> str:
    slug = "".join(c if c in _SLUG_SAFE else "_" for c in key)
    return slug[:max_len] or "session"


class LocalWorkspaceManager(WorkspaceManager):
    """Local Workspace Manager — provisions directories on the host.

    Directory structure:
        base_path/
          sessions/<session_slug>/   # Session workspaces (persistent)
          ephemeral/<uuid>/          # Ephemeral workspaces (deleted on release)

    Session workspaces are cached in-memory. At process end, session
    directories remain on disk (intentional — the next session finds the
    same state).

    Ephemeral workspaces are deleted asynchronously on release().
    """

    def __init__(self, base_path: Path) -> None:
        self._base_path = base_path
        self._sessions: dict[str, WorkspaceContext] = {}
        base_path.mkdir(parents=True, exist_ok=True)
        (base_path / "sessions").mkdir(exist_ok=True)
        (base_path / "ephemeral").mkdir(exist_ok=True)
        logger.info("workspace.manager.init | base_path=%s", base_path)

    async def get_or_create(self, session_key: str | None) -> WorkspaceContext:
        if session_key is not None:
            return await self._get_or_create_session(session_key)
        return await self._create_ephemeral()

    async def release(self, workspace: WorkspaceContext) -> None:
        if workspace.strategy == WorkspaceStrategy.EPHEMERAL:
            await asyncio.to_thread(self._delete_dir, workspace.root_path)
            logger.debug(
                "workspace.released.ephemeral | id=%s path=%s",
                workspace.workspace_id, workspace.root_path,
            )
        else:
            self._sessions.pop(workspace.session_key or "", None)
            logger.debug(
                "workspace.released.session | id=%s key=%s (dir kept)",
                workspace.workspace_id, workspace.session_key,
            )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _get_or_create_session(self, session_key: str) -> WorkspaceContext:
        if session_key in self._sessions:
            ctx = self._sessions[session_key]
            logger.debug(
                "workspace.session.reused | key=%s id=%s path=%s",
                session_key, ctx.workspace_id, ctx.root_path,
            )
            return ctx

        slug = _slugify(session_key)
        root = self._base_path / "sessions" / slug
        await asyncio.to_thread(root.mkdir, parents=True, exist_ok=True)

        ctx = WorkspaceContext(
            workspace_id=str(uuid.uuid4()),
            root_path=root,
            strategy=WorkspaceStrategy.SESSION,
            session_key=session_key,
            created_at=datetime.now(UTC),
        )
        self._sessions[session_key] = ctx
        logger.info(
            "workspace.session.created | key=%s id=%s path=%s",
            session_key, ctx.workspace_id, root,
        )
        return ctx

    async def _create_ephemeral(self) -> WorkspaceContext:
        workspace_id = str(uuid.uuid4())
        root = self._base_path / "ephemeral" / workspace_id
        await asyncio.to_thread(root.mkdir, parents=True, exist_ok=True)

        ctx = WorkspaceContext(
            workspace_id=workspace_id,
            root_path=root,
            strategy=WorkspaceStrategy.EPHEMERAL,
            session_key=None,
            created_at=datetime.now(UTC),
        )
        logger.debug("workspace.ephemeral.created | id=%s path=%s", workspace_id, root)
        return ctx

    @staticmethod
    def _delete_dir(path: Path) -> None:
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
