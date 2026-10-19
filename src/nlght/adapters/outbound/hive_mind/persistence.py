# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Persist session snapshots as JSON files for the hive-mind tier.

Snapshots are stored below the configured base directory.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from nlght.core.hive_mind.models import SessionSnapshot
from nlght.ports.outbound.session_backend import SessionBackend

logger = logging.getLogger(__name__)


class FileSystemBackend(SessionBackend):
    """Store each session as one JSON file below ``base_dir``."""

    def __init__(self, base_dir: str = "./sessions") -> None:
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)
        logger.info("hive_mind.filesystem.init | base_dir=%s", self.base_dir)

    def _path(self, session_id: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id)
        return self.base_dir / f"{safe}.json"

    def save(self, snapshot: SessionSnapshot) -> None:
        path = self._path(snapshot.session_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(snapshot.to_dict(), f, ensure_ascii=False, indent=2)
        logger.debug("hive_mind.filesystem.saved | id=%s", snapshot.session_id)

    def load(self, session_id: str) -> SessionSnapshot | None:
        path = self._path(session_id)
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return SessionSnapshot.from_dict(data)

    def delete(self, session_id: str) -> bool:
        path = self._path(session_id)
        if path.exists():
            path.unlink()
            return True
        return False

    def list_sessions(self) -> list[str]:
        return [p.stem for p in self.base_dir.glob("*.json")]

    def exists(self, session_id: str) -> bool:
        return self._path(session_id).exists()
