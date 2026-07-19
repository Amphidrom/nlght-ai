# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""FileSystemBackend — JSON-file-based SessionSnapshot persistence.

Part of the [hive-mind] extra.  For the simple in-memory tier this
module is not used.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from nlght.core.hive_mind.models import SessionSnapshot
from nlght.ports.outbound.session_backend import SessionBackend

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FileSystemBackend
# ---------------------------------------------------------------------------

class FileSystemBackend(SessionBackend):
    """Implements SessionBackend — stores each session as a single JSON file under ``base_dir``."""

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
