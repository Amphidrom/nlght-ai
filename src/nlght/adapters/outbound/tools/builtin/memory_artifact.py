# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from nlght.adapters.outbound.tools.builtin.action_semantics import READ_SESSION
from nlght.core.tools.tool import ToolBase, ToolParameter, ToolSignature

if TYPE_CHECKING:
    from nlght.core.workspace.workspace import WorkspaceContext
    from nlght.ports.outbound.store_coordinator import StoreCoordinator

logger = logging.getLogger(__name__)


class LoadMemoryArtifactTool(ToolBase):
    """Loads content from workspace-promoted memory artifacts.

    Artifacts are written to ``<workspace>/memory_artifacts/<uuid><ext>`` by
    ``MemoryIngestionPipeline`` when a workspace is provided.  HiveMind stores
    only a compact ``[memory_artifact]`` reference; this tool resolves that
    reference back to the full content on demand.

    Configuration:
      ``artifacts_dir`` — absolute path to the artifacts directory.
      Overridden by the workspace context when injected at catalog build time.
    """

    KIND: ClassVar[str]     = "memory_artifact"
    PROVIDER: ClassVar[str] = "platform"

    def __init__(
        self,
        *,
        name: str,
        config: dict[str, Any],
        workspace: WorkspaceContext | None = None,
        store_coordinator: StoreCoordinator | None = None,
        **kwargs: Any,  # noqa: ANN401 (forwarded into ToolBase.__init__ typed kw surface)
    ) -> None:
        super().__init__(name=name, config=config, **kwargs)
        self._workspace = workspace
        self._store = store_coordinator

    @classmethod
    def signatures(cls) -> list[ToolSignature]:
        return [
            ToolSignature(
                name="load_memory_artifact",
                description=(
                    "Load the full content of a stored memory artifact by ID. "
                    "Use when you need the complete payload of a [memory_artifact] reference."
                ),
                method_name="load_artifact",
                parameters=[
                    ToolParameter(
                        name="id",
                        type="string",
                        description="Artifact ID from the [memory_artifact] reference.",
                    ),
                ],
                action=READ_SESSION,
            ),
            ToolSignature(
                name="load_memory_artifact_range",
                description=(
                    "Load a specific line range from a stored memory artifact. "
                    "Use to inspect only part of a large artifact without loading the full content."
                ),
                method_name="load_artifact_range",
                parameters=[
                    ToolParameter(
                        name="id",
                        type="string",
                        description="Artifact ID from the [memory_artifact] reference.",
                    ),
                    ToolParameter(
                        name="from_line",
                        type="number",
                        description="First line to return (1-based, inclusive).",
                    ),
                    ToolParameter(
                        name="to_line",
                        type="number",
                        description="Last line to return (1-based, inclusive).",
                    ),
                ],
                action=READ_SESSION,
            ),
        ]

    # ------------------------------------------------------------------

    def _artifacts_dir(self) -> Path | None:
        if self._workspace is not None:
            return self._workspace.root_path / "memory_artifacts"
        raw = self.config.get("artifacts_dir")
        return Path(str(raw)) if raw else None

    def _resolve_path(self, artifact_id: str) -> Path | None:
        d = self._artifacts_dir()
        if d is None or not d.is_dir():
            return None
        matches = list(d.glob(f"{artifact_id}*"))
        return matches[0] if matches else None

    async def load_artifact(self, *, id: str) -> str:
        if self._store is not None:
            result = self._store.load_artifact(id.strip())
            if result is not None:
                total = len(result.content.splitlines())
                logger.info("load_memory_artifact.ok | id=%s chars=%d lines=%d", id, len(result.content), total)
                return json.dumps({"id": id, "total_lines": total, "content": result.content, "extension": result.extension})
            logger.warning("load_memory_artifact.not_found | id=%s", id)
            return json.dumps({"id": id, "error": "Artifact not found.", "content": ""})
        path = self._resolve_path(id.strip())
        if path is None:
            return json.dumps({"id": id, "error": "Artifact not found.", "content": ""})
        try:
            content = path.read_text(encoding="utf-8")
            total = len(content.splitlines())
            logger.info("load_memory_artifact.ok | id=%s chars=%d lines=%d", id, len(content), total)
            return json.dumps({"id": id, "total_lines": total, "content": content, "extension": path.suffix})
        except Exception as exc:
            logger.warning("load_memory_artifact.failed | id=%s error=%s", id, exc)
            return json.dumps({"id": id, "error": str(exc), "content": ""})

    async def load_artifact_range(self, *, id: str, from_line: int, to_line: int) -> str:
        if self._store is not None:
            result = self._store.load_artifact(id.strip())
            if result is None:
                return json.dumps({"id": id, "error": "Artifact not found.", "content": ""})
            lines = result.content.splitlines()
            total = len(lines)
            start = max(0, int(from_line) - 1)
            end = min(total, int(to_line))
            excerpt = "\n".join(lines[start:end])
            logger.info(
                "load_memory_artifact_range.ok | id=%s from=%d to=%d total=%d chars=%d",
                id, start + 1, end, total, len(excerpt),
            )
            return json.dumps({
                "id": id,
                "from_line": start + 1,
                "to_line": end,
                "total_lines": total,
                "content": excerpt,
                "extension": result.extension,
            })
        path = self._resolve_path(id.strip())
        if path is None:
            return json.dumps({"id": id, "error": "Artifact not found.", "content": ""})
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
            total = len(lines)
            start = max(0, int(from_line) - 1)
            end = min(total, int(to_line))
            excerpt = "\n".join(lines[start:end])
            logger.info(
                "load_memory_artifact_range.ok | id=%s from=%d to=%d total=%d chars=%d",
                id, start + 1, end, total, len(excerpt),
            )
            return json.dumps({
                "id": id,
                "from_line": start + 1,
                "to_line": end,
                "total_lines": total,
                "content": excerpt,
                "extension": path.suffix,
            })
        except Exception as exc:
            logger.warning("load_memory_artifact_range.failed | id=%s error=%s", id, exc)
            return json.dumps({"id": id, "error": str(exc), "content": ""})
