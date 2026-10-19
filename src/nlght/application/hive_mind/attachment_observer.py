# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Detect and ingest attached documents from messages."""

from __future__ import annotations

import hashlib
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from nlght.ports.outbound.store_coordinator import StoreCoordinator

if TYPE_CHECKING:
    from nlght.core.workspace.workspace import WorkspaceContext

logger = logging.getLogger(__name__)

_ATTACHMENT_HEADERS = ("Attached document", "Attached file")


@dataclass
class DetectedAttachment:
    filename: str
    content: str

    @property
    def extension(self) -> str:
        return Path(self.filename).suffix

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.content.encode("utf-8", errors="replace")).hexdigest()[:16]


def _parse_attachments(content: str) -> list[DetectedAttachment]:
    """Parse named attachments from fenced message content.

    Args:
        content (str): Message text with quoted filenames in fence openings.

    Returns:
        list[DetectedAttachment]: Detected filename-content pairs.
    """
    result: list[DetectedAttachment] = []
    lines = content.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("```") and '"' in line:
            q1 = line.index('"') + 1
            q2 = line.find('"', q1)
            if q2 > q1:
                filename = line[q1:q2].strip()
                i += 1
                body_lines: list[str] = []
                while i < len(lines) and not lines[i].startswith("```"):
                    body_lines.append(lines[i])
                    i += 1
                if filename:
                    result.append(DetectedAttachment(
                        filename=filename,
                        content="\n".join(body_lines),
                    ))
        i += 1
    return result


def scan_attachments(messages: list[dict[str, Any]]) -> list[DetectedAttachment]:
    """Find named attachments in recognized message wrappers.

    Args:
        messages (list[dict[str, Any]]): Raw system and user message mappings.

    Returns:
        list[DetectedAttachment]: Last content per filename in first-seen filename order.
    """
    by_name: dict[str, DetectedAttachment] = {}
    for msg in messages:
        if msg.get("role") not in ("system", "user"):
            continue
        content = str(msg.get("content") or "")
        if not any(h in content for h in _ATTACHMENT_HEADERS):
            continue
        for att in _parse_attachments(content):
            by_name[att.filename] = att
    return list(by_name.values())


def ingest_attachments(
    attachments: list[DetectedAttachment],
    *,
    store: StoreCoordinator,
    turn_nr: int,
    workspace: WorkspaceContext | None = None,
) -> list[str]:
    """Store attachments whose content changed since the last sighting.

    Args:
        attachments (list[DetectedAttachment]): Detected attachments in this turn.
        store (StoreCoordinator): Session coordinator used for remembered payloads.
        turn_nr (int): Current conversation turn number.
        workspace (WorkspaceContext | None): Optional workspace for artifact files.

    Returns:
        list[str]: Filenames of newly stored or updated attachments.

    Side Effects:
        Writes workspace files and promotes session payloads.
    """
    if workspace is not None:
        store.set_slot("workspace", workspace)

    updated: list[str] = []
    for att in attachments:
        existing = store.get_payload([att.filename])
        if existing is not None and f"content_hash:{att.content_hash}" in existing.tags:
            logger.debug("attachment.unchanged | filename=%s", att.filename)
            continue

        artifact_id = str(uuid.uuid4())
        compact = _write_artifact(att, artifact_id, workspace)
        store.promote_payload(
            content=compact,
            entities=[att.filename],
            tags=["attachment", f"content_hash:{att.content_hash}"],
            turn_nr=turn_nr,
        )
        action = "updated" if existing is not None else "stored"
        logger.info("attachment.%s | filename=%s id=%s", action, att.filename, artifact_id)
        updated.append(att.filename)

    return updated


def _write_artifact(
    att: DetectedAttachment,
    artifact_id: str,
    workspace: WorkspaceContext | None,
) -> str:
    """Create a compact reference for one attachment.

    Args:
        att (DetectedAttachment): Attachment to store.
        artifact_id (str): Stable artifact identity.
        workspace (WorkspaceContext | None): Optional workspace for the file.

    Returns:
        str: Compact memory-artifact reference string.

    Side Effects:
        Writes an artifact file when a workspace is available.
    """
    if workspace is not None:
        artifacts_dir = workspace.root_path / "memory_artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        path = artifacts_dir / f"{artifact_id}{att.extension or '.txt'}"
        path.write_text(att.content, encoding="utf-8")
    size = len(att.content.encode("utf-8"))
    ext_tag = att.extension.lstrip(".") or "text"
    return (
        f"[memory_artifact] id={artifact_id}"
        f" | label={att.filename}"
        f" | type={ext_tag}"
        f" | size={size}"
    )
