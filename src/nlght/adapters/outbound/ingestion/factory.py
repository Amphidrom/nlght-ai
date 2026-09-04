# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Building a document source from a block of configuration.

One definition of what a source configuration looks like, because there are now
two callers: the `ingestion.source` step, which acquires a snapshot to process,
and a configured watcher, which acquires one to notice that something changed.
They describe the same thing and must keep describing it the same way — a
watcher pointed at a different tree than the run it triggers is a fault nobody
sees, because the symptom is nothing happening.

`SourceConfigurationError` is deliberately plain: each caller wraps it in the
error its own layer speaks, a workflow configuration error for the step and a
startup failure for the runtime.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import httpx2

from nlght.adapters.outbound.ingestion.confluence_source import (
    ConfluenceDocumentSource,
    ConfluenceSourceSettings,
    HttpxConfluenceTransport,
)
from nlght.adapters.outbound.ingestion.filesystem_source import (
    FilesystemDocumentSource,
    FilesystemRoot,
    FilesystemSourceSettings,
)
from nlght.ports.outbound.document_source import DocumentSource


class SourceConfigurationError(ValueError):
    """A source configuration that cannot produce a source."""


def _require(config: Mapping[str, Any], key: str, where: str) -> Any:  # noqa: ANN401 (source config is heterogeneous)
    if key not in config:
        raise SourceConfigurationError(f"{where} is missing required config key '{key}'.")
    return config[key]


def _filesystem(config: Mapping[str, Any], where: str) -> DocumentSource:
    raw_roots = _require(config, "roots", where)
    if not isinstance(raw_roots, list) or not raw_roots:
        raise SourceConfigurationError(f"{where} expects 'roots' to be a non-empty list.")
    roots = tuple(
        FilesystemRoot(path=Path(str(_require(root, "path", where))), alias=root.get("alias"))
        for root in raw_roots
    )
    return FilesystemDocumentSource(
        FilesystemSourceSettings(
            source_id=str(_require(config, "source_id", where)),
            roots=roots,
            include=tuple(str(item) for item in config.get("include", ("**/*",))),
            exclude=tuple(str(item) for item in config.get("exclude", ())),
            respect_ignore_files=bool(config.get("respect_ignore_files", True)),
        )
    )


def _confluence(config: Mapping[str, Any], where: str) -> DocumentSource:
    settings = ConfluenceSourceSettings(
        source_id=str(_require(config, "source_id", where)),
        base_url=str(_require(config, "base_url", where)),
        space_keys=tuple(str(key) for key in _require(config, "space_keys", where)),
        user_email=str(_require(config, "user_email", where)),
        api_token=str(_require(config, "api_token", where)),
        cql=config.get("cql"),
        max_pages=int(config.get("max_pages", 1000)),
        page_size=int(config.get("page_size", 50)),
        timeout_seconds=float(config.get("timeout_seconds", 30.0)),
    )
    return ConfluenceDocumentSource(settings, HttpxConfluenceTransport(httpx2.AsyncClient()))


SOURCE_BUILDERS: dict[str, Callable[[Mapping[str, Any], str], DocumentSource]] = {
    "filesystem": _filesystem,
    "confluence": _confluence,
}


def build_source(kind: str, config: Mapping[str, Any], *, where: str) -> DocumentSource:
    """Build the source named by ``kind``, or say which kinds exist.

    ``where`` names the thing being configured — a step or a watcher — so the
    message points at the block to fix rather than at this function.
    """
    builder = SOURCE_BUILDERS.get(kind)
    if builder is None:
        raise SourceConfigurationError(
            f"{where} names unknown source type '{kind}'. "
            f"Known types: {', '.join(sorted(SOURCE_BUILDERS))}."
        )
    return builder(config, where)
