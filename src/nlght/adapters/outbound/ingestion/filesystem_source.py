# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path

import pathspec

from nlght.core.ingestion import AcquisitionDiagnostic, SourceDocument, SourceSnapshot

_DEFAULT_EXCLUDES = (
    "**/.git/**",
    "**/__pycache__/**",
    "**/*.pyc",
    "**/.venv/**",
    "**/node_modules/**",
    "**/.idea/**",
    "**/.vscode/**",
    "**/.pytest_cache/**",
    "**/.mypy_cache/**",
)


@dataclass(slots=True, frozen=True)
class FilesystemRoot:
    path: Path
    alias: str | None = None

    @property
    def identity_alias(self) -> str:
        return (self.alias or self.path.name).strip()


@dataclass(slots=True, frozen=True)
class FilesystemSourceSettings:
    source_id: str
    roots: tuple[FilesystemRoot, ...]
    include: tuple[str, ...] = ("**/*",)
    exclude: tuple[str, ...] = ()
    respect_ignore_files: bool = True

    def __post_init__(self) -> None:
        if not self.source_id.strip() or not self.roots:
            raise ValueError("filesystem source_id and roots must not be empty")
        aliases = [root.identity_alias for root in self.roots]
        if any(not alias for alias in aliases):
            raise ValueError("filesystem root aliases must not be empty")
        if len(set(aliases)) != len(aliases):
            raise ValueError("filesystem root aliases must be unique")
        if not self.include or any(not pattern.strip() for pattern in (*self.include, *self.exclude)):
            raise ValueError("filesystem include patterns must not be empty")


class FilesystemDocumentSource:
    def __init__(self, settings: FilesystemSourceSettings) -> None:
        self._settings = settings
        self._include = pathspec.GitIgnoreSpec.from_lines(settings.include)
        self._exclude = pathspec.GitIgnoreSpec.from_lines((*_DEFAULT_EXCLUDES, *settings.exclude))

    @property
    def source_id(self) -> str:
        return self._settings.source_id

    async def acquire(self) -> SourceSnapshot:
        return await asyncio.to_thread(self._acquire_sync)

    def _acquire_sync(self) -> SourceSnapshot:
        documents: list[SourceDocument] = []
        observed: set[str] = set()
        diagnostics: list[AcquisitionDiagnostic] = []
        complete = True

        for configured_root in self._settings.roots:
            root = configured_root.path.resolve()
            if not root.is_dir():
                complete = False
                diagnostics.append(AcquisitionDiagnostic("root-unavailable", str(root), "configured root is not a readable directory"))
                continue
            root_documents, root_observed, root_diagnostics, root_complete = self._scan_root(root, configured_root.identity_alias)
            documents.extend(root_documents)
            observed.update(root_observed)
            diagnostics.extend(root_diagnostics)
            complete = complete and root_complete

        documents.sort(key=lambda item: item.external_id)
        return SourceSnapshot(
            source_id=self.source_id,
            documents=tuple(documents),
            observed_external_ids=tuple(sorted(observed)),
            complete=complete,
            diagnostics=tuple(diagnostics),
        )

    def _scan_root(
        self,
        root: Path,
        alias: str,
    ) -> tuple[list[SourceDocument], set[str], list[AcquisitionDiagnostic], bool]:
        documents: list[SourceDocument] = []
        observed: set[str] = set()
        diagnostics: list[AcquisitionDiagnostic] = []
        complete = True

        for dirpath, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
            directory = Path(dirpath)
            ignore = self._ignore_spec(directory, root)
            kept_dirs: list[str] = []
            for name in sorted(dirnames):
                relative = (directory / name).relative_to(root).as_posix() + "/"
                if self._exclude.match_file(relative) or (ignore is not None and ignore.match_file(relative)):
                    continue
                kept_dirs.append(name)
            dirnames[:] = kept_dirs

            for name in sorted(filenames):
                candidate = directory / name
                relative = candidate.relative_to(root).as_posix()
                if not self._include.match_file(relative) or self._exclude.match_file(relative):
                    continue
                if ignore is not None and ignore.match_file(relative):
                    continue

                external_id = f"fs:{alias}/{relative}"
                observed.add(external_id)
                try:
                    resolved = candidate.resolve(strict=True)
                    resolved.relative_to(root)
                    content = resolved.read_bytes()
                except (OSError, ValueError) as exc:
                    complete = False
                    diagnostics.append(AcquisitionDiagnostic("file-unreadable", external_id, type(exc).__name__))
                    continue

                global_path = f"{alias}/{relative}"
                documents.append(
                    SourceDocument(
                        source=self.source_id,
                        external_id=external_id,
                        path=global_path,
                        content=content,
                        metadata={
                            "source_type": "filesystem",
                            "root": alias,
                            "path": global_path,
                            "root_relative_path": relative,
                            "filename": candidate.name,
                            "location": str(resolved),
                        },
                    )
                )

        return documents, observed, diagnostics, complete

    def _ignore_spec(self, directory: Path, root: Path) -> pathspec.GitIgnoreSpec | None:
        if not self._settings.respect_ignore_files:
            return None
        patterns: list[str] = []
        # `parents` of the *relative* path are themselves relative (e.g. `src`,
        # `src/foo`); rejoin them onto `root` so every link in the chain is
        # absolute — the loop below calls `current.relative_to(root)`, which
        # fails on a bare relative path once the walk descends two levels.
        intermediate = reversed(directory.relative_to(root).parents[:-1])
        chain = [root, *(root / parent for parent in intermediate)]
        if directory != root:
            chain.append(directory)
        for current in chain:
            relative_dir = current.relative_to(root).as_posix()
            prefix = "" if relative_dir == "." else f"{relative_dir}/"
            for filename in (".gitignore", ".botignore"):
                ignore_file = current / filename
                if not ignore_file.is_file():
                    continue
                try:
                    lines = ignore_file.read_text(encoding="utf-8", errors="replace").splitlines()
                except OSError:
                    continue
                for line in lines:
                    normalized = _normalize_ignore_pattern(line, prefix)
                    if normalized is not None:
                        patterns.append(normalized)
        return pathspec.GitIgnoreSpec.from_lines(patterns) if patterns else None


def _normalize_ignore_pattern(line: str, prefix: str) -> str | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    negated = stripped.startswith("!")
    pattern = stripped[1:] if negated else stripped
    if not pattern:
        return None
    if pattern.startswith("/"):
        normalized = prefix + pattern[1:]
    elif "/" not in pattern.rstrip("/"):
        normalized = f"{prefix}**/{pattern}"
    else:
        normalized = prefix + pattern
    return f"!{normalized}" if negated else normalized
