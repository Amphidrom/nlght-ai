# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Turning the `watchers` block into running observers.

The watcher itself was tested long before anything built one. This is the part
that was missing: reading the configuration, refusing a broken entry, and
handing each watcher the queue to submit into.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from nlght.adapters.outbound.config.file.source import FileConfigurationSource
from nlght.bootstrap.wiring import _build_watchers
from nlght.core.config.snapshot import WatcherConfig


def _context(*watchers: WatcherConfig) -> SimpleNamespace:
    return SimpleNamespace(watchers=list(watchers))


def _filesystem(tmp_path: Path, **overrides: object) -> WatcherConfig:
    config: dict[str, object] = {
        "workflow": "ingest-data",
        "source_id": "docs",
        "roots": [{"path": str(tmp_path), "alias": "docs"}],
        "include": ["**/*.md"],
    }
    config.update(overrides)
    return WatcherConfig(name="docs-tree", kind="filesystem", config=config)


def test_a_configured_watcher_is_built_and_given_the_queue(tmp_path: Path) -> None:
    dispatcher = object()

    watchers = _build_watchers(_context(_filesystem(tmp_path)), dispatcher)

    assert len(watchers) == 1
    # Without the dispatcher a watched corpus would run inside whichever process
    # noticed the change, so this is the assertion that matters.
    assert watchers[0]._dispatcher is dispatcher
    assert watchers[0]._settings.workflow == "ingest-data"


def test_a_disabled_watcher_stays_declared_and_does_not_run(tmp_path: Path) -> None:
    disabled = _filesystem(tmp_path)
    disabled.enabled = False

    assert _build_watchers(_context(disabled), None) == []


def test_no_watchers_configured_is_not_an_error() -> None:
    assert _build_watchers(_context(), None) == []


def test_a_watcher_without_a_workflow_stops_startup(tmp_path: Path) -> None:
    # Skipping it with a warning would be worse than failing: a watcher that
    # does not exist and a source that never changes are the same silence.
    broken = _filesystem(tmp_path)
    broken.config.pop("workflow")

    with pytest.raises(RuntimeError, match="docs-tree"):
        _build_watchers(_context(broken), None)


def test_a_watcher_with_an_unbuildable_source_names_the_problem(tmp_path: Path) -> None:
    broken = _filesystem(tmp_path)
    broken.config.pop("roots")

    with pytest.raises(RuntimeError, match="roots"):
        _build_watchers(_context(broken), None)


def test_an_unknown_kind_lists_the_kinds_that_exist(tmp_path: Path) -> None:
    unknown = _filesystem(tmp_path)
    unknown.kind = "carrier-pigeon"

    with pytest.raises(RuntimeError, match="filesystem"):
        _build_watchers(_context(unknown), None)


async def test_the_yaml_block_reaches_the_snapshot(tmp_path: Path) -> None:
    path = tmp_path / "platform.yaml"
    path.write_text(
        """
watchers:
  - name: docs-tree
    kind: filesystem
    enabled: true
    config:
      workflow: ingest-data
      source_id: docs
      roots:
        - path: /srv/docs
          alias: docs
      include: ["**/*.md"]
      debounce_seconds: 5
  - name: wiki
    kind: confluence
    enabled: false
    config:
      workflow: ingest-knowledge
""",
        encoding="utf-8",
    )

    snapshot = await FileConfigurationSource(str(path)).load()

    assert [w.name for w in snapshot.watchers] == ["docs-tree", "wiki"]
    assert snapshot.watchers[0].kind == "filesystem"
    assert snapshot.watchers[0].config["workflow"] == "ingest-data"
    assert snapshot.watchers[0].config["debounce_seconds"] == 5
    assert snapshot.watchers[1].enabled is False
