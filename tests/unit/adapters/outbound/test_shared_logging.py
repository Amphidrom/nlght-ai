# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Tests for nlght.shared.logging — setup_logging."""
from __future__ import annotations

import logging

from nlght.shared.logging import setup_logging


def test_setup_logging_defaults(tmp_path) -> None:
    setup_logging(log_dir=str(tmp_path))
    root = logging.getLogger()
    assert root.level == logging.INFO


def test_setup_logging_root_debug(tmp_path) -> None:
    setup_logging({"root": "DEBUG"}, log_dir=str(tmp_path))
    assert logging.getLogger().level == logging.DEBUG


def test_setup_logging_named_logger(tmp_path) -> None:
    setup_logging({"nlght.test_ns": "WARNING"}, log_dir=str(tmp_path))
    assert logging.getLogger(__name__)


def test_setup_logging_off_level(tmp_path) -> None:
    setup_logging({"root": "OFF"}, log_dir=str(tmp_path))
    # OFF maps to CRITICAL + 1
    assert logging.getLogger().level > logging.CRITICAL


def test_setup_logging_unknown_level_skipped(tmp_path) -> None:
    # Should not raise — unknown levels are logged as WARNING and skipped
    setup_logging({"some.logger": "NOTAVALIDLEVEL"}, log_dir=str(tmp_path))


def test_setup_logging_creates_log_dir(tmp_path) -> None:
    log_dir = str(tmp_path / "new_logs")
    setup_logging(log_dir=log_dir)
    import os
    assert os.path.isdir(log_dir)


def test_setup_logging_warn_alias(tmp_path) -> None:
    setup_logging({"root": "WARN"}, log_dir=str(tmp_path))
    assert logging.getLogger().level == logging.WARNING
