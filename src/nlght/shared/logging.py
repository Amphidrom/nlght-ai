# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import logging
import os
from datetime import datetime

_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "WARN": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
    "OFF": logging.CRITICAL + 1,
}


def setup_logging(
    level_overrides: dict[str, str] | None = None,
    app_name: str = "nlght",
    log_dir: str = "logs",
) -> None:
    """Configure logging from a flat dict of logger-name → level strings.

    The special key ``root`` controls the root logger, equivalent to Spring Boot's
    ``logging.level.root``.  All other keys are applied as named-logger overrides,
    e.g. ``{"nlght": "DEBUG", "sqlalchemy.engine": "WARNING"}``.
    """
    overrides: dict[str, str] = level_overrides or {}
    root_level_str = overrides.get("root", "INFO").upper()
    root_level = _LEVEL_MAP.get(root_level_str, logging.INFO)

    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(log_dir, f"{app_name}_{timestamp}.log")

    logging.basicConfig(
        level=root_level,
        format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,  # override any earlier basicConfig calls
    )

    for logger_name, level_str in overrides.items():
        if logger_name == "root":
            continue
        level = _LEVEL_MAP.get(level_str.upper())
        if level is None:
            logging.getLogger(__name__).warning(
                "Unknown log level %r for logger %r — skipping", level_str, logger_name
            )
            continue
        logging.getLogger(logger_name).setLevel(level)
