# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from pathlib import Path

COPYRIGHT_HEADER = "# Copyright (c) 2026 Amphidrom GmbH. All rights reserved."

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXCLUDED_DIRECTORIES = {
    ".git",
    ".idea",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "venv",
}


def is_excluded(file_path: Path) -> bool:
    relative_path = file_path.relative_to(PROJECT_ROOT)
    return any(part in EXCLUDED_DIRECTORIES for part in relative_path.parts)


def has_copyright_header(file_path: Path) -> bool:
    try:
        with file_path.open(encoding="utf-8") as file:
            first_line = file.readline().rstrip("\r\n")
    except UnicodeDecodeError:
        return False

    return first_line == COPYRIGHT_HEADER


def test_all_python_files_have_copyright_header() -> None:
    files_without_header = [
        file_path.relative_to(PROJECT_ROOT)
        for file_path in PROJECT_ROOT.rglob("*.py")
        if not is_excluded(file_path)
        and not has_copyright_header(file_path)
    ]

    assert not files_without_header, (
        "The following Python files do not have the required copyright "
        "header as their first line:\n"
        + "\n".join(f"- {file_path}" for file_path in files_without_header)
    )