# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Tests for LocalOsRuntime.

Exec-Tests verwenden ``sys.executable`` damit sie plattformunabhängig laufen.
Dateisystem-Tests verwenden ``tmp_path`` (pytest built-in fixture).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from nlght.adapters.outbound.os_runtime.local import LocalOsRuntime

# ---------------------------------------------------------------------------
# exec()
# ---------------------------------------------------------------------------


async def test_exec_returns_exit_code_zero_on_success(tmp_path: Path) -> None:
    runtime = LocalOsRuntime(workdir=str(tmp_path))

    exit_code, _, _ = await runtime.exec([sys.executable, "-c", ""])

    assert exit_code == 0


async def test_exec_captures_stdout(tmp_path: Path) -> None:
    runtime = LocalOsRuntime(workdir=str(tmp_path))

    _, stdout, _ = await runtime.exec([sys.executable, "-c", "print('hello')"])

    assert stdout.strip() == "hello"


async def test_exec_captures_stderr(tmp_path: Path) -> None:
    runtime = LocalOsRuntime(workdir=str(tmp_path))

    _, _, stderr = await runtime.exec(
        [sys.executable, "-c", "import sys; sys.stderr.write('err\\n')"]
    )

    assert stderr.strip() == "err"


async def test_exec_returns_nonzero_exit_code_on_failure(tmp_path: Path) -> None:
    runtime = LocalOsRuntime(workdir=str(tmp_path))

    exit_code, _, _ = await runtime.exec([sys.executable, "-c", "raise SystemExit(1)"])

    assert exit_code == 1


async def test_exec_uses_cwd(tmp_path: Path) -> None:
    runtime = LocalOsRuntime(workdir=".")
    marker = tmp_path / "marker.txt"
    marker.write_text("x")

    # list files in tmp_path via Python, using cwd
    _, stdout, _ = await runtime.exec(
        [sys.executable, "-c", "import os; print(os.listdir('.'))"],
        cwd=str(tmp_path),
    )

    assert "marker.txt" in stdout


async def test_exec_uses_default_workdir(tmp_path: Path) -> None:
    runtime = LocalOsRuntime(workdir=str(tmp_path))

    _, stdout, _ = await runtime.exec(
        [sys.executable, "-c", "import os; print(os.getcwd())"]
    )

    assert str(tmp_path) in stdout


# ---------------------------------------------------------------------------
# read_text / write_text
# ---------------------------------------------------------------------------


async def test_write_text_creates_file(tmp_path: Path) -> None:
    runtime = LocalOsRuntime()
    target = tmp_path / "out.txt"

    await runtime.write_text(str(target), "hello world")

    assert target.read_text() == "hello world"


async def test_write_text_creates_missing_parent_dirs(tmp_path: Path) -> None:
    runtime = LocalOsRuntime()
    target = tmp_path / "a" / "b" / "c.txt"

    await runtime.write_text(str(target), "nested")

    assert target.read_text() == "nested"


async def test_read_text_returns_file_content(tmp_path: Path) -> None:
    runtime = LocalOsRuntime()
    src = tmp_path / "input.txt"
    src.write_text("content here")

    result = await runtime.read_text(str(src))

    assert result == "content here"


async def test_read_write_roundtrip(tmp_path: Path) -> None:
    runtime = LocalOsRuntime()
    path = str(tmp_path / "rw.txt")

    await runtime.write_text(path, "roundtrip")
    result = await runtime.read_text(path)

    assert result == "roundtrip"


# ---------------------------------------------------------------------------
# file_exists
# ---------------------------------------------------------------------------


async def test_file_exists_true_for_existing_file(tmp_path: Path) -> None:
    runtime = LocalOsRuntime()
    f = tmp_path / "exists.txt"
    f.write_text("x")

    assert await runtime.file_exists(str(f)) is True


async def test_file_exists_false_for_missing_file(tmp_path: Path) -> None:
    runtime = LocalOsRuntime()

    assert await runtime.file_exists(str(tmp_path / "nope.txt")) is False


# ---------------------------------------------------------------------------
# list_dir
# ---------------------------------------------------------------------------


async def test_list_dir_returns_direct_children(tmp_path: Path) -> None:
    runtime = LocalOsRuntime()
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.txt").write_text("b")

    entries = await runtime.list_dir(str(tmp_path))

    assert set(entries) == {"a.txt", "b.txt"}


async def test_list_dir_does_not_recurse(tmp_path: Path) -> None:
    runtime = LocalOsRuntime()
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "nested.txt").write_text("n")

    entries = await runtime.list_dir(str(tmp_path))

    assert "nested.txt" not in entries
    assert "sub" in entries


# ---------------------------------------------------------------------------
# make_dir
# ---------------------------------------------------------------------------


async def test_make_dir_creates_directory(tmp_path: Path) -> None:
    runtime = LocalOsRuntime()
    new_dir = tmp_path / "new"

    await runtime.make_dir(str(new_dir))

    assert new_dir.is_dir()


async def test_make_dir_creates_nested_dirs(tmp_path: Path) -> None:
    runtime = LocalOsRuntime()
    nested = tmp_path / "x" / "y" / "z"

    await runtime.make_dir(str(nested))

    assert nested.is_dir()


async def test_make_dir_idempotent(tmp_path: Path) -> None:
    runtime = LocalOsRuntime()
    d = tmp_path / "existing"
    d.mkdir()

    # must not raise
    await runtime.make_dir(str(d))

    assert d.is_dir()


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


async def test_delete_removes_file(tmp_path: Path) -> None:
    runtime = LocalOsRuntime()
    f = tmp_path / "to_delete.txt"
    f.write_text("bye")

    await runtime.delete(str(f))

    assert not f.exists()


async def test_delete_raises_for_nonexistent_file(tmp_path: Path) -> None:
    runtime = LocalOsRuntime()

    with pytest.raises(FileNotFoundError):
        await runtime.delete(str(tmp_path / "ghost.txt"))
