# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Tests for ShellTool — the built-in shell/filesystem tool."""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from nlght.adapters.outbound.tools.builtin.shell import ShellTool
from nlght.core.errors.errors import ToolExecutionError

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_tool(os_runtime: object = None, config: dict | None = None) -> ShellTool:
    return ShellTool(name="shell", config=config or {}, os_runtime=os_runtime)


def _mock_runtime(
    *,
    exec_result: tuple[int, str, str] = (0, "", ""),
    read_result: str = "",
    exists_result: bool = True,
    list_result: list[str] | None = None,
) -> MagicMock:
    rt = MagicMock()
    rt.exec = AsyncMock(return_value=exec_result)
    rt.read_text = AsyncMock(return_value=read_result)
    rt.write_text = AsyncMock(return_value=None)
    rt.file_exists = AsyncMock(return_value=exists_result)
    rt.list_dir = AsyncMock(return_value=list_result or [])
    rt.make_dir = AsyncMock(return_value=None)
    rt.delete = AsyncMock(return_value=None)
    return rt


# ---------------------------------------------------------------------------
# KIND and signatures
# ---------------------------------------------------------------------------


def test_kind_is_shell() -> None:
    assert ShellTool.KIND == "shell"


def test_signatures_returns_nine_entries() -> None:
    sigs = ShellTool.signatures()
    assert len(sigs) == 9


def test_signature_names() -> None:
    names = {s.name for s in ShellTool.signatures()}
    assert names == {
        "run_script",
        "shell_exec",
        "shell_read_chunk",
        "shell_read_text",
        "shell_write_text",
        "shell_file_exists",
        "shell_list_dir",
        "shell_make_dir",
        "shell_delete",
    }


def test_each_signature_has_matching_method() -> None:
    tool = _make_tool()
    for sig in ShellTool.signatures():
        assert hasattr(tool, sig.method_name), f"missing method: {sig.method_name}"


# ---------------------------------------------------------------------------
# no OsRuntime → ToolExecutionError
# ---------------------------------------------------------------------------


async def test_exec_raises_when_no_runtime() -> None:
    tool = _make_tool(os_runtime=None)
    with pytest.raises(ToolExecutionError):
        await tool.exec(command="echo hi")


async def test_read_text_raises_when_no_runtime() -> None:
    tool = _make_tool(os_runtime=None)
    with pytest.raises(ToolExecutionError):
        await tool.read_text(path="/tmp/x")


# ---------------------------------------------------------------------------
# exec
# ---------------------------------------------------------------------------


async def test_exec_returns_json_with_exit_code_stdout_stderr() -> None:
    rt = _mock_runtime(exec_result=(0, "hello\n", ""))
    tool = _make_tool(os_runtime=rt)

    result = json.loads(await tool.exec(command="echo hello"))

    assert result == {"exit_code": 0, "stdout": "hello\n", "stderr": ""}


async def test_exec_passes_command_via_bash_c() -> None:
    rt = _mock_runtime()
    tool = _make_tool(os_runtime=rt)

    await tool.exec(command="ls -la")

    rt.exec.assert_awaited_once_with(["bash", "-c", "ls -la"], cwd=None)


async def test_exec_forwards_explicit_cwd() -> None:
    rt = _mock_runtime()
    tool = _make_tool(os_runtime=rt)

    await tool.exec(command="pwd", cwd="/workspace")

    rt.exec.assert_awaited_once_with(["bash", "-c", "pwd"], cwd="/workspace")


async def test_exec_uses_config_default_cwd_when_no_cwd_given() -> None:
    rt = _mock_runtime()
    tool = _make_tool(os_runtime=rt, config={"default_cwd": "/default"})

    await tool.exec(command="pwd")

    rt.exec.assert_awaited_once_with(["bash", "-c", "pwd"], cwd="/default")


async def test_exec_explicit_cwd_overrides_default_cwd() -> None:
    rt = _mock_runtime()
    tool = _make_tool(os_runtime=rt, config={"default_cwd": "/default"})

    await tool.exec(command="pwd", cwd="/explicit")

    rt.exec.assert_awaited_once_with(["bash", "-c", "pwd"], cwd="/explicit")


async def test_exec_wraps_runtime_error_in_tool_execution_error() -> None:
    rt = MagicMock()
    rt.exec = AsyncMock(side_effect=RuntimeError("boom"))
    tool = _make_tool(os_runtime=rt)

    with pytest.raises(ToolExecutionError, match="shell_exec failed"):
        await tool.exec(command="bad")


async def test_exec_nonzero_exit_code_is_returned_not_raised() -> None:
    rt = _mock_runtime(exec_result=(1, "", "error"))
    tool = _make_tool(os_runtime=rt)

    result = json.loads(await tool.exec(command="false"))

    assert result["exit_code"] == 1
    assert result["stderr"] == "error"


# ---------------------------------------------------------------------------
# read_text
# ---------------------------------------------------------------------------


async def test_read_text_returns_json_with_content() -> None:
    rt = _mock_runtime(read_result="hello world")
    tool = _make_tool(os_runtime=rt)

    result = json.loads(await tool.read_text(path="/tmp/f.txt"))

    assert result == {"content": "hello world"}


async def test_read_text_re_raises_file_not_found() -> None:
    rt = MagicMock()
    rt.read_text = AsyncMock(side_effect=FileNotFoundError("/tmp/missing"))
    tool = _make_tool(os_runtime=rt)

    with pytest.raises(FileNotFoundError):
        await tool.read_text(path="/tmp/missing")


async def test_read_text_wraps_other_errors() -> None:
    rt = MagicMock()
    rt.read_text = AsyncMock(side_effect=OSError("io error"))
    tool = _make_tool(os_runtime=rt)

    with pytest.raises(ToolExecutionError, match="shell_read_text failed"):
        await tool.read_text(path="/tmp/x")


# ---------------------------------------------------------------------------
# write_text
# ---------------------------------------------------------------------------


async def test_write_text_returns_ok_json() -> None:
    rt = _mock_runtime()
    tool = _make_tool(os_runtime=rt)

    result = json.loads(await tool.write_text(path="/tmp/out.txt", content="data"))

    assert result == {"ok": True, "path": "/tmp/out.txt"}
    rt.write_text.assert_awaited_once_with("/tmp/out.txt", "data")


async def test_write_text_wraps_errors() -> None:
    rt = MagicMock()
    rt.write_text = AsyncMock(side_effect=PermissionError("denied"))
    tool = _make_tool(os_runtime=rt)

    with pytest.raises(ToolExecutionError, match="shell_write_text failed"):
        await tool.write_text(path="/root/x", content="x")


async def test_write_text_generates_scratch_path_when_omitted() -> None:
    rt = _mock_runtime()
    tool = _make_tool(os_runtime=rt)

    result = json.loads(await tool.write_text(content="data"))

    assert result["ok"] is True
    assert result["path"].startswith("/workspace/tmp/nlght/script_")
    # The generated path is the one actually written to.
    rt.write_text.assert_awaited_once_with(result["path"], "data")


# ---------------------------------------------------------------------------
# run_script
# ---------------------------------------------------------------------------


async def test_run_script_writes_runs_and_cleans_up() -> None:
    rt = _mock_runtime(exec_result=(0, "HIT | sqli", ""))
    tool = _make_tool(os_runtime=rt)

    result = json.loads(await tool.run_script(content="print('x')"))

    assert result == {"exit_code": 0, "stdout": "HIT | sqli", "stderr": ""}
    # script written under the scratch dir with a .py extension by default …
    written_path = rt.write_text.await_args.args[0]
    assert written_path.startswith("/workspace/tmp/nlght/script_")
    assert written_path.endswith(".py")
    # … executed via python3 through bash …
    cmd = rt.exec.await_args.args[0]
    assert cmd[0] == "bash" and "python3" in cmd[2] and written_path in cmd[2]
    # … and the scratch file is deleted afterwards.
    rt.delete.assert_awaited_once_with(written_path)


async def test_run_script_sh_uses_bash_interpreter() -> None:
    rt = _mock_runtime(exec_result=(0, "ok", ""))
    tool = _make_tool(os_runtime=rt)

    await tool.run_script(content="echo hi", ext="sh")

    written_path = rt.write_text.await_args.args[0]
    assert written_path.endswith(".sh")
    cmd = rt.exec.await_args.args[0]
    assert "bash " in cmd[2] and "python3" not in cmd[2]


async def test_run_script_cleans_up_even_on_exec_error() -> None:
    rt = _mock_runtime()
    rt.exec = AsyncMock(side_effect=RuntimeError("boom"))
    tool = _make_tool(os_runtime=rt)

    with pytest.raises(ToolExecutionError, match="run_script exec failed"):
        await tool.run_script(content="print('x')")

    rt.delete.assert_awaited_once()


# ---------------------------------------------------------------------------
# file_exists
# ---------------------------------------------------------------------------


async def test_file_exists_true() -> None:
    rt = _mock_runtime(exists_result=True)
    tool = _make_tool(os_runtime=rt)

    result = json.loads(await tool.file_exists(path="/tmp/exists.txt"))

    assert result == {"exists": True, "path": "/tmp/exists.txt"}


async def test_file_exists_false() -> None:
    rt = _mock_runtime(exists_result=False)
    tool = _make_tool(os_runtime=rt)

    result = json.loads(await tool.file_exists(path="/tmp/nope.txt"))

    assert result["exists"] is False


# ---------------------------------------------------------------------------
# list_dir
# ---------------------------------------------------------------------------


async def test_list_dir_returns_entries() -> None:
    rt = _mock_runtime(list_result=["a.txt", "b.txt"])
    tool = _make_tool(os_runtime=rt)

    result = json.loads(await tool.list_dir(path="/tmp"))

    assert result == {"entries": ["a.txt", "b.txt"], "path": "/tmp"}


async def test_list_dir_re_raises_file_not_found() -> None:
    rt = MagicMock()
    rt.list_dir = AsyncMock(side_effect=FileNotFoundError("/tmp/missing"))
    tool = _make_tool(os_runtime=rt)

    with pytest.raises(FileNotFoundError):
        await tool.list_dir(path="/tmp/missing")


# ---------------------------------------------------------------------------
# make_dir
# ---------------------------------------------------------------------------


async def test_make_dir_returns_ok_json() -> None:
    rt = _mock_runtime()
    tool = _make_tool(os_runtime=rt)

    result = json.loads(await tool.make_dir(path="/tmp/new"))

    assert result == {"ok": True, "path": "/tmp/new"}
    rt.make_dir.assert_awaited_once_with("/tmp/new")


async def test_make_dir_wraps_errors() -> None:
    rt = MagicMock()
    rt.make_dir = AsyncMock(side_effect=RuntimeError("boom"))
    tool = _make_tool(os_runtime=rt)

    with pytest.raises(ToolExecutionError, match="shell_make_dir failed"):
        await tool.make_dir(path="/tmp/x")


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


async def test_delete_returns_ok_json() -> None:
    rt = _mock_runtime()
    tool = _make_tool(os_runtime=rt)

    result = json.loads(await tool.delete(path="/tmp/old.txt"))

    assert result == {"ok": True, "path": "/tmp/old.txt"}
    rt.delete.assert_awaited_once_with("/tmp/old.txt")


async def test_delete_re_raises_file_not_found() -> None:
    rt = MagicMock()
    rt.delete = AsyncMock(side_effect=FileNotFoundError("/tmp/ghost"))
    tool = _make_tool(os_runtime=rt)

    with pytest.raises(FileNotFoundError):
        await tool.delete(path="/tmp/ghost")


async def test_delete_wraps_other_errors() -> None:
    rt = MagicMock()
    rt.delete = AsyncMock(side_effect=PermissionError("denied"))
    tool = _make_tool(os_runtime=rt)

    with pytest.raises(ToolExecutionError, match="shell_delete failed"):
        await tool.delete(path="/tmp/x")
