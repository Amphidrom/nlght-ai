# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Tests for GitTool — the built-in git VCS tool."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nlght.adapters.outbound.tools.builtin.git import (
    GitTool,
    _normalise_repo_relative_path,
    _parse_branches,
    _parse_log_formatted,
    _parse_status_porcelain,
)
from nlght.core.errors.errors import ToolExecutionError

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_tool(os_runtime: object = None, config: dict | None = None) -> GitTool:
    return GitTool(name="git", config=config or {}, os_runtime=os_runtime)


def _mock_runtime(*, default_result: tuple[int, str, str] = (0, "", "")) -> MagicMock:
    rt = MagicMock()
    rt.exec = AsyncMock(return_value=default_result)
    rt.write_text = AsyncMock(return_value=None)
    return rt


# ---------------------------------------------------------------------------
# KIND / signatures
# ---------------------------------------------------------------------------


def test_kind_is_git() -> None:
    assert GitTool.KIND == "git"


def test_signatures_count() -> None:
    assert len(GitTool.signatures()) == 13


def test_signature_names() -> None:
    names = {s.name for s in GitTool.signatures()}
    assert names == {
        "git_clone", "git_checkout", "git_status", "git_diff",
        "git_commit", "git_push", "git_pull", "git_branch_list",
        "git_log", "git_apply_patch", "git_apply_file_content",
        "git_read_file", "git_list_dir",
    }


def test_each_signature_has_matching_method() -> None:
    tool = _make_tool()
    for sig in GitTool.signatures():
        assert hasattr(tool, sig.method_name), f"missing method: {sig.method_name}"


def test_default_workdir_uses_workspace_repo_when_available() -> None:
    workspace = type("Workspace", (), {"root_path": Path("workspaces/session-1")})()
    rt = type("LocalOsRuntime", (), {})()

    tool = GitTool(name="git", config={}, os_runtime=rt, workspace=workspace)

    assert tool._workdir == str(Path("workspaces/session-1") / "repo")


def test_default_workdir_uses_workspace_only_for_local_runtime() -> None:
    workspace = type("Workspace", (), {"root_path": Path("workspaces/session-1")})()
    rt = type("DockerOsRuntime", (), {})()

    tool = GitTool(name="git", config={}, os_runtime=rt, workspace=workspace)

    assert tool._workdir == "/workspace"


def test_configured_workdir_overrides_workspace_default() -> None:
    workspace = type("Workspace", (), {"root_path": Path("workspaces/session-1")})()
    rt = type("DockerOsRuntime", (), {})()

    tool = GitTool(
        name="git",
        config={"workdir": "/custom/repo"},
        os_runtime=rt,
        workspace=workspace,
    )

    assert tool._workdir == "/custom/repo"


def test_repo_relative_path_rejects_absolute_paths_cross_platform() -> None:
    with pytest.raises(ToolExecutionError):
        _normalise_repo_relative_path("/workspace/repo/file.md")
    with pytest.raises(ToolExecutionError):
        _normalise_repo_relative_path("C:/workspace/repo/file.md")


def test_repo_relative_path_accepts_diff_prefixed_relative_path() -> None:
    assert _normalise_repo_relative_path("a/challenges/x.md") == "challenges/x.md"
    assert _normalise_repo_relative_path("b/challenges/x.md") == "challenges/x.md"


# ---------------------------------------------------------------------------
# no runtime → ToolExecutionError
# ---------------------------------------------------------------------------


async def test_clone_raises_without_runtime() -> None:
    tool = _make_tool()
    with pytest.raises(ToolExecutionError):
        await tool.clone(repo_url="https://example.com/repo.git")


async def test_status_raises_without_runtime() -> None:
    tool = _make_tool()
    with pytest.raises(ToolExecutionError):
        await tool.status()


# ---------------------------------------------------------------------------
# parse helpers
# ---------------------------------------------------------------------------


def test_parse_status_porcelain_empty() -> None:
    assert _parse_status_porcelain("") == []


def test_parse_status_porcelain_modified() -> None:
    raw = " M src/foo.py\n?? new_file.txt\n"
    entries = _parse_status_porcelain(raw)
    assert len(entries) == 2
    assert entries[0] == {"index": "", "worktree": "M", "path": "src/foo.py"}
    assert entries[1] == {"index": "?", "worktree": "?", "path": "new_file.txt"}


def test_parse_log_formatted_parses_entries() -> None:
    raw = "abc123|Alice|2024-01-01|Fix bug\ndef456|Bob|2024-01-02|Add feature\n"
    entries = _parse_log_formatted(raw)
    assert len(entries) == 2
    assert entries[0] == {"hash": "abc123", "author": "Alice", "date": "2024-01-01", "message": "Fix bug"}


def test_parse_log_formatted_skips_malformed() -> None:
    raw = "bad line\nabc|Alice|2024|msg\n"
    entries = _parse_log_formatted(raw)
    assert len(entries) == 1


def test_parse_branches_current_marker() -> None:
    raw = "* main\n  feature/x\n  remotes/origin/main\n"
    branches = _parse_branches(raw)
    assert branches[0] == {"name": "main", "current": True, "remote": False}
    assert branches[1] == {"name": "feature/x", "current": False, "remote": False}
    assert branches[2]["remote"] is True


# ---------------------------------------------------------------------------
# clone
# ---------------------------------------------------------------------------


async def test_clone_builds_correct_command() -> None:
    rt = _mock_runtime()
    tool = _make_tool(os_runtime=rt)

    await tool.clone(repo_url="https://example.com/repo.git")

    rt.exec.assert_awaited_once()
    cmd, *_ = rt.exec.call_args.args
    assert cmd[:3] == ["git", "clone", "https://example.com/repo.git"]


async def test_clone_with_branch_and_depth() -> None:
    rt = _mock_runtime()
    tool = _make_tool(os_runtime=rt)

    await tool.clone(repo_url="https://example.com/repo.git", branch="dev", depth=1)

    cmd = rt.exec.call_args.args[0]
    assert "-b" in cmd
    assert "dev" in cmd
    assert "--depth" in cmd
    assert "1" in cmd


async def test_clone_updates_workdir_on_success() -> None:
    rt = _mock_runtime(default_result=(0, "", ""))
    tool = _make_tool(os_runtime=rt, config={"workdir": "/workspace"})

    await tool.clone(repo_url="https://example.com/my-repo.git")

    assert tool._workdir == "/workspace/my-repo"


async def test_clone_returns_json_with_success_flag() -> None:
    rt = _mock_runtime(default_result=(0, "Cloning...", ""))
    tool = _make_tool(os_runtime=rt)

    result = json.loads(await tool.clone(repo_url="https://example.com/repo.git"))

    assert result["success"] is True
    assert result["exit_code"] == 0


async def test_clone_does_not_update_workdir_on_failure() -> None:
    rt = _mock_runtime(default_result=(1, "", "error"))
    tool = _make_tool(os_runtime=rt, config={"workdir": "/workspace"})

    await tool.clone(repo_url="https://example.com/repo.git")

    assert tool._workdir == "/workspace"


# ---------------------------------------------------------------------------
# checkout
# ---------------------------------------------------------------------------


async def test_checkout_calls_git_checkout() -> None:
    rt = _mock_runtime(default_result=(0, "", ""))
    tool = _make_tool(os_runtime=rt)

    result = json.loads(await tool.checkout(ref="main"))

    assert result["success"] is True
    cmd = rt.exec.call_args.args[0]
    assert "checkout" in cmd
    assert "main" in cmd


async def test_checkout_create_adds_b_flag() -> None:
    rt = _mock_runtime(default_result=(0, "", ""))
    tool = _make_tool(os_runtime=rt)

    await tool.checkout(ref="feature/new", create=True)

    cmd = rt.exec.call_args.args[0]
    assert "-b" in cmd


async def test_checkout_raises_on_failure() -> None:
    rt = _mock_runtime(default_result=(1, "", "pathspec not found"))
    tool = _make_tool(os_runtime=rt)

    with pytest.raises(ToolExecutionError, match="checkout"):
        await tool.checkout(ref="nonexistent")


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


async def test_status_returns_structured_json() -> None:
    rt = MagicMock()
    rt.exec = AsyncMock(side_effect=[
        (0, " M foo.py\n?? bar.py\n", ""),  # git status
        (0, "main\n", ""),                  # git branch --show-current
    ])
    tool = _make_tool(os_runtime=rt)

    result = json.loads(await tool.status())

    assert result["branch"] == "main"
    assert result["clean"] is False
    assert result["summary"]["untracked"] == 1


async def test_status_clean_repo() -> None:
    rt = MagicMock()
    rt.exec = AsyncMock(side_effect=[
        (0, "", ""),     # git status
        (0, "main\n", ""),
    ])
    tool = _make_tool(os_runtime=rt)

    result = json.loads(await tool.status())

    assert result["clean"] is True


# ---------------------------------------------------------------------------
# diff
# ---------------------------------------------------------------------------


async def test_diff_returns_output() -> None:
    rt = _mock_runtime(default_result=(0, "diff --git a/x.py b/x.py\n...", ""))
    tool = _make_tool(os_runtime=rt)

    result = await tool.diff()

    assert "diff --git" in result


async def test_diff_staged_flag() -> None:
    rt = _mock_runtime(default_result=(0, "", ""))
    tool = _make_tool(os_runtime=rt)

    await tool.diff(staged=True)

    cmd = rt.exec.call_args.args[0]
    assert "--cached" in cmd


async def test_diff_no_changes_returns_placeholder() -> None:
    rt = _mock_runtime(default_result=(0, "", ""))
    tool = _make_tool(os_runtime=rt)

    result = await tool.diff()

    assert result == "(no changes)"


# ---------------------------------------------------------------------------
# commit
# ---------------------------------------------------------------------------


async def test_commit_stages_all_and_commits() -> None:
    rt = MagicMock()
    rt.exec = AsyncMock(side_effect=[
        (0, "", ""),           # git add -A
        (0, "", ""),           # git commit
        (0, "abc1234\n", ""),  # git rev-parse
    ])
    tool = _make_tool(os_runtime=rt)

    result = json.loads(await tool.commit(message="fix: something"))

    assert result["success"] is True
    assert result["hash"] == "abc1234"
    assert result["message"] == "fix: something"


async def test_commit_raises_if_add_fails() -> None:
    rt = MagicMock()
    rt.exec = AsyncMock(return_value=(1, "", "permission denied"))
    tool = _make_tool(os_runtime=rt)

    with pytest.raises(ToolExecutionError, match="git add"):
        await tool.commit(message="oops")


async def test_commit_with_specific_paths() -> None:
    rt = MagicMock()
    rt.exec = AsyncMock(side_effect=[
        (0, "", ""),
        (0, "", ""),
        (0, "def5678\n", ""),
    ])
    tool = _make_tool(os_runtime=rt)

    await tool.commit(message="partial", paths=["src/foo.py"])

    add_cmd = rt.exec.call_args_list[0].args[0]
    assert "src/foo.py" in add_cmd
    assert "--" in add_cmd


async def test_commit_author_added_when_configured() -> None:
    rt = MagicMock()
    rt.exec = AsyncMock(side_effect=[
        (0, "", ""),
        (0, "", ""),
        (0, "abc\n", ""),
    ])
    tool = _make_tool(os_runtime=rt, config={"commit_author": "Bot <bot@example.com>"})

    await tool.commit(message="auto")

    commit_cmd = rt.exec.call_args_list[1].args[0]
    assert "--author" in commit_cmd
    assert "Bot <bot@example.com>" in commit_cmd


# ---------------------------------------------------------------------------
# push / pull
# ---------------------------------------------------------------------------


async def test_push_returns_json_with_success_flag() -> None:
    rt = _mock_runtime(default_result=(0, "pushed", ""))
    tool = _make_tool(os_runtime=rt, config={"default_remote": "origin"})

    result = json.loads(await tool.push())

    assert result["success"] is True
    cmd = rt.exec.call_args.args[0]
    assert "origin" in cmd


async def test_push_force_adds_flag() -> None:
    rt = _mock_runtime(default_result=(0, "", ""))
    tool = _make_tool(os_runtime=rt)

    await tool.push(force=True)

    assert "--force-with-lease" in rt.exec.call_args.args[0]


async def test_pull_rebase_adds_flag() -> None:
    rt = _mock_runtime(default_result=(0, "", ""))
    tool = _make_tool(os_runtime=rt)

    await tool.pull(rebase=True)

    assert "--rebase" in rt.exec.call_args.args[0]


# ---------------------------------------------------------------------------
# branch_list
# ---------------------------------------------------------------------------


async def test_branch_list_returns_json_array() -> None:
    raw = "* main\n  dev\n"
    rt = _mock_runtime(default_result=(0, raw, ""))
    tool = _make_tool(os_runtime=rt)

    result = json.loads(await tool.branch_list())

    assert isinstance(result, list)
    assert result[0]["current"] is True


# ---------------------------------------------------------------------------
# log
# ---------------------------------------------------------------------------


async def test_log_returns_commit_list() -> None:
    raw = "abc|Alice|2024-01-01|Init\ndef|Bob|2024-01-02|Add tests\n"
    rt = _mock_runtime(default_result=(0, raw, ""))
    tool = _make_tool(os_runtime=rt)

    result = json.loads(await tool.log())

    assert len(result) == 2
    assert result[0]["hash"] == "abc"


async def test_log_with_count_limit() -> None:
    rt = _mock_runtime(default_result=(0, "", ""))
    tool = _make_tool(os_runtime=rt)

    await tool.log(count=5)

    cmd = rt.exec.call_args.args[0]
    assert "-n5" in cmd


# ---------------------------------------------------------------------------
# apply_patch
# ---------------------------------------------------------------------------


async def test_apply_patch_writes_file_then_runs_git_apply() -> None:
    rt = MagicMock()
    rt.exec = AsyncMock(return_value=(0, "", ""))
    rt.write_text = AsyncMock()
    tool = _make_tool(os_runtime=rt)

    result = json.loads(await tool.apply_patch(patch_content="--- a/x\n+++ b/x\n"))

    rt.write_text.assert_awaited_once()
    path_arg = rt.write_text.call_args.args[0]
    assert path_arg.endswith(".diff")
    cmd = rt.exec.call_args.args[0]
    assert "--whitespace=nowarn" in cmd
    assert result["success"] is True


async def test_apply_patch_check_only_adds_flag() -> None:
    rt = MagicMock()
    rt.exec = AsyncMock(return_value=(0, "", ""))
    rt.write_text = AsyncMock()
    tool = _make_tool(os_runtime=rt)

    result = json.loads(await tool.apply_patch(patch_content="diff", check_only=True))

    cmd = rt.exec.call_args.args[0]
    assert "--check" in cmd
    assert result["check_only"] is True


async def test_apply_file_content_generates_patch_and_writes_file() -> None:
    rt = MagicMock()
    rt.read_text = AsyncMock(return_value="old\n")
    rt.write_text = AsyncMock()
    rt.exec = AsyncMock(return_value=(0, "", ""))
    tool = _make_tool(os_runtime=rt, config={"workdir": "/workspace/repo"})

    result = json.loads(await tool.apply_file_content(
        path="docs/file.md",
        content="new\n",
    ))

    assert result["success"] is True
    assert result["changed"] is True
    assert "--- a/docs/file.md" in result["patch"]
    assert "+++ b/docs/file.md" in result["patch"]
    rt.read_text.assert_awaited_once_with("/workspace/repo/docs/file.md")
    rt.write_text.assert_any_await("/workspace/repo/docs/file.md", "new\n")


async def test_apply_file_content_check_only_does_not_write_target() -> None:
    rt = MagicMock()
    rt.read_text = AsyncMock(return_value="old\n")
    rt.write_text = AsyncMock()
    rt.exec = AsyncMock(return_value=(0, "", ""))
    tool = _make_tool(os_runtime=rt, config={"workdir": "/workspace/repo"})

    result = json.loads(await tool.apply_file_content(
        path="docs/file.md",
        content="new\n",
        check_only=True,
    ))

    assert result["success"] is True
    assert result["changed"] is False
    target_writes = [
        call for call in rt.write_text.await_args_list
        if call.args[0] == "/workspace/repo/docs/file.md"
    ]
    assert target_writes == []
