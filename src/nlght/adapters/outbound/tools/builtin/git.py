# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import difflib
import json
import logging
import os
import tempfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, Any, ClassVar

from nlght.adapters.outbound.tools.builtin.action_semantics import (
    ARBITRARY_EXTERNAL_READ_WRITE,
    ARBITRARY_EXTERNAL_WRITE,
    ARBITRARY_NETWORK_CLONE,
    CHECK_ONLY_WRITE,
    READ_RUNTIME,
    WRITE_RUNTIME,
    RelativePathBinder,
)
from nlght.core.errors.errors import ToolExecutionError
from nlght.core.tools.tool import ToolBase, ToolParameter, ToolSignature

if TYPE_CHECKING:
    from nlght.core.workspace.workspace import WorkspaceContext
    from nlght.ports.outbound.os_runtime import OsRuntime

logger = logging.getLogger(__name__)


def _default_git_workdir(workspace: WorkspaceContext | None, os_runtime: OsRuntime | None) -> str:
    runtime_name = type(os_runtime).__name__ if os_runtime is not None else ""
    if (
        runtime_name == "LocalOsRuntime"
        and workspace is not None
        and getattr(workspace, "root_path", None) is not None
    ):
        return str(Path(workspace.root_path) / "repo")
    return "/workspace"


def _normalise_repo_relative_path(path: str) -> str:
    rel = path.replace("\\", "/").strip()
    if rel.startswith("a/") or rel.startswith("b/"):
        rel = rel[2:]
    if (
        not rel
        or PurePosixPath(rel).is_absolute()
        or PureWindowsPath(rel).is_absolute()
    ):
        raise ToolExecutionError("git path must be relative to the repository workdir")
    parts = [part for part in rel.split("/") if part]
    if any(part == ".." for part in parts):
        raise ToolExecutionError("git path must not contain '..'")
    return "/".join(parts)


def _join_runtime_path(workdir: str, rel_path: str) -> str:
    base = workdir.rstrip("/\\")
    return f"{base}/{rel_path}"


def _unified_diff_for_content(
    *,
    rel_path: str,
    old_content: str,
    new_content: str,
) -> str:
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    return "".join(difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=f"a/{rel_path}",
        tofile=f"b/{rel_path}",
    ))


# ---------------------------------------------------------------------------
# Parse helpers
# ---------------------------------------------------------------------------


def _parse_status_porcelain(raw: str) -> list[dict[str, str]]:
    entries = []
    for line in raw.splitlines():
        if len(line) < 4:
            continue
        entries.append(
            {
                "index": line[0].strip(),
                "worktree": line[1].strip(),
                "path": line[3:].strip(),
            }
        )
    return entries


def _parse_log_formatted(raw: str) -> list[dict[str, str]]:
    entries = []
    for line in raw.strip().splitlines():
        parts = line.split("|", 3)
        if len(parts) == 4:
            entries.append(
                {
                    "hash": parts[0],
                    "author": parts[1],
                    "date": parts[2],
                    "message": parts[3],
                }
            )
    return entries


def _parse_branches(raw: str) -> list[dict[str, Any]]:
    branches = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        current = line.startswith("* ")
        name = line.lstrip("* ").strip()
        branches.append(
            {
                "name": name,
                "current": current,
                "remote": name.startswith("remotes/"),
            }
        )
    return branches


# ---------------------------------------------------------------------------
# GitTool
# ---------------------------------------------------------------------------


class GitTool(ToolBase):
    """Built-in tool for Git VCS operations inside an OsRuntime.

    Requires an OsRuntime with ``git`` available in the runtime environment.

    Configuration (optional):
      ``workdir``              — default working directory for git commands (default ``/workspace``)
      ``default_remote``       — remote name used when none is specified (default ``origin``)
      ``default_clone_depth``  — shallow clone depth; ``None`` = full clone
      ``commit_author``        — author override for commits (``Name <email>`` format)
    """

    KIND: ClassVar[str]     = "git"
    PROVIDER: ClassVar[str] = "local"

    def __init__(
        self,
        *,
        name: str,
        config: dict[str, Any],
        os_runtime: OsRuntime | None = None,
        workspace: WorkspaceContext | None = None,
        **_: object,
    ) -> None:
        super().__init__(name=name, config=config, os_runtime=os_runtime)
        self._workdir: str = str(
            config.get("workdir") or _default_git_workdir(workspace, os_runtime)
        )
        self._default_remote: str = str(config.get("default_remote", "origin"))
        self._clone_depth: int | None = config.get("default_clone_depth")
        self._commit_author: str | None = config.get("commit_author")

    @classmethod
    def signatures(cls) -> list[ToolSignature]:
        return [
            ToolSignature(
                name="git_clone",
                description=(
                    "Clone a git repository into the runtime container. "
                    "Token auth must be pre-configured in the container environment."
                ),
                method_name="clone",
                parameters=[
                    ToolParameter(name="repo_url", type="string", description="HTTPS repository URL."),
                    ToolParameter(name="target_dir", type="string", description="Target directory inside workdir. Defaults to repo name.", required=False),
                    ToolParameter(name="branch", type="string", description="Branch to checkout after clone.", required=False),
                    ToolParameter(name="depth", type="number", description="Shallow clone depth. 1 = fastest.", required=False),
                ],
                action=ARBITRARY_NETWORK_CLONE,
                argument_binder=RelativePathBinder(fields=("target_dir",)),
            ),
            ToolSignature(
                name="git_checkout",
                description="Switch to an existing branch/tag/commit, or create a new branch.",
                method_name="checkout",
                parameters=[
                    ToolParameter(name="ref", type="string", description="Branch name, tag, or commit SHA."),
                    ToolParameter(name="create", type="boolean", description="Create new branch (-b).", required=False, default=False),
                ],
                action=WRITE_RUNTIME,
            ),
            ToolSignature(
                name="git_status",
                description="Structured working-tree status: current branch, changed files, counts.",
                method_name="status",
                parameters=[],
                action=READ_RUNTIME,
            ),
            ToolSignature(
                name="git_diff",
                description="Show file changes. Unstaged by default, staged with staged=true, or diff against a ref.",
                method_name="diff",
                parameters=[
                    ToolParameter(name="staged", type="boolean", description="Show staged (--cached) changes.", required=False, default=False),
                    ToolParameter(name="ref", type="string", description="Diff against ref, e.g. 'HEAD~3', 'main'.", required=False),
                    ToolParameter(name="paths", type="array", description="Limit to specific file paths.", required=False),
                    ToolParameter(name="stat_only", type="boolean", description="File-level stats only.", required=False, default=False),
                ],
                action=READ_RUNTIME,
                argument_binder=RelativePathBinder(list_fields=("paths",)),
            ),
            ToolSignature(
                name="git_commit",
                description="Stage files and create a commit. Stages all changes by default.",
                method_name="commit",
                parameters=[
                    ToolParameter(name="message", type="string", description="Commit message."),
                    ToolParameter(name="add_all", type="boolean", description="Stage all changes (git add -A). Ignored when paths given.",
                          required=False, default=True),
                    ToolParameter(name="paths", type="array", description="Stage only these paths. Overrides add_all.", required=False),
                ],
                action=WRITE_RUNTIME,
                argument_binder=RelativePathBinder(list_fields=("paths",)),
            ),
            ToolSignature(
                name="git_push",
                description="Push commits to remote. Uses --force-with-lease for force pushes.",
                method_name="push",
                parameters=[
                    ToolParameter(name="remote", type="string", description="Remote name. Defaults to configured default_remote.", required=False),
                    ToolParameter(name="branch", type="string", description="Branch to push. Defaults to current branch.", required=False),
                    ToolParameter(name="force", type="boolean", description="Force push (--force-with-lease).", required=False, default=False),
                    ToolParameter(name="set_upstream", type="boolean", description="Set upstream tracking (-u).", required=False, default=False),
                ],
                action=ARBITRARY_EXTERNAL_WRITE,
            ),
            ToolSignature(
                name="git_pull",
                description="Pull changes from remote. Merge by default, rebase with rebase=true.",
                method_name="pull",
                parameters=[
                    ToolParameter(name="remote", type="string", description="Remote name. Defaults to configured default_remote.", required=False),
                    ToolParameter(name="branch", type="string", description="Branch to pull.", required=False),
                    ToolParameter(name="rebase", type="boolean", description="Use rebase instead of merge.", required=False, default=False),
                ],
                action=ARBITRARY_EXTERNAL_READ_WRITE,
            ),
            ToolSignature(
                name="git_branch_list",
                description="List all branches. Structured JSON with current/remote indicators.",
                method_name="branch_list",
                parameters=[
                    ToolParameter(name="include_remote", type="boolean", description="Include remote-tracking branches.", required=False, default=True),
                ],
                action=READ_RUNTIME,
            ),
            ToolSignature(
                name="git_log",
                description="Commit history as structured JSON (hash, author, date, message).",
                method_name="log",
                parameters=[
                    ToolParameter(name="count", type="number", description="Max commits to return.", required=False, default=20),
                    ToolParameter(name="ref", type="string", description="Starting ref (branch, tag, SHA).", required=False),
                    ToolParameter(name="paths", type="array", description="Only commits touching these paths.", required=False),
                ],
                action=READ_RUNTIME,
                argument_binder=RelativePathBinder(list_fields=("paths",)),
            ),
            ToolSignature(
                name="git_apply_patch",
                description="Apply a unified diff/patch to the working tree. Supports dry-run via check_only.",
                method_name="apply_patch",
                parameters=[
                    ToolParameter(name="patch_content", type="string", description="Unified diff content."),
                    ToolParameter(name="check_only", type="boolean", description="Dry-run: check without modifying files.", required=False, default=False),
                ],
                action=CHECK_ONLY_WRITE,
            ),
            ToolSignature(
                name="git_apply_file_content",
                description=(
                    "Replace one repository-relative file with complete content. "
                    "The tool generates and validates a unified diff before writing."
                ),
                method_name="apply_file_content",
                parameters=[
                    ToolParameter(name="path", type="string", description="Repository-relative file path."),
                    ToolParameter(name="content", type="string", description="Complete desired file content."),
                    ToolParameter(
                        name="check_only",
                        type="boolean",
                        description="Dry-run: generate and check patch without modifying files.",
                        required=False,
                        default=False,
                    ),
                ],
                action=CHECK_ONLY_WRITE,
                argument_binder=RelativePathBinder(fields=("path",)),
            ),
            ToolSignature(
                name="git_read_file",
                description=(
                    "Read the content of a file at a specific git ref (default HEAD). "
                    "Pass start_line/end_line to read only a section (1-indexed, inclusive). "
                    "Prefer narrowing with start_line/end_line on large files rather than reading the whole file."
                ),
                method_name="read_file",
                parameters=[
                    ToolParameter(name="path", type="string",
                                  description="Path relative to the repository root."),
                    ToolParameter(name="ref", type="string",
                                  description="Git ref (branch, tag, commit SHA). Defaults to HEAD.",
                                  required=False, default="HEAD"),
                    ToolParameter(name="start_line", type="number",
                                  description="First line to read (1-indexed, inclusive). Omit to start from line 1.",
                                  required=False),
                    ToolParameter(name="end_line", type="number",
                                  description="Last line to read (1-indexed, inclusive). Omit to read to end of file.",
                                  required=False),
                ],
                action=READ_RUNTIME,
                argument_binder=RelativePathBinder(fields=("path",)),
            ),
            ToolSignature(
                name="git_list_dir",
                description=(
                    "List files and subdirectories tracked by git at a specific ref (default HEAD). "
                    "Use path='.' for the repository root."
                ),
                method_name="list_dir_at_ref",
                parameters=[
                    ToolParameter(name="path", type="string",
                                  description="Directory path relative to repo root, or '.' for root.",
                                  required=False, default="."),
                    ToolParameter(name="ref", type="string", description="Git ref. Defaults to HEAD.", required=False, default="HEAD"),
                ],
                action=READ_RUNTIME,
                argument_binder=RelativePathBinder(fields=("path",)),
            ),
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _runtime(self) -> OsRuntime:
        if self.os_runtime is None:
            raise ToolExecutionError(
                f"GitTool '{self.name}' requires an OsRuntime but none was injected."
            )
        return self.os_runtime

    async def _run(
        self, args: list[str], *, cwd: str | None = None
    ) -> tuple[int, str, str]:
        runtime = self._runtime()
        return await runtime.exec(args, cwd=cwd or self._workdir)

    async def _run_ok(
        self,
        args: list[str],
        *,
        cwd: str | None = None,
        error_msg: str = "git command failed",
    ) -> str:
        exit_code, stdout, stderr = await self._run(args, cwd=cwd)
        if exit_code != 0:
            raise ToolExecutionError(f"{error_msg} (exit {exit_code}): {stderr.strip()}")
        return stdout

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    async def clone(
        self,
        *,
        repo_url: str,
        target_dir: str | None = None,
        branch: str | None = None,
        depth: int | None = None,
    ) -> str:
        cmd = ["git", "clone"]
        if branch:
            cmd += ["-b", branch]
        effective_depth = depth or self._clone_depth
        if effective_depth:
            cmd += ["--depth", str(effective_depth)]
        cmd.append(repo_url)
        if target_dir:
            cmd.append(target_dir)

        exit_code, stdout, stderr = await self._run(cmd, cwd=self._workdir)

        if target_dir:
            clone_path = (
                target_dir if target_dir.startswith("/") else f"{self._workdir}/{target_dir}"
            )
        else:
            repo_name = repo_url.rstrip("/").split("/")[-1].removesuffix(".git")
            clone_path = f"{self._workdir}/{repo_name}"

        # Update workdir so subsequent operations target the cloned repo
        if exit_code == 0:
            self._workdir = clone_path

        return json.dumps(
            {
                "exit_code": exit_code,
                "success": exit_code == 0,
                "clone_path": clone_path,
                "stdout": stdout,
                "stderr": stderr,
            }
        )

    async def checkout(self, *, ref: str, create: bool = False) -> str:
        cmd = ["git", "checkout"]
        if create:
            cmd.append("-b")
        cmd.append(ref)
        stdout = await self._run_ok(cmd, error_msg=f"checkout '{ref}' failed")
        return json.dumps(
            {
                "success": True,
                "ref": ref,
                "created": create,
                "output": stdout.strip() or f"Switched to {'new ' if create else ''}branch '{ref}'",
            }
        )

    async def status(self) -> str:
        status_out = await self._run_ok(
            ["git", "status", "--porcelain=v1"], error_msg="git status failed"
        )
        _, branch_raw, _ = await self._run(["git", "branch", "--show-current"])
        entries = _parse_status_porcelain(status_out)
        return json.dumps(
            {
                "branch": branch_raw.strip(),
                "clean": len(entries) == 0,
                "files": entries,
                "summary": {
                    "modified": sum(1 for e in entries if "M" in (e["index"], e["worktree"])),
                    "added": sum(1 for e in entries if e["index"] == "A"),
                    "deleted": sum(1 for e in entries if "D" in (e["index"], e["worktree"])),
                    "untracked": sum(1 for e in entries if e["index"] == "?" and e["worktree"] == "?"),
                },
            }
        )

    async def diff(
        self,
        *,
        staged: bool = False,
        ref: str | None = None,
        paths: list[str] | None = None,
        stat_only: bool = False,
    ) -> str:
        cmd = ["git", "diff"]
        if staged:
            cmd.append("--cached")
        if ref:
            cmd.append(ref)
        if stat_only:
            cmd.append("--stat")
        if paths:
            cmd.append("--")
            cmd.extend(paths)
        output = await self._run_ok(cmd, error_msg="git diff failed")
        return output if output else "(no changes)"

    async def commit(
        self,
        *,
        message: str,
        add_all: bool = True,
        paths: list[str] | None = None,
    ) -> str:
        if paths:
            await self._run_ok(["git", "add", "--"] + paths, error_msg="git add failed")
        elif add_all:
            await self._run_ok(["git", "add", "-A"], error_msg="git add -A failed")

        commit_cmd = ["git", "commit", "-m", message]
        if self._commit_author:
            commit_cmd += ["--author", self._commit_author]
        await self._run_ok(commit_cmd, error_msg="git commit failed")

        hash_out = await self._run_ok(
            ["git", "rev-parse", "--short", "HEAD"], error_msg="rev-parse failed"
        )
        return json.dumps({"hash": hash_out.strip(), "message": message, "success": True})

    async def push(
        self,
        *,
        remote: str | None = None,
        branch: str | None = None,
        force: bool = False,
        set_upstream: bool = False,
    ) -> str:
        cmd = ["git", "push"]
        if force:
            cmd.append("--force-with-lease")
        if set_upstream:
            cmd.append("--set-upstream")
        cmd.append(remote or self._default_remote)
        if branch:
            cmd.append(branch)
        exit_code, stdout, stderr = await self._run(cmd)
        return json.dumps(
            {"exit_code": exit_code, "success": exit_code == 0, "output": (stdout or "") + (stderr or "")}
        )

    async def pull(
        self,
        *,
        remote: str | None = None,
        branch: str | None = None,
        rebase: bool = False,
    ) -> str:
        cmd = ["git", "pull"]
        if rebase:
            cmd.append("--rebase")
        cmd.append(remote or self._default_remote)
        if branch:
            cmd.append(branch)
        exit_code, stdout, stderr = await self._run(cmd)
        return json.dumps(
            {"exit_code": exit_code, "success": exit_code == 0, "output": (stdout or "") + (stderr or "")}
        )

    async def branch_list(self, *, include_remote: bool = True) -> str:
        cmd = ["git", "branch"]
        if include_remote:
            cmd.append("-a")
        stdout = await self._run_ok(cmd, error_msg="git branch failed")
        return json.dumps(_parse_branches(stdout))

    async def log(
        self,
        *,
        count: int = 20,
        ref: str | None = None,
        paths: list[str] | None = None,
    ) -> str:
        cmd = ["git", "log", "--format=%H|%an|%ai|%s", f"-n{count}"]
        if ref:
            cmd.append(ref)
        if paths:
            cmd.append("--")
            cmd.extend(paths)
        stdout = await self._run_ok(cmd, error_msg="git log failed")
        return json.dumps(_parse_log_formatted(stdout))

    async def apply_patch(self, *, patch_content: str, check_only: bool = False) -> str:
        runtime = self._runtime()
        fd, patch_path = tempfile.mkstemp(suffix=".diff", prefix="_nlght_git_patch_")
        os.close(fd)
        await runtime.write_text(patch_path, patch_content)

        cmd = ["git", "apply", "--whitespace=nowarn"]
        if check_only:
            cmd.append("--check")
        cmd.append(patch_path)

        exit_code, stdout, stderr = await self._run(cmd)
        return json.dumps(
            {
                "exit_code": exit_code,
                "success": exit_code == 0,
                "check_only": check_only,
                "output": (stdout or "") + (stderr or ""),
            }
        )

    async def read_file(
        self,
        *,
        path: str,
        ref: str = "HEAD",
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> str:
        rel = path.lstrip("/\\").replace("\\", "/")
        ec, out, err = await self._run(["git", "show", f"{ref}:{rel}"])
        if ec != 0:
            ec, out, err = await self._run(["git", "cat-file", "blob", f"{ref}:{rel}"])
        if ec != 0:
            raise ToolExecutionError(f"git_read_file '{rel}' at '{ref}' failed: {err.strip()}")
        if not out.strip():
            return json.dumps({"path": rel, "ref": ref, "content": "", "note": "empty file"})
        file_lines = out.splitlines()
        total = len(file_lines)
        if start_line is not None or end_line is not None:
            s = max(0, int(start_line) - 1) if start_line is not None else 0
            e = min(total, int(end_line))    if end_line   is not None else total
            return json.dumps({
                "path": rel, "ref": ref,
                "lines": f"{s + 1}-{e}", "total_lines": total,
                "content": "\n".join(file_lines[s:e]),
            })
        return json.dumps({"path": rel, "ref": ref, "total_lines": total, "content": out})

    async def list_dir_at_ref(
        self,
        *,
        path: str = ".",
        ref: str = "HEAD",
    ) -> str:
        rel = path.lstrip("/\\").replace("\\", "/") or "."
        if rel == ".":
            ec, out, _ = await self._run(["git", "ls-tree", "--name-only", ref])
        else:
            ec, out, _ = await self._run(["git", "ls-tree", "--name-only", f"{ref}:{rel}"])
        if ec != 0:
            ec, out, _ = await self._run(["git", "ls-files", rel])
        return json.dumps({
            "path": rel, "ref": ref,
            "entries": [e for e in out.strip().splitlines() if e],
        })

    async def apply_file_content(
        self,
        *,
        path: str,
        content: str,
        check_only: bool = False,
    ) -> str:
        runtime = self._runtime()
        rel_path = _normalise_repo_relative_path(path)
        file_path = _join_runtime_path(self._workdir, rel_path)
        old_content = await runtime.read_text(file_path)
        patch_content = _unified_diff_for_content(
            rel_path=rel_path,
            old_content=old_content,
            new_content=content,
        )

        if not patch_content:
            return json.dumps({
                "success": True,
                "check_only": check_only,
                "changed": False,
                "path": rel_path,
                "patch": "",
                "output": "no changes",
            })

        if check_only:
            return json.dumps({
                "success": True,
                "check_only": True,
                "changed": False,
                "path": rel_path,
                "patch": patch_content,
                "output": "check_only — no subprocess",
            })

        await runtime.write_text(file_path, content)
        return json.dumps({
            "success": True,
            "check_only": False,
            "changed": True,
            "path": rel_path,
            "patch": patch_content,
            "output": "",
        })
