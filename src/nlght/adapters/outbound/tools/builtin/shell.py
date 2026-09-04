# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import json
import shlex
import uuid
from typing import TYPE_CHECKING, ClassVar

from nlght.adapters.outbound.tools.builtin.action_semantics import (
    DELETE_RUNTIME,
    EXECUTE_RUNTIME,
    OVERWRITING_WRITE_RUNTIME,
    READ_RUNTIME,
    WRITE_RUNTIME,
)
from nlght.core.errors.errors import ToolExecutionError
from nlght.core.tools.tool import ToolBase, ToolParameter, ToolSignature

if TYPE_CHECKING:
    from nlght.ports.outbound.os_runtime import OsRuntime

_DEFAULT_MAX_STDOUT_BYTES = 8192
_STDERR_MAX = 4096


def _is_binary_stdout(text: str, sample_size: int = 1024) -> bool:
    """Return True when stdout appears to be binary data decoded with errors='replace'.

    The Docker runtime decodes bytes with errors='replace', so binary data shows
    up as a high density of U+FFFD replacement characters and control characters.
    """
    if not text:
        return False
    sample = text[:sample_size]
    non_printable = sum(
        1 for c in sample
        if c == "�" or ord(c) < 0x09 or (0x0E <= ord(c) <= 0x1F)
    )
    return non_printable / len(sample) > 0.10


class ShellTool(ToolBase):
    """Built-in tool that exposes OsRuntime operations to LLM workflows.

    Each method returns a JSON-serialisable string so the result can be
    forwarded to an LLM as a tool-call response without further processing.

    Configuration (optional):
      ``default_cwd`` — working directory used when the caller omits ``cwd``.
    """

    KIND: ClassVar[str]     = "shell"
    PROVIDER: ClassVar[str] = "local"

    @classmethod
    def signatures(cls) -> list[ToolSignature]:
        return [
            ToolSignature(
                name="shell_exec",
                description=(
                    "Execute a shell command and return exit_code, stdout, and stderr. "
                    "Large text outputs (>max_stdout_bytes) are saved to a file inside the "
                    "container and returned as stdout_saved_to + stdout_preview. "
                    "Binary stdout (images, archives, …) is summarised as stdout_binary=true. "
                    "Use shell_read_chunk to read large saved outputs in parts."
                ),
                method_name="exec",
                parameters=[
                    ToolParameter(name="command", type="string", description="Shell command string to execute."),
                    ToolParameter(name="cwd",     type="string", description="Working directory.", required=False),
                ],
                action=EXECUTE_RUNTIME,
            ),
            ToolSignature(
                name="run_script",
                description=(
                    "Write a script to a scratch file inside the container, execute it, "
                    "and return exit_code, stdout, and stderr in one call — then the file "
                    "is deleted. Replaces the shell_write_text + shell_exec two-step for "
                    "PoC scripts. The script inherits the container environment (including "
                    "the injected NLGHT_AUTH_* session variables), so it runs authenticated. "
                    "Large/binary stdout is handled exactly like shell_exec."
                ),
                method_name="run_script",
                parameters=[
                    ToolParameter(name="content", type="string", description="Full script source to write and run."),
                    ToolParameter(name="ext", type="string", description="File extension / language: 'py' (python3, default) or 'sh' (bash).", required=False),
                    ToolParameter(name="cwd", type="string", description="Working directory.", required=False),
                ],
                action=EXECUTE_RUNTIME,
            ),
            ToolSignature(
                name="shell_read_chunk",
                description=(
                    "Read a chunk of a text file at a character offset. "
                    "Use after shell_exec returns stdout_saved_to to read large output in parts. "
                    "Returns content, offset, length_read, total_chars, has_more."
                ),
                method_name="read_chunk",
                parameters=[
                    ToolParameter(name="path",   type="string", description="Absolute path to the file inside the container."),
                    ToolParameter(name="offset", type="number", description="Character offset to start from (default: 0).",  required=False),
                    ToolParameter(name="length", type="number", description="Characters to read (default: 4096).", required=False),
                ],
                action=READ_RUNTIME,
            ),
            ToolSignature(
                name="shell_read_text",
                description="Read the UTF-8 content of a file.",
                method_name="read_text",
                parameters=[
                    ToolParameter(name="path", type="string", description="Absolute path to the file."),
                ],
                action=READ_RUNTIME,
            ),
            ToolSignature(
                name="shell_write_text",
                description=(
                    "Write UTF-8 text to a file, creating parent directories as needed. "
                    "If path is omitted, a unique scratch path under /workspace/tmp/nlght/ "
                    "is generated and returned."
                ),
                method_name="write_text",
                parameters=[
                    ToolParameter(name="content", type="string", description="Text content to write."),
                    ToolParameter(name="path", type="string", description="Absolute path to the file. Omit to auto-generate a scratch path.", required=False),
                ],
                action=OVERWRITING_WRITE_RUNTIME,
            ),
            ToolSignature(
                name="shell_file_exists",
                description="Check whether a path exists in the filesystem.",
                method_name="file_exists",
                parameters=[
                    ToolParameter(name="path", type="string", description="Path to check."),
                ],
                action=READ_RUNTIME,
            ),
            ToolSignature(
                name="shell_list_dir",
                description="List direct children of a directory.",
                method_name="list_dir",
                parameters=[
                    ToolParameter(name="path", type="string", description="Directory path."),
                ],
                action=READ_RUNTIME,
            ),
            ToolSignature(
                name="shell_make_dir",
                description="Create a directory and all missing parents.",
                method_name="make_dir",
                parameters=[
                    ToolParameter(name="path", type="string", description="Directory path to create."),
                ],
                action=WRITE_RUNTIME,
            ),
            ToolSignature(
                name="shell_delete",
                description="Delete a file or directory recursively.",
                method_name="delete",
                parameters=[
                    ToolParameter(name="path", type="string", description="Path to delete."),
                ],
                action=DELETE_RUNTIME,
            ),
        ]

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _runtime(self) -> OsRuntime:
        if self.os_runtime is None:
            raise ToolExecutionError(
                f"ShellTool '{self.name}' requires an OsRuntime but none was injected."
            )
        return self.os_runtime

    def _resolve_cwd(self, cwd: str | None) -> str | None:
        if cwd is not None:
            return cwd
        default = self.config.get("default_cwd")
        return str(default) if default is not None else None

    # ------------------------------------------------------------------
    # operations
    # ------------------------------------------------------------------

    async def exec(self, *, command: str, cwd: str | None = None) -> str:
        runtime      = self._runtime()
        resolved_cwd = self._resolve_cwd(cwd)

        try:
            # Run through bash (not sh/dash) so the container's BASH_ENV command
            # logger (DEBUG trap) fires — this is the "captured shell" that makes
            # every executed command show up in the container log.
            exit_code, stdout, stderr = await runtime.exec(
                ["bash", "-c", command], cwd=resolved_cwd,
            )
        except Exception as exc:
            raise ToolExecutionError(f"shell_exec failed: {exc}") from exc

        return await self._format_exec_result(runtime, exit_code, stdout, stderr)

    async def _format_exec_result(
        self, runtime: OsRuntime, exit_code: int, stdout: str, stderr: str,
    ) -> str:
        """Shape an (exit_code, stdout, stderr) tuple into the JSON tool result.

        Shared by shell_exec and run_script: binary stdout is summarised, oversized
        text stdout is saved to a scratch file for shell_read_chunk, and stderr is
        capped. Keeps both tools' output contract identical.
        """
        max_bytes  = int(self.config.get("max_stdout_bytes", _DEFAULT_MAX_STDOUT_BYTES))
        stderr_out = stderr[:_STDERR_MAX] if len(stderr) > _STDERR_MAX else stderr

        if _is_binary_stdout(stdout):
            return json.dumps({
                "exit_code":     exit_code,
                "stdout_binary": True,
                "stdout_bytes":  len(stdout.encode("utf-8", errors="replace")),
                "stderr":        stderr_out,
            })

        if len(stdout) <= max_bytes:
            return json.dumps({"exit_code": exit_code, "stdout": stdout, "stderr": stderr_out})

        # stdout exceeds limit — save to file inside container for chunk reading.
        # Unique name under the shared workspace scratch dir → never collides.
        save_path = f"/workspace/tmp/nlght/stdout_{uuid.uuid4().hex}.txt"
        try:
            await runtime.write_text(save_path, stdout)
            return json.dumps({
                "exit_code":          exit_code,
                "stdout_saved_to":    save_path,
                "stdout_preview":     stdout[:512],
                "stdout_total_chars": len(stdout),
                "hint": (
                    f"Output exceeds {max_bytes} chars and was saved to {save_path}. "
                    "Use shell_read_chunk(path, offset, length) to read it in parts."
                ),
                "stderr": stderr_out,
            })
        except Exception:
            return json.dumps({
                "exit_code":          exit_code,
                "stdout":             stdout[:max_bytes],
                "stdout_truncated":   True,
                "stdout_total_chars": len(stdout),
                "stderr":             stderr_out,
            })

    async def run_script(self, *, content: str, ext: str = "py", cwd: str | None = None) -> str:
        runtime      = self._runtime()
        resolved_cwd = self._resolve_cwd(cwd)

        ext = (ext or "py").lstrip(".").lower()
        interpreter = "bash" if ext in ("sh", "bash") else "python3"
        # Scratch lives under /workspace/tmp/nlght — same convention as http_probe.
        script_path = f"/workspace/tmp/nlght/script_{uuid.uuid4().hex}.{ext}"

        try:
            await runtime.write_text(script_path, content)
        except Exception as exc:
            raise ToolExecutionError(f"run_script write failed: {exc}") from exc

        try:
            # Run through bash so the container BASH_ENV command logger records it,
            # and so the script inherits the auth env set via runtime.add_env().
            exit_code, stdout, stderr = await runtime.exec(
                ["bash", "-c", f"{interpreter} {shlex.quote(script_path)}"],
                cwd=resolved_cwd,
            )
        except Exception as exc:
            raise ToolExecutionError(f"run_script exec failed: {exc}") from exc
        finally:
            # Best-effort cleanup — scratch scripts would otherwise pile up.
            try:
                await runtime.delete(script_path)
            except Exception:
                pass

        return await self._format_exec_result(runtime, exit_code, stdout, stderr)

    async def read_chunk(self, *, path: str, offset: int = 0, length: int = 4096) -> str:
        runtime = self._runtime()
        try:
            content = await runtime.read_text(path)
        except FileNotFoundError:
            return json.dumps({"error": f"file not found: {path}"})
        except Exception as exc:
            raise ToolExecutionError(f"shell_read_chunk failed: {exc}") from exc

        chunk = content[offset: offset + length]
        total = len(content)
        return json.dumps({
            "content":     chunk,
            "offset":      offset,
            "length_read": len(chunk),
            "total_chars": total,
            "has_more":    offset + len(chunk) < total,
        })

    async def read_text(self, *, path: str) -> str:
        runtime = self._runtime()
        try:
            content = await runtime.read_text(path)
        except FileNotFoundError:
            raise
        except Exception as exc:
            raise ToolExecutionError(f"shell_read_text failed: {exc}") from exc
        return json.dumps({"content": content})

    async def write_text(self, *, content: str, path: str | None = None) -> str:
        runtime = self._runtime()
        if not path:
            # No path given → generate a scratch path under the shared workspace dir
            # and return it so the caller can run/read it afterwards.
            path = f"/workspace/tmp/nlght/script_{uuid.uuid4().hex}.txt"
        try:
            await runtime.write_text(path, content)
        except Exception as exc:
            raise ToolExecutionError(f"shell_write_text failed: {exc}") from exc
        return json.dumps({"ok": True, "path": path})

    async def file_exists(self, *, path: str) -> str:
        runtime = self._runtime()
        try:
            exists = await runtime.file_exists(path)
        except Exception as exc:
            raise ToolExecutionError(f"shell_file_exists failed: {exc}") from exc
        return json.dumps({"exists": exists, "path": path})

    async def list_dir(self, *, path: str) -> str:
        runtime = self._runtime()
        try:
            entries = await runtime.list_dir(path)
        except FileNotFoundError:
            raise
        except Exception as exc:
            raise ToolExecutionError(f"shell_list_dir failed: {exc}") from exc
        return json.dumps({"entries": entries, "path": path})

    async def make_dir(self, *, path: str) -> str:
        runtime = self._runtime()
        try:
            await runtime.make_dir(path)
        except Exception as exc:
            raise ToolExecutionError(f"shell_make_dir failed: {exc}") from exc
        return json.dumps({"ok": True, "path": path})

    async def delete(self, *, path: str) -> str:
        runtime = self._runtime()
        try:
            await runtime.delete(path)
        except FileNotFoundError:
            raise
        except Exception as exc:
            raise ToolExecutionError(f"shell_delete failed: {exc}") from exc
        return json.dumps({"ok": True, "path": path})
