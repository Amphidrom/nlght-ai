# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
from pathlib import Path

from nlght.core.tools.action import ExecutionCapabilities
from nlght.ports.outbound.os_runtime import OsRuntime, OsRuntimeFactory

logger = logging.getLogger(__name__)

_WINDOWS = sys.platform == "win32"


class LocalOsRuntime(OsRuntimeFactory, OsRuntime):
    """Implements OsRuntimeFactory and OsRuntime for local host execution.

    Uses ``asyncio.create_subprocess_exec`` for process execution and
    ``asyncio.to_thread`` for blocking filesystem operations.

    As a factory, ``bind()`` and ``create()`` always return ``self`` — the
    local runtime is not isolated per invocation. Intended for local
    development, tests, and deployments without containers.

    ``workdir`` is the default working directory for ``exec()`` calls
    without an explicit ``cwd``.
    """

    def __init__(self, *, workdir: str = ".") -> None:
        self._workdir = workdir
        self._env: dict[str, str] = {}

    def add_env(self, env: dict[str, str]) -> None:
        self._env.update(env or {})

    def _resolve(self, path: str) -> Path:
        p = Path(path)
        return p if p.is_absolute() else Path(self._workdir) / p

    async def exec(
        self,
        command: list[str],
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> tuple[int, str, str]:
        effective_cwd = str(self._resolve(cwd)) if cwd else self._workdir
        logger.debug("local_runtime.exec | cmd=%s cwd=%s", command, effective_cwd)

        # subprocess env= REPLACES the whole environment, so merge onto the parent
        # env: os.environ < runtime-added env (add_env) < per-call env.
        effective_env: dict[str, str] | None = None
        if self._env or env:
            effective_env = {**os.environ, **self._env, **(env or {})}

        if _WINDOWS:
            # On Windows, built-ins like `date`, `echo`, `dir` are cmd.exe
            # commands, not executables — run via shell so they are found.
            proc = await asyncio.create_subprocess_shell(
                subprocess.list2cmdline(command),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=effective_cwd,
                env=effective_env,
            )
        else:
            proc = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=effective_cwd,
                env=effective_env,
            )
        stdout_bytes, stderr_bytes = await proc.communicate()
        exit_code = proc.returncode or 0
        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")

        logger.debug(
            "local_runtime.exec.done | exit_code=%d stdout_len=%d",
            exit_code,
            len(stdout),
        )
        return exit_code, stdout, stderr

    async def read_text(self, path: str) -> str:
        return await asyncio.to_thread(self._resolve(path).read_text, encoding="utf-8")

    async def write_text(self, path: str, content: str) -> None:
        def _write() -> None:
            p = self._resolve(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")

        await asyncio.to_thread(_write)

    async def file_exists(self, path: str) -> bool:
        return await asyncio.to_thread(self._resolve(path).exists)

    async def list_dir(self, path: str) -> list[str]:
        def _list() -> list[str]:
            return [entry.name for entry in self._resolve(path).iterdir()]

        return await asyncio.to_thread(_list)

    async def make_dir(self, path: str) -> None:
        await asyncio.to_thread(self._resolve(path).mkdir, parents=True, exist_ok=True)

    async def delete(self, path: str) -> None:
        await asyncio.to_thread(self._resolve(path).unlink)

    def shell(self) -> tuple[str, str]:
        return ("powershell", "-Command") if _WINDOWS else ("sh", "-c")

    def security_capabilities(self) -> ExecutionCapabilities:
        """Local execution is host execution; no weaker claim is supportable."""
        return ExecutionCapabilities.unconfined()

    # ── OsRuntimeFactory — shared instance, bind/create return self ──────────

    def bind(self, cid: str) -> LocalOsRuntime:
        return self

    async def create(self, name: str) -> LocalOsRuntime:
        return self

    async def release(self, runtime: OsRuntime) -> None:
        pass
