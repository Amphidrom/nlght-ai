# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from nlght.core.tools.action import ExecutionCapabilities

if TYPE_CHECKING:
    pass


@runtime_checkable
class OsRuntimeFactory(Protocol):
    """Factory that creates and releases an isolated OsRuntime instance per invocation."""

    def bind(self, cid: str) -> OsRuntime:
        """Returns a lazy runtime — the container/resource is only created on first access."""
        ...

    async def create(self, name: str) -> OsRuntime:
        """Creates and starts a new runtime instance (e.g. a Docker container) with the given name."""
        ...

    async def release(self, runtime: OsRuntime) -> None:
        """Stops and cleans up the runtime instance after the invocation completes."""
        ...


@runtime_checkable
class OsRuntime(Protocol):
    """Abstract interface for an isolated OS execution environment.

    Implementations provide command execution and filesystem access —
    regardless of whether the command runs locally, in Docker, or in
    Kubernetes.

    Tools that use this port receive an instance via ``ToolBase.__init__``
    (injected by the ``ToolCatalogBuilder``). They never call the
    implementation directly — only this interface.
    """

    async def exec(
        self,
        command: list[str],
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> tuple[int, str, str]:
        """Executes a command. Returns ``(exit_code, stdout, stderr)``."""
        ...

    def add_env(self, env: dict[str, str]) -> None:
        """Adds env variables that are carried along with every subsequent ``exec``."""
        ...

    async def read_text(self, path: str) -> str:
        """Reads a text file."""
        ...

    async def write_text(self, path: str, content: str) -> None:
        """Writes a text file (creates missing directories)."""
        ...

    async def file_exists(self, path: str) -> bool:
        """Returns ``True`` if the path exists."""
        ...

    async def list_dir(self, path: str) -> list[str]:
        """Lists the contents of a directory (direct children only)."""
        ...

    async def make_dir(self, path: str) -> None:
        """Creates a directory including all missing parents."""
        ...

    async def delete(self, path: str) -> None:
        """Deletes a file or an empty directory."""
        ...

    def shell(self) -> tuple[str, str]:
        """Returns ``(executable, flag)`` for shell command execution.

        Examples:
          ``("powershell", "-Command")`` on Windows
          ``("sh", "-c")`` on Linux / macOS / Docker
        """
        ...

    def security_capabilities(self) -> ExecutionCapabilities:
        """Return guarantees enforced by this runtime, never operator intent."""
        ...
