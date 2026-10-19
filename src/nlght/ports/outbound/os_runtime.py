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
        """Bind an invocation to a lazily created runtime.

        Args:
            cid (str): Invocation correlation ID.

        Returns:
            OsRuntime: Bound OS runtime.

        Side Effects:
            May cache the bound runtime in the factory.
        """
        ...

    async def create(self, name: str) -> OsRuntime:
        """Create an isolated runtime for one invocation.

        Args:
            name (str): Runtime instance name.

        Returns:
            OsRuntime: OS runtime ready for use.

        Side Effects:
            May start and cache an isolated runtime resource.
        """
        ...

    async def release(self, runtime: OsRuntime) -> None:
        """Release a runtime after its invocation.

        Args:
            runtime (OsRuntime): Runtime instance to release.

        Side Effects:
            May stop and clean up the runtime resource.
        """
        ...


@runtime_checkable
class OsRuntime(Protocol):
    """Abstract interface for an isolated OS execution environment.

    Implementations provide command execution and filesystem access,
    regardless of whether the command runs locally, in Docker, or in
    Kubernetes.

    ``ToolCatalogBuilder`` injects this port through ``ToolBase.__init__``.
    Tools call the interface provided here.
    """

    async def exec(
        self,
        command: list[str],
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> tuple[int, str, str]:
        """Execute a command in the isolated environment.

        Args:
            command (list[str]): Command arguments without a shell string.
            cwd (str | None): Optional runtime working directory.
            env (dict[str, str] | None): Optional process environment.

        Returns:
            tuple[int, str, str]: Exit code, standard output, and standard error.

        Side Effects:
            Starts a process and applies its command effects.
        """
        ...

    def add_env(self, env: dict[str, str]) -> None:
        """Set environment values for subsequent commands.

        Args:
            env (dict[str, str]): Environment variable mapping.

        Side Effects:
            Changes the runtime's command environment.
        """
        ...

    async def read_text(self, path: str) -> str:
        """Read text from the runtime filesystem.

        Args:
            path (str): File path.

        Returns:
            str: File contents as text.

        Side Effects:
            Reads the runtime filesystem.
        """
        ...

    async def write_text(self, path: str, content: str) -> None:
        """Write text in the runtime filesystem.

        Args:
            path (str): Destination path in the runtime.
            content (str): Text to write.

        Side Effects:
            Creates missing directories and writes the file.
        """
        ...

    async def file_exists(self, path: str) -> bool:
        """Check for a path in the runtime filesystem.

        Args:
            path (str): File path.

        Returns:
            bool: Whether the path exists.

        Side Effects:
            Reads filesystem metadata.
        """
        ...

    async def list_dir(self, path: str) -> list[str]:
        """List direct children of a runtime directory.

        Args:
            path (str): Directory path.

        Returns:
            list[str]: Names of direct children.

        Side Effects:
            Reads the runtime filesystem.
        """
        ...

    async def make_dir(self, path: str) -> None:
        """Create a runtime directory.

        Args:
            path (str): Directory path.

        Side Effects:
            Creates the directory and missing parents.
        """
        ...

    async def delete(self, path: str) -> None:
        """Remove a file or empty runtime directory.

        Args:
            path (str): File or directory path.

        Side Effects:
            Deletes the specified filesystem entry.
        """
        ...

    def shell(self) -> tuple[str, str]:
        """Select the shell invocation for this runtime.

        Returns:
            tuple[str, str]: Executable and command flag, such as ``("sh", "-c")``.
        """
        ...

    def security_capabilities(self) -> ExecutionCapabilities:
        """Report capabilities enforced by this runtime.

        Returns:
            ExecutionCapabilities: Execution capability contract.
        """
        ...
