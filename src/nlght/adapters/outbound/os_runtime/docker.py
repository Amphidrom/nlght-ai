# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import asyncio
import io
import logging
import os
import sys as _sys
import tarfile
import threading
import time
from typing import TYPE_CHECKING

from nlght.ports.outbound.os_runtime import OsRuntime, OsRuntimeFactory

if TYPE_CHECKING:
    # The repo's own top-level docker/ directory (Dockerfile assets) shadows
    # the docker SDK package name when it isn't installed (no [docker] extra
    # in the dev environment). mypy never analyzes either candidate — the
    # docker.* override in pyproject sets follow_imports = "skip", so these
    # names are Any regardless of which package resolution wins.
    from docker import DockerClient
    from docker.models.containers import Container

try:
    import docker as _docker_sdk
    import docker.errors as _docker_errors
    _DOCKER_AVAILABLE = True
except ImportError:
    _docker_sdk = None
    _docker_errors = None
    _DOCKER_AVAILABLE = False

logger = logging.getLogger(__name__)

_DOCKER_MISSING = (
    "docker package not installed — install with: pip install 'nlght-ai[docker]'"
)


def _require_docker() -> None:
    if not _DOCKER_AVAILABLE:
        raise ImportError(_DOCKER_MISSING)


class _NeverRaised(Exception):
    """Placeholder except-clause type for the docker.errors.NotFound fallback.

    ``except type(None)`` is invalid Python (TypeError: catching classes that
    do not inherit from BaseException) — this is a valid, never-actually-raised
    stand-in for when the docker package (and _docker_errors) isn't installed.
    """


class DockerOsRuntimeFactory(OsRuntimeFactory):
    """Creates a dedicated Docker container per invocation (named after the CID).

    Implements OsRuntimeFactory and SubsystemLifecycle — ``start()`` initializes
    the Docker client, ``stop()`` is a no-op (containers are cleaned up
    individually via ``release()``).
    """

    def __init__(
        self,
        *,
        base_image: str,
        workdir: str = "/workspace",
        workspace_path: str | None = None,
        extra_hosts: list[str] | None = None,
    ) -> None:
        self._base_image = base_image
        self._workdir = workdir
        self._workspace_path = os.path.abspath(workspace_path) if workspace_path else None
        self._extra_hosts: dict[str, str] = {}
        for entry in (extra_hosts or []):
            if ":" in entry:
                host, ip = entry.split(":", 1)
                self._extra_hosts[host] = ip
        self._client: DockerClient | None = None
        self._instances: dict[str, DockerOsRuntime] = {}

    async def start(self) -> None:
        await asyncio.to_thread(self._sync_init_client)

    def _sync_init_client(self) -> None:
        _require_docker()
        self._client = _docker_sdk.from_env()
        logger.info("docker_factory.client_ready")

    async def stop(self) -> None:
        pass

    def bind(self, cid: str) -> DockerOsRuntime:
        """Returns a lazy runtime for this CID — the container is created only on first access."""
        if cid in self._instances:
            return self._instances[cid]
        rt = DockerOsRuntime(
            name=cid,
            base_image=self._base_image,
            workdir=self._workdir,
            workspace_path=self._workspace_path,
            extra_hosts=self._extra_hosts,
            client=self._client,
        )
        self._instances[cid] = rt
        return rt

    async def create(self, name: str) -> DockerOsRuntime:
        if name in self._instances:
            return self._instances[name]
        rt = DockerOsRuntime(
            name=name,
            base_image=self._base_image,
            workdir=self._workdir,
            workspace_path=self._workspace_path,
            extra_hosts=self._extra_hosts,
            client=self._client,
        )
        await rt.start()
        self._instances[name] = rt
        return rt

    async def release(self, runtime: OsRuntime) -> None:
        assert isinstance(runtime, DockerOsRuntime)
        self._instances.pop(runtime._name, None)
        await runtime.stop()

    async def release_by_name(self, name: str) -> None:
        rt = self._instances.pop(name, None)
        if rt is not None:
            await rt.stop()


class DockerOsRuntime(OsRuntime):
    """OsRuntime that executes commands in a dedicated Docker container.

    Implements OsRuntime. Created per invocation via
    ``DockerOsRuntimeFactory.bind(cid)`` (lazy) or via ``create(name)``
    (eager, with immediate start). ``start()`` creates the container,
    ``stop()`` stops and removes it.
    """

    def __init__(
        self,
        *,
        name: str,
        base_image: str,
        workdir: str = "/workspace",
        workspace_path: str | None = None,
        extra_hosts: dict[str, str] | None = None,
        client: DockerClient | None = None,
    ) -> None:
        self._name = name
        self._base_image = base_image
        self._workdir = workdir
        self._workspace_path = workspace_path
        self._extra_hosts: dict[str, str] = extra_hosts or {}
        self._client: DockerClient | None = client
        self._container: Container | None = None
        self._env: dict[str, str] = {}
        # docker-py shares one HTTP connection per client; concurrent exec_run /
        # archive calls interleave on the socket and corrupt the demux stream
        # ("N is not a valid stream"). Serialise all container I/O on this lock.
        self._io_lock = threading.Lock()

    def _resolve(self, path: str) -> str:
        return path if path.startswith("/") else f"{self._workdir}/{path}"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        await asyncio.to_thread(self._sync_start)

    def _sync_start(self) -> None:
        if self._client is None:
            _require_docker()
            # Use sys.modules so tests can monkeypatch the docker module
            _docker_mod = _sys.modules["docker"]
            self._client = _docker_mod.from_env()

        # Try to reuse an existing container with this name
        _errors_mod = _sys.modules.get("docker.errors", _docker_errors)
        try:
            existing = self._client.containers.get(self._name)
            if existing.status == "exited":
                existing.start()
            self._container = existing
            logger.info("docker_runtime.reused | name=%s", self._name)
            return
        except Exception as _exc:
            if _errors_mod is not None and not isinstance(_exc, _errors_mod.NotFound):
                raise

        volumes: dict[str, dict[str, str]] | None = None
        if self._workspace_path:
            host_dir = os.path.join(self._workspace_path, self._name)
            os.makedirs(host_dir, exist_ok=True)
            volumes = {host_dir: {"bind": self._workdir, "mode": "rw"}}
            logger.info(
                "docker_runtime.creating | name=%s image=%s workspace=%s",
                self._name, self._base_image, host_dir,
            )
        else:
            logger.info("docker_runtime.creating | name=%s image=%s", self._name, self._base_image)

        self._container = self._client.containers.run(
            self._base_image,
            name=self._name,
            command="sleep infinity",
            detach=True,
            tty=True,
            volumes=volumes or {},
            extra_hosts=self._extra_hosts or None,
        )
        logger.info("docker_runtime.created | name=%s image=%s", self._name, self._base_image)

    async def stop(self) -> None:
        if self._container is not None:
            await asyncio.to_thread(self._sync_stop)

    def _sync_stop(self) -> None:
        assert self._container is not None
        try:
            self._container.stop()
            self._container.remove()
            logger.info("docker_runtime.removed | name=%s", self._name)
        except Exception as exc:
            logger.warning("docker_runtime.stop_failed | name=%s error=%s", self._name, exc)

    # ------------------------------------------------------------------
    # OsRuntime — exec
    # ------------------------------------------------------------------

    def add_env(self, env: dict[str, str]) -> None:
        self._env.update(env or {})

    async def exec(
        self,
        command: list[str],
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> tuple[int, str, str]:
        return await asyncio.to_thread(self._sync_exec, command, cwd, env)

    def _sync_exec(
        self,
        command: list[str],
        cwd: str | None,
        env: dict[str, str] | None,
    ) -> tuple[int, str, str]:
        self._ensure_started_sync()
        assert self._container is not None

        base_env = {"BASH_ENV": "/etc/nlght-bash-logger.sh"}
        base_env.update(self._env)
        base_env.update(env or {})

        # No _io_lock here: exec_run with demux=True uses its own HTTP exchange
        # per exec-id and is safe to call concurrently. Holding the lock during
        # long-running commands (nuclei, sqlmap) would cause every other exec()
        # call to block in a thread-pool thread, eventually exhausting the pool
        # once asyncio wait_for cancels the coroutine but not the underlying thread.
        # _io_lock is only needed for get_archive / put_archive (streaming tar).
        result = self._container.exec_run(
            cmd=command,
            stdout=True,
            stderr=True,
            demux=True,
            workdir=self._resolve(cwd) if cwd else self._workdir,
            environment=base_env,
        )
        stdout_b, stderr_b = result.output or (b"", b"")
        return (
            result.exit_code,
            stdout_b.decode("utf-8", errors="replace") if stdout_b else "",
            stderr_b.decode("utf-8", errors="replace") if stderr_b else "",
        )

    # ------------------------------------------------------------------
    # OsRuntime — file system
    # ------------------------------------------------------------------

    async def read_text(self, path: str) -> str:
        return await asyncio.to_thread(self._sync_read_text, path)

    def _sync_read_text(self, path: str) -> str:
        self._ensure_started_sync()
        assert self._container is not None
        _errors_mod = _sys.modules.get("docker.errors", _docker_errors)
        _not_found_cls = _errors_mod.NotFound if _errors_mod is not None else _NeverRaised
        try:
            with self._io_lock:
                stream, _ = self._container.get_archive(path)
        except _not_found_cls as exc:
            raise FileNotFoundError(path) from exc
        buf = io.BytesIO(b"".join(stream))
        with tarfile.open(fileobj=buf) as tar:
            member = tar.getmembers()[0]
            extracted = tar.extractfile(member)
            return (extracted.read() if extracted else b"").decode("utf-8")

    async def write_text(self, path: str, content: str) -> None:
        await asyncio.to_thread(self._sync_write_text, path, content)

    def _sync_write_text(self, path: str, content: str) -> None:
        self._ensure_started_sync()
        assert self._container is not None
        parent = os.path.dirname(path) or "/"

        data = content.encode("utf-8")
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            info = tarfile.TarInfo(name=os.path.basename(path))
            info.size = len(data)
            info.mtime = int(time.time())
            tar.addfile(info, io.BytesIO(data))
        buf.seek(0)

        with self._io_lock:
            self._container.exec_run(["mkdir", "-p", parent])
            self._container.put_archive(parent, buf)

    async def file_exists(self, path: str) -> bool:
        exit_code, _, _ = await self.exec(["test", "-e", path])
        return exit_code == 0

    async def list_dir(self, path: str) -> list[str]:
        exit_code, stdout, stderr = await self.exec(["ls", "-1", path])
        if exit_code != 0:
            raise FileNotFoundError(f"{path}: {stderr.strip()}")
        return [line.strip() for line in stdout.splitlines() if line.strip()]

    async def make_dir(self, path: str) -> None:
        resolved = self._resolve(path)
        exit_code, _, stderr = await self.exec(["mkdir", "-p", resolved])
        if exit_code != 0:
            raise RuntimeError(f"make_dir '{resolved}' failed: {stderr.strip()}")

    async def delete(self, path: str) -> None:
        exists = await self.file_exists(path)
        if not exists:
            raise FileNotFoundError(path)
        exit_code, _, stderr = await self.exec(["rm", "-rf", path])
        if exit_code != 0:
            raise RuntimeError(f"delete '{path}' failed: {stderr.strip()}")

    # ------------------------------------------------------------------

    def shell(self) -> tuple[str, str]:
        return ("bash", "-c")

    def _ensure_started_sync(self) -> None:
        """Starts the container on first access (lazy init)."""
        if self._container is None:
            self._sync_start()

    def _assert_running(self) -> None:
        if self._container is None:
            raise RuntimeError(
                "DockerOsRuntime is not started — call start() before use."
            )
