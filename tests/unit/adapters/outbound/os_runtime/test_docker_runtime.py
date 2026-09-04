# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import io
import sys
import tarfile
from types import SimpleNamespace

import pytest

from nlght.adapters.outbound.os_runtime import docker as docker_module
from nlght.adapters.outbound.os_runtime.docker import DockerOsRuntime, DockerOsRuntimeFactory
from nlght.core.tools.action import (
    FilesystemCapability,
    NetworkCapability,
    ProcessCapability,
    SecretsCapability,
)


class _NotFound(Exception):
    pass


class _Container:
    def __init__(self, *, status: str = "running", network_mode: str = "bridge") -> None:
        self.status = status
        self.attrs = {"HostConfig": {"NetworkMode": network_mode}}
        self.started = False
        self.stopped = False
        self.removed = False
        self.exec_calls: list[dict] = []
        self.archives: list[tuple[str, bytes]] = []

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def remove(self) -> None:
        self.removed = True

    def exec_run(self, cmd: list[str], **kwargs: object) -> SimpleNamespace:
        self.exec_calls.append({"cmd": cmd, **kwargs})
        return SimpleNamespace(exit_code=0, output=(b"out", b"err"))

    def get_archive(self, path: str) -> tuple[list[bytes], object]:
        if path == "/missing.txt":
            raise _NotFound()
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            data = b"hello"
            info = tarfile.TarInfo("file.txt")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        return [buf.getvalue()], object()

    def put_archive(self, parent: str, data: io.BytesIO) -> None:
        self.archives.append((parent, data.getvalue()))


class _Containers:
    def __init__(self) -> None:
        self.existing: _Container | None = None
        self.created: _Container | None = None
        self.run_args: tuple[str, dict[str, object]] | None = None
        self.get_error: Exception | None = None

    def get(self, name: str) -> _Container:
        if self.get_error is not None:
            raise self.get_error
        if self.existing is None:
            raise _NotFound()
        return self.existing

    def run(self, image: str, **kwargs: object) -> _Container:
        self.run_args = (image, kwargs)
        self.created = _Container()
        return self.created


class _DockerModule:
    def __init__(self, containers: _Containers) -> None:
        self._client = SimpleNamespace(containers=containers)
        self.errors = SimpleNamespace(NotFound=_NotFound)

    def from_env(self) -> object:
        return self._client


def _install_docker(monkeypatch: pytest.MonkeyPatch, containers: _Containers) -> None:
    docker_module = _DockerModule(containers)
    monkeypatch.setitem(sys.modules, "docker", docker_module)
    monkeypatch.setitem(sys.modules, "docker.errors", docker_module.errors)
    # The adapter resolves the docker module lazily via sys.modules, but
    # _require_docker() consults this import-time flag — force it so the
    # tests are hermetic even when the real docker package is not installed.
    monkeypatch.setattr("nlght.adapters.outbound.os_runtime.docker._DOCKER_AVAILABLE", True)
    monkeypatch.setattr("nlght.adapters.outbound.os_runtime.docker._docker_sdk", docker_module)


def test_sync_start_reuses_or_creates_container(monkeypatch: pytest.MonkeyPatch) -> None:
    containers = _Containers()
    containers.existing = _Container(status="exited")
    _install_docker(monkeypatch, containers)

    runtime = DockerOsRuntime(name="nlght", base_image="python:3")
    runtime._sync_start()

    assert runtime._container is containers.existing
    assert containers.existing.started is True

    containers.existing = None
    runtime = DockerOsRuntime(name="nlght", base_image="python:3")
    runtime._sync_start()

    assert runtime._container is containers.created


def test_sync_exec_read_write(monkeypatch: pytest.MonkeyPatch) -> None:
    containers = _Containers()
    _install_docker(monkeypatch, containers)
    runtime = DockerOsRuntime(name="nlght", base_image="python:3")

    # exec auto-starts the container lazily on first call
    assert runtime._sync_exec(["pwd"], "/work", {"A": "B"}) == (0, "out", "err")
    assert runtime._container is containers.created
    assert runtime._container.exec_calls[0]["workdir"] == "/work"

    assert runtime._sync_read_text("/file.txt") == "hello"
    with pytest.raises(FileNotFoundError):
        runtime._sync_read_text("/missing.txt")

    runtime._sync_write_text("/work/file.txt", "payload")
    assert runtime._container.exec_calls[-1]["cmd"] == ["mkdir", "-p", "/work"]
    assert runtime._container.archives[0][0] == "/work"


async def test_async_file_helpers_raise_on_failed_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = DockerOsRuntime(name="nlght", base_image="python:3")
    calls: list[list[str]] = []
    responses = [(1, "", ""), (1, "", "missing"), (1, "", "mkdir failed"), (0, "", ""), (1, "", "rm failed")]

    async def fake_exec(command: list[str], **_: object) -> tuple[int, str, str]:
        calls.append(command)
        return responses.pop(0)

    monkeypatch.setattr(runtime, "exec", fake_exec)

    assert await runtime.file_exists("/missing") is False
    with pytest.raises(FileNotFoundError):
        await runtime.list_dir("/missing")
    with pytest.raises(RuntimeError, match="make_dir"):
        await runtime.make_dir("/bad")
    with pytest.raises(RuntimeError, match="delete"):
        await runtime.delete("/bad")
    assert calls[-2:] == [["test", "-e", "/bad"], ["rm", "-rf", "/bad"]]


def test_require_docker_explains_optional_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(docker_module, "_DOCKER_AVAILABLE", False)
    with pytest.raises(ImportError, match=r"nlght-ai\[docker\]"):
        docker_module._require_docker()


async def test_factory_initializes_binds_creates_releases_and_reuses(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    containers = _Containers()
    _install_docker(monkeypatch, containers)
    factory = DockerOsRuntimeFactory(
        base_image="python:3",
        workdir="/work",
        workspace_path=str(tmp_path),
        extra_hosts=["host.docker.internal:127.0.0.1", "invalid"],
    )

    await factory.start()
    first = factory.bind("cid")
    assert factory.bind("cid") is first
    created = await factory.create("eager")
    assert await factory.create("eager") is created
    image, kwargs = containers.run_args
    assert image == "python:3"
    assert kwargs["name"] == "eager"
    assert kwargs["volumes"] == {str(tmp_path / "eager"): {"bind": "/work", "mode": "rw"}}
    assert kwargs["extra_hosts"] == {"host.docker.internal": "127.0.0.1"}

    await factory.release(created)
    assert containers.created.stopped and containers.created.removed
    assert "eager" not in factory._instances
    await factory.release_by_name("missing")
    await factory.stop()


async def test_factory_release_by_name_stops_bound_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    containers = _Containers()
    _install_docker(monkeypatch, containers)
    factory = DockerOsRuntimeFactory(base_image="python:3")
    runtime = factory.bind("bound")
    runtime._container = _Container()

    await factory.release_by_name("bound")
    assert runtime._container.stopped and runtime._container.removed


def test_sync_start_propagates_non_not_found_lookup_error(monkeypatch: pytest.MonkeyPatch) -> None:
    containers = _Containers()
    containers.get_error = RuntimeError("daemon unavailable")
    _install_docker(monkeypatch, containers)

    with pytest.raises(RuntimeError, match="daemon unavailable"):
        DockerOsRuntime(name="nlght", base_image="python:3")._sync_start()


def test_confined_runtime_refuses_to_reuse_networked_container(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    containers = _Containers()
    containers.existing = _Container(network_mode="bridge")
    _install_docker(monkeypatch, containers)

    with pytest.raises(RuntimeError, match="does not enforce network_mode='none'"):
        DockerOsRuntime(
            name="nlght",
            base_image="python:3",
            network_mode="none",
            allow_runtime_env=False,
        )._sync_start()


def test_runtime_resolves_paths_environment_and_running_assertion(monkeypatch: pytest.MonkeyPatch) -> None:
    containers = _Containers()
    _install_docker(monkeypatch, containers)
    runtime = DockerOsRuntime(name="nlght", base_image="python:3", workdir="/work")
    assert runtime._resolve("relative") == "/work/relative"
    assert runtime._resolve("/absolute") == "/absolute"
    assert runtime.shell() == ("bash", "-c")
    with pytest.raises(RuntimeError, match="not started"):
        runtime._assert_running()

    runtime.add_env({"BASE": "one"})
    runtime._sync_exec(["env"], None, {"CALL": "two"})
    environment = runtime._container.exec_calls[-1]["environment"]
    assert environment == {
        "BASH_ENV": "/etc/nlght-bash-logger.sh",
        "BASE": "one",
        "CALL": "two",
    }


def test_confined_runtime_capabilities_are_enforced_not_inferred(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    containers = _Containers()
    _install_docker(monkeypatch, containers)
    runtime = DockerOsRuntime(
        name="confined",
        base_image="python:3",
        network_mode="none",
        allow_runtime_env=False,
    )

    capabilities = runtime.security_capabilities()
    assert capabilities.filesystem == FilesystemCapability.SANDBOX
    assert capabilities.network == NetworkCapability.NONE
    assert capabilities.process == ProcessCapability.SANDBOXED
    assert capabilities.secrets == SecretsCapability.NONE

    with pytest.raises(PermissionError, match="environment injection is disabled"):
        runtime.add_env({"TOKEN": "secret"})
    with pytest.raises(PermissionError, match="environment injection is disabled"):
        runtime._sync_exec(["env"], None, {"TOKEN": "secret"})

    runtime._sync_start()
    assert containers.run_args is not None
    assert containers.run_args[1]["network_mode"] == "none"


def test_default_docker_capabilities_do_not_claim_confinement() -> None:
    capabilities = DockerOsRuntime(
        name="default", base_image="python:3"
    ).security_capabilities()

    assert capabilities.network == NetworkCapability.ARBITRARY
    assert capabilities.secrets == SecretsCapability.MAY_READ


def test_sync_stop_swallows_container_cleanup_error() -> None:
    class _BrokenContainer(_Container):
        def stop(self) -> None:
            raise RuntimeError("stop failed")

    runtime = DockerOsRuntime(name="nlght", base_image="python:3")
    runtime._container = _BrokenContainer()
    runtime._sync_stop()
