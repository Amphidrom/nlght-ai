# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

import pytest

from nlght.adapters.outbound.tools.loader import ToolLoader
from nlght.core.errors.errors import WorkflowConfigurationError
from nlght.core.tools.tool import ToolBase, ToolSignature


class _EchoTool(ToolBase):
    KIND = "echo"

    @classmethod
    def signatures(cls) -> list[ToolSignature]:
        return []

    async def echo(self, *, message: str) -> str:
        return message


class _OtherTool(ToolBase):
    KIND = "other"

    @classmethod
    def signatures(cls) -> list[ToolSignature]:
        return []


def test_register_and_contains() -> None:
    loader = ToolLoader()
    loader.register(_EchoTool)

    assert "echo" in loader


def test_registered_kinds_returns_sorted_list() -> None:
    loader = ToolLoader()
    loader.register(_OtherTool)
    loader.register(_EchoTool)

    assert loader.registered_kinds() == ["echo", "other"]


def test_instantiate_returns_correct_type() -> None:
    loader = ToolLoader()
    loader.register(_EchoTool)

    instance = loader.instantiate(kind="echo", name="my-echo", config={"key": "val"})

    assert isinstance(instance, _EchoTool)


def test_instantiate_passes_name_and_config() -> None:
    loader = ToolLoader()
    loader.register(_EchoTool)

    instance = loader.instantiate(kind="echo", name="my-echo", config={"key": "val"})

    assert instance.name == "my-echo"
    assert instance.config == {"key": "val"}


def test_instantiate_raises_for_unknown_kind() -> None:
    loader = ToolLoader()

    with pytest.raises(WorkflowConfigurationError, match="Unknown tool kind"):
        loader.instantiate(kind="unknown", name="x", config={})


def test_register_overwrites_previous() -> None:
    class _EchoV2(ToolBase):
        KIND = "echo"

        @classmethod
        def signatures(cls) -> list[ToolSignature]:
            return []

    loader = ToolLoader()
    loader.register(_EchoTool)
    loader.register(_EchoV2)

    instance = loader.instantiate(kind="echo", name="x", config={})

    assert isinstance(instance, _EchoV2)


def test_not_contains_unregistered_kind() -> None:
    loader = ToolLoader()

    assert "echo" not in loader
