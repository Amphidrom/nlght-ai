# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""A provider's own tool loop tells the catalog what the model had read (ADR-0074)."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from nlght.adapters.outbound.model.ollama_cloud import OllamaCloudClient
from nlght.adapters.outbound.signals.buffering import BufferingSignalEmitter
from nlght.core.model.messages import (
    ContextKind,
    ContextProvenance,
    ContextRecord,
    UntrustedContextMessage,
    UserMessage,
)
from nlght.core.model.trust import (
    UNANNOTATED,
    ContentLedger,
    ContentOrigin,
    ContentTrust,
    SourceRef,
)
from nlght.core.tools.tool import ToolOutput

MODEL_DIR = Path(__file__).parents[5] / "src/nlght/adapters/outbound/model"
HANDBOOK = SourceRef("data_source", "store-a", "handbook", ContentTrust.TRUSTED)
WEB = SourceRef("web", "", "https://attacker.example", ContentTrust.UNTRUSTED)


class _CapturingCatalog:
    def __init__(self) -> None:
        self.origins: list[ContentOrigin | None] = []
        self.ledger = ContentLedger()

    def all(self) -> list[Any]:
        return []

    async def execute_with_origin(
        self, tc: dict[str, Any], *, context_origin: ContentOrigin | None = None,
    ) -> ToolOutput:
        self.origins.append(context_origin)
        return ToolOutput("22C", UNANNOTATED)

    async def execute(self, tc: dict[str, Any], **_: object) -> str:
        raise AssertionError("the provider loop must use the origin-carrying path")


def _client() -> OllamaCloudClient:
    client = OllamaCloudClient.__new__(OllamaCloudClient)
    client._temperature = None
    client._http = MagicMock()
    client._http.base_url = "https://ollama.test"
    client._default_model = "gpt-oss:20b"
    client._http.complete_message = AsyncMock(side_effect=[
        ({"content": "", "tool_calls": [{
            "id": "call_01", "type": "function",
            "function": {"name": "get_weather", "arguments": "{}"},
        }]}, {}),
        ({"content": "It is 22C."}, {}),
    ])
    return client


async def test_the_loop_passes_the_origin_of_what_the_model_read() -> None:
    catalog = _CapturingCatalog()
    bound = _client().bind(
        model=None, emitter=BufferingSignalEmitter(), stream=False, tool_catalog=catalog,
    )
    retrieved = ContextRecord(
        record_id="passage:1",
        kind=ContextKind.RETRIEVAL,
        content="Weather is fetched from wttr.in",
        provenance=ContextProvenance(source="passage", origin=ContentOrigin.of(WEB)),
    )

    await bound.call([
        UntrustedContextMessage(records=(retrieved,)),
        UserMessage("weather?", origin=ContentOrigin.of(HANDBOOK)),
    ])

    (origin,) = catalog.origins
    assert origin is not None
    assert set(origin.sources) == {HANDBOOK, WEB}


def test_no_provider_loop_bypasses_the_origin_path() -> None:
    """Every provider and debug twin executes through ``execute_tool_call``."""
    offenders = []
    for path in sorted(MODEL_DIR.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "execute"
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == "_tool_catalog"
            ):
                offenders.append(f"{path.name}:{node.lineno}")

    assert offenders == []
