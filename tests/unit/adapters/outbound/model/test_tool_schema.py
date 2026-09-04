# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""ToolParameter lowers losslessly to standard JSON Schema.

The canonical wire form for a tool's parameters is plain JSON Schema; ToolParameter
is just a typed builder that lowers to it (items/enum/required), keeping us
deckungsgleich mit OpenAI / Anthropic / MCP.
"""
from __future__ import annotations

from nlght.adapters.outbound.model._tool_helpers import (
    contract_to_openai_tool,
    params_to_json_schema,
)
from nlght.adapters.outbound.model.anthropic import _to_anthropic_tool
from nlght.adapters.outbound.tools.builtin.action_semantics import READ_REQUEST
from nlght.adapters.outbound.tools.contract import CallbackToolContract
from nlght.core.tools.tool import ToolParameter


def _params() -> list[ToolParameter]:
    return [
        ToolParameter(name="chain_of", type="array", items={"type": "integer"},
                      description="indices"),
        ToolParameter(name="severity", type="string", enum=["critical", "high"]),
        ToolParameter(name="note", type="string", required=False),
    ]


def _contract() -> CallbackToolContract:
    return CallbackToolContract(
        name="report_chain",
        description="Report a chain.",
        parameters=_params(),
        callback=lambda **_: "ok",
        action=READ_REQUEST,
    )


def test_params_to_json_schema_emits_items_enum_and_required() -> None:
    schema = params_to_json_schema(_params())

    assert schema["type"] == "object"
    assert schema["properties"]["chain_of"] == {
        "type": "array", "description": "indices", "items": {"type": "integer"},
    }
    assert schema["properties"]["severity"]["enum"] == ["critical", "high"]
    # required omits the optional `note`
    assert schema["required"] == ["chain_of", "severity"]


def test_plain_parameter_schema_is_unchanged() -> None:
    # A parameter without enum/items lowers exactly as before (no surprises).
    schema = params_to_json_schema([ToolParameter(name="x", type="string", description="d")])
    assert schema["properties"]["x"] == {"type": "string", "description": "d"}


def test_contract_to_openai_tool_wraps_the_schema() -> None:
    tool = contract_to_openai_tool(_contract())
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "report_chain"
    assert tool["function"]["parameters"]["properties"]["chain_of"]["items"] == {"type": "integer"}


def test_anthropic_input_schema_passes_json_schema_through() -> None:
    tool = _to_anthropic_tool(contract_to_openai_tool(_contract()))
    props = tool["input_schema"]["properties"]
    assert props["chain_of"]["items"] == {"type": "integer"}
    assert props["severity"]["enum"] == ["critical", "high"]
