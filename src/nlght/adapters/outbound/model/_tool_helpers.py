# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from nlght.core.model.messages import MessageLike, origin_of
from nlght.core.model.trust import UNANNOTATED, ContentOrigin

if TYPE_CHECKING:
    from nlght.core.tools.tool import ToolParameter
    from nlght.ports.outbound.tool_catalog import ToolCatalog, ToolContract

logger = logging.getLogger(__name__)


def terminal_tool_names(tool_catalog: ToolCatalog | None) -> set[str]:
    """Read terminal tool names from the catalog contract.

    Args:
        tool_catalog (ToolCatalog | None): Active tool catalog, if one is bound.

    Returns:
        set[str]: Names of calls that end the current model turn.

    Notes:
        Terminal metadata remains outside provider wire schemas.
    """
    if tool_catalog is None:
        return set()
    return {c.name for c in tool_catalog.all() if getattr(c, "terminal", False)}


def _param_to_json_schema(p: ToolParameter) -> dict[str, Any]:
    """Lower one tool parameter to a JSON Schema property.

    Args:
        p (ToolParameter): Tool parameter contract.

    Returns:
        dict[str, Any]: JSON Schema property for the parameter.
    """
    prop: dict[str, Any] = {"type": getattr(p, "type", None) or "string"}
    desc = getattr(p, "description", "")
    if desc:
        prop["description"] = desc
    enum = getattr(p, "enum", None)
    if enum is not None:
        prop["enum"] = list(enum)
    items = getattr(p, "items", None)
    if items is not None:
        prop["items"] = items
    return prop


def params_to_json_schema(parameters: list[ToolParameter] | None) -> dict[str, Any]:
    """Lower tool parameters to one canonical JSON Schema object.

    Args:
        parameters (list[ToolParameter] | None): Parameter contracts, or None for an empty schema.

    Returns:
        dict[str, Any]: Object schema with properties and required fields.

    Notes:
        All model clients use this lowering for nested items and enums.
    """
    properties: dict[str, Any] = {}
    required: list[str] = []
    for p in (parameters or []):
        properties[p.name] = _param_to_json_schema(p)
        if getattr(p, "required", True):
            required.append(p.name)
    return {"type": "object", "properties": properties, "required": required}


def contract_to_openai_tool(contract: ToolContract) -> dict[str, Any]:
    """Serialize a tool contract in OpenAI tool format.

    Args:
        contract (ToolContract): Tool contract to expose to the model.

    Returns:
        dict[str, Any]: Provider tool definition containing its JSON Schema.
    """
    return {
        "type": "function",
        "function": {
            "name": contract.name,
            "description": getattr(contract, "description", "") or "",
            "parameters": params_to_json_schema(getattr(contract, "parameters", None)),
        },
    }


def context_origin_of(messages: Sequence[MessageLike]) -> ContentOrigin:
    """Combine the origins of messages about to reach the model.

    Args:
        messages (Sequence[MessageLike]): Workflow messages or raw provider messages.

    Returns:
        ContentOrigin: Combined content origin. Unreadable messages remain unannotated.

    Notes:
        Provenance decoding does not interrupt the tool loop.
    """
    try:
        return origin_of(messages)
    except (TypeError, ValueError):
        return UNANNOTATED


async def execute_tool_call(
    tool_catalog: ToolCatalog,
    tc: dict[str, Any],
    context_origin: ContentOrigin | None = None,
) -> str:
    """Execute a model-proposed call through the catalog.

    Args:
        tool_catalog (ToolCatalog): Catalog that authorizes and executes the call.
        tc (dict[str, Any]): Provider tool-call payload.
        context_origin (ContentOrigin | None): Origin of content the model read before
            calling the tool.

    Returns:
        str: Text returned by the tool.

    Side Effects:
        Executes the call and records input and result origins when supported.
    """
    execute_with_origin = getattr(tool_catalog, "execute_with_origin", None)
    if execute_with_origin is None:
        return await tool_catalog.execute(tc)
    output = await execute_with_origin(tc, context_origin=context_origin)
    return str(output.content)


def append_ollama_native_tool_turn(
    messages: list[dict[str, Any]],
    tool_calls_raw: list[dict[str, Any]],
    results: list[str],
    assistant_text: str = "",
) -> list[dict[str, Any]]:
    """Append tool calls and results in Ollama native chat format.

    Args:
        messages (list[dict[str, Any]]): Existing provider conversation.
        tool_calls_raw (list[dict[str, Any]]): Tool calls proposed by the model.
        results (list[str]): Tool results paired with the calls.
        assistant_text (str): Assistant text accompanying the calls.

    Returns:
        list[dict[str, Any]]: Messages with native assistant calls and plain tool results.

    Notes:
        Native calls use dictionary arguments without OpenAI id/type fields.
    """
    tc_list: list[dict[str, Any]] = []
    for tc in tool_calls_raw:
        entry: dict[str, Any] = {"function": {"name": tc["name"], "arguments": tc.get("input", {})}}
        # Provider-opaque continuation token (Gemini 3.x thought_signature, base64).
    # Other providers ignore the extra key. Google re-emits it in
        # _to_google_contents. Kept at the tool_call level so it survives the
        # canonical round-trip.
        if tc.get("thought_signature"):
            entry["thought_signature"] = tc["thought_signature"]
        tc_list.append(entry)
    updated = list(messages) + [{"role": "assistant", "content": assistant_text or "", "tool_calls": tc_list}]
    for result_text in results:
        updated.append({"role": "tool", "content": result_text})
    return updated


def append_openai_tool_turn(
    messages: list[dict[str, Any]],
    tool_calls_raw: list[dict[str, Any]],
    results: list[str],
    assistant_text: str = "",
) -> list[dict[str, Any]]:
    """Append tool calls and results in OpenAI chat format.

    Args:
        messages (list[dict[str, Any]]): Existing provider conversation.
        tool_calls_raw (list[dict[str, Any]]): Tool calls proposed by the model.
        results (list[str]): Tool results paired with the calls.
        assistant_text (str): Assistant text accompanying the calls.

    Returns:
        list[dict[str, Any]]: Messages with assistant tool calls and paired tool results.
    """
    tc_list = [
        {
            "id": tc["id"],
            "type": "function",
            "function": {
                "name": tc["name"],
                "arguments": json.dumps(tc.get("input", {})),
            },
        }
        for tc in tool_calls_raw
    ]
    updated = list(messages) + [{"role": "assistant", "content": assistant_text or "", "tool_calls": tc_list}]
    for tc, result_text in zip(tool_calls_raw, results, strict=False):
        updated.append({"role": "tool", "tool_call_id": tc["id"], "content": result_text, "name": tc["name"]})
    return updated
