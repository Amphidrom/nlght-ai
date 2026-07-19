# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nlght.core.tools.tool import ToolParameter
    from nlght.ports.outbound.tool_catalog import ToolCatalog, ToolContract

logger = logging.getLogger(__name__)


def terminal_tool_names(tool_catalog: ToolCatalog | None) -> set[str]:
    """Names of tools whose invocation ends the model turn.

    ``terminal`` is contract-level metadata only — it is never serialised into the
    tool schema sent to a provider (see ``contract_to_openai_tool``), so honouring
    it stays fully wire-compatible with OpenAI / Anthropic / MCP tool definitions.
    A client uses this set to stop its tool loop (and, where the wire protocol
    allows, close the stream) once such a tool is called.
    """
    if tool_catalog is None:
        return set()
    return {c.name for c in tool_catalog.all() if getattr(c, "terminal", False)}


def _param_to_json_schema(p: ToolParameter) -> dict[str, Any]:
    """Lower a single ToolParameter to a JSON-Schema property object."""
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
    """Lower a list[ToolParameter] to a standard JSON-Schema object.

    This is the single canonical lowering: every model client serialises tools
    through here, so the wire schema is plain JSON Schema (``items``/``enum``/
    nested) — deckungsgleich mit OpenAI/Anthropic/MCP.  Parameters without
    ``enum``/``items`` produce exactly the same schema as before.
    """
    properties: dict[str, Any] = {}
    required: list[str] = []
    for p in (parameters or []):
        properties[p.name] = _param_to_json_schema(p)
        if getattr(p, "required", True):
            required.append(p.name)
    return {"type": "object", "properties": properties, "required": required}


def contract_to_openai_tool(contract: ToolContract) -> dict[str, Any]:
    """Convert a ToolContract to an OpenAI-format tool definition."""
    return {
        "type": "function",
        "function": {
            "name": contract.name,
            "description": getattr(contract, "description", "") or "",
            "parameters": params_to_json_schema(getattr(contract, "parameters", None)),
        },
    }


async def execute_tool_call(tool_catalog: ToolCatalog, tc: dict[str, Any]) -> str:
    """Thin backward-compatible wrapper — delegates to ``catalog.execute(tc)``."""
    return await tool_catalog.execute(tc)


def append_ollama_native_tool_turn(
    messages: list[dict[str, Any]],
    tool_calls_raw: list[dict[str, Any]],
    results: list[str],
    assistant_text: str = "",
) -> list[dict[str, Any]]:
    """Append tool turns in Ollama native /api/chat format.

    Ollama expects tool_calls without id/type, arguments as a dict (not JSON string),
    and tool result messages as plain {"role": "tool", "content": "..."}.
    """
    tc_list: list[dict[str, Any]] = []
    for tc in tool_calls_raw:
        entry: dict[str, Any] = {"function": {"name": tc["name"], "arguments": tc.get("input", {})}}
        # Provider-opaque continuation token (Gemini 3.x thought_signature, base64).
        # Other providers ignore the extra key; Google re-emits it in
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
    """Append assistant-with-tool_calls + tool-result messages (canonical OpenAI format)."""
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
