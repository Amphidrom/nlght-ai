# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from nlght.adapters.outbound.tools.builtin.http_probe import HttpProbeTool
from nlght.core.errors.errors import ToolExecutionError


def _tool(runtime=None, config=None) -> HttpProbeTool:
    return HttpProbeTool(name="probe", config=config or {}, os_runtime=runtime)


def test_http_probe_signature_and_runtime_contract() -> None:
    signature = HttpProbeTool.signatures()[0]
    assert signature.name == "http_probe"
    assert [parameter.name for parameter in signature.parameters] == [
        "url", "method", "headers", "body", "follow_redirects",
    ]
    assert _tool(config={"max_body_bytes": 12})._max_body() == 12
    with pytest.raises(ToolExecutionError, match="requires an OsRuntime"):
        _tool()._runtime()


async def test_http_probe_rejects_invalid_url_without_runtime() -> None:
    result = json.loads(await _tool().probe(url="file:///etc/passwd"))
    assert result == {"error": "url must start with http:// or https://"}


async def test_http_probe_writes_parameters_executes_and_cleans_up() -> None:
    runtime = AsyncMock()
    runtime.exec.return_value = (0, '{"status_code": 200, "body": "ok"}\n', "")
    result = await _tool(runtime, {"max_body_bytes": 123}).probe(
        url="https://example.com/api",
        method="POST",
        headers={"X-Test": "yes"},
        body="payload",
        follow_redirects=False,
    )

    assert json.loads(result)["status_code"] == 200
    params = json.loads(runtime.write_text.await_args_list[0].args[1])
    assert params == {
        "url": "https://example.com/api",
        "method": "POST",
        "headers": {"X-Test": "yes"},
        "body": "payload",
        "follow_redirects": False,
        "max_body": 123,
    }
    assert runtime.exec.await_args.args[0][:2] == ["bash", "-c"]
    assert runtime.delete.await_count == 2


async def test_http_probe_reports_execution_failure_and_ignores_cleanup_failure() -> None:
    runtime = AsyncMock()
    runtime.write_text.side_effect = RuntimeError("disk full")
    runtime.delete.side_effect = RuntimeError("cleanup failed")

    result = json.loads(await _tool(runtime).probe(url="https://example.com"))
    assert result == {
        "error": "probe execution failed: disk full",
        "url": "https://example.com",
    }
    assert runtime.delete.await_count == 2


async def test_http_probe_reports_empty_output_and_truncates_stderr() -> None:
    runtime = AsyncMock()
    runtime.exec.return_value = (7, "  ", "x" * 700)
    result = json.loads(await _tool(runtime).probe(url="http://example.com"))
    assert result["error"] == "probe script produced no output (exit=7)"
    assert len(result["stderr"]) == 500


async def test_http_probe_returns_non_json_output_verbatim() -> None:
    runtime = AsyncMock()
    runtime.exec.return_value = (0, "not-json\n", "")
    assert await _tool(runtime).probe(url="http://example.com") == "not-json"
