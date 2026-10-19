# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import re
from pathlib import Path

from nlght.adapters.outbound.tools.builtin.activate_playbook import ActivatePlaybookTool
from nlght.adapters.outbound.tools.builtin.fetch_url import FetchUrlTool
from nlght.adapters.outbound.tools.builtin.git import GitTool
from nlght.adapters.outbound.tools.builtin.http_probe import HttpProbeTool
from nlght.adapters.outbound.tools.builtin.shell import ShellTool
from nlght.adapters.outbound.tools.builtin.web_search import WebSearchTool
from nlght.core.model import trust

REFERENCE = Path(__file__).parents[3] / "docs/documentation/content-trust.mdx"


def _section(name: str) -> str:
    text = REFERENCE.read_text(encoding="utf-8")
    start, end = f"{{/* {name}:start */}}", f"{{/* {name}:end */}}"
    assert text.count(start) == 1
    assert text.count(end) == 1
    return text.split(start, 1)[1].split(end, 1)[0]


def test_every_source_kind_a_policy_can_meet_is_documented() -> None:
    body = _section("content-source-kinds").split("|---|---|", 1)[1]
    documented = set(re.findall(r"^\| `([a-z_]+)` \|", body, flags=re.MULTILINE))
    declared = {
        value for name, value in vars(trust).items()
        if name.startswith("SOURCE_") and isinstance(value, str)
    }

    # Platform text is trusted, so it never appears among untrusted sources.
    assert documented == declared - {trust.SOURCE_PLATFORM}


def test_the_built_in_labels_match_what_the_tools_declare() -> None:
    rows = {
        line.split("|")[1]: line.split("|")[2]
        for line in _section("builtin-tool-labels").splitlines()
        if line.startswith("| `")
    }
    for tool in (FetchUrlTool, HttpProbeTool, WebSearchTool, GitTool, ShellTool, ActivatePlaybookTool):
        (row,) = [label for tools, label in rows.items() if f"`{tool.KIND}`" in tools]
        assert f"`{tool.UNTRUSTED_CONTENT_SOURCE}`" in row, tool.KIND
