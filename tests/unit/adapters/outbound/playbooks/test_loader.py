# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Tests for nlght.adapters.outbound.playbooks.loader."""
from __future__ import annotations

from dataclasses import dataclass

import yaml

from nlght.adapters.outbound.playbooks.loader import load_playbook_definitions


@dataclass(frozen=True)
class _FakePlaybookFile:
    name: str
    text: str

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, _FakePlaybookFile):
            return NotImplemented
        return self.name < other.name

    def read_text(self, encoding: str = "utf-8") -> str:
        assert encoding == "utf-8"
        return self.text


class _FakePlaybookDir:
    def __init__(self, files: dict[str, str] | None = None, *, exists: bool = True) -> None:
        self._files = dict(files or {})
        self._exists = exists

    def exists(self) -> bool:
        return self._exists

    def glob(self, pattern: str):
        suffix = pattern.replace("*", "")
        matches = [
            _FakePlaybookFile(name=name, text=text)
            for name, text in self._files.items()
            if name.endswith(suffix)
        ]
        return sorted(matches, key=lambda item: item.name)

    def __str__(self) -> str:
        return "<fake-playbook-dir>"


def _playbook_dir(files: dict[str, dict], *, exists: bool = True) -> _FakePlaybookDir:
    return _FakePlaybookDir(
        {
            name: yaml.dump(content)
            for name, content in files.items()
        },
        exists=exists,
    )


def _minimal_playbook(name: str = "my_playbook") -> dict:
    return {
        "name": name,
        "description_hint": "A test playbook",
        "phases": [],
    }


def test_missing_directory_returns_empty() -> None:
    result = load_playbook_definitions(_FakePlaybookDir(exists=False))
    assert result == {}


def test_empty_directory_returns_empty() -> None:
    result = load_playbook_definitions(_FakePlaybookDir())
    assert result == {}


def test_loads_yaml_playbook() -> None:
    result = load_playbook_definitions(_playbook_dir({"s1.yaml": _minimal_playbook("s1")}))
    assert "s1" in result
    assert result["s1"].name == "s1"


def test_loads_yml_extension() -> None:
    result = load_playbook_definitions(_playbook_dir({"s2.yml": _minimal_playbook("s2")}))
    assert "s2" in result


def test_loads_multiple_playbooks() -> None:
    result = load_playbook_definitions(
        _playbook_dir(
            {
                "a.yaml": _minimal_playbook("a"),
                "b.yaml": _minimal_playbook("b"),
                "c.yaml": _minimal_playbook("c"),
            }
        )
    )
    assert set(result.keys()) == {"a", "b", "c"}


def test_playbook_with_phases() -> None:
    playbook = {
        "name": "phased",
        "description_hint": "hint",
        "phases": [
            {"name": "p1", "goal": "do x", "tools": ["shell"],
             "role": "executor", "guidance": "careful", "exit_condition": "done"},
        ],
        "constraints": ["no hallucinations"],
        "fallback": "ask user",
    }
    result = load_playbook_definitions(_playbook_dir({"phased.yaml": playbook}))
    defn = result["phased"]
    assert len(defn.phases) == 1
    assert defn.phases[0].name == "p1"
    assert defn.phases[0].role == "executor"
    assert defn.constraints == ["no hallucinations"]
    assert defn.fallback == "ask user"


def test_invalid_yaml_skipped() -> None:
    result = load_playbook_definitions(
        _FakePlaybookDir(
            {
                "bad.yaml": "not: a: valid: playbook",
                "good.yaml": yaml.dump(_minimal_playbook("good")),
            }
        )
    )
    assert "good" in result
    assert "bad" not in result


def test_loader_normalizes_string_and_list_fields_consistently() -> None:
    playbook = {
        "name": "mixed_playbook",
        "description_hint": ["Line 1", "Line 2"],
        "requires": "web-search",
        "optional": "fetch-url",
        "constraints": "No hallucinations",
        "fallback": ["Use fallback", "Explain uncertainty"],
        "phases": {
            "name": "collect",
            "goal": "Collect data",
            "tools": "web-search",
            "role": ["retrieval_external"],
            "guidance": ["Check dates", "Prefer primary sources"],
            "exit_condition": ["Enough evidence collected"],
        },
    }

    result = load_playbook_definitions(_playbook_dir({"mixed_playbook.yaml": playbook}))
    defn = result["mixed_playbook"]

    assert defn.description_hint == "Line 1\nLine 2"
    assert defn.requires == [["web_search"]]
    assert defn.optional == ["fetch_url"]
    assert defn.constraints == ["No hallucinations"]
    assert defn.fallback == "Use fallback\nExplain uncertainty"
    assert len(defn.phases) == 1
    assert defn.phases[0].tools == ["web_search"]
    assert defn.phases[0].role == "retrieval_external"
    assert defn.phases[0].guidance == "Check dates\nPrefer primary sources"
    assert defn.phases[0].exit_condition == "Enough evidence collected"


def test_loader_flattens_nested_tool_lists() -> None:
    playbook = {
        "name": "nested_playbook",
        "description_hint": "hint",
        "requires": [["web-search", "fetch-url"]],
        "optional": [["fetch-url"]],
        "phases": [
            {
                "name": "collect",
                "goal": "Collect data",
                "tools": [["web-search"], ["fetch-url"]],
            }
        ],
    }

    result = load_playbook_definitions(_playbook_dir({"nested_playbook.yaml": playbook}))
    defn = result["nested_playbook"]

    assert defn.requires == [["web_search", "fetch_url"]]
    assert defn.optional == ["fetch_url"]
    assert defn.phases[0].tools == ["web_search", "fetch_url"]


def test_loader_reads_phase_exit_policy_and_visible_tags() -> None:
    playbook = {
        "name": "exit_playbook",
        "description_hint": "hint",
        "phases": [
            {
                "name": "fetch",
                "goal": "Fetch source documents",
                "tools": ["fetch-url"],
                "input_tags": ["kind:selected_source"],
                "output_tags": ["kind:source_document"],
                "visible_tags": ["kind:selected_source", "kind:source_document"],
                "output_example": "[kind:source_document] document_id=doc_abcd1234 | candidate_id=cand_abcd1234_0 | url=https://example.com | title=Example",
                "exit": {
                    "type": "io_coverage",
                    "input_tag": "kind:selected_source",
                    "terminal_tags": ["kind:source_document", "kind:source_fetch_failed"],
                    "key_fields": ["candidate_id"],
                },
            }
        ],
    }

    result = load_playbook_definitions(_playbook_dir({"exit_playbook.yaml": playbook}))
    phase = result["exit_playbook"].phases[0]

    assert phase.visible_tags == ["kind:selected_source", "kind:source_document"]
    assert phase.output_example == "[kind:source_document] document_id=doc_abcd1234 | candidate_id=cand_abcd1234_0 | url=https://example.com | title=Example"
    assert phase.exit is not None
    assert phase.exit.type == "io_coverage"
    assert phase.exit.input_tag == "kind:selected_source"
    assert phase.exit.terminal_tags == ["kind:source_document", "kind:source_fetch_failed"]
    assert phase.exit.key_fields == ["candidate_id"]


def test_loader_reads_output_exists_exit_policy() -> None:
    playbook = {
        "name": "evaluate_playbook",
        "description_hint": "hint",
        "phases": [
            {
                "name": "evaluate",
                "goal": "Evaluate evidence",
                "tools": [],
                "input_tags": ["kind:evidence"],
                "output_tags": ["kind:evaluated_evidence"],
                "visible_tags": ["kind:evidence", "kind:source_document", "kind:evaluated_evidence"],
                "exit": {
                    "type": "output_exists",
                    "output_tags": ["kind:evaluated_evidence"],
                    "min_outputs": 1,
                },
            }
        ],
    }

    result = load_playbook_definitions(_playbook_dir({"evaluate_playbook.yaml": playbook}))
    phase = result["evaluate_playbook"].phases[0]

    assert phase.visible_tags == ["kind:evidence", "kind:source_document", "kind:evaluated_evidence"]
    assert phase.exit is not None
    assert phase.exit.type == "output_exists"
    assert phase.exit.output_tags == ["kind:evaluated_evidence"]
    assert phase.exit.min_outputs == 1


def test_loader_preserves_multi_source_min_outputs_for_selection() -> None:
    playbook = {
        "name": "selection_playbook",
        "description_hint": "hint",
        "phases": [
            {
                "name": "select",
                "goal": "Select sources",
                "tools": [],
                "input_tags": ["kind:source_candidate"],
                "output_tags": ["kind:selected_source"],
                "exit": {
                    "type": "output_exists",
                    "output_tags": ["kind:selected_source"],
                    "min_outputs": 3,
                },
            }
        ],
    }

    result = load_playbook_definitions(_playbook_dir({"selection_playbook.yaml": playbook}))
    phase = result["selection_playbook"].phases[0]

    assert phase.exit is not None
    assert phase.exit.type == "output_exists"
    assert phase.exit.min_outputs == 3
