# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal, cast

import yaml

from nlght.core.playbooks.playbook import Phase, PhaseExitPolicy, PlaybookDefinition

logger = logging.getLogger(__name__)


def load_playbook_definitions(directory: Path) -> dict[str, PlaybookDefinition]:
    """Loads all *.yaml / *.yml Playbook definitions from the given directory.

    Invalid files are skipped with a WARNING — a broken playbook does not
    block platform startup.

    Returns a dict {playbook_name: PlaybookDefinition}.
    """
    definitions: dict[str, PlaybookDefinition] = {}

    if not directory.exists():
        logger.warning("playbook.loader.dir_missing | path=%s", directory)
        return definitions

    for path in sorted(directory.glob("*.yaml")) + sorted(directory.glob("*.yml")):
        try:
            defn = _load_one(path)
            definitions[defn.name] = defn
            logger.debug("playbook.loader.loaded | playbook=%s file=%s", defn.name, path.name)
        except Exception as exc:
            logger.warning("playbook.loader.skip | file=%s error=%s", path.name, exc)

    logger.info("playbook.loader.done | loaded=%d dir=%s", len(definitions), directory)
    return definitions


def _as_list(value: object) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple | set):
        return list(value)
    return [value]


def _as_text(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    parts = [str(item).strip() for item in _as_list(value)]
    parts = [part for part in parts if part]
    if not parts:
        return None
    return "\n".join(parts)


def _normalize_name(name: str) -> str:
    """Normalizes identifiers: hyphens → underscores (``web-search`` → ``web_search``)."""
    return name.replace("-", "_")


def _as_string_list(value: object, *, normalize: bool = False) -> list[str]:
    items: list[str] = []
    for item in _as_list(value):
        if isinstance(item, list | tuple | set):
            items.extend(_as_string_list(list(item), normalize=normalize))
            continue
        text = str(item).strip()
        if normalize:
            text = _normalize_name(text)
        if not text:
            continue
        items.append(text)
    return items


def _as_requires(value: object) -> list[list[str]]:
    groups: list[list[str]] = []
    for item in _as_list(value):
        group = _as_string_list(item, normalize=True)
        if group:
            groups.append(group)
    return groups


def _load_phase(raw_phase: object) -> Phase:
    if not isinstance(raw_phase, dict):
        raise TypeError(f"phase must be a mapping, got {type(raw_phase).__name__}")
    max_rounds = raw_phase.get("max_rounds")
    if max_rounds is not None:
        try:
            max_rounds = int(max_rounds)
        except (TypeError, ValueError):
            max_rounds = None
    exit_policy = _load_phase_exit(raw_phase.get("exit"))
    return Phase(
        name=str(raw_phase["name"]).strip(),
        goal=str(raw_phase["goal"]).strip(),
        tools=_as_string_list(raw_phase.get("tools"), normalize=True),
        role=_as_text(raw_phase.get("role")),
        guidance=_as_text(raw_phase.get("guidance")),
        exit_condition=_as_text(raw_phase.get("exit_condition")),
        input_tags=_as_string_list(raw_phase.get("input_tags")),
        output_tags=_as_string_list(raw_phase.get("output_tags")),
        visible_tags=_as_string_list(raw_phase.get("visible_tags")),
        output_example=_as_text(raw_phase.get("output_example")),
        max_rounds=max_rounds,
        exit=exit_policy,
        execution=_as_text(raw_phase.get("execution")),
        query_inputs=raw_phase.get("query_inputs") if isinstance(raw_phase.get("query_inputs"), dict) else None,
    )


def _load_phase_exit(raw_exit: object) -> PhaseExitPolicy | None:
    if raw_exit is None:
        return None
    if not isinstance(raw_exit, dict):
        raise TypeError(f"phase exit must be a mapping, got {type(raw_exit).__name__}")
    exit_type = str(raw_exit.get("type", "")).strip()
    if exit_type not in {"output_exists", "io_coverage"}:
        raise ValueError(f"unsupported phase exit type: {exit_type!r}")
    # Just validated above -- the runtime membership check guarantees this,
    # mypy can't narrow a str `in {...}` check into a Literal on its own.
    validated_exit_type = cast(Literal["output_exists", "io_coverage"], exit_type)
    min_outputs = raw_exit.get("min_outputs", 1)
    try:
        min_outputs = int(min_outputs)
    except (TypeError, ValueError):
        min_outputs = 1
    return PhaseExitPolicy(
        type=validated_exit_type,
        input_tag=_as_text(raw_exit.get("input_tag")),
        output_tags=_as_string_list(raw_exit.get("output_tags")),
        terminal_tags=_as_string_list(raw_exit.get("terminal_tags")),
        key_fields=_as_string_list(raw_exit.get("key_fields")),
        min_outputs=max(0, min_outputs),
    )


def _load_one(path: Path) -> PlaybookDefinition:
    raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"playbook file must contain a mapping, got {type(raw).__name__}")

    phases = [_load_phase(p) for p in _as_list(raw.get("phases"))]
    activation = raw.get("activation") or {}

    return PlaybookDefinition(
        name=str(raw["name"]).strip(),
        description_hint=_as_text(raw.get("description_hint")) or "",
        requires=_as_requires(raw.get("requires")),
        optional=_as_string_list(raw.get("optional"), normalize=True),
        phases=phases,
        constraints=_as_string_list(raw.get("constraints")),
        fallback=_as_text(raw.get("fallback")),
        use_when=_as_string_list(activation.get("use_when")),
        use_none_only_if=_as_string_list(activation.get("use_none_only_if")),
    )
