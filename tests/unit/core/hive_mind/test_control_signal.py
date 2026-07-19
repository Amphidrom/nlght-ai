# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Tests for nlght.core.hive_mind.control_signal — check_control_signals."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from nlght.core.hive_mind.control_signal import check_control_signals

# ---------------------------------------------------------------------------
# Minimal test fakes
# ---------------------------------------------------------------------------

class _SignalType(StrEnum):
    CONTROL = "control"
    INFO    = "info"


@dataclass
class _Metadata:
    intent: str = "info"


@dataclass
class _Classification:
    type:     _SignalType = _SignalType.INFO
    metadata: _Metadata  = field(default_factory=_Metadata)
    entities: list[str]  = field(default_factory=list)


def _ctrl(intent: str, entities: list[str] | None = None) -> _Classification:
    return _Classification(
        type=_SignalType.CONTROL,
        metadata=_Metadata(intent=intent),
        entities=entities or [],
    )

def _info() -> _Classification:
    return _Classification(type=_SignalType.INFO)


# ---------------------------------------------------------------------------
# No control signals → pass_through
# ---------------------------------------------------------------------------

def test_no_control_signals_pass_through() -> None:
    decision = check_control_signals([_info(), _info()], mental_model=None, cid="cid-1")
    assert decision.pass_through is True
    assert decision.should_stop is False


def test_empty_classifications_pass_through() -> None:
    decision = check_control_signals([], mental_model=None, cid="cid-2")
    assert decision.pass_through is True


# ---------------------------------------------------------------------------
# Stop intents → should_stop=True
# ---------------------------------------------------------------------------

def test_stop_intent_returns_stop_decision() -> None:
    decision = check_control_signals([_ctrl("stop")], mental_model=None, cid="cid-3")
    assert decision.should_stop is True
    assert decision.stop_reason == "stop"
    assert decision.pass_through is False


def test_cancel_intent_returns_stop_decision() -> None:
    decision = check_control_signals([_ctrl("cancel")], mental_model=None, cid="cid-4")
    assert decision.should_stop is True
    assert decision.stop_reason == "cancel"


def test_abort_intent_returns_stop_decision() -> None:
    decision = check_control_signals([_ctrl("abort")], mental_model=None, cid="cid-5")
    assert decision.should_stop is True


def test_reset_intent_returns_stop_decision() -> None:
    decision = check_control_signals([_ctrl("reset")], mental_model=None, cid="cid-6")
    assert decision.should_stop is True


# ---------------------------------------------------------------------------
# Priority boost intents
# ---------------------------------------------------------------------------

def test_search_intent_boosts_web_search() -> None:
    decision = check_control_signals([_ctrl("search")], mental_model=None, cid="cid-7")
    assert decision.priority_boost == "web_search"
    assert decision.should_stop is False
    assert decision.pass_through is False


def test_execute_intent_boosts_code_executor() -> None:
    decision = check_control_signals([_ctrl("execute")], mental_model=None, cid="cid-8")
    assert decision.priority_boost == "code_executor"


# ---------------------------------------------------------------------------
# Scope change
# ---------------------------------------------------------------------------

def test_scope_change_from_entity_prefix() -> None:
    sc = _ctrl("info", entities=["scope:coding"])
    decision = check_control_signals([sc], mental_model=None, cid="cid-9")
    assert decision.scope_change == "coding"


def test_scope_change_from_entity_equals() -> None:
    sc = _ctrl("info", entities=["set_scope=analysis"])
    decision = check_control_signals([sc], mental_model=None, cid="cid-10")
    assert decision.scope_change == "analysis"


# ---------------------------------------------------------------------------
# Control signal with no actionable directive → pass_through
# ---------------------------------------------------------------------------

def test_control_signal_with_unknown_intent_pass_through() -> None:
    sc = _ctrl("unknown_directive_xyz")
    decision = check_control_signals([sc], mental_model=None, cid="cid-11")
    assert decision.pass_through is True


# ---------------------------------------------------------------------------
# Stop takes priority over everything else
# ---------------------------------------------------------------------------

def test_stop_takes_priority_over_boost() -> None:
    decision = check_control_signals(
        [_ctrl("search"), _ctrl("stop")], mental_model=None, cid="cid-12"
    )
    assert decision.should_stop is True
