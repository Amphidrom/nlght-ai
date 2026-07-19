# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Decision semantics of the rule-based access-policy engine."""
from __future__ import annotations

import uuid
from datetime import UTC, datetime

from nlght.adapters.outbound.access_policy.rule_engine import AccessRuleEngine
from nlght.core.access.rule import AccessRule, Effect, SubjectType
from nlght.core.entry.context import RequestContext
from nlght.core.runtime.resource import ResourceDef


class _StaticRepo:
    def __init__(self, rules: list[AccessRule]) -> None:
        self.rules = rules
        self.calls = 0

    async def list_enabled(self) -> list[AccessRule]:
        self.calls += 1
        return self.rules


def _rule(
    subject_type: SubjectType = "tool",
    subject: str = "*",
    effect: Effect = "allow",
    conditions: dict[str, list[str]] | None = None,
    priority: int = 0,
    enabled: bool = True,
) -> AccessRule:
    return AccessRule(
        rule_id=uuid.uuid4(),
        subject_type=subject_type,
        subject=subject,
        effect=effect,
        conditions=conditions or {},
        priority=priority,
        enabled=enabled,
    )


def _engine(rules: list[AccessRule], ttl: float = 300.0) -> AccessRuleEngine:
    return AccessRuleEngine(_StaticRepo(rules), cache_ttl_seconds=ttl)


def _caller(
    headers: dict[str, str] | None = None,
    client_host: str | None = "10.0.0.1",
) -> RequestContext:
    return RequestContext(
        correlation_id="cid-1",
        request_id="rid-1",
        received_at=datetime(2026, 1, 1, tzinfo=UTC),
        path="/",
        method="POST",
        headers=headers or {},
        query_params={},
        client_host=client_host,
    )


def _resource(name: str = "web_search") -> ResourceDef:
    return ResourceDef(
        resource_id=uuid.uuid4(),
        name=name,
        kind="tool",
        provider="builtin",
        config={},
    )


# ---------------------------------------------------------------------------
# Defaults — no rules, no restriction
# ---------------------------------------------------------------------------


async def test_empty_rule_set_allows_everything() -> None:
    engine = _engine([])
    assert await engine.decide("tool", "anything", _caller(), "llama3") is True
    assert await engine.decide("model", "gpt-4", _caller(), "gpt-4") is True
    assert await engine.decide("playbook", "research", None, None) is True


async def test_no_rule_for_subject_allows() -> None:
    engine = _engine([_rule(subject_type="tool", subject="other_tool", effect="deny")])
    assert await engine.decide("tool", "web_search", _caller(), None) is True


async def test_rules_for_other_subject_type_do_not_apply() -> None:
    engine = _engine([_rule(subject_type="model", subject="*", effect="deny")])
    assert await engine.decide("tool", "web_search", _caller(), None) is True


# ---------------------------------------------------------------------------
# Allowlist semantics
# ---------------------------------------------------------------------------


async def test_allow_rule_with_matching_condition_allows() -> None:
    engine = _engine([
        _rule(subject="web_search", conditions={"model": ["llama3"]}),
    ])
    assert await engine.decide("tool", "web_search", _caller(), "llama3") is True


async def test_allow_rule_with_unmatched_condition_denies() -> None:
    # Rules exist for the subject but none match → deny (allowlist).
    engine = _engine([
        _rule(subject="web_search", conditions={"model": ["llama3"]}),
    ])
    assert await engine.decide("tool", "web_search", _caller(), "gpt-4") is False


async def test_unconditional_allow_rule_allows() -> None:
    engine = _engine([_rule(subject="web_search")])
    assert await engine.decide("tool", "web_search", None, None) is True


async def test_deny_rule_denies() -> None:
    engine = _engine([_rule(subject="web_search", effect="deny")])
    assert await engine.decide("tool", "web_search", _caller(), "llama3") is False


# ---------------------------------------------------------------------------
# Globs
# ---------------------------------------------------------------------------


async def test_subject_glob_matches() -> None:
    engine = _engine([
        _rule(subject="web_*", conditions={"model": ["llama3"]}),
    ])
    assert await engine.decide("tool", "web_search", _caller(), "llama3") is True
    assert await engine.decide("tool", "web_search", _caller(), "mistral") is False
    assert await engine.decide("tool", "shell", _caller(), "mistral") is True


async def test_condition_value_glob_matches() -> None:
    engine = _engine([
        _rule(subject="*", conditions={"model": ["llama3*"]}),
    ])
    assert await engine.decide("tool", "x", _caller(), "llama3.2") is True
    assert await engine.decide("tool", "x", _caller(), "gpt-4") is False


# ---------------------------------------------------------------------------
# Conditions — AND across keys, OR within values
# ---------------------------------------------------------------------------


async def test_condition_values_are_or_combined() -> None:
    engine = _engine([
        _rule(subject="t", conditions={"model": ["llama3", "mistral"]}),
    ])
    assert await engine.decide("tool", "t", _caller(), "mistral") is True
    assert await engine.decide("tool", "t", _caller(), "gpt-4") is False


async def test_condition_keys_are_and_combined() -> None:
    engine = _engine([
        _rule(subject="t", conditions={"model": ["llama3"], "header:x-org": ["acme"]}),
    ])
    ok = _caller(headers={"X-Org": "acme"})
    wrong_org = _caller(headers={"X-Org": "evil"})
    assert await engine.decide("tool", "t", ok, "llama3") is True
    assert await engine.decide("tool", "t", wrong_org, "llama3") is False
    assert await engine.decide("tool", "t", ok, "gpt-4") is False


async def test_header_condition_is_case_insensitive_on_name() -> None:
    engine = _engine([
        _rule(subject="t", conditions={"header:X-ORG": ["acme"]}),
    ])
    assert await engine.decide("tool", "t", _caller(headers={"x-org": "acme"}), None) is True


async def test_client_host_condition() -> None:
    engine = _engine([
        _rule(subject="t", conditions={"client_host": ["10.0.*"]}),
    ])
    assert await engine.decide("tool", "t", _caller(client_host="10.0.0.7"), None) is True
    assert await engine.decide("tool", "t", _caller(client_host="192.168.0.1"), None) is False


async def test_missing_context_value_fails_condition() -> None:
    engine = _engine([
        _rule(subject="t", conditions={"model": ["llama3"]}),
    ])
    # No model bound → the condition cannot be satisfied → deny.
    assert await engine.decide("tool", "t", _caller(), None) is False


async def test_no_caller_fails_caller_conditions() -> None:
    engine = _engine([
        _rule(subject="t", conditions={"header:x-org": ["acme"]}),
    ])
    assert await engine.decide("tool", "t", None, "llama3") is False


async def test_unknown_condition_key_never_matches() -> None:
    engine = _engine([
        _rule(subject="t", conditions={"moon_phase": ["full"]}),
    ])
    assert await engine.decide("tool", "t", _caller(), "llama3") is False


# ---------------------------------------------------------------------------
# Priority and deny-precedence
# ---------------------------------------------------------------------------


async def test_higher_priority_rule_wins() -> None:
    engine = _engine([
        _rule(subject="t", effect="deny", priority=10, conditions={"model": ["llama3"]}),
        _rule(subject="t", effect="allow", priority=0),
    ])
    assert await engine.decide("tool", "t", _caller(), "llama3") is False
    assert await engine.decide("tool", "t", _caller(), "gpt-4") is True


async def test_deny_wins_on_equal_priority() -> None:
    engine = _engine([
        _rule(subject="t", effect="allow"),
        _rule(subject="t", effect="deny"),
    ])
    assert await engine.decide("tool", "t", _caller(), None) is False


async def test_allow_all_except_denied_model() -> None:
    # "allow everything except tool access from gpt-*" — deny needs priority.
    engine = _engine([
        _rule(subject="*", effect="deny", priority=10, conditions={"model": ["gpt-*"]}),
        _rule(subject="*", effect="allow", priority=0),
    ])
    assert await engine.decide("tool", "shell", _caller(), "gpt-4") is False
    assert await engine.decide("tool", "shell", _caller(), "llama3") is True


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


async def test_rules_are_cached_within_ttl() -> None:
    repo = _StaticRepo([])
    engine = AccessRuleEngine(repo, cache_ttl_seconds=300.0)
    await engine.decide("tool", "a", None, None)
    await engine.decide("tool", "b", None, None)
    await engine.decide("model", "c", None, "c")
    assert repo.calls == 1


async def test_invalidate_cache_forces_reload() -> None:
    repo = _StaticRepo([])
    engine = AccessRuleEngine(repo, cache_ttl_seconds=300.0)
    await engine.decide("tool", "a", None, None)
    engine.invalidate_cache()
    await engine.decide("tool", "a", None, None)
    assert repo.calls == 2


# ---------------------------------------------------------------------------
# Port facades
# ---------------------------------------------------------------------------


async def test_tool_facade_uses_resource_name() -> None:
    engine = _engine([_rule(subject_type="tool", subject="web_search", effect="deny")])
    policy = engine.for_tools()
    assert await policy.is_allowed(_resource("web_search"), _caller(), "llama3") is False
    assert await policy.is_allowed(_resource("shell"), _caller(), "llama3") is True


async def test_model_facade_matches_model_name_as_subject_and_condition() -> None:
    engine = _engine([
        _rule(subject_type="model", subject="gpt-*", effect="deny"),
        _rule(subject_type="model", subject="*", effect="allow", conditions={"model": ["llama*"]}),
    ])
    policy = engine.for_models()
    assert await policy.is_allowed("gpt-4", _caller()) is False
    assert await policy.is_allowed("llama3", _caller()) is True


async def test_playbook_facade() -> None:
    engine = _engine([
        _rule(subject_type="playbook", subject="research", conditions={"model": ["llama3"]}),
    ])
    policy = engine.for_playbooks()
    assert await policy.is_allowed("research", _caller(), "llama3") is True
    assert await policy.is_allowed("research", _caller(), "gpt-4") is False
    assert await policy.is_allowed("other", _caller(), "gpt-4") is True
