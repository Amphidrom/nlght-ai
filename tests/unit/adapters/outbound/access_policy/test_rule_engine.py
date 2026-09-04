# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Decision semantics of the rule-based access-policy engine."""
from __future__ import annotations

import uuid
from datetime import UTC, datetime

from nlght.adapters.outbound.access_policy.rule_engine import AccessRuleEngine
from nlght.core.access.rule import TOOL_SEPARATOR, AccessRule, Effect, SubjectType, tool_subject
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
    workflow: str | None = None,
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
        workflow=workflow,
    )


def _resource(name: str = "web_search", kind: str = "http") -> ResourceDef:
    return ResourceDef(
        resource_id=uuid.uuid4(),
        name=name,
        kind=kind,
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


async def test_the_resource_facade_decides_on_the_address() -> None:
    """A whole configured resource, named by `<kind>/<name>`."""
    engine = _engine([_rule(subject_type="resource", subject="http/web_search", effect="deny")])
    policy = engine.for_resources()
    assert await policy.is_allowed(_resource("web_search", kind="http"), _caller(), "llama3") is False
    assert await policy.is_allowed(_resource("shell", kind="local"), _caller(), "llama3") is True


async def test_the_resource_facade_globs_over_a_kind() -> None:
    """`data_store/*` is every data store, which is what the shape is for."""
    engine = _engine([_rule(subject_type="resource", subject="data_store/*", effect="deny")])
    policy = engine.for_resources()
    assert await policy.is_allowed(_resource("main", kind="data_store"), _caller(), None) is False
    assert await policy.is_allowed(_resource("main", kind="knowledge_store"), _caller(), None) is True


async def test_the_tool_facade_decides_on_one_signature_of_one_resource() -> None:
    """The correction this slice exists for.

    A rule naming a tool used to be compared against the resource's *name*, so
    it authorized every operation the resource offers and its use inside a
    workflow step where no tool call happens. It now names one signature of one
    activation.
    """
    engine = _engine([
        _rule(subject_type="tool", subject="data_store/main::data-search", effect="deny"),
    ])
    policy = engine.for_tools()
    resource = _resource("main", kind="data_store")

    assert await policy.is_allowed(resource, "data-search", _caller(), None) is False
    # A sibling operation of the same resource is untouched...
    assert await policy.is_allowed(resource, "data-search-keyword", _caller(), None) is True
    # ...and so is the same operation on a different activation.
    other = _resource("secondary", kind="data_store")
    assert await policy.is_allowed(other, "data-search", _caller(), None) is True


async def test_a_tool_rule_can_glob_one_operation_across_activations() -> None:
    engine = _engine([
        _rule(subject_type="tool", subject="data_store/*::data-search", effect="deny"),
    ])
    policy = engine.for_tools()

    assert await policy.is_allowed(_resource("main", kind="data_store"), "data-search", _caller(), None) is False
    assert await policy.is_allowed(_resource("two", kind="data_store"), "data-search", _caller(), None) is False
    assert await policy.is_allowed(_resource("main", kind="data_store"), "data-search-keyword", _caller(), None) is True


async def test_a_resource_rule_scoped_to_a_workflow_denies_every_other() -> None:
    """The contract this whole slice is for, at the engine level.

    Allowlist semantics do the work: the rule matches the subject, so the
    subject is governed; a caller whose workflow does not match satisfies no
    rule and is denied rather than falling through to allowed.
    """
    engine = _engine([
        _rule(subject_type="resource", subject="data_store/main", effect="allow",
              conditions={"workflow": ["answer"]}),
    ])
    policy = engine.for_resources()
    resource = _resource("main", kind="data_store")

    assert await policy.is_allowed(resource, _caller(workflow="answer"), None) is True
    assert await policy.is_allowed(resource, _caller(workflow="ingest"), None) is False
    # No workflow at all is denied too — deliberately, so nothing running
    # outside a flow becomes a way past a workflow-scoped rule.
    assert await policy.is_allowed(resource, _caller(workflow=None), None) is False


async def test_an_unconditional_resource_rule_allows_every_workflow() -> None:
    engine = _engine([
        _rule(subject_type="resource", subject="data_store/shared", effect="allow"),
    ])
    policy = engine.for_resources()
    resource = _resource("shared", kind="data_store")

    assert await policy.is_allowed(resource, _caller(workflow="answer"), None) is True
    assert await policy.is_allowed(resource, _caller(workflow="ingest"), None) is True
    assert await policy.is_allowed(resource, _caller(workflow=None), None) is True


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


# ---------------------------------------------------------------------------
# workflow condition
# ---------------------------------------------------------------------------


async def test_workflow_condition_allows_only_the_named_flow() -> None:
    engine = AccessRuleEngine(
        _StaticRepo([
            _rule(subject="index-writer", effect="allow", conditions={"workflow": ["ingest-*"]}),
        ]),
        cache_ttl_seconds=0.0,
    )

    assert await engine.decide(
        "tool", "index-writer", _caller(workflow="ingest-data"), None
    ) is True
    # Allowlist semantics: a rule exists for the subject but no rule matches.
    assert await engine.decide(
        "tool", "index-writer", _caller(workflow="chat"), None
    ) is False


async def test_workflow_condition_denies_a_caller_outside_any_workflow() -> None:
    engine = AccessRuleEngine(
        _StaticRepo([
            _rule(subject="index-writer", conditions={"workflow": ["ingest-data"]}),
        ]),
        cache_ttl_seconds=0.0,
    )

    assert await engine.decide("tool", "index-writer", _caller(workflow=None), None) is False
    assert await engine.decide("tool", "index-writer", None, None) is False


async def test_workflow_deny_rule_beats_allow_at_equal_priority() -> None:
    engine = AccessRuleEngine(
        _StaticRepo([
            _rule(subject="*", effect="allow"),
            _rule(subject="*", effect="deny", conditions={"workflow": ["untrusted"]}),
        ]),
        cache_ttl_seconds=0.0,
    )

    assert await engine.decide("tool", "shell", _caller(workflow="untrusted"), None) is False
    assert await engine.decide("tool", "shell", _caller(workflow="ingest-data"), None) is True


async def test_workflow_condition_combines_with_other_conditions() -> None:
    engine = AccessRuleEngine(
        _StaticRepo([
            _rule(
                subject="index-writer",
                conditions={"workflow": ["ingest-*"], "client_host": ["10.0.0.*"]},
            ),
        ]),
        cache_ttl_seconds=0.0,
    )

    assert await engine.decide(
        "tool", "index-writer", _caller(workflow="ingest-data", client_host="10.0.0.7"), None
    ) is True
    # All condition keys must match (AND).
    assert await engine.decide(
        "tool", "index-writer", _caller(workflow="ingest-data", client_host="192.168.1.1"), None
    ) is False


async def test_workflow_condition_applies_to_models_and_playbooks_too() -> None:
    engine = AccessRuleEngine(
        _StaticRepo([
            _rule(subject_type="model", subject="*", conditions={"workflow": ["ingest-*"]}),
            _rule(subject_type="playbook", subject="*", conditions={"workflow": ["ingest-*"]}),
        ]),
        cache_ttl_seconds=0.0,
    )

    assert await engine.decide("model", "gpt-4o", _caller(workflow="ingest-knowledge"), "gpt-4o")
    assert not await engine.decide("model", "gpt-4o", _caller(workflow="chat"), "gpt-4o")
    assert await engine.decide("playbook", "rag", _caller(workflow="ingest-knowledge"), None)
    assert not await engine.decide("playbook", "rag", _caller(workflow="chat"), None)


def test_every_subject_type_is_accepted_wherever_the_list_is_needed() -> None:
    """The list of subject types is derived, never restated.

    It was written out in three places: the domain type, the repository's
    validity check, and the admin form's validation. Adding `resource` to one of
    them left the other two rejecting it — the repository dropped every rule
    naming a resource as "malformed" with a warning, and the subject then read as
    ungoverned, which is the failure mode where a policy silently allows.
    """
    from typing import get_args

    from nlght.adapters.inbound.http.admin.router import _POLICY_SUBJECT_TYPES
    from nlght.adapters.outbound.persistence.access_rule_repository import _SUBJECT_TYPES

    declared = set(get_args(SubjectType))

    assert declared == {"resource", "tool", "model", "playbook"}
    assert set(_SUBJECT_TYPES) == declared, "the repository would drop rules it does not know"
    assert set(_POLICY_SUBJECT_TYPES) == declared, "the admin form would refuse to create them"


def test_the_tool_subject_is_formatted_in_one_place() -> None:
    """A separator a name can contain is one that splits in the wrong place."""
    assert tool_subject("data_store/data-main", "data-search") == (
        "data_store/data-main::data-search"
    )
    assert TOOL_SEPARATOR == "::"
