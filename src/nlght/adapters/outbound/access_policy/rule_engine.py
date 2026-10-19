# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Evaluate stored access rules for resources, tools, models, and playbooks.

One engine caches enabled rules for the four access-policy ports. An empty
policy table allows access. Otherwise the highest-priority matching rule
decides, with denials winning ties. A subject with rules but no matching
conditions is denied (ADR-0022).
"""

from __future__ import annotations

import asyncio
import logging
import time
from fnmatch import fnmatchcase
from typing import TYPE_CHECKING

from nlght.core.access.rule import AccessRule, SubjectType, tool_subject
from nlght.ports.outbound.access_rule_repository import AccessRuleRepository

if TYPE_CHECKING:
    from nlght.core.entry.context import RequestContext
    from nlght.core.runtime.resource import ResourceDef

logger = logging.getLogger(__name__)

_KNOWN_CONDITION_KEYS = ("model", "client_host", "workflow")
_HEADER_PREFIX = "header:"


class AccessRuleEngine:
    """Evaluate stored access rules for the policy port facades."""

    def __init__(
        self,
        repository: AccessRuleRepository,
        *,
        cache_ttl_seconds: float = 5.0,
    ) -> None:
        self._repository = repository
        self._cache_ttl = cache_ttl_seconds
        self._cache: list[AccessRule] | None = None
        self._cache_at: float = 0.0
        self._lock = asyncio.Lock()

    def for_resources(self) -> ResourceRuleAccessPolicy:
        return ResourceRuleAccessPolicy(self)

    def for_tools(self) -> ToolRuleAccessPolicy:
        return ToolRuleAccessPolicy(self)

    def for_models(self) -> ModelRuleAccessPolicy:
        return ModelRuleAccessPolicy(self)

    def for_playbooks(self) -> PlaybookRuleAccessPolicy:
        return PlaybookRuleAccessPolicy(self)

    async def decide(
        self,
        subject_type: SubjectType,
        subject_name: str,
        caller: RequestContext | None,
        model: str | None,
    ) -> bool:
        rules = await self._rules()
        if not rules:
            return True

        candidates = [
            r for r in rules
            if r.subject_type == subject_type and fnmatchcase(subject_name, r.subject)
        ]
        if not candidates:
            return True

        candidates.sort(key=lambda r: (-r.priority, r.effect != "deny"))
        for rule in candidates:
            if self._conditions_match(rule, caller, model):
                allowed = rule.effect == "allow"
                if not allowed:
                    logger.info(
                        "access_policy.deny | type=%s subject=%s rule=%s",
                        subject_type, subject_name, rule.rule_id,
                    )
                return allowed

        logger.info(
            "access_policy.deny_unmatched | type=%s subject=%s rules=%d",
            subject_type, subject_name, len(candidates),
        )
        return False

    def invalidate_cache(self) -> None:
        """Drop cached rules so the next check reads the repository."""
        self._cache = None
        self._cache_at = 0.0

    # -- internals -----------------------------------------------------------

    def _conditions_match(
        self,
        rule: AccessRule,
        caller: RequestContext | None,
        model: str | None,
    ) -> bool:
        for key, patterns in rule.conditions.items():
            actual = self._condition_value(key, rule, caller, model)
            if actual is None:
                return False
            if not any(fnmatchcase(actual, p) for p in patterns):
                return False
        return True

    def _condition_value(
        self,
        key: str,
        rule: AccessRule,
        caller: RequestContext | None,
        model: str | None,
    ) -> str | None:
        if key == "model":
            return model
        if key == "client_host":
            return caller.client_host if caller is not None else None
        if key == "workflow":
            return caller.workflow if caller is not None else None
        if key.startswith(_HEADER_PREFIX):
            if caller is None:
                return None
            wanted = key[len(_HEADER_PREFIX):].strip().lower()
            for name, value in caller.headers.items():
                if name.lower() == wanted:
                    return value
            return None
        logger.warning(
            "access_policy.unknown_condition | rule=%s key=%r (known: %s, %s<name>): treating as unmatched",
            rule.rule_id, key, ", ".join(_KNOWN_CONDITION_KEYS), _HEADER_PREFIX,
        )
        return None

    async def _rules(self) -> list[AccessRule]:
        now = time.monotonic()
        if self._cache is not None and now - self._cache_at < self._cache_ttl:
            return self._cache
        async with self._lock:
            now = time.monotonic()
            if self._cache is not None and now - self._cache_at < self._cache_ttl:
                return self._cache
            self._cache = await self._repository.list_enabled()
            self._cache_at = now
            return self._cache


class ResourceRuleAccessPolicy:
    """ResourceAccessPolicy facade for a resource address.

    Authorizes the resource as a whole: whether this caller may activate it at
    all. That is the only question for a resource used inside a workflow step,
    and the first of two for one whose signatures a model may call.
    """

    def __init__(self, engine: AccessRuleEngine) -> None:
        self._engine = engine

    async def is_allowed(
        self,
        resource: ResourceDef,
        caller: RequestContext | None,
        model: str | None,
    ) -> bool:
        return await self._engine.decide("resource", resource.address, caller, model)


class ToolRuleAccessPolicy:
    """ToolAccessPolicy facade for one signature of one resource.

    This used to decide on `resource.name`, which meant a rule named a tool and
    authorized a whole resource. It now decides on
    `<kind>/<name>::<signature>`, so a rule can allow one operation of one
    activation without allowing the rest.

    It is the second of two checks, never the only one: a resource the caller
    may not activate offers nothing to allow, and no tool rule can put it back.
    """

    def __init__(self, engine: AccessRuleEngine) -> None:
        self._engine = engine

    async def is_allowed(
        self,
        resource: ResourceDef,
        signature_name: str,
        caller: RequestContext | None,
        model: str | None,
    ) -> bool:
        subject = tool_subject(resource.address, signature_name)
        return await self._engine.decide("tool", subject, caller, model)


class ModelRuleAccessPolicy:
    """ModelAccessPolicy facade for a model name.

    A ``model`` condition on a model rule matches against the model name,
    which lets e.g. ``subject='*'`` rules stay expressible. Header/host
    conditions apply to the caller as usual.
    """

    def __init__(self, engine: AccessRuleEngine) -> None:
        self._engine = engine

    async def is_allowed(
        self,
        model_name: str,
        caller: RequestContext,
    ) -> bool:
        return await self._engine.decide("model", model_name, caller, model_name)


class PlaybookRuleAccessPolicy:
    """Playbook access policy keyed by playbook name."""



    def __init__(self, engine: AccessRuleEngine) -> None:
        self._engine = engine

    async def is_allowed(
        self,
        playbook_name: str,
        caller: RequestContext | None,
        model: str | None,
    ) -> bool:
        return await self._engine.decide("playbook", playbook_name, caller, model)
