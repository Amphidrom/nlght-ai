# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import logging
from typing import cast

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from nlght.adapters.outbound.persistence.models import AccessPolicyRule
from nlght.core.access.rule import AccessRule, Effect, SubjectType
from nlght.ports.outbound.access_rule_repository import AccessRuleRepository

logger = logging.getLogger(__name__)

_SUBJECT_TYPES = {"tool", "model", "playbook"}
_EFFECTS = {"allow", "deny"}


def _to_access_rule(row: AccessPolicyRule) -> AccessRule | None:
    """Map a DB row to the domain rule; malformed rows are dropped with a warning.

    Dropping (instead of raising) keeps one bad row from disabling the whole
    policy engine; the affected subject stays governed by the remaining rules.
    """
    if row.subject_type not in _SUBJECT_TYPES or row.effect not in _EFFECTS:
        logger.warning(
            "access_rule.malformed | rule_id=%s subject_type=%r effect=%r — ignoring",
            row.rule_id, row.subject_type, row.effect,
        )
        return None
    conditions: dict[str, list[str]] = {}
    for key, value in (row.conditions or {}).items():
        if isinstance(value, str):
            conditions[str(key)] = [value]
        elif isinstance(value, list):
            conditions[str(key)] = [str(v) for v in value]
        else:
            logger.warning(
                "access_rule.malformed_condition | rule_id=%s key=%r — ignoring rule",
                row.rule_id, key,
            )
            return None
    return AccessRule(
        rule_id=row.rule_id,
        subject_type=cast(SubjectType, row.subject_type),
        subject=row.subject,
        effect=cast(Effect, row.effect),
        conditions=conditions,
        priority=row.priority,
        enabled=row.enabled,
    )


class SqlAlchemyAccessRuleRepository(AccessRuleRepository):
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def list_enabled(self) -> list[AccessRule]:
        async with AsyncSession(self._engine) as session:
            result = await session.execute(
                select(AccessPolicyRule).where(AccessPolicyRule.enabled.is_(True))
            )
            rules = (_to_access_rule(row) for row in result.scalars())
            return [r for r in rules if r is not None]
