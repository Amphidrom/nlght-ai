# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from nlght.adapters.outbound.persistence.access_rule_repository import SqlAlchemyAccessRuleRepository
from nlght.adapters.outbound.persistence.models import AccessPolicyRule, Base


@pytest.fixture
async def engine():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


async def _seed(engine, **kwargs) -> uuid.UUID:
    rule_id = kwargs.pop("rule_id", uuid.uuid4())
    defaults = {
        "subject_type": "tool",
        "subject": "*",
        "effect": "allow",
        "conditions": {},
        "priority": 0,
        "enabled": True,
    }
    defaults.update(kwargs)
    async with AsyncSession(engine) as session:
        async with session.begin():
            session.add(AccessPolicyRule(rule_id=rule_id, **defaults))
    return rule_id


async def test_list_enabled_maps_rows_to_domain_rules(engine) -> None:
    rid = await _seed(
        engine,
        subject_type="model",
        subject="gpt-*",
        effect="deny",
        conditions={"header:x-org": ["acme"]},
        priority=7,
    )
    repo = SqlAlchemyAccessRuleRepository(engine)

    rules = await repo.list_enabled()

    assert len(rules) == 1
    rule = rules[0]
    assert rule.rule_id == rid
    assert rule.subject_type == "model"
    assert rule.subject == "gpt-*"
    assert rule.effect == "deny"
    assert rule.conditions == {"header:x-org": ["acme"]}
    assert rule.priority == 7
    assert rule.enabled is True


async def test_list_enabled_excludes_disabled(engine) -> None:
    await _seed(engine, enabled=False)
    repo = SqlAlchemyAccessRuleRepository(engine)

    assert await repo.list_enabled() == []


async def test_scalar_condition_values_are_normalized_to_lists(engine) -> None:
    await _seed(engine, conditions={"model": "llama3"})
    repo = SqlAlchemyAccessRuleRepository(engine)

    rules = await repo.list_enabled()

    assert rules[0].conditions == {"model": ["llama3"]}


async def test_malformed_rows_are_dropped_not_raised(engine) -> None:
    await _seed(engine, subject_type="bogus")
    await _seed(engine, effect="bogus")
    await _seed(engine, conditions={"model": 42})
    good = await _seed(engine, subject="web_*")
    repo = SqlAlchemyAccessRuleRepository(engine)

    rules = await repo.list_enabled()

    assert [r.rule_id for r in rules] == [good]
