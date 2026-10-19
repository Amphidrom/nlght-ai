# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Every context record says where its content came from (ADR-0074)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import SimpleNamespace

from nlght.core.context.elements import passage_elements
from nlght.core.context.passage import ContextPassage
from nlght.core.hive_mind.model_context import ModelContextBuilder, PromptInputPolicy
from nlght.core.hive_mind.models import MentalModel, TurnSummary
from nlght.core.model.trust import (
    PLATFORM_TEXT,
    UNANNOTATED,
    ContentLedger,
    ContentOrigin,
    ContentTrust,
    SourceRef,
)
from nlght.core.retrieval.hit import Provenance

HANDBOOK = SourceRef("data_source", "store-a", "handbook", ContentTrust.TRUSTED)
WEB = SourceRef("web", "fetch-1", "", ContentTrust.UNTRUSTED)


@dataclass
class _Ctx:
    ledger: ContentLedger = field(default_factory=ContentLedger)
    tools: object = None
    playbooks: object = None


def _records(model: MentalModel, ctx: object = None) -> dict[str, object]:
    records = ModelContextBuilder(ctx=ctx).build(  # type: ignore[arg-type]
        model,
        input_policy=PromptInputPolicy(
            include_conversation_store=True,
            include_current_request_interpretation=True,
        ),
    )
    return {record.record_id: record.provenance.origin for record in records}


def test_a_passage_keeps_the_label_its_store_gave_it() -> None:
    passage = ContextPassage(
        text="TLS is required.",
        carrier="chunk",
        carrier_id="c1",
        found_by=("vector",),
        rank=1,
        provenance=Provenance(document_id="d1", origin=ContentOrigin.of(HANDBOOK)),
    )
    model = MentalModel(
        turn_id="t", built_at=datetime.now(UTC),
        elements=tuple(passage_elements([passage])),
    )

    origins = _records(model)

    assert origins["passage:chunk:c1"] == ContentOrigin.of(HANDBOOK)


def test_a_remembered_turn_keeps_its_stamp_and_an_unstamped_one_is_unannotated() -> None:
    model = MentalModel(
        turn_id="t", built_at=datetime.now(UTC),
        recent_turns=[
            TurnSummary(turn_nr=1, user_input="a", intent="i", topic="t",
                        origin=ContentOrigin.of(WEB)),
            TurnSummary(turn_nr=2, user_input="b", intent="i", topic="t"),
        ],
    )

    origins = _records(model)

    assert origins["turn:1"] == ContentOrigin.of(WEB)
    assert origins["turn:2"] == UNANNOTATED


def test_an_interpretation_inherits_what_the_session_remembers_and_the_run_read() -> None:
    model = MentalModel(
        turn_id="t", built_at=datetime.now(UTC),
        intents=["deploy"],
        recent_turns=[TurnSummary(turn_nr=1, user_input="a", intent="i", topic="t",
                                  origin=ContentOrigin.of(WEB))],
    )
    ctx = _Ctx()
    ctx.ledger.record(ContentOrigin.of(HANDBOOK))

    origins = _records(model, ctx)

    assert set(origins["request:intents"].sources) == {HANDBOOK, WEB}  # type: ignore[attr-defined]


class _Tools:
    def all(self) -> list[object]:
        return [SimpleNamespace(name="data-search", parameters=[], description="Search.")]

    def __bool__(self) -> bool:
        return True


class _Playbooks:
    def all(self) -> list[object]:
        return []

    def to_contract(self) -> str:
        return "- triage: sort incoming tickets"

    def __bool__(self) -> bool:
        return True


def test_tool_guidance_is_platform_text_and_playbook_text_is_untrusted() -> None:
    ctx = _Ctx(tools=_Tools(), playbooks=_Playbooks())
    records = ModelContextBuilder(ctx=ctx).build(  # type: ignore[arg-type]
        MentalModel(turn_id="t", built_at=datetime.now(UTC)),
        input_policy=PromptInputPolicy(include_tools=True, include_playbooks=True),
    )

    origins = {record.record_id: record.provenance.origin for record in records}

    assert origins["runtime:tools"] == PLATFORM_TEXT
    assert origins["runtime:playbooks"].trust is ContentTrust.UNTRUSTED
    assert origins["runtime:playbooks"].sources[0].kind == "playbook"
