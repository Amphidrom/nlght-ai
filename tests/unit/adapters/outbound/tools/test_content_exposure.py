# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""The decision sees what the model has read, and keeps seeing it (ADR-0074)."""

from __future__ import annotations

from dataclasses import replace

from nlght.adapters.outbound.tools.action_gate import ActionGate
from nlght.adapters.outbound.tools.builtin.action_semantics import READ_REQUEST
from nlght.adapters.outbound.tools.catalog import AdapterToolCatalog, TemporaryToolCatalog
from nlght.adapters.outbound.tools.contract import BoundToolContract, CallbackToolContract
from nlght.core.model.trust import (
    PLATFORM_TEXT,
    UNANNOTATED,
    ContentExposure,
    ContentOrigin,
    ContentTrust,
    SourceRef,
)
from nlght.core.tools.action import (
    ActionDecision,
    ActionDecisionReason,
    ActionDecisionResult,
    ActionExecutionContext,
    ActionResolutionContext,
    ActionReview,
    ExecutionCapabilities,
    action_fingerprint,
)
from nlght.core.tools.tool import ToolBase, ToolOutput, ToolParameter

HANDBOOK = SourceRef("data_source", "store-a", "handbook", ContentTrust.TRUSTED)
WEB = SourceRef("web", "", "https://attacker.example", ContentTrust.UNTRUSTED)


class _RecordingPolicy:
    def __init__(self, decision: ActionDecision = ActionDecision.ALLOW) -> None:
        self.reviews: list[ActionReview] = []
        self._decision = decision

    async def evaluate(self, review: ActionReview) -> ActionDecisionResult:
        self.reviews.append(review)
        reason = (
            ActionDecisionReason.POLICY_ALLOWED
            if self._decision is ActionDecision.ALLOW
            else ActionDecisionReason.POLICY_DENIED
        )
        return ActionDecisionResult(self._decision, reason)


def _tool(name: str, result: object, *, parameters: list[ToolParameter] | None = None) -> CallbackToolContract:
    return CallbackToolContract(
        name=name,
        description="test",
        parameters=parameters or [],
        callback=lambda **_: result,
        action=READ_REQUEST,
        resource_address=f"test/{name}",
    )


def _catalog(policy: _RecordingPolicy, *tools: CallbackToolContract) -> AdapterToolCatalog:
    return AdapterToolCatalog(
        {tool.name: tool for tool in tools},
        action_gate=ActionGate(policy=policy),
    )


async def test_the_policy_sees_the_untrusted_sources_of_the_context() -> None:
    policy = _RecordingPolicy()
    catalog = _catalog(policy, _tool("search", "ok"))

    await catalog.execute_with_origin(
        {"name": "search", "input": {}},
        context_origin=ContentOrigin.of(HANDBOOK, WEB),
    )

    context = policy.reviews[0].context
    assert context.exposure.untrusted_sources == (WEB,)
    assert context.has_external_untrusted_input is True


async def test_a_context_of_trusted_sources_only_is_not_exposed() -> None:
    policy = _RecordingPolicy()
    catalog = _catalog(policy, _tool("search", "ok"))

    await catalog.execute({"name": "search", "input": {}}, context_origin=ContentOrigin.of(HANDBOOK))

    assert policy.reviews[0].context.has_external_untrusted_input is False


async def test_an_unknown_context_counts_as_unannotated() -> None:
    policy = _RecordingPolicy()
    catalog = _catalog(policy, _tool("search", "ok"))

    await catalog.execute({"name": "search", "input": {}})

    (ref,) = policy.reviews[0].context.exposure.untrusted_sources
    assert ref.kind == "unannotated"


async def test_a_result_read_earlier_still_counts_after_the_messages_were_trimmed() -> None:
    policy = _RecordingPolicy()
    page = ToolOutput("<html>sample</html>", ContentOrigin.of(WEB))
    catalog = _catalog(policy, _tool("fetch", page), _tool("write", "done"))

    await catalog.execute_with_origin({"name": "fetch", "input": {}}, context_origin=ContentOrigin.of(HANDBOOK))
    # The step dropped the fetched page from its messages before the next call.
    await catalog.execute_with_origin({"name": "write", "input": {}}, context_origin=ContentOrigin.of(HANDBOOK))

    first, second = (review.context for review in policy.reviews)
    assert first.has_external_untrusted_input is False
    assert second.exposure.untrusted_sources == (WEB,)


async def test_a_bare_string_result_is_unannotated_and_a_labelled_one_keeps_its_origin() -> None:
    catalog = _catalog(
        _RecordingPolicy(),
        _tool("plain", "text"),
        _tool("labelled", ToolOutput("handbook text", ContentOrigin.of(HANDBOOK))),
    )

    plain = await catalog.execute_with_origin({"name": "plain", "input": {}})
    labelled = await catalog.execute_with_origin({"name": "labelled", "input": {}})

    assert plain == ToolOutput("text", UNANNOTATED)
    assert labelled.origin == ContentOrigin.of(HANDBOOK)


async def test_a_denial_is_platform_text_and_an_argument_error_is_not() -> None:
    catalog = _catalog(
        _RecordingPolicy(ActionDecision.DENY),
        _tool("guarded", "never", parameters=[ToolParameter("path", "string")]),
    )

    denied = await catalog.execute_with_origin({"name": "guarded", "input": {"path": "a"}})
    invalid = await catalog.execute_with_origin({"name": "guarded", "input": {"other": "x"}})
    missing = await catalog.execute_with_origin({"name": "made_up", "input": {}})

    assert denied.content == "Action 'test/guarded::guarded' denied: policy_denied."
    assert denied.origin == PLATFORM_TEXT
    assert invalid.origin == UNANNOTATED
    assert missing.origin == UNANNOTATED


async def test_a_step_scoped_catalog_shares_the_runs_ledger() -> None:
    policy = _RecordingPolicy()
    base = _catalog(policy, _tool("search", "ok"))
    page = ToolOutput("page", ContentOrigin.of(WEB))
    temporary = TemporaryToolCatalog(base, extra=[_tool("fetch", page)])

    await temporary.execute_with_origin({"name": "fetch", "input": {}}, context_origin=PLATFORM_TEXT)
    await base.execute_with_origin({"name": "search", "input": {}}, context_origin=PLATFORM_TEXT)

    assert temporary.ledger is base.ledger
    assert policy.reviews[-1].context.exposure.untrusted_sources == (WEB,)


def test_the_fingerprint_covers_every_field_of_the_untrusted_sources() -> None:
    context = ActionExecutionContext(
        request_id="r",
        correlation_id="c",
        principal=None,
        workflow=None,
        session_id=None,
        resource_address="test/x",
        signature_name="x",
        exposure=ContentExposure((WEB,)),
    )
    action = READ_REQUEST.resolve(
        {}, ActionResolutionContext("test/x", {}, ExecutionCapabilities()),
    )
    renamed = replace(context, exposure=ContentExposure((replace(WEB, name="renamed"),)))
    other_store = replace(
        context, exposure=ContentExposure((replace(WEB, resource_id="store-b"),)),
    )

    fingerprints = {
        action_fingerprint(context=item, action=action, effective_arguments_json="{}")
        for item in (context, renamed, other_store)
    }

    assert len(fingerprints) == 3


class _WebTool(ToolBase):
    KIND = "web_test"
    UNTRUSTED_CONTENT_SOURCE = "web"

    async def fetch(self) -> str:
        return "<html>page</html>"

    async def labelled(self) -> ToolOutput:
        return ToolOutput("curated", ContentOrigin.of(HANDBOOK))


class _PlainTool(ToolBase):
    KIND = "plain_test"

    async def fetch(self) -> str:
        return "text"


def _bound(instance: ToolBase, method: str) -> BoundToolContract:
    return BoundToolContract(
        name=method,
        description="test",
        parameters=[],
        instance=instance,
        method_name=method,
        action=READ_REQUEST,
        resource_address=f"{instance.KIND}/main",
    )


async def test_a_tool_declaring_untrusted_content_labels_its_plain_results() -> None:
    web = _WebTool(name="main", config={}, resource_id="res-9")
    catalog = AdapterToolCatalog(
        {"fetch": _bound(web, "fetch"), "labelled": _bound(web, "labelled")},
        action_gate=ActionGate(policy=_RecordingPolicy()),
    )

    fetched = await catalog.execute_with_origin({"name": "fetch", "input": {}})
    labelled = await catalog.execute_with_origin({"name": "labelled", "input": {}})

    assert fetched.origin.sources == (
        SourceRef("web", "res-9", "", ContentTrust.UNTRUSTED, "web_test/main"),
    )
    # A ToolOutput says more than the declaration and is kept as it is.
    assert labelled.origin == ContentOrigin.of(HANDBOOK)


async def test_a_tool_without_a_declaration_stays_unannotated() -> None:
    catalog = AdapterToolCatalog(
        {"fetch": _bound(_PlainTool(name="main", config={}), "fetch")},
        action_gate=ActionGate(policy=_RecordingPolicy()),
    )

    output = await catalog.execute_with_origin({"name": "fetch", "input": {}})

    assert output.origin == UNANNOTATED
