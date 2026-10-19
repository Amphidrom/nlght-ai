# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Compose trusted instructions with budgeted, untrusted model context.

Retrieved passages and optional session memory enter the same element reduction.
``ModelContextBuilder`` renders selected knowledge as untrusted records, while
``SystemPromptBuilder`` supplies deployment-authored instructions. The step
writes the composed messages to ``ctx.messages`` for a later model call.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from nlght.core.context import ContextBudget, build_context
from nlght.core.context.elements import passage_elements
from nlght.core.hive_mind.builder import ContextSnapshot, MentalModelBuilder
from nlght.core.hive_mind.model_context import (
    ModelContextBuilder,
    PromptInputPolicy,
)
from nlght.core.hive_mind.models import MentalModel
from nlght.core.hive_mind.relevance import RelevanceEngine
from nlght.core.hive_mind.system_prompt import DEFAULT_CONSTRAINTS, SystemPromptBuilder
from nlght.core.model.budget import TokenBudget
from nlght.core.model.messages import (
    compose_messages,
    compose_prompt,
    normalize_workflow_messages,
)
from nlght.core.workflow.step import (
    StepBase,
    StepOption,
    StepResult,
    WorkflowStepContext,
)

logger = logging.getLogger(__name__)

DEFAULT_EGO = "You are a precise assistant. Answer from what you are given."

#: How much passage text may be selected before anything is weighed. A cap on
#: *selection*, in characters, and not a budget (ADR-0052): the model's context
#: window is `prompt_budget`'s business and is applied afterwards, over every
#: element at once.
DEFAULT_MAX_PASSAGE_CHARS = 20000


class PromptComposeStep(StepBase):
    """Composes trusted framing and untrusted retrieval/session context.

    Step config:
        max_passage_chars: cap on selected passage text (characters)
        max_system_prompt_tokens: the prompt budget, when no model budget is bound
        include_memory: whether to read session memory at all (default true)
    """

    TYPE = "prompt.compose"

    @classmethod
    def options(cls) -> list[StepOption]:
        return [
            StepOption("max_passage_chars", "string",
                       "Characters of retrieved passage text to select before weighing",
                       placeholder=str(DEFAULT_MAX_PASSAGE_CHARS)),
            StepOption("max_system_prompt_tokens", "string",
                       "Prompt budget in estimated tokens; the reduction decides "
                       "what fits inside it"),
            StepOption("include_memory", "string",
                       "Whether to read session memory (default: true)"),
        ]

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        model = self._mental_model(ctx)
        passages = self._passages(ctx)
        model = replace(model, elements=tuple(passage_elements(passages)))

        if self.config.get("ego") not in (None, "", DEFAULT_EGO):
            raise ValueError(
                "prompt.compose ego is runtime configuration and cannot author trusted instructions"
            )

        trusted_instructions = SystemPromptBuilder().build(
            DEFAULT_EGO,
            constraints=list(DEFAULT_CONSTRAINTS),
        )
        context = ModelContextBuilder(ctx=ctx).build(
            model,
            input_policy=PromptInputPolicy(
                include_conversation_store=True,
                include_session_result_store=True,
                include_current_request_interpretation=True,
            ),
            # The client that will make the call is the authority on its own
            # budget, and it is passed through rather than read: whatever
            # `prompt_budget` means is settled in one place (ADR-0012), and this
            # step must not acquire a second opinion about it.
            token_budget=self._authority(ctx),
            max_prompt_tokens=self._configured_budget(ctx),
            reserved_instruction_tokens=max(1, int(len(trusted_instructions) / 3.5)),
        )

        envelope = compose_prompt(trusted_instructions, context)
        ctx.messages = compose_messages(
            envelope,
            normalize_workflow_messages(ctx.messages),
        )
        ctx.metadata["prompt.trusted_instruction_chars"] = len(trusted_instructions)
        ctx.metadata["prompt.context_records"] = len(context)
        logger.info(
            "[%s] prompt.compose.done | passages=%d memory=%s trusted_chars=%d context_records=%d",
            ctx.correlation_id, len(passages),
            "yes" if model.recent_turns or model.known_results else "no",
            len(trusted_instructions), len(context),
        )
        return StepResult(ctx=ctx, verdict=self.Verdict.DEFAULT)

    @staticmethod
    def _authority(ctx: WorkflowStepContext) -> TokenBudget | None:
        """Read the bound model's authoritative token budget.

        Args:
            ctx (WorkflowStepContext): Context containing the bound model.

        Returns:
            TokenBudget | None: Model budget, or ``None`` without a bound model.
        """
        return ctx.llm.token_budget if ctx.llm is not None else None

    def _configured_budget(self, ctx: WorkflowStepContext) -> int | None:
        """Read the configured prompt limit when the model has no known budget.

        Args:
            ctx (WorkflowStepContext): Context containing the bound model.

        Returns:
            int | None: Configured token limit, or ``None`` when the model
            supplies the authoritative budget or no limit is configured.

        Side Effects:
            Logs a configured limit that the bound model supersedes.
        """
        raw = self.config.get("max_system_prompt_tokens")
        configured = int(raw) if raw not in (None, "") else None
        authority = self._authority(ctx)
        if configured is not None and authority is not None:
            logger.info(
                "[%s] prompt.compose.budget | using the bound model's %s and not the "
                "configured %s",
                ctx.correlation_id, authority.prompt_budget, configured,
            )
            return None
        return configured

    def _passages(self, ctx: WorkflowStepContext) -> list[Any]:
        """Select retrieved passages within the configured character limit.

        Args:
            ctx (WorkflowStepContext): Context holding retrieval hits.

        Returns:
            list[Any]: Selected passages, or an empty list without hits.
        """
        hits = ctx.metadata.get("retrieval.hits")
        if not hits:
            return []
        raw = self.config.get("max_passage_chars")
        selection = build_context(
            hits,
            ContextBudget(max_chars=int(raw) if raw not in (None, "") else
                          DEFAULT_MAX_PASSAGE_CHARS),
        )
        return list(selection.passages)

    def _mental_model(self, ctx: WorkflowStepContext) -> MentalModel:
        """Build a mental model from recent conversation turns.

        Args:
            ctx (WorkflowStepContext): Context with an optional store coordinator.

        Returns:
            MentalModel: Scored recent turns, or an empty model when memory is
            disabled, unavailable, or unbound.

        Side Effects:
            Reads recent turns and logs failures to read session memory.
        """
        coordinator = ctx.store_coordinator
        wanted = str(self.config.get("include_memory", "true")).lower() != "false"
        if coordinator is None or not wanted:
            return MentalModel(turn_id=ctx.correlation_id, built_at=datetime.now(UTC))

        try:
            snapshot = ContextSnapshot(
                recent_turns=list(coordinator.get_recent_turns(5)),
            )
        except Exception:
            logger.warning(
                "[%s] prompt.compose.memory_unavailable: composing without it",
                ctx.correlation_id, exc_info=True,
            )
            return MentalModel(turn_id=ctx.correlation_id, built_at=datetime.now(UTC))

        # The engine's own defaults: this step chooses what is *in* the model,
        # never how relevant any of it is.
        return MentalModelBuilder(relevance_engine=RelevanceEngine()).build(
            turn_id=ctx.correlation_id,
            context=snapshot,
            signal_entities=[],
            intents=[],
        )
