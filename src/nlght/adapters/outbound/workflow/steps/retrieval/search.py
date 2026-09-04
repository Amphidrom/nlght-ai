# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Ask the index a question, across whichever stores this platform has.

The step owns the round trips and the translation, and no judgement. What to
ask, how deep, and how the answers are ranked is decided in `core.retrieval` —
pure, and tested without a store.

    plan    →  requested ∩ available, and whether the vector call is worth it
    routes  →  one call per store, each failure isolated
    rank    →  positions within each store are fused; scores never are
    hits    →  three granularities, kept apart, ranked together

Search produces *findings*: these systems found these things. Which of them
becomes text a model is shown is a separate question — it needs a token budget
and a context window — and it is not answered here.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from nlght.core.errors.errors import WorkflowConfigurationError
from nlght.core.model.messages import UserMessage, normalize_workflow_messages
from nlght.core.retrieval import (
    ASSERTION,
    KNOWLEDGE,
    LEXICAL,
    SOURCES,
    VECTOR,
    FusionWeights,
    Provenance,
    RetrievalHit,
    RetrievalPlanError,
    fuse,
    plan_from_config,
    rank_within_sources,
)
from nlght.core.workflow.step import (
    StepBase,
    StepOption,
    StepResult,
    WorkflowStepContext,
)

if TYPE_CHECKING:
    from nlght.adapters.outbound.stores.connections import StoreConnections
    from nlght.adapters.outbound.tools.activator import ResourceActivator
    from nlght.adapters.outbound.tools.loader import ToolLoader
    from nlght.core.entry.context import RequestContext
    from nlght.ports.outbound.embedding_client import EmbeddingClient
    from nlght.ports.outbound.resource_repository import ResourceRepository

logger = logging.getLogger(__name__)

NOTHING_FOUND = "empty"
"""Verdict when every searched store answered and none of them held anything."""

#: Which activation answers each capability. Two stores, three capabilities:
#: `data_store` spans keyword and semantic discovery over one corpus, and a
#: capability is not a store — splitting them into a store each is what the
#: retired `lexical_store`/`vector_store` kinds did, and it made the corpus the
#: ingestion writes and the corpus retrieval reads two configurations that
#: nothing kept aligned.
_KIND = {
    LEXICAL: "data_store",
    VECTOR: "data_store",
    KNOWLEDGE: "knowledge_store",
}

#: The step option naming each activation. Both discovery capabilities come
#: from the one `data_store`, so they are named once.
_OPTION = {
    LEXICAL: "data_store",
    VECTOR: "data_store",
    KNOWLEDGE: "knowledge_store",
}


class RetrievalSearchStep(StepBase):
    """Search the index and leave ranked findings in the step context.

    Reads ``retrieval.question`` from the context metadata, or the last user
    message where no step put one there. Writes ``retrieval.hits`` and
    ``retrieval.stats``.

    Step config:
        sources:          stores to search (default: those with an activation)
        required_sources: stores whose absence is a failure (default: none)
        fetch_k:          how deep each store is asked
        fusion_k:         how many of each store's hits take part in the ranking
        final_k:          how many findings leave the step
        source_weights:   what each store's opinion is worth in the fusion
        damping:          how hard the top of each store's list is damped
        kinds:            knowledge assertion kinds to query
        lexical_floor / lexical_ceiling: when the vector call is worth making
        data_store / knowledge_store: resource names
    """

    TYPE = "retrieval.search"

    def __init__(
        self,
        *,
        config: dict[str, Any],
        resource_repository: ResourceRepository | None = None,
        resource_activator: ResourceActivator | None = None,
        tool_loader: ToolLoader | None = None,
        store_connections: StoreConnections | None = None,
        embedding_client: EmbeddingClient | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(config=config, **kwargs)
        self._resources = resource_repository
        self._activator = resource_activator
        self._tools = tool_loader
        self._connections = store_connections
        self._embedding = embedding_client

    @classmethod
    def options(cls) -> list[StepOption]:
        return [
            StepOption("sources", "array",
                       "Stores to search: lexical, vector, knowledge. "
                       "Defaults to the ones this step names an activation for.",
                       placeholder='["lexical", "vector", "knowledge"]'),
            StepOption("required_sources", "array",
                       "Stores whose absence fails the step rather than being reported",
                       placeholder='["knowledge"]'),
            StepOption(
                "fetch_k", "object",
                "Per-source fetch depth before fusion. Raw JSON also accepts one integer "
                "for all sources.",
                default={LEXICAL: 50, VECTOR: 50, KNOWLEDGE: 20},
            ),
            StepOption("final_k", "integer", "Findings returned by the step", default=20),
            StepOption("fusion_k", "integer",
                       "Hits per store taking part in the ranking; 0 means all fetched",
                       default=0),
            StepOption(
                "source_weights", "object",
                "Relative contribution of each source to reciprocal-rank fusion",
                default={LEXICAL: 1.0, VECTOR: 1.0, KNOWLEDGE: 1.0},
            ),
            StepOption("damping", "number",
                       "How hard the top of each store's list is damped in the fusion",
                       default=60.0),
            StepOption(
                "kinds", "array", "Knowledge assertion kinds to query",
                default=["fact", "rule", "pattern", "decision"],
            ),
            StepOption("lexical_floor", "integer",
                       "Below this many lexical hits the vector store is asked as well",
                       default=1),
            StepOption("lexical_ceiling", "integer",
                       "Above this many lexical hits the vector store is asked as well",
                       default=100),
            StepOption("data_store", "string",
                       "Resource name of the data store activation; answers both "
                       "keyword and semantic discovery over the document corpus"),
            StepOption("knowledge_store", "string",
                       "Resource name of the knowledge store activation"),
        ]

    async def _activate(self, source: str, caller: RequestContext | None) -> Any:  # noqa: ANN401
        """The store activation this workflow named, or `None` where it named none.

        Named per source rather than discovered, so a workflow searching one
        index cannot pick up another's.

        Three outcomes, and they are not the same thing. Naming nothing means
        this platform does not have that capability — reported, never an error.
        Naming something that does not resolve is a misconfiguration. Naming
        something this workflow may not use is a denial, and it raises: the
        workflow asked for it.
        """
        name = str(self.config.get(_OPTION[source], "")).strip()
        if not name:
            return None
        if self._activator is None:
            raise WorkflowConfigurationError(
                f"Step '{self.TYPE}' has no resource activator, so it cannot reach '{name}'."
            )
        return await self._activator.activate(
            kind=_KIND[source], name=name, caller=caller
        )

    async def _stores(self, caller: RequestContext | None) -> dict[str, Any]:
        """Every capability this platform actually has, by source.

        One activation can answer two capabilities: `data_store` serves both
        keyword and semantic discovery over one corpus, so naming it makes both
        available and it is activated once. That is also why the two cannot be
        available separately any more — they are two questions asked of one
        store, not two stores, and a plan requiring one without the other has
        nothing left to distinguish.
        """
        activated: dict[str, Any] = {}
        cache: dict[str, Any] = {}
        for source in SOURCES:
            name = str(self.config.get(_OPTION[source], "")).strip()
            if not name:
                continue
            if name not in cache:
                store = await self._activate(source, caller)
                if store is None:
                    continue
                cache[name] = store
            activated[source] = cache[name]
        return activated

    def _question(self, ctx: WorkflowStepContext) -> str:
        stated = str(ctx.metadata.get("retrieval.question", "") or "").strip()
        if stated:
            return stated
        for message in reversed(normalize_workflow_messages(ctx.messages or [])):
            if isinstance(message, UserMessage):
                return str(message.content or "").strip()
        return ""

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        stores = await self._stores(ctx.trigger.context)
        try:
            plan = plan_from_config(self.config, available=tuple(stores))
            weights = FusionWeights.from_config(self.config)
        except (RetrievalPlanError, ValueError) as error:
            raise WorkflowConfigurationError(f"Step '{self.TYPE}': {error}") from error

        if not plan.sources:
            raise WorkflowConfigurationError(
                f"Step '{self.TYPE}' has no store to search. Name at least one of "
                f"{', '.join(sorted(_KIND.values()))}, or narrow 'sources' to one it has."
            )

        question = self._question(ctx)
        if not question:
            raise WorkflowConfigurationError(
                f"Step '{self.TYPE}' has no question to ask: set 'retrieval.question' "
                f"in the step context, or place it after a step that carries user messages."
            )

        started = time.monotonic()
        stats: dict[str, Any] = {
            "question": question,
            "sources": list(plan.sources),
            "unavailable": list(plan.skipped),
            "routes": {},
        }

        def note(source: str, **values: Any) -> None:  # noqa: ANN401
            stats["routes"].setdefault(source, {}).update(values)

        found: list[RetrievalHit] = []
        lexical_hits = 0
        if LEXICAL in plan.sources:
            hits = await self._lexical(stores[LEXICAL], question, plan.fetch_for(LEXICAL), note)
            lexical_hits = len(hits)
            found.extend(hits)

        if plan.wants_vector(lexical_hits):
            found.extend(
                await self._vector(stores[VECTOR], question, plan.fetch_for(VECTOR), note)
            )
        elif VECTOR in plan.sources:
            note(VECTOR, asked=False, because="lexical_answered")

        if KNOWLEDGE in plan.sources:
            found.extend(await self._knowledge(stores[KNOWLEDGE], question, plan, note))

        ranked = rank_within_sources(found)
        if plan.fusion_k > 0:
            ranked = [hit for hit in ranked if hit.source_rank <= plan.fusion_k]
        fused = fuse(ranked, weights=weights)[: plan.final_k]

        stats["found"] = len(found)
        stats["returned"] = len(fused)
        stats["duration_ms"] = int((time.monotonic() - started) * 1000)
        ctx.metadata["retrieval.hits"] = tuple(fused)
        ctx.metadata["retrieval.stats"] = stats

        logger.info(
            "[%s] retrieval.search | sources=%s found=%d returned=%d ms=%d",
            ctx.correlation_id, ",".join(plan.sources), len(found), len(fused),
            stats["duration_ms"],
        )
        return StepResult(ctx=ctx, verdict=None if fused else NOTHING_FOUND)

    async def _lexical(
        self, store: Any, question: str, limit: int, note: Any  # noqa: ANN401
    ) -> list[RetrievalHit]:
        """Keyword discovery, as candidates the store has already vouched for.

        The store returns findings rather than backend rows, and everything it
        returns is on the published revision — it drops the rest before anything
        here can rank it (ADR-0062). So there is no translation left to do, and
        nothing here re-checks currency: a second check after the fusion would be
        both duplicate and too late to matter.
        """
        try:
            hits = await store.search_keyword_candidates(query=question, limit=limit)
        except Exception as error:  # noqa: BLE001 (one store failing must not lose the others)
            logger.exception("retrieval.lexical.failed")
            note(LEXICAL, asked=True, ok=False, error=str(error))
            return []
        note(LEXICAL, asked=True, ok=True, hits=len(hits))
        return list(hits)

    async def _vector(
        self, store: Any, question: str, limit: int, note: Any  # noqa: ANN401
    ) -> list[RetrievalHit]:
        """Semantic discovery, from the same activation and the same corpus.

        The store embeds the question itself, which is why this no longer holds
        an embedding client of its own for the call. A platform without an
        embedding provider gets an empty list rather than a failure — a lexical
        index and a knowledge store are still a working platform.
        """
        try:
            hits = await store.search_semantic_candidates(query=question, limit=limit)
        except Exception as error:  # noqa: BLE001 (one store failing must not lose the others)
            logger.exception("retrieval.vector.failed")
            note(VECTOR, asked=True, ok=False, error=str(error))
            return []
        if not hits:
            note(VECTOR, asked=True, ok=True, hits=0)
            return []
        note(VECTOR, asked=True, ok=True, hits=len(hits))
        return list(hits)

    async def _knowledge(
        self, store: Any, question: str, plan: Any, note: Any  # noqa: ANN401
    ) -> list[RetrievalHit]:
        """Reviewed claims, and only the ones a reader is allowed to see.

        The store's read path resolves identities through the repository, which
        is where the quarantine lives (ADR-0032) — a claim awaiting review or
        merged into another cannot come back from it. Nothing here re-implements
        that, and nothing here may work around it.
        """
        hits: list[RetrievalHit] = []
        limit = plan.fetch_for(KNOWLEDGE)
        for kind in plan.kinds:
            method = getattr(store, f"query_{kind}s", None)
            if method is None:
                note(KNOWLEDGE, unknown_kind=kind)
                continue
            try:
                claims = await method(query=question, limit=limit)
            except Exception as error:  # noqa: BLE001 (one kind failing must not lose the rest)
                logger.exception("retrieval.knowledge.failed kind=%s", kind)
                note(KNOWLEDGE, asked=True, ok=False, error=str(error))
                continue
            hits.extend(
                RetrievalHit(
                    source=KNOWLEDGE,
                    carrier=ASSERTION,
                    carrier_id=str(claim.identity),
                    content=_wording(claim),
                    raw_score=float(getattr(claim, "confidence", 0.0) or 0.0),
                    # The claim, the state cited, and every sighting supporting
                    # it. `document_id` stays empty deliberately: a claim two
                    # documents assert has two supports and no home, and picking
                    # one for the flat field would invent an owner.
                    provenance=Provenance(
                        assertion_id=str(
                            getattr(claim, "assertion_id", "") or claim.identity
                        ),
                        knowledge_revision_id=str(
                            getattr(claim, "revision_id", "") or ""
                        ),
                        support=tuple(getattr(claim, "support", ()) or ()),
                    ),
                    fields={"kind": (kind,), "type": (str(getattr(claim, "type", "")),)},
                )
                for claim in claims
            )
        note(KNOWLEDGE, asked=True, ok=True, hits=len(hits))
        return hits



def _wording(claim: Any) -> str:  # noqa: ANN401 (a KnowledgeObject of any kind)
    """What a reviewed claim says, in the source's own words.

    `observed_text` is the authority (ADR-0048) and the store carries it now, so
    this is a read rather than a search through the payload for whichever field
    happened to hold prose.

    A claim written before revisions existed has none, and the payload is then
    the only thing left to show. That is worse and it is still better than an
    empty passage — but nothing here assembles a sentence out of field values,
    which is the one thing ADR-0048 forbids.
    """
    observed = getattr(claim, "observed_text", "")
    if isinstance(observed, str) and observed.strip():
        return observed.strip()
    payload = getattr(claim, "payload", None) or {}
    for name in ("rule_text", "decision", "description"):
        value = payload.get(name) if isinstance(payload, dict) else getattr(payload, name, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return str(getattr(claim, "identity", ""))
