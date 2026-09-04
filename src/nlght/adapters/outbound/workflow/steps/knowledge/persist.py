# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

from nlght.adapters.outbound.knowledge.equivalence_classifier import (
    ModelPropositionEquivalence,
)
from nlght.adapters.outbound.knowledge.semantic_classifier import (
    ModelPropositionClassifier,
)
from nlght.adapters.outbound.stores.connections import StoreConnections
from nlght.adapters.outbound.stores.knowledge_writer import KnowledgeIndexWriterTool
from nlght.adapters.outbound.tools.activator import ResourceActivator
from nlght.adapters.outbound.tools.loader import ToolLoader
from nlght.core.entry.context import RequestContext
from nlght.core.errors.errors import WorkflowConfigurationError
from nlght.core.knowledge import (
    NO_ANCHOR,
    ExtractedItem,
    KnowledgeWrite,
    LineageWrite,
    ObservedSection,
    entity_key_or_none,
)
from nlght.core.workflow.step import (
    StepBase,
    StepOption,
    StepResult,
    WorkflowStepContext,
)
from nlght.ports.outbound.resource_repository import ResourceRepository

logger = logging.getLogger(__name__)

NOTHING_PERSISTED = "empty"


def _assertion_text(item: ExtractedItem) -> str:
    """The wording a citation quotes: what the source said, and nothing else.

    `observed_text` and no fallback to the fields. This used to pick the first
    non-empty of `rule_text`/`decision`/`description`/`object` and otherwise join
    every value — so a fact's citation text was "requires spring_boot java_17",
    a string assembled here that appears in no document. Structure may enrich a
    claim; it may never restate it.

    Empty is possible and is left empty. A candidate extracted before the wording
    was asked for has none, and inventing one from its fields is precisely what
    this stopped doing.
    """
    return item.observed_text.strip()


class KnowledgePersistStep(StepBase):
    """Writes classified candidates into the knowledge graph.

    Identity-addressed and idempotent: the same assertion found again
    reinforces the existing node and records another observation, so re-running
    a source strengthens knowledge instead of duplicating it.

    Also indexes each assertion for lexical search when a ``writer`` resource is
    configured. Without it the knowledge store's search side stays empty and
    every query returns nothing, so the step logs plainly when none is set.

    Step config:
        writer: resource name of the knowledge_index_writer activation

    Reads ``knowledge.classified``; falls back to ``knowledge.extracted`` only
    when no classification step ran, in which case everything is quarantined —
    persisting unreviewed assertions as immediately retrievable would defeat the
    review boundary.
    """

    TYPE = "knowledge.persist"

    def __init__(
        self,
        *,
        config: dict[str, Any],
        resource_repository: ResourceRepository | None = None,
        resource_activator: ResourceActivator | None = None,
        tool_loader: ToolLoader | None = None,
        store_connections: StoreConnections | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(config=config, **kwargs)
        self._resources = resource_repository
        self._activator = resource_activator
        self._tools = tool_loader
        # A step is loaded per hop, so its writer is activated per hop too — the
        # runtime's shared pools are what keep that from opening connections.
        self._connections = store_connections

    @classmethod
    def options(cls) -> list[StepOption]:
        return [
            StepOption("writer", "string",
                       "Resource name of the knowledge_index_writer activation; "
                       "without it assertions are stored but not searchable",
                       placeholder="knowledge-index"),
        ]

    async def _writer(self, caller: RequestContext | None) -> KnowledgeIndexWriterTool | None:
        name = str(self.config.get("writer", "")).strip()
        if not name or self._resources is None or self._tools is None:
            return None
        if self._activator is None:
            raise WorkflowConfigurationError(
                f"Step '{self.TYPE}' has no resource activator, so it cannot "
                f"reach '{name}'."
            )
        instance = await self._activator.activate(
            kind=KnowledgeIndexWriterTool.KIND,
            name=name,
            caller=caller,
        )
        if not isinstance(instance, KnowledgeIndexWriterTool):
            raise WorkflowConfigurationError(
                f"Resource '{name}' resolves to {type(instance).__name__}, "
                f"which is not a {KnowledgeIndexWriterTool.__name__}."
            )
        instance.ensure_ready()
        return instance

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        classified = self._classified(ctx)
        # The writer resource owns the knowledge database. There is no central
        # fallback: a connection reached past the activation would let one
        # deployment-wide URL silently override where a pipeline writes.
        writer = await self._writer(ctx.trigger.context)
        if writer is None:
            raise WorkflowConfigurationError(
                f"Step '{self.TYPE}' requires a knowledge database. Name a "
                f"'{KnowledgeIndexWriterTool.KIND}' resource in its 'writer' config."
            )
        repository = writer.repository

        run_id = ctx.correlation_id

        # Two different faults, counted apart. One says the extraction could not
        # name a business question — a prompt or schema problem. The other says
        # the evidence has no document, so the claim could never be diffed —
        # a pipeline problem. A single number for both says which run had
        # trouble and nothing about what to fix.
        without_entity = 0
        without_document = 0
        without_wording = 0
        quarantined = 0
        extraction = str(ctx.metadata.get("knowledge.extraction_version", "unknown"))

        # Grouped by slot, because retraction is a set operation: a claim is gone
        # because the slot no longer holds it, and that is only knowable once
        # everything the slot does hold has arrived. Deciding absence while the
        # group is still being filled decides it on a partial slot.
        grouped: dict[tuple[str, str], list[tuple[KnowledgeWrite, LineageWrite | None]]] = {}
        loose: list[tuple[KnowledgeWrite, LineageWrite | None]] = []
        readable: dict[str, ExtractedItem] = {}
        unplaceable = 0

        for item, reason in classified:
            lineage, gap = self._lineage(item, run_id, extraction)
            if gap == "no_entity_key":
                without_entity += 1
            elif gap == "missing_document_evidence":
                without_document += 1
            elif gap == "no_observed_text":
                without_wording += 1
            if lineage is None and item.payload() is None:
                # Nowhere to put what this claim says.
                #
                # A proposition lives on the revision a sighting produces, so a
                # candidate with no lineage has no revision to keep it on. That
                # is survivable while the *legacy payload* can hold the whole
                # claim — the payload row then carries everything, which is the
                # case for an older `rule` that never had its canonical pair.
                #
                # It is not survivable when both are missing: the graph row then
                # has no payload, no proposition and no revision, and says
                # nothing at all. Storing that is not a milder failure than
                # dropping the claim, only a quieter one.
                #
                # The test is what the claim *has*, not what kind it is. Every
                # kind has a proposition now, so a rule with four fields the
                # payload cannot hold is in exactly the position an n-ary fact
                # was, and is treated the same way.
                unplaceable += 1
                logger.warning(
                    "[%s] knowledge.persist.unplaceable | reason=%s fields=%s document=%s",
                    ctx.correlation_id, gap, sorted(item.content),
                    item.evidence.get("document_id", "?"),
                )
                continue
            graph = KnowledgeWrite(
                identity=item.fingerprint,
                # The claim, and the old row derived from it — never two
                # independent writes. Two can disagree and nothing afterwards
                # says which is right; a derivation cannot disagree with itself.
                proposition=item.proposition,
                payload=item.payload(),
                # Stated, because the payload is a projection and may be absent
                # for any kind now — read off it, a rule whose row cannot hold
                # its claim would enter the graph as a fact.
                _kind=item.kind,
                type=item.type,
                confidence=item.confidence,
                source_id=str(item.evidence.get("source_id", "unknown")),
                run_id=run_id,
                product=item.tags.get("product"),
                version=item.tags.get("version"),
                review_required=bool(reason),
                review_reason=reason,
                evidence=dict(item.evidence),
                metadata=dict(item.metadata),
            )
            slot = (
                str(item.evidence.get("document_id") or ""),
                str(item.evidence.get("unit_ordinal") or ""),
            )
            if all(slot):
                grouped.setdefault(slot, []).append((graph, lineage))
            else:
                # No document or no unit is no slot, so this candidate cannot take
                # part in a set comparison and must never retire anything. It is
                # still stored: a claim the diff cannot reach is bad, losing it is
                # worse.
                #
                # With its lineage, where it has one. This dropped the pair and
                # kept only the graph write, so a candidate whose evidence named
                # its document but not its section lost a lineage row that had
                # already been built successfully — and with it the revision its
                # structure lives on. A slot is optional on a lineage row; a
                # document is not.
                loose.append((graph, lineage))
            # Only what may be read reaches the index, and a quarantined
            # assertion simply does not go there. Removal belongs to the two
            # things that really take knowledge back: a reviewer withdrawing an
            # approval, and a source no longer carrying the claim.
            if not reason:
                readable[item.fingerprint] = item
            quarantined += 1 if reason else 0

        # Which units this run actually put to the model, and which units each
        # document has. A document may be compared only when every one of its
        # units was looked at: an unchanged document is skipped whole and yields
        # nothing, and one truncated by `max_units` is half read — either would
        # otherwise look like a source that dropped everything.
        observed = {
            (str(document), str(unit))
            for document, unit in ctx.metadata.get("knowledge.observed", ())
        }
        parsed: dict[str, set[str]] = {}
        for unit in ctx.metadata.get("knowledge.units", ()):
            parsed.setdefault(str(unit.document_id), set()).add(str(unit.unit_ordinal))
        complete = {
            document
            for document, units in parsed.items()
            if units and all((document, unit) in observed for unit in units)
        }

        # What each section of each document can be found by. The unit carries
        # it as fields, so a candidate is placed by where its section stands
        # rather than by the number the parser gave that section — which
        # renumbers whenever a paragraph is inserted above it.
        observed_sections: dict[str, ObservedSection] = {}
        for unit in ctx.metadata.get("knowledge.units", ()):
            # Both fingerprints from the one text, because they answer different
            # questions about it: which slot this is, and whether the model was
            # asked exactly this. Deriving them separately is how one of them
            # ended up answering the other's question.
            observed_sections[str(unit.unit_ordinal)] = ObservedSection.as_read(
                anchor=unit.anchor,
                anchor_strength=unit.anchor_strength,
                content=unit.content,
                unit_ordinal=str(unit.unit_ordinal),
            )

        # Grouped by document, because that is the container retraction is
        # decided in. Comparing per slot retired each unchanged claim from the
        # slot it used to occupy and re-created it in the one it had moved to, on
        # an edit that touched none of them. The slot matches a claim; the
        # document decides one is gone.
        by_document: dict[str, dict[str, list[tuple[KnowledgeWrite, LineageWrite | None]]]] = {}
        for (document, section), slot_writes in grouped.items():
            by_document.setdefault(document, {}).setdefault(section, []).extend(slot_writes)
        for document in complete:
            by_document.setdefault(document, {})

        # The one judgement about meaning this pipeline delegates: whether a
        # changed body is the same rule said differently or a different rule.
        # Nothing here can settle it — `must` becoming `must not` changed the
        # rule and `must` becoming `shall` did not, and no rule over characters
        # separates those. Without a model the deterministic answer stands.
        classify = (
            ModelPropositionClassifier(ctx.llm).classify if ctx.llm is not None else None
        )
        # And the second, asked far more rarely: whether a claim standing in this
        # slot is this claim said differently. Only where the matching ladder
        # found nothing, only against candidates in the same slot, and only for
        # the band structure cannot settle — so a corpus whose sources are stable
        # never asks it at all. Without a model every pair is `unjudged`, which
        # leaves the claims apart and says so.
        assess = ModelPropositionEquivalence(ctx.llm) if ctx.llm is not None else None

        persisted = 0
        retired = 0
        for document in sorted(by_document):
            slots = by_document[document]
            if document not in complete:
                # Written, but not compared: this run did not see all of it.
                #
                # Indexed all the same. Not being comparable is a statement about
                # *retraction* — nothing here may decide a claim is gone — and it
                # was quietly being read as a statement about retrieval too, so
                # every claim from a partially observed document was stored and
                # unsearchable. Only the review boundary decides what is readable,
                # and it says the same thing here as anywhere else.
                for slot_writes in slots.values():
                    for graph, lineage in slot_writes:
                        stored, _ = await repository.persist(graph, lineage)
                        approved = readable.get(stored.identity)
                        if approved is not None:
                            await writer.index(
                                stored, keywords=approved.metadata.get("keywords")
                            )
                    persisted += len(slot_writes)
                continue
            outcome = await repository.record_document(
                document_id=document,
                sections=[
                    replace(
                        observed_sections.get(unit_ordinal)
                        or ObservedSection(
                            anchor="", anchor_strength=NO_ANCHOR,
                            content_fingerprint="", unit_ordinal=unit_ordinal,
                        ),
                        assertions=tuple(slot_writes),
                    )
                    for unit_ordinal, slot_writes in slots.items()
                ],
                run_id=run_id,
                classify=classify,
                assess=assess,
            )
            persisted += sum(len(slot_writes) for slot_writes in slots.values())
            retired += len(outcome.retired)
            for stored in outcome.stored:
                approved = readable.get(stored.identity)
                if approved is not None:
                    await writer.index(stored, keywords=approved.metadata.get("keywords"))
            for gone in outcome.retired:
                # A claim no source carries must leave the index in the same
                # breath, or it stays retrievable while being unfindable.
                if gone.fingerprint:
                    await writer.remove(identity=gone.fingerprint, kind=gone.kind)

        for graph, lineage in loose:
            stored, _ = await repository.persist(graph, lineage)
            persisted += 1
            approved = readable.get(stored.identity)
            if approved is not None:
                await writer.index(stored, keywords=approved.metadata.get("keywords"))

        ctx.metadata["knowledge.persisted"] = persisted
        ctx.metadata["knowledge.retired"] = retired
        ctx.metadata["knowledge.unplaceable"] = unplaceable
        logger.info(
            "[%s] knowledge.persist.done | persisted=%d quarantined=%d retired=%d "
            "documents=%d/%d units=%d no_entity_key=%d missing_document_evidence=%d "
            "no_observed_text=%d unplaceable=%d",
            ctx.correlation_id, persisted, quarantined, retired,
            len(complete), len(by_document), len(observed),
            without_entity, without_document, without_wording, unplaceable,
        )
        return StepResult(ctx=ctx, verdict="DEFAULT" if persisted else NOTHING_PERSISTED)

    @staticmethod
    def _lineage(
        item: ExtractedItem, run_id: str, extraction_version: str
    ) -> tuple[LineageWrite | None, str | None]:
        """The lineage for one candidate, or nothing when it cannot say what it is.

        An assertion whose kind carries no identifying fields — a rule extracted
        before `subject` and `rule_property` were asked for, or one the model
        left them out of — has no entity key. It gets no lineage row and no
        substitute: a key derived from its wording is one a later run cannot
        reproduce, so every run would open a new assertion for the same claim.
        That is the drift this design removes, arriving through the migration
        meant to end it.
        """
        entity = entity_key_or_none(item.kind, **item.content)
        if entity is None:
            return None, "no_entity_key"
        text = _assertion_text(item)
        if not text:
            # No observed wording, so nothing a citation could quote. A gap like
            # the others rather than an exception: the wording is asked for at
            # extraction and verified there, so reaching here without one means a
            # producer that predates the field — which is a candidate to count,
            # not a run to crash.
            #
            # And emphatically not a licence to rebuild one from the fields. That
            # is what produced "requires spring_boot java_17" as the sentence a
            # reviewer was shown.
            return None, "no_observed_text"
        document_id = str(item.evidence.get("document_id") or "").strip()
        document_revision = str(item.evidence.get("processing_revision_id") or "").strip()
        if not document_id or not document_revision:
            # Evidence without a document cannot answer "what did document D at
            # revision R assert", which is the question the diff is built on.
            return None, "missing_document_evidence"
        scope = item.content.get("applies_to") or item.content.get("scope") or ()
        return LineageWrite(
            entity=entity,
            kind=item.kind,
            text=text,
            # Identity is still content-derived; it serves as the fingerprint
            # until the matcher that replaces it is wired in.
            fingerprint=item.fingerprint,
            extraction_version=extraction_version,
            document_id=document_id,
            document_revision=document_revision,
            # Provenance, carried so a reader and a report can name the file.
            # Never used to resolve anything: a renamed document is the same
            # document.
            document_path=str(item.evidence.get("document_path") or ""),
            run_id=run_id,
            scope=tuple(str(value) for value in scope) if isinstance(scope, (list, tuple)) else (),
            slot_id=str(item.evidence.get("unit_ordinal") or "") or None,
            # The complete structure, travelling with the sighting that carries
            # it — so it lands on the revision this sighting produces rather than
            # on the assertion it continues. One assertion has many revisions and
            # a reworded claim keeps every earlier form.
            proposition=item.proposition,
        ), None

    def _classified(
        self, ctx: WorkflowStepContext
    ) -> tuple[tuple[ExtractedItem, str | None], ...]:
        classified = ctx.metadata.get("knowledge.classified")
        if isinstance(classified, tuple):
            return classified

        extracted = ctx.metadata.get("knowledge.extracted")
        if isinstance(extracted, tuple):
            reason = "no review classification step ran"
            return tuple((item, reason) for item in extracted)

        raise WorkflowConfigurationError(
            f"Step '{self.TYPE}' requires extracted candidates; "
            f"place a 'knowledge.extract' step before it."
        )
