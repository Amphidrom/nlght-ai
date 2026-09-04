# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any

from nlght.adapters.outbound.stores.knowledge_writer import KnowledgeIndexWriterTool
from nlght.core.errors.errors import WorkflowConfigurationError
from nlght.core.ingestion import SourceSnapshot, stable_digest
from nlght.core.knowledge import (
    BUILTIN_KINDS,
    ExtractedItem,
    ExtractionStateWrite,
    contract,
    extraction_version,
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
    from nlght.ports.outbound.knowledge_repository import KnowledgeRepository
    from nlght.ports.outbound.resource_repository import ResourceRepository

logger = logging.getLogger(__name__)

NOTHING_EXTRACTED = "empty"

UNCHANGED = "unchanged"
"""Verdict when every document was already extracted by this same extraction.

Distinct from `empty`, which means the model found nothing to say about text it
did read. Here the model was not asked at all."""

MORE_UNITS = "more"
"""Verdict when units of this pass remain unextracted. Route it back to this
step: one invocation extracts a bounded batch, not every unit of the run."""

_JSON_BLOCK = re.compile(r"\[.*\]", re.DOTALL)

_PROMPT_TEMPLATE = """Extract atomic knowledge assertions from the text below.

Return ONLY a JSON array. Each element must be an object with:
  "kind":       one of {kinds}
  "type":       a short category, e.g. "dependency", "security_policy"
  "confidence": a number between 0 and 1
  "observed_text": the sentence from TEXT that states this claim, copied
                exactly — the same characters, nothing added, shortened,
                corrected or rephrased. It must appear verbatim in TEXT; an
                element whose "observed_text" does not is discarded.
  "content":    the assertion, shaped by kind:
                  fact     -> {{"predicate": ..., "subject": ..., "object": ...}}
                  rule     -> {{"subject": ..., "rule_property": ..., "rule_text": ...}}
                  pattern  -> {{"pattern_name": ..., "description": ...}}
                  decision -> {{"subject": ..., "decision_type": ..., "decision": ..., "effect": ...}}

"observed_text" is what the source says; "content" is what you made of it. The
first is quoted back to a reader and must never be reassembled from the second —
so it is required for every kind, including "fact", where the structured fields
alone cannot reproduce a sentence.

For "rule" and "decision", "subject" and "rule_property"/"decision_type" name
what the claim is about and which question about it — short, lower_snake_case,
one or two words, and the same words for the same question every time:

  "Expenses above CHF 500 require approval."
      subject: expense           rule_property: approval_threshold
  "Expenses above CHF 500 are prohibited."
      subject: expense           rule_property: prohibition_threshold
  "Access tokens must not be logged."
      subject: access_token      rule_property: logging_prohibition

They are not a summary of the sentence. Two differently worded statements of the
same requirement must give the same pair; two different requirements about one
subject must give different ones. Put the wording itself in "rule_text",
"decision" and "effect", unchanged.

A "fact" names a relation and the things standing in it: "predicate" for the
relation, then "subject" and "object". Where a relation carries more than two
things, add short lower_snake_case roles rather than crowding them into
"object". Use the same predicate and the same role names for the same kind of
statement every time — two runs over one unchanged sentence must produce the
same proposition.


Choosing the kind:

{contract}

Return [] when the text asserts nothing.

TEXT:
{text}
"""

#: The prompt as it is sent and as it is hashed.
#:
#: The contract is substituted **here**, at import, and not at call time. It is
#: what `extraction_version` hashes (see `_version`), so a contract that arrived
#: only in the formatted string would change what the model is asked while every
#: document went on being skipped as already extracted under the old version.
#:
#: Braces are escaped on the way in so the contract cannot disturb the `{kinds}`
#: and `{text}` substitution that still happens per call.
_PROMPT = _PROMPT_TEMPLATE.replace(
    "{contract}", contract().replace("{", "{{").replace("}", "}}")
)


class KnowledgeExtractStep(StepBase):
    """Extracts candidate assertions from knowledge units with the platform model.

    Uses ``ctx.llm`` — the workflow's configured provider with its access policy,
    metering, and token budget — rather than the prototype's raw Ollama endpoint.

    Step config:
        max_units:      cap on units processed per extraction pass (default: all)
        unit_batch_size: units processed by one step invocation (default: 1)
        min_confidence: drop candidates below this (default 0.0)

    Reads ``knowledge.units``, accumulates ``knowledge.extracted``.
    """

    TYPE = "knowledge.extract"

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
        # The same three the persist step takes, and for the same reason: the
        # knowledge database is reached through a writer activation and nowhere
        # else. Here they are optional — without them nothing is skipped.
        self._resources = resource_repository
        self._activator = resource_activator
        self._tools = tool_loader
        self._connections = store_connections

    @classmethod
    def options(cls) -> list[StepOption]:
        return [
            StepOption("model_provider", "string", "Model provider to extract with",
                       placeholder="ollama"),
            StepOption("model", "string", "Model name", placeholder="llama3"),
            StepOption("max_units", "integer",
                       "Cap on units processed per extraction pass; 0 means all", default=0),
            StepOption("unit_batch_size", "integer",
                       "Units processed by one step invocation", default=1),
            StepOption("min_confidence", "number",
                       "Drop candidates below this confidence", default=0.0),
            StepOption("temperature", "number",
                       "Sampling temperature. 0 pins the model, which is what keeps two runs "
                       "over one unchanged document agreeing on what they found.",
                       default=0.0),
            StepOption("writer", "string",
                       "Resource name of the knowledge_index_writer activation. Without it "
                       "every run re-extracts every document, because there is nowhere to "
                       "record what was already extracted.",
                       placeholder="knowledge-index"),
        ]

    async def _repository(self, caller: RequestContext | None) -> KnowledgeRepository | None:
        """The knowledge database, reached the only way there is.

        Through the writer activation, exactly as `knowledge.persist` reaches
        it. Optional here: without one there is nowhere to record what was
        extracted, so nothing is skipped and the step behaves as it always did.
        """
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
        return instance.repository

    def _temperature(self) -> float:
        try:
            return float(self.config.get("temperature", 0.0))
        except (TypeError, ValueError) as exc:
            raise WorkflowConfigurationError(
                f"Step '{self.TYPE}' has an invalid temperature: {exc}"
            ) from exc

    def _version(self, ctx: WorkflowStepContext) -> str:
        """What only this step knows; the rest is folded in by the core.

        The assertion schema, the segmentation and the kind list belong to the
        extraction itself and are added there, so a new one of those cannot be
        forgotten here.
        """
        model = str(self.config.get("model", "")).strip() or ctx.model or "unknown"
        provider = str(self.config.get("model_provider", "")).strip() or "unknown"
        return extraction_version(
            workflow=ctx.trigger.context.workflow or "unknown",
            model=f"{provider}/{model}",
            prompt=_PROMPT,
            # Only the settings that shape what comes out. `unit_batch_size` and
            # `max_units` decide how the work is paced, not what it yields.
            settings={
                "min_confidence": self.config.get("min_confidence", 0.0),
                "temperature": self._temperature(),
            },
        )

    @staticmethod
    def _content_hashes(ctx: WorkflowStepContext) -> dict[str, str]:
        """What each document's *parser* was given, keyed by document.

        Not the source bytes: a classifier that starts extracting a PDF
        differently changes what extraction sees while the file on disk is
        untouched. And, since ADR-0046, not the searchable text either — for a
        source that transforms its documents, the parser reads the form the
        source holds and the searchable text is a derivation of it.

        The two can move independently, and that is the case worth catching. A
        heading given an `id` changes the markup and not one word of the text, so
        hashing the text would skip the document and the new anchor would arrive
        only with the next real edit. Slot identity resolves against exactly
        those anchors.

            markup present      hash the markup      — the parser's input
            markup absent       hash the text        — the same thing

        The second line is most documents, and it is why this does not re-extract
        an existing corpus: only a source that keeps a separate form gets a
        different answer than before.
        """
        documents = ctx.metadata.get("ingestion.processed")
        if not isinstance(documents, tuple):
            return {}
        snapshot = ctx.metadata.get("ingestion.snapshot")
        source_forms: dict[str, bytes] = {}
        if isinstance(snapshot, SourceSnapshot):
            source_forms = {
                acquired.external_id: acquired.source_form
                for acquired in snapshot.documents
                if acquired.source_form is not None
            }
        return {
            document.document_id: stable_digest(
                source_forms[document.external_id].decode("utf-8", errors="replace")
                if document.external_id in source_forms
                else (document.text or "")
            )
            for document in documents
        }

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        units = ctx.metadata.get("knowledge.units")
        if not isinstance(units, tuple):
            raise WorkflowConfigurationError(
                f"Step '{self.TYPE}' requires knowledge units; "
                f"place a 'knowledge.parse' step before it."
            )
        if ctx.llm is None:
            raise WorkflowConfigurationError(
                f"Step '{self.TYPE}' requires a model; set 'model_provider' in its step config."
            )

        try:
            configured_max_units = int(self.config.get("max_units", 0))
            unit_batch_size = int(self.config.get("unit_batch_size", 1))
        except ValueError as exc:
            raise WorkflowConfigurationError(
                f"Step '{self.TYPE}' has invalid unit limits: {exc}"
            ) from exc
        if configured_max_units < 0:
            raise WorkflowConfigurationError(
                f"Step '{self.TYPE}' requires max_units to be zero or positive."
            )
        if unit_batch_size < 1:
            raise WorkflowConfigurationError(
                f"Step '{self.TYPE}' requires unit_batch_size to be at least 1."
            )

        # Decided once per run, before the first batch, and carried in metadata
        # so the batches that follow read the same answer. Re-asking per batch
        # would let a concurrent run change what this one is doing halfway.
        version = self._version(ctx)
        # Carried forward: `knowledge.persist` records it on every revision it
        # writes, so a stored assertion says which extraction produced it.
        ctx.metadata["knowledge.extraction_version"] = version
        repository = await self._repository(ctx.trigger.context)
        hashes = self._content_hashes(ctx)
        if "knowledge.extract.skipped_documents" not in ctx.metadata:
            skipped = await self._already_extracted(repository, hashes, version)
            ctx.metadata["knowledge.extract.skipped_documents"] = skipped
            if skipped:
                logger.info(
                    "[%s] knowledge.extract.skip | documents=%d unchanged",
                    ctx.correlation_id, len(skipped),
                )
        skipped_documents = set(ctx.metadata["knowledge.extract.skipped_documents"])

        # Filtered before the limits are applied: max_units counts work, and a
        # document nobody has to re-read is not work.
        if skipped_documents:
            units = tuple(
                unit for unit in units
                if unit.document_id not in skipped_documents
            )
            if not units:
                # Every step after this one reads `knowledge.extracted` and
                # refuses to run without it — a guard that catches a workflow
                # wired without an extract step. Returning before establishing
                # the key made a second run over an unchanged corpus fail on a
                # configuration error that was not one. `empty` never had the
                # problem because it wrote its tuple on the way out; the skip
                # has to leave the same trace. The contract is that this step
                # establishes the key, whether or not it did any work.
                ctx.metadata.setdefault("knowledge.extracted", ())
                # Nothing was looked at, so no slot may be closed. This is the
                # skip path, and it is the normal case on a settled corpus.
                ctx.metadata.setdefault("knowledge.observed", ())
                logger.info(
                    "[%s] knowledge.extract.unchanged | documents=%d",
                    ctx.correlation_id, len(skipped_documents),
                )
                return StepResult(ctx=ctx, verdict=UNCHANGED)

        max_units = configured_max_units or len(units)
        unit_limit = min(len(units), max_units)
        min_confidence = float(self.config.get("min_confidence", 0.0))
        next_unit = int(ctx.metadata.get("knowledge.extract.next_unit", 0))
        if next_unit >= unit_limit:
            # The same reason as above: a document with no units — an empty
            # page, or one whose text the classifier could not read — leaves
            # here, and the chain behind it still expects the key to exist.
            existing = ctx.metadata.setdefault("knowledge.extracted", ())
            ctx.metadata.setdefault("knowledge.observed", ())
            extracted_count = len(existing) if isinstance(existing, tuple) else 0
            return StepResult(ctx=ctx, verdict="DEFAULT" if extracted_count else NOTHING_EXTRACTED)

        existing = ctx.metadata.get("knowledge.extracted")
        extracted: list[ExtractedItem] = list(existing) if isinstance(existing, tuple) else []
        batch_end = min(next_unit + unit_batch_size, unit_limit)
        batch_candidates = 0
        # A funnel rather than one number. Four candidates where a previous run
        # produced eighteen is a behaviour change, and "candidates=4" cannot say
        # whether the model said less, said it wrongly, or said it below the
        # floor. Each stage is counted so the answer is in the log.
        funnel = {"returned": 0, "malformed": 0, "unknown_kind": 0, "below_confidence": 0}
        # Which slots this run actually looked at. Retraction is decided per
        # slot by what the slot no longer contains, so it may only run where the
        # model was asked — and the commonest case is that it was not: an
        # unchanged document is skipped whole and yields nothing, which without
        # this would read as a slot that lost every claim it had.
        observed: list[tuple[str, str]] = list(ctx.metadata.get("knowledge.observed", ()))
        for unit in units[next_unit:batch_end]:
            observed.append((unit.document_id, unit.unit_ordinal))
            parsed, tally = await self._extract(ctx, unit.content)
            for name, count in tally.items():
                funnel[name] = funnel.get(name, 0) + count
            for item in parsed:
                if item.confidence < min_confidence:
                    funnel["below_confidence"] += 1
                    continue
                # Evidence ties every candidate back to the exact unit it came
                # from, which is what makes a later review decidable.
                extracted.append(
                    ExtractedItem(
                        kind=item.kind,
                        type=item.type,
                        content=item.content,
                        confidence=item.confidence,
                        # Carried, not rebuilt. This copied field by field and
                        # dropped the one it did not know about, so every
                        # candidate reached persistence with no wording and was
                        # counted as having none — the parser verified it and
                        # this threw it away three lines later.
                        observed_text=item.observed_text,
                        evidence={
                            "unit_ordinal": unit.unit_ordinal,
                            "source_id": unit.source_id,
                            "excerpt": unit.content[:500],
                            # Fields rather than whatever the parser happened to
                            # put in its dict: the lineage needs both, and a
                            # parser that forgot one used to leave every
                            # candidate unlinkable.
                            **unit.provenance,
                            **unit.metadata,
                        },
                        metadata=dict(unit.metadata),
                        tags=dict(unit.tags),
                    )
                )
                batch_candidates += 1

        ctx.metadata["knowledge.extracted"] = tuple(extracted)
        ctx.metadata["knowledge.observed"] = tuple(observed)
        ctx.metadata["knowledge.extract.next_unit"] = batch_end
        logger.info(
            "[%s] knowledge.extract.batch | units=%d/%d returned=%d malformed=%d "
            "unknown_kind=%d below_confidence=%d candidates=%d total=%d",
            ctx.correlation_id, batch_end, unit_limit,
            funnel["returned"], funnel["malformed"], funnel["unknown_kind"],
            funnel["below_confidence"], batch_candidates, len(extracted),
        )
        if batch_end < unit_limit:
            return StepResult(ctx=ctx, verdict=MORE_UNITS)

        # Recorded only now, once every unit of every document has been read.
        # Recording per batch would mark a document extracted while half of it
        # still was not, and a crash in between would then skip the remainder
        # for good — the same ordering `ingestion.write` keeps for the index.
        await self._record(repository, hashes, version, ctx, skipped_documents, len(extracted))
        return StepResult(ctx=ctx, verdict="DEFAULT" if extracted else NOTHING_EXTRACTED)

    @staticmethod
    async def _already_extracted(
        repository: KnowledgeRepository | None,
        hashes: dict[str, str],
        version: str,
    ) -> list[str]:
        """Which documents this extraction has already read, unchanged.

        Nothing is skipped without both a repository to have recorded it and a
        content hash to compare against — a missing record is not "current",
        and neither is a document whose text this step cannot see.
        """
        if repository is None or not hashes:
            return []
        skipped: list[str] = []
        for document_id, content_hash in hashes.items():
            state = await repository.extraction_state(document_id)
            if state is not None and state.is_current(
                content_hash=content_hash, extraction_version=version
            ):
                skipped.append(document_id)
        return skipped

    @staticmethod
    async def _record(
        repository: KnowledgeRepository | None,
        hashes: dict[str, str],
        version: str,
        ctx: WorkflowStepContext,
        skipped: set[str],
        assertion_count: int,
    ) -> None:
        if repository is None or not hashes:
            return
        for document_id, content_hash in hashes.items():
            if document_id in skipped:
                continue
            await repository.record_extraction(
                ExtractionStateWrite(
                    document_id=document_id,
                    content_hash=content_hash,
                    extraction_version=version,
                    run_id=ctx.correlation_id,
                    # The run's total rather than this document's share: the
                    # step accumulates across documents and the number is for
                    # reading, not for deciding anything.
                    assertion_count=assertion_count,
                )
            )

    async def _extract(
        self, ctx: WorkflowStepContext, text: str
    ) -> tuple[list[ExtractedItem], dict[str, int]]:
        assert ctx.llm is not None
        prompt = _PROMPT.format(kinds=", ".join(BUILTIN_KINDS), text=text)

        collected: list[str] = []
        # Pinned by default. A sampling model answers the same question in
        # slightly different words each time, and identity is derived from those
        # words — so every run produced entity keys the run before it had never
        # seen, and a corpus nobody edited grew assertions. Temperature is part
        # of the extraction version, so changing it re-extracts rather than
        # leaving a corpus half written by each setting.
        async for event in ctx.llm.stream(
            [{"role": "user", "content": prompt}], temperature=self._temperature()
        ):
            if event.kind == "token" and event.content:
                collected.append(event.content)
            elif event.kind == "done":
                break

        return self._parse("".join(collected), text)

    @staticmethod
    def _parse(raw: str, source: str = "") -> tuple[list[ExtractedItem], dict[str, int]]:
        """Tolerate prose around the JSON, but never invent an assertion.

        A model that answers with an explanation plus a JSON array is common;
        a malformed element is dropped with a warning rather than failing the
        whole unit, because one bad candidate must not lose the good ones.
        """
        tally = {"returned": 0, "malformed": 0}
        match = _JSON_BLOCK.search(raw)
        if match is None:
            return [], tally
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            logger.warning("knowledge.extract.unparsable_response")
            return [], tally
        if not isinstance(payload, list):
            return [], tally

        tally["returned"] = len(payload)
        items: list[ExtractedItem] = []
        for element in payload:
            if not isinstance(element, dict):
                tally["malformed"] += 1
                continue
            # Asked before the item is built, because building it raises for an
            # unknown kind too — and then a model inventing kinds would be
            # indistinguishable from one writing malformed content, which want
            # different fixes.
            if str(element.get("kind", "")).strip().lower() not in BUILTIN_KINDS:
                tally["unknown_kind"] = tally.get("unknown_kind", 0) + 1
                continue
            try:
                item = ExtractedItem(
                    kind=str(element["kind"]).strip().lower(),
                    type=str(element.get("type", "general")),
                    content=dict(element["content"]),
                    confidence=float(element.get("confidence", 0.5)),
                    observed_text=str(element.get("observed_text", "")),
                )
                # Rejected for saying nothing its kind can hold — not for
                # failing to fit the legacy three-column row. A fact with four
                # roles was counted here as malformed and dropped, so the corpus
                # kept only the claims an old table happened to have space for.
                item.well_formed()
            except (KeyError, TypeError, ValueError) as exc:
                # A bare `'object'` said nothing about what had gone wrong. The
                # kind and the fields that did arrive are what tell a prompt
                # problem from a model having a bad day.
                logger.warning(
                    "knowledge.extract.invalid_item | kind=%s type=%s missing_or_invalid=%s "
                    "fields=%s",
                    element.get("kind", "?"),
                    element.get("type", "?"),
                    exc,
                    sorted(element.get("content", {}))
                    if isinstance(element.get("content"), dict)
                    else "no content object",
                )
                tally["malformed"] += 1
                continue
            # The wording has to be the source's, not the model's account of it.
            # Asked for in the prompt and checked here, because a field a model
            # fills in is another model-generated string — and one stored under
            # the name "observed" while holding a paraphrase is worse than no
            # wording at all: everything downstream quotes it as if a person
            # could find it in the document.
            #
            # Counted apart from `malformed`. A candidate whose structure is
            # wrong and one whose quotation is wrong want different fixes, and a
            # single number would say which run had trouble and nothing about
            # what to do.
            if source and not item.verified_against(source):
                tally["unverified_wording"] = tally.get("unverified_wording", 0) + 1
                logger.warning(
                    "knowledge.extract.unverified_wording | kind=%s observed=%.80r",
                    item.kind, item.observed_text,
                )
                continue
            items.append(item)
        return items, tally
