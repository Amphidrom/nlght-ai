# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Refinement steps between extraction and persistence.

Ported from ``next-integration/knowledge-indexer/pipeline/``. Each step is an
independent transformation over ``ctx.metadata['knowledge.extracted']``, so a
workflow composes only the ones it wants and in whatever order suits its source.

All of them are structural rather than language-based: they inspect the shape of
an assertion, never its natural-language meaning, so behaviour does not drift
with the extraction model.
"""

from __future__ import annotations

import logging
import re
from collections import Counter, defaultdict
from dataclasses import replace
from typing import Any

from nlght.core.errors.errors import WorkflowConfigurationError
from nlght.core.knowledge import ExtractedItem
from nlght.core.workflow.step import (
    StepBase,
    StepOption,
    StepResult,
    WorkflowStepContext,
)

logger = logging.getLogger(__name__)

ALL_KINDS = ("fact", "rule", "pattern", "decision")
_TECHNICAL = re.compile(r"[.@:/_\-()\[\]=]|::|->|\d")
_SPLIT = re.compile(r"\s*(?:,|;|\band\b|\bund\b|/)\s*", re.IGNORECASE)
_PLACEHOLDERS = frozenset({"-", "_", "?", "N/A", "n/a", "TBD", "tbd", "none", "None"})


def _norm(value: Any) -> str:  # noqa: ANN401 (content values are heterogeneous)
    return " ".join(str(value or "").strip().split()).lower()


def _structural_key(item: ExtractedItem, *, case_sensitive: bool = False) -> tuple[str, ...]:
    """The whole structured assertion, so two different claims stay two.

    Every field the extraction produced, for every kind. It used to be a fixed
    projection per kind — a fact keyed on `(subject, predicate, object)`, a rule
    on `rule_text` alone — which reached past the n-ary work in front of it:

        transfers 500 records from staging to archive
        transfers 500 records from staging to backup

    are one key and were collapsed here, before anything was persisted, while
    `Proposition` and the entity key had long since carried every role. Two rules
    about one subject with different thresholds went the same way, since neither
    `subject` nor `rule_property` was in the key.

    Not the entity identity, deliberately. This step exists to drop *structurally
    identical* extractions inside one pass — the same sentence read twice — and
    reaching for identity here would let it answer questions that belong to the
    repository, where the slot, the history and the equivalence judgement are.
    Two claims that share an entity key and differ in their fields are a
    disagreement to resolve there, not a duplicate to drop here.

    `observed_text` stays out. One sentence may state several atomic claims
    (ADR-0049), so they share their wording by design, and keying on it would
    fold them back into one.
    """

    def value(raw: Any) -> str:  # noqa: ANN401 (content values are heterogeneous)
        text = str(raw).strip()
        return text if case_sensitive else _norm(text)

    # Sorted by field name, so the key describes the assertion rather than the
    # order the model happened to emit its fields in. Blank values are dropped
    # for the same reason `entity_key` drops them: a field the model left empty
    # says nothing, and keeping it would make two identical claims differ.
    fields = tuple(
        f"{name}={value(raw)}"
        for name, raw in sorted(item.content.items())
        if str(raw).strip()
    )
    return (item.kind, *fields)


def _symbol_ratio(text: str) -> float:
    if not text:
        return 0.0
    symbols = sum(1 for char in text if not char.isalnum() and not char.isspace())
    return symbols / max(len(text), 1)


class _RefineStep(StepBase):
    """Shared plumbing: read candidates, transform, write back."""

    def _items(self, ctx: WorkflowStepContext) -> tuple[ExtractedItem, ...]:
        items = ctx.metadata.get("knowledge.extracted")
        if not isinstance(items, tuple):
            raise WorkflowConfigurationError(
                f"Step '{self.TYPE}' requires extracted candidates; "
                f"place a 'knowledge.extract' step before it."
            )
        return items

    def _kinds(self) -> set[str]:
        return {str(kind) for kind in self.config.get("apply_to_kinds", ALL_KINDS)}

    def _finish(
        self,
        ctx: WorkflowStepContext,
        items: tuple[ExtractedItem, ...],
        before: tuple[ExtractedItem, ...],
    ) -> StepResult:
        """Write the candidates back, and say what this step did to identity.

        Counts alone cannot answer the question a rerun raises. When a corpus
        that was only reworded comes back with an assertion more, something
        between the model and the key produced a different canonical
        proposition — and this chain is four steps long. `in`/`out` stays flat
        while `canonicalize` lowercases an object or `normalize` strips a
        trailing stop, either of which moves the entity key the model is judged
        by, so a chain reporting only counts leaves the model carrying the blame
        for what a rewrite rule did.

        `rewritten` is how many candidates left carrying a fingerprint this
        step was not handed. It was called `identity_changed`, from before
        identity stopped being derived from content — these steps cannot change
        an assertion's identity at all now, only the wording it is recorded
        under. Counted against a multiset rather than a set,
        because the interesting case is exactly the one a set misses: a step
        that rewrites one candidate onto a fingerprint *another* candidate
        already had — two spellings of one claim collapsing onto each other —
        is the merge that makes an assertion disappear, and it would report
        nothing. A split counts its new halves, which is right: they are claims
        the step invented.
        """
        ctx.metadata["knowledge.extracted"] = items
        handed_in = Counter(item.fingerprint for item in before)
        rewritten = sum(
            (Counter(item.fingerprint for item in items) - handed_in).values()
        )
        logger.info(
            "[%s] %s.done | in=%d out=%d rewritten=%d",
            ctx.correlation_id, self.TYPE, len(before), len(items), rewritten,
        )
        return StepResult(ctx=ctx, verdict="DEFAULT")

    @classmethod
    def options(cls) -> list[StepOption]:
        return [
            StepOption("apply_to_kinds", "array", "Kinds this step applies to",
                       default=list(ALL_KINDS)),
        ]


class KnowledgeCanonicalizeStep(_RefineStep):
    """Collapses whitespace and strips trailing punctuation in every field.

    Purely textual: two extractions that differ only in spacing must reach the
    same structural key, or deduplication and identity merging silently miss.
    """

    TYPE = "knowledge.canonicalize"

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        items = self._items(ctx)
        kinds = self._kinds()
        canonical = tuple(
            item
            if item.kind not in kinds
            else replace(
                item,
                content={
                    key: " ".join(str(value).split()).rstrip(" .;,")
                    if isinstance(value, str)
                    else value
                    for key, value in item.content.items()
                },
            )
            for item in items
        )
        return self._finish(ctx, canonical, items)


class KnowledgeNormalizeStep(_RefineStep):
    """Reclassifies an assertion's ``type`` from its structural shape.

    A long, prose-like predicate is a different kind of claim from a short
    symbolic one, and the extraction model is unreliable about saying so.
    """

    TYPE = "knowledge.normalize"

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        items = self._items(ctx)
        kinds = self._kinds()
        long_predicate = int(self.config.get("long_predicate_tokens", 6))
        rule_long = int(self.config.get("rule_long_tokens", 12))

        normalized: list[ExtractedItem] = []
        for item in items:
            if item.kind not in kinds:
                normalized.append(item)
                continue
            metadata = dict(item.metadata)
            if item.kind == "fact":
                predicate = str(item.content.get("predicate", ""))
                metadata["form"] = (
                    "narrative" if len(predicate.split()) > long_predicate else "relational"
                )
            elif item.kind == "rule":
                text = str(item.content.get("rule_text", ""))
                metadata["form"] = "detailed" if len(text.split()) > rule_long else "concise"
            normalized.append(replace(item, metadata=metadata))
        return self._finish(ctx, tuple(normalized), items)

    @classmethod
    def options(cls) -> list[StepOption]:
        return [
            *super().options(),
            StepOption("long_predicate_tokens", "integer",
                       "Predicates longer than this are narrative, not relational", default=6),
            StepOption("rule_long_tokens", "integer",
                       "Rules longer than this count as detailed", default=12),
        ]


class KnowledgeAtomicityStep(_RefineStep):
    """Splits conjunctions so one candidate carries one reviewable claim.

    "A requires B and C" becomes two assertions, because a reviewer must be able
    to accept one and reject the other. Expansion is capped: a sentence with
    many conjunctions is more likely prose than a list, and splitting it would
    manufacture nonsense.
    """

    TYPE = "knowledge.atomicity"

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        items = self._items(ctx)
        kinds = self._kinds()
        max_splits = int(self.config.get("max_splits", 6))
        max_token_span = int(self.config.get("max_token_span", 6))

        out: list[ExtractedItem] = []
        for item in items:
            if item.kind != "fact" or item.kind not in kinds:
                out.append(item)
                continue
            subjects = self._split(str(item.content.get("subject", "")), max_token_span)
            objects = self._split(str(item.content.get("object", "")), max_token_span)
            if len(subjects) * len(objects) > max_splits:
                out.append(item)
                continue
            if len(subjects) == 1 and len(objects) == 1:
                out.append(item)
                continue
            for subject in subjects:
                for obj in objects:
                    out.append(
                        replace(item, content={**item.content, "subject": subject, "object": obj})
                    )
        return self._finish(ctx, tuple(out), items)

    @staticmethod
    def _split(value: str, max_token_span: int) -> list[str]:
        if not value.strip():
            return [value]
        parts = [part.strip() for part in _SPLIT.split(value) if part.strip()]
        if len(parts) < 2:
            return [value]
        # Long fragments are prose, not list items.
        if any(len(part.split()) > max_token_span for part in parts):
            return [value]
        return parts

    @classmethod
    def options(cls) -> list[StepOption]:
        return [
            *super().options(),
            StepOption("max_splits", "integer",
                       "Give up splitting beyond this many combinations", default=6),
            StepOption("max_token_span", "integer",
                       "Fragments longer than this are prose, not list items", default=6),
        ]


class KnowledgeIdentityMergeStep(_RefineStep):
    """Merges structurally identical candidates, keeping the strongest.

    Confidence is the maximum rather than the sum: repetition within one run is
    the same evidence seen twice, not independent corroboration. Cross-run
    reinforcement happens later, in the repository.
    """

    TYPE = "knowledge.identity_merge"

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        items = self._items(ctx)
        merged: dict[tuple[str, ...], ExtractedItem] = {}
        order: list[tuple[str, ...]] = []

        for item in items:
            key = _structural_key(item)
            existing = merged.get(key)
            if existing is None:
                merged[key] = item
                order.append(key)
                continue
            merged[key] = replace(
                existing,
                confidence=max(existing.confidence, item.confidence),
                evidence={**existing.evidence, **item.evidence},
                metadata={**existing.metadata, **item.metadata},
            )
        return self._finish(ctx, tuple(merged[key] for key in order), items)

    @classmethod
    def options(cls) -> list[StepOption]:
        return []


class KnowledgeCollapseStep(_RefineStep):
    """Collapses parallel relations that share subject and predicate.

    "X supports A", "X supports B" become one assertion listing both objects,
    which reads as one claim to a reviewer instead of a wall of near-duplicates.
    """

    TYPE = "knowledge.collapse"

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        items = self._items(ctx)
        buckets: dict[tuple[str, str], list[ExtractedItem]] = defaultdict(list)
        passthrough: list[ExtractedItem] = []

        for item in items:
            if item.kind != "fact":
                passthrough.append(item)
                continue
            buckets[
                (_norm(item.content.get("subject")), _norm(item.content.get("predicate")))
            ].append(item)

        collapsed: list[ExtractedItem] = list(passthrough)
        for group in buckets.values():
            if len(group) == 1:
                collapsed.append(group[0])
                continue
            objects = sorted({str(entry.content.get("object", "")) for entry in group})
            base = group[0]
            collapsed.append(
                replace(
                    base,
                    content={**base.content, "object": ", ".join(objects)},
                    confidence=max(entry.confidence for entry in group),
                )
            )
        return self._finish(ctx, tuple(collapsed), items)

    @classmethod
    def options(cls) -> list[StepOption]:
        return []


class KnowledgeDedupStep(_RefineStep):
    """Drops structurally identical candidates, keeping the first."""

    TYPE = "knowledge.dedup"

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        items = self._items(ctx)
        case_sensitive = bool(self.config.get("case_sensitive", False))
        kinds = self._kinds()

        seen: set[tuple[str, ...]] = set()
        unique: list[ExtractedItem] = []
        for item in items:
            if item.kind not in kinds:
                unique.append(item)
                continue
            key = _structural_key(item, case_sensitive=case_sensitive)
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)
        return self._finish(ctx, tuple(unique), items)

    @classmethod
    def options(cls) -> list[StepOption]:
        return [
            *super().options(),
            StepOption("case_sensitive", "boolean",
                       "Treat differently-cased assertions as distinct", default=False),
        ]


class KnowledgeNoiseFilterStep(_RefineStep):
    """Drops candidates whose shape marks them as extraction noise.

    Placeholder subjects, subjects echoing their own section heading, prose with
    no technical marker, and runaway predicates are all signs the model
    described the document rather than asserting something about the subject.
    """

    TYPE = "knowledge.noise_filter"

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        items = self._items(ctx)
        kinds = self._kinds()
        min_object_length = int(self.config.get("min_object_length", 0))
        max_text_length = int(self.config.get("max_text_length", 10_000))
        max_predicate_tokens = int(self.config.get("max_predicate_tokens", 20))

        kept = tuple(
            item
            for item in items
            if item.kind not in kinds
            or not self._is_noise(item, min_object_length, max_text_length, max_predicate_tokens)
        )
        return self._finish(ctx, kept, items)

    @staticmethod
    def _is_noise(
        item: ExtractedItem,
        min_object_length: int,
        max_text_length: int,
        max_predicate_tokens: int,
    ) -> bool:
        content = item.content
        if item.kind == "fact":
            subject = str(content.get("subject", "")).strip()
            predicate = str(content.get("predicate", "")).strip()
            obj = str(content.get("object", "")).strip()

            if subject.startswith("<") and subject.endswith(">"):
                return True
            if subject and subject == str(item.evidence.get("section", "")).strip():
                return True
            if len(predicate.split()) > max_predicate_tokens:
                return True
            if len(obj) < min_object_length:
                return True
            combined = f"{subject} {predicate} {obj}"
            narrative = len(combined) >= 15 and _symbol_ratio(combined) < 0.05
            if narrative and not _TECHNICAL.search(combined):
                return True
            return len(obj) > 120 and not _TECHNICAL.search(obj)

        text = " ".join(str(value) for value in content.values())
        return len(text) > max_text_length

    @classmethod
    def options(cls) -> list[StepOption]:
        return [
            *super().options(),
            StepOption("min_object_length", "integer", "Drop facts with shorter objects",
                       default=0),
            StepOption("max_text_length", "integer", "Drop candidates longer than this",
                       default=10000),
            StepOption("max_predicate_tokens", "integer",
                       "Drop facts whose predicate runs longer than this", default=20),
        ]


class KnowledgeQualityScoreStep(_RefineStep):
    """Adjusts confidence from class, type, and how often a relation recurs.

    A relation asserted about many subjects is usually boilerplate rather than a
    strong claim, so recurrence lowers rather than raises confidence.
    """

    TYPE = "knowledge.quality_score"

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        items = self._items(ctx)
        kinds = self._kinds()
        weights = dict(self.config.get("weights", {}))
        type_weight = dict(self.config.get("type_weight", {}))
        repetition_penalty = float(self.config.get("repetition_penalty", 0.02))

        relations = Counter(
            (_norm(item.content.get("predicate")), _norm(item.content.get("object")))
            for item in items
            if item.kind == "fact"
        )

        scored: list[ExtractedItem] = []
        for item in items:
            if item.kind not in kinds:
                scored.append(item)
                continue
            score = item.confidence
            score += float(type_weight.get(item.type, 0.0))
            score += float(weights.get(f"{item.kind}_bonus", 0.0))
            if item.kind == "fact":
                recurrence = relations[
                    (_norm(item.content.get("predicate")), _norm(item.content.get("object")))
                ]
                score -= repetition_penalty * max(0, recurrence - 1)
            scored.append(replace(item, confidence=min(1.0, max(0.0, score))))
        return self._finish(ctx, tuple(scored), items)

    @classmethod
    def options(cls) -> list[StepOption]:
        return [
            *super().options(),
            StepOption("weights", "object", "Per-kind bonus, e.g. {\"rule_bonus\": 0.12}",
                       default={}),
            StepOption("type_weight", "object", "Per-type confidence adjustment", default={}),
            StepOption("repetition_penalty", "number",
                       "Subtracted per extra occurrence of the same relation", default=0.02),
        ]


class KnowledgeGraphQualityStep(_RefineStep):
    """Penalises structurally weak assertions.

    A missing object, an overlong predicate, or a subject that appears in
    implausibly many facts all indicate the extraction drifted from asserting to
    summarising.
    """

    TYPE = "knowledge.graph_quality"

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        items = self._items(ctx)
        penalties = dict(self.config.get("penalties", {}))
        limits = dict(self.config.get("limits", {}))
        predicate_max = int(limits.get("predicate_max_len", 40))
        object_max = int(limits.get("object_max_len", 400))
        subject_max_facts = int(limits.get("subject_max_facts", 50))

        subjects = Counter(
            _norm(item.content.get("subject")) for item in items if item.kind == "fact"
        )

        scored: list[ExtractedItem] = []
        for item in items:
            if item.kind != "fact":
                scored.append(item)
                continue
            score = item.confidence
            predicate = str(item.content.get("predicate", ""))
            obj = str(item.content.get("object", ""))
            if not obj:
                score -= float(penalties.get("missing_object", 0.2))
            if len(predicate) > predicate_max:
                score -= float(penalties.get("predicate_too_long", 0.1))
            if len(obj) > object_max:
                score -= float(penalties.get("object_too_long", 0.1))
            if subjects[_norm(item.content.get("subject"))] > subject_max_facts:
                score -= float(penalties.get("subject_overloaded", 0.05))
            scored.append(replace(item, confidence=min(1.0, max(0.0, score))))
        return self._finish(ctx, tuple(scored), items)

    @classmethod
    def options(cls) -> list[StepOption]:
        return [
            StepOption("penalties", "object",
                       "Confidence subtracted per structural defect", default={}),
            StepOption("limits", "object",
                       "Structural limits: predicate_max_len, object_max_len, subject_max_facts",
                       default={}),
        ]


class KnowledgeMetaEnrichmentStep(_RefineStep):
    """Derives searchable keywords for each candidate.

    The knowledge store's BM25 side ranks on the observed wording and matches
    ``keywords`` for explicit term lookups; this is where the second comes from.

    It also used to assemble ``text_all``, a bag of every field value. That is
    gone: the wording is indexed now, and the bag was neither observed nor
    quotable — for a pattern it was `description + pattern_name`, both already
    indexed beside it.
    """

    TYPE = "knowledge.meta_enrichment"

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        items = self._items(ctx)
        min_length = int(self.config.get("min_keyword_length", 3))
        max_keywords = int(self.config.get("max_keywords", 24))

        enriched: list[ExtractedItem] = []
        for item in items:
            # Sorted by field name, so the same claim yields the same keywords
            # however the model happened to order its JSON keys — walking
            # `values()` made a claim nobody edited re-index differently on the
            # next run. Nothing may read a role position out of this: it is a
            # term list for explicit lookups, and the language lives in
            # `observed_text`.
            text = " ".join(str(item.content[name]) for name in sorted(item.content))
            tokens = [
                token.lower()
                for token in re.findall(r"[A-Za-z0-9_.\-]+", text)
                if len(token) >= min_length
            ]
            keywords = sorted(dict.fromkeys(tokens))[:max_keywords]
            enriched.append(
                replace(
                    item,
                    metadata={**item.metadata, "keywords": keywords},
                )
            )
        return self._finish(ctx, tuple(enriched), items)

    @classmethod
    def options(cls) -> list[StepOption]:
        return [
            StepOption("min_keyword_length", "integer", "Ignore shorter tokens", default=3),
            StepOption("max_keywords", "integer", "Cap per candidate", default=24),
        ]


class KnowledgeDomainClassifyStep(_RefineStep):
    """Scores structural completeness and labels the resulting domain.

    An assertion missing a field, or carrying a runaway predicate, is a
    process-flavoured observation rather than a technical fact. Recording that
    as a domain label plus explicit penalty reasons lets a later review policy
    quarantine by domain instead of by a bare confidence number.
    """

    TYPE = "knowledge.domain_classify"

    _REQUIRED: dict[str, tuple[str, ...]] = {
        "fact": ("subject", "predicate", "object"),
        "rule": ("rule_text",),
        "pattern": ("pattern_name", "description"),
        "decision": ("decision", "effect"),
    }
    _LONG_FIELD: dict[str, str] = {
        "rule": "rule_text",
        "pattern": "description",
        "decision": "effect",
    }

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        items = self._items(ctx)
        penalties = dict(self.config.get("process_penalty", {}))
        limits = dict(self.config.get("limits", {}))
        thresholds = dict(self.config.get("domain_thresholds", {}))

        low_structure = float(penalties.get("low_structure_penalty", 0.25))
        empty_object = float(penalties.get("empty_object_penalty", 0.3))
        predicate_length = float(penalties.get("predicate_length_penalty", 0.15))
        text_length = float(penalties.get("text_too_long_penalty", 0.2))
        predicate_max = int(limits.get("predicate_max_len", 40))
        text_max = int(limits.get("max_text_len", 500))
        technical = float(thresholds.get("technical", 0.7))
        process = float(thresholds.get("process", 0.4))

        classified: list[ExtractedItem] = []
        for item in items:
            score = item.confidence
            reasons: list[str] = []
            content = item.content

            required = self._REQUIRED.get(item.kind, ())
            if any(not str(content.get(field, "")).strip() for field in required):
                score -= low_structure
                reasons.append(f"incomplete_{item.kind}")

            if item.kind == "fact":
                if not str(content.get("object", "")).strip():
                    score -= empty_object
                    reasons.append("empty_object")
                if len(str(content.get("predicate", ""))) > predicate_max:
                    score -= predicate_length
                    reasons.append("predicate_too_long")
            else:
                field = self._LONG_FIELD.get(item.kind)
                if field and len(str(content.get(field, ""))) > text_max:
                    score -= text_length
                    reasons.append(f"{item.kind}_text_too_long")

            score = min(1.0, max(0.0, score))
            domain = (
                "technical" if score >= technical else "process" if score >= process else "unclear"
            )
            classified.append(
                replace(
                    item,
                    confidence=score,
                    metadata={**item.metadata, "domain": domain, "penalties": reasons},
                )
            )
        return self._finish(ctx, tuple(classified), items)

    @classmethod
    def options(cls) -> list[StepOption]:
        return [
            StepOption("process_penalty", "object",
                       "Confidence subtracted per structural defect", default={}),
            StepOption("limits", "object", "predicate_max_len and max_text_len", default={}),
            StepOption("domain_thresholds", "object",
                       "Score at or above which a candidate is technical / process", default={}),
        ]


class KnowledgeValidateStep(_RefineStep):
    """Drops candidates that are not worth reviewing.

    Three independent gates, all structural: the kind and type must be allowed,
    every required field must carry enough information to mean something, and
    confidence must clear a per-kind threshold. Rules are held to a higher bar
    than facts because a wrong normative statement is more damaging than a wrong
    descriptive one.

    A candidate carrying a technical signal — a flag, path, version, annotation,
    dotted identifier — is exempt from the low-information gate, since short
    technical tokens are precisely what makes those assertions valuable.
    """

    TYPE = "knowledge.validate"

    _REQUIRED = KnowledgeDomainClassifyStep._REQUIRED

    async def run(self, ctx: WorkflowStepContext) -> StepResult:
        items = self._items(ctx)
        allowed_kinds = {str(kind) for kind in self.config.get("allowed_kinds", ALL_KINDS)}
        allowed_types = {str(item) for item in self.config.get("allowed_types", [])}
        min_chars = int(self.config.get("min_chars", 2))
        min_tokens = int(self.config.get("min_tokens", 1))
        fact_threshold = float(self.config.get("confidence_threshold_fact", 0.0))
        rule_threshold = float(
            self.config.get("confidence_threshold_rule", max(fact_threshold, 0.0))
        )

        kept: list[ExtractedItem] = []
        for item in items:
            if item.kind not in allowed_kinds:
                continue
            if allowed_types and item.type not in allowed_types:
                continue

            threshold = rule_threshold if item.kind == "rule" else fact_threshold
            if item.confidence < threshold:
                continue

            fields = [
                str(item.content.get(name, "")).strip()
                for name in self._REQUIRED.get(item.kind, ())
            ]
            if any(not value for value in fields):
                continue
            # Placeholders are rejected unconditionally: "n/a" contains a slash
            # and would otherwise pass as a technical token.
            if any(value in _PLACEHOLDERS for value in fields):
                continue
            if not _TECHNICAL.search(" ".join(fields)) and any(
                self._low_information(value, min_chars, min_tokens) for value in fields
            ):
                continue
            kept.append(item)
        return self._finish(ctx, tuple(kept), items)

    @staticmethod
    def _low_information(value: str, min_chars: int, min_tokens: int) -> bool:
        if len(value) < min_chars:
            return True
        return len([token for token in re.split(r"[\s,;:()\[\]{}]+", value) if token]) < min_tokens

    @classmethod
    def options(cls) -> list[StepOption]:
        return [
            StepOption("allowed_kinds", "array", "Kinds that may pass", default=list(ALL_KINDS)),
            StepOption("allowed_types", "array",
                       "Types that may pass; empty means any", default=[]),
            StepOption("confidence_threshold_fact", "number",
                       "Minimum confidence for non-rule kinds", default=0.0),
            StepOption("confidence_threshold_rule", "number",
                       "Minimum confidence for rules, normally stricter", default=0.0),
            StepOption("min_chars", "integer", "Shortest meaningful field value", default=2),
            StepOption("min_tokens", "integer", "Fewest meaningful tokens per field", default=1),
        ]
