# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""
nlght/application/hive_mind/layered_observe_extractor.py
---------------------------------------------------------
Layered observe stage — drop-in replacement for SpacyMemoryExtractor.

Implements the MemoryExtractor protocol and can be passed directly to
MemoryIngestionPipeline:

    pipeline = MemoryIngestionPipeline(
        extractor=LayeredObserveExtractor()
    )

Stages:
    1. Structural pre-filter — isolates code blocks, extracts labels
                               purely via regex/structure, < 1ms
    2. Intent classifier     — rule-based on reference_text, < 1ms
                               optional: FastText model injectable
    3. NER (selective)       — spaCy only on reference_text (without code),
                               lazy-loaded, skipped when there is no prose text
    4. Dedup guard           — hash-based, eliminates duplicate candidates
                               before they reach the pipeline
"""

from __future__ import annotations

import ast
import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from nlght.core.hive_mind.extraction import (
    ExtractedArtifact,
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
    MemoryCandidate,
    MemoryCandidateConfidence,
    MemoryCandidateKind,
)

# ---------------------------------------------------------------------------
# Stage 1 — Structural pre-filter
# ---------------------------------------------------------------------------

# Recognized label markers (DE + EN), order: most specific first
_LABEL_MARKERS = (
    "hier hast du",
    "hier ist",
    "das ist",
    "this is",
    "speichere",
    "merke dir",
    "remember",
    "store",
)

# Markers indicating a request, not an artifact label
_REQUEST_MARKERS = (
    "vergleiche",
    "vergleich",
    "prüfe",
    "pruefe",
    "analysiere",
    "was ",
    "welche",
    "zeige",
    "show",
    "compare",
    "check",
    "analyse",
    "analyze",
)

# Languages recognized as code
_CODE_LANGUAGES = frozenset({
    "python", "py", "javascript", "js", "typescript", "ts",
    "java", "csharp", "cs", "go", "rust", "rs",
    "bash", "sh", "yaml", "yml", "json", "toml", "sql",
    "html", "css", "xml", "dockerfile", "text",
})

_FENCED_BLOCK_RE = re.compile(
    r"```(?P<lang>[^\n`]*)\n(?P<body>.*?)```",
    re.DOTALL,
)

_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")

_WHITESPACE_RE = re.compile(r"\s+")


def _normalise(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text.strip(" .,:;!?\"'`\u201c\u201d"))


def _label_slug(label: str) -> str:
    chars = []
    for ch in label.strip().lower():
        if ch.isalnum() or ch in {"_", ".", "-"}:
            chars.append(ch)
        elif chars and chars[-1] != "_":
            chars.append("_")
    return "".join(chars).strip("_") or "artifact"


def _is_request_label(label: str) -> bool:
    lo = label.lower()
    return any(lo.startswith(m) for m in _REQUEST_MARKERS)


def _extract_label_from_prefix(prefix: str) -> str:
    """Extracts a label from the text preceding a fenced block.

    Strategy (in this order):
    1. Last non-empty sentence / last line
    2. Marker strip (e.g. "Hier hast du Code 1:" → "Code 1")
    3. Colon split (last part before ":")
    4. Fallback: empty → no artifact stored
    """
    if not prefix.strip():
        return ""

    # Last non-empty line
    lines = [ln.strip() for ln in prefix.splitlines() if ln.strip()]
    if not lines:
        return ""
    line = lines[-1].strip("\"'`\u201c\u201d")

    # Marker strip
    lo = line.lower()
    for marker in _LABEL_MARKERS:
        if lo.startswith(marker):
            line = line[len(marker):].strip(" :")
            break

    # Colon split: "Hier hast du meinen Service A:" → "meinen Service A"
    if ":" in line:
        line = line.rsplit(":", 1)[0].strip()

    result = _normalise(line)

    # Minimum length: ignore too-short or obvious request fragments
    if len(result) < 2 or _is_request_label(result):
        return ""
    return result


@dataclass
class _FencedBlock:
    label: str          # can be empty
    language: str
    content: str
    start: int          # char offset in the original text
    end: int


def _extract_fenced_blocks(text: str) -> list[_FencedBlock]:
    blocks: list[_FencedBlock] = []
    for m in _FENCED_BLOCK_RE.finditer(text):
        lang = m.group("lang").strip().lower() or "text"
        if lang not in _CODE_LANGUAGES:
            lang = "text"
        content = m.group("body").strip()
        if not content:
            continue
        prefix = text[: m.start()]
        label = _extract_label_from_prefix(prefix)
        blocks.append(_FencedBlock(
            label=label,
            language=lang,
            content=content,
            start=m.start(),
            end=m.end(),
        ))
    return blocks


def _reference_text(text: str, blocks: list[_FencedBlock]) -> str:
    """Text without fenced blocks and without inline code — remaining prose."""
    result = _FENCED_BLOCK_RE.sub(" ", text)
    result = _INLINE_CODE_RE.sub(" ", result)
    return _WHITESPACE_RE.sub(" ", result).strip()


# ---------------------------------------------------------------------------
# Stage 2 — Intent classifier (rule-based, injectable)
# ---------------------------------------------------------------------------

_COMPARE_KEYWORDS = frozenset({
    "vergleiche", "vergleich", "gemeinsamkeiten", "unterschiede",
    "vergleichen", "ähnlichkeiten", "compare", "comparison", "diff",
    "versus", "vs",
})
_RECALL_KEYWORDS = frozenset({
    "gemerkt", "weißt", "weisst", "erinnerst", "recall",
    "was weißt", "was weiß", "what do you know", "remember",
})
_STORE_KEYWORDS = frozenset({
    "speichere", "merke", "hier hast du", "hier ist", "store",
    "remember this", "keep this",
})
_UNCERTAINTY_WORDS = frozenset({
    "vielleicht", "glaube", "eventuell", "evtl", "später",
    "maybe", "perhaps", "possibly", "might",
})


@runtime_checkable
class IntentClassifier(Protocol):
    def classify(self, text: str) -> str: ...


class RuleBasedIntentClassifier:
    """Fast rule-based classifier. Replaceable with FastText/LLM."""

    def classify(self, text: str) -> str:
        tokens = set(_WHITESPACE_RE.sub(" ", text.lower()).split())
        bigrams = self._bigrams(text.lower())
        if tokens & _COMPARE_KEYWORDS or bigrams & _COMPARE_KEYWORDS:
            return "compare"
        if tokens & _RECALL_KEYWORDS or bigrams & _RECALL_KEYWORDS:
            return "recall"
        return "store"

    @staticmethod
    def _bigrams(text: str) -> set[str]:
        words = text.split()
        return {f"{a} {b}" for a, b in zip(words, words[1:], strict=False)}

    def has_uncertainty(self, text: str) -> bool:
        tokens = set(text.lower().split())
        return bool(tokens & _UNCERTAINTY_WORDS)


# ---------------------------------------------------------------------------
# Stage 3 — NER (selective, only on reference_text)
# ---------------------------------------------------------------------------

_SPACY_MODELS = ("de_core_news_sm", "en_core_web_sm")

# POS tags considered entities (spaCy universal POS)
_ENTITY_POS = frozenset({"PROPN", "NOUN"})

# Token texts incorrectly recognized as entities in a code context
_CODE_STOPWORDS = frozenset({
    "if", "else", "elif", "for", "while", "return", "import", "from",
    "class", "def", "async", "await", "with", "as", "try", "except",
    "true", "false", "none", "null", "undefined", "var", "let", "const",
    "self", "cls", "args", "kwargs", "text", "data", "result", "value",
    "str", "int", "bool", "list", "dict", "set", "tuple", "any",
    "currently", "a", "an", "the",  # common false positives from the log
})


def _load_spacy(model_names: tuple[str, ...] = _SPACY_MODELS) -> Callable[[str], Any]:
    try:
        import spacy  # noqa: PLC0415 (lazy import: optional extra or deliberate startup-cost/cycle avoidance)
    except ImportError as exc:
        raise ImportError(
            "NER stage requires spaCy. Install nlght-ai[nlp] "
            "or inject an IntentClassifier without NER."
        ) from exc
    for name in model_names:
        try:
            nlp: Callable[[str], Any] = spacy.load(name)
            return nlp
        except Exception:
            continue
    raise ImportError(f"Kein spaCy-Modell ladbar: {model_names}")


def _ner_entities_from_text(
    text: str,
    nlp: Callable[[str], Any],
) -> list[ExtractedEntity]:
    """NER only on cleaned prose text, with a code-stopword filter."""
    if not text.strip():
        return []
    doc = nlp(text)
    entities: list[ExtractedEntity] = []
    seen: set[str] = set()
    for ent in doc.ents:
        raw = str(getattr(ent, "text", "")).strip()
        lo = raw.lower()
        if not raw or lo in _CODE_STOPWORDS or len(raw) < 2:
            continue
        if lo not in seen:
            seen.add(lo)
            entities.append(ExtractedEntity(
                text=raw,
                kind=str(getattr(ent, "label_", "entity") or "entity"),
                canonical=raw,
            ))
    return entities


# ---------------------------------------------------------------------------
# Stufe 4 — Dedup-Guard
# ---------------------------------------------------------------------------

def _candidate_hash(content: str) -> str:
    return hashlib.sha1(content.encode("utf-8", errors="replace")).hexdigest()[:12]


class _Deduplicator:
    def __init__(self) -> None:
        self._seen: set[str] = set()

    def is_duplicate(self, candidate: MemoryCandidate) -> bool:
        h = _candidate_hash(candidate.content)
        if h in self._seen:
            return True
        self._seen.add(h)
        return False

    def filter(self, candidates: list[MemoryCandidate]) -> list[MemoryCandidate]:
        return [c for c in candidates if not self.is_duplicate(c)]


# ---------------------------------------------------------------------------
# Haupt-Extraktor
# ---------------------------------------------------------------------------

@dataclass
class LayeredObserveExtractor:
    """Layered observe stage — drop-in for SpacyMemoryExtractor.

    Args:
        intent_classifier:  Overridable with a FastText/LLM classifier.
        spacy_models:       Order of spaCy models (lazy-loaded).
        use_ner:            Enable the NER stage (default: True).
                            False = pure rule-based mode, maximum speed.
        min_label_length:   Minimum length for artifact labels.
    """

    intent_classifier: IntentClassifier = field(
        default_factory=RuleBasedIntentClassifier
    )
    spacy_models: tuple[str, ...] = _SPACY_MODELS
    use_ner: bool = True
    min_label_length: int = 2

    _nlp: Any = field(default=None, init=False, repr=False)

    async def extract(self, text: str) -> ExtractionResult:
        # --- Stufe 1: Struktur ---
        blocks = _extract_fenced_blocks(text)
        ref_text = _reference_text(text, blocks)

        # --- Stufe 2: Intent ---
        intent = self.intent_classifier.classify(ref_text or text)
        uncertain = (
            self.intent_classifier.has_uncertainty(ref_text)
            if isinstance(self.intent_classifier, RuleBasedIntentClassifier)
            else False
        )

        # --- Stufe 3: NER (selektiv) ---
        ner_entities: list[ExtractedEntity] = []
        if self.use_ner and ref_text.strip():
            try:
                if self._nlp is None:
                    self._nlp = _load_spacy(self.spacy_models)
                ner_entities = _ner_entities_from_text(ref_text, self._nlp)
            except ImportError:
                pass  # No spaCy → rule-based extraction only

        # --- Artifacts from blocks ---
        artifacts = self._artifacts_from_blocks(blocks)
        artifact_labels = {a.label for a in artifacts if a.label}

        # Entities: NER results + artifact labels
        entities: list[ExtractedEntity] = list(ner_entities)
        for label in artifact_labels:
            if not any(e.text == label for e in entities):
                entities.append(ExtractedEntity(
                    text=label,
                    kind="artifact_label",
                    canonical=label,
                ))

        # --- Relations from artifact symbols ---
        relations = self._artifact_relations(artifacts)

        # --- Referenced labels ---
        referenced_labels = self._referenced_labels(ref_text, artifact_labels)

        # --- Assemble candidates ---
        dedup = _Deduplicator()
        candidates: list[MemoryCandidate] = []

        # Artifact candidates
        for artifact in artifacts:
            c = self._artifact_candidate(artifact)
            if not dedup.is_duplicate(c):
                candidates.append(c)

        # Relation candidates (only from clean artifact relations)
        for relation in relations:
            c = self._relation_candidate(relation, uncertain)
            if not dedup.is_duplicate(c):
                candidates.append(c)

        # Request candidate for compare/recall
        if intent in {"compare", "recall"}:
            c = self._request_candidate(intent, referenced_labels)
            if not dedup.is_duplicate(c):
                candidates.append(c)

        return ExtractionResult(
            intent=intent,
            entities=entities,
            artifacts=artifacts,
            relations=relations,
            candidates=candidates,
            referenced_labels=referenced_labels,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _artifacts_from_blocks(
        self, blocks: list[_FencedBlock]
    ) -> list[ExtractedArtifact]:
        artifacts: list[ExtractedArtifact] = []
        seen_labels: set[str] = set()
        for block in blocks:
            label = block.label
            if not label or len(label) < self.min_label_length:
                continue
            lo = label.lower()
            if lo in seen_labels:
                continue
            seen_labels.add(lo)
            content_type = "code" if block.language != "text" else "text"
            artifacts.append(ExtractedArtifact(
                label=label,
                content=block.content,
                content_type=content_type,
                language=block.language,
            ))
        return artifacts

    def _artifact_relations(
        self, artifacts: list[ExtractedArtifact]
    ) -> list[ExtractedRelation]:
        relations: list[ExtractedRelation] = []
        seen: set[tuple[str, str, str]] = set()
        for artifact in artifacts:
            for symbol in self._python_symbols(artifact):
                key = (artifact.label.lower(), "contains", symbol.lower())
                if key in seen:
                    continue
                seen.add(key)
                relations.append(ExtractedRelation(
                    subject=artifact.label,
                    predicate="contains",
                    object=symbol,
                    confidence=0.95,
                ))
        return relations

    @staticmethod
    def _python_symbols(artifact: ExtractedArtifact) -> list[str]:
        if artifact.language not in {"python", "py"}:
            return []
        try:
            tree = ast.parse(artifact.content)
        except SyntaxError:
            return []
        symbols = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                symbols.append(node.name)
        return symbols

    def _referenced_labels(
        self,
        ref_text: str,
        artifact_labels: set[str],
    ) -> list[str]:
        """Finds references to known labels in the prose text."""
        found: list[str] = []
        lo_text = ref_text.lower()
        for label in artifact_labels:
            if label.lower() in lo_text:
                found.append(label)
        return found

    @staticmethod
    def _artifact_candidate(artifact: ExtractedArtifact) -> MemoryCandidate:
        slug = _label_slug(artifact.label)
        return MemoryCandidate(
            key=f"artifact.{slug}",
            kind=MemoryCandidateKind.ARTIFACT,
            content=(
                f"Artefakt {artifact.label} ({artifact.language}):\n"
                f"```{artifact.language}\n{artifact.content}\n```"
            ),
            entities=[artifact.label, "artifact"],
            tags=["user_fact", "artifact", slug],
            confidence=MemoryCandidateConfidence.VALIDATED,
        )

    @staticmethod
    def _relation_candidate(
        relation: ExtractedRelation,
        uncertain: bool,
    ) -> MemoryCandidate:
        predicates = {
            "uses": "nutzt",
            "contains": "enthält",
            "depends_on": "hängt ab von",
            "is_a": "ist",
        }
        predicate_de = predicates.get(relation.predicate, relation.predicate)
        content = f"{relation.subject} {predicate_de} {relation.object}."
        confidence = (
            MemoryCandidateConfidence.UNCERTAIN
            if uncertain and relation.predicate != "contains"
            else MemoryCandidateConfidence.VALIDATED
        )
        slug = f"relation.{_label_slug(relation.subject)}.{_label_slug(relation.predicate)}.{_label_slug(relation.object)}"
        return MemoryCandidate(
            key=slug,
            kind=MemoryCandidateKind.RELATION,
            content=content,
            entities=[relation.subject, relation.object],
            tags=["user_fact", "relation", _label_slug(relation.predicate)],
            confidence=confidence,
        )

    @staticmethod
    def _request_candidate(
        intent: str,
        referenced_labels: list[str],
    ) -> MemoryCandidate:
        if intent == "compare":
            refs = ", ".join(referenced_labels) if referenced_labels else ""
            content = (
                "Der Nutzer fordert einen Vergleich an"
                + (f" und referenziert: {refs}." if refs else ".")
            )
            return MemoryCandidate(
                key="comparison.request",
                kind=MemoryCandidateKind.REQUEST,
                content=content,
                entities=referenced_labels or ["comparison"],
                tags=["request", "comparison"],
                confidence=MemoryCandidateConfidence.TRANSIENT,
            )
        return MemoryCandidate(
            key="recall.request",
            kind=MemoryCandidateKind.REQUEST,
            content="Der Nutzer fragt nach bereits gespeichertem Session-Wissen.",
            entities=["memory"],
            tags=["request", "recall"],
            confidence=MemoryCandidateConfidence.TRANSIENT,
        )


# ---------------------------------------------------------------------------
# Smoke test (python -m layered_observe_extractor)
# ---------------------------------------------------------------------------

async def _smoke_test() -> None:
    extractor = LayeredObserveExtractor(use_ner=False)

    tests = [
        # Turn 1: code with label
        'Hier hast du Code 1:\n```python\nclass ServiceA:\n    def run(self): pass\n```',
        # Turn 2: code without label (former problem)
        '```python\nif text.startswith("```"):\n    text = text.split("\\n", 1)[-1]\n```',
        # Turn 3: comparison
        'Vergleiche Code 1 mit diesem Code:\n```python\nclass ServiceB:\n    def run(self): pass\n```\nWelche Gemeinsamkeiten bestehen mit Code 2?',
        # Turn 4: dedup test (identical content)
        'Hier hast du Code 1:\n```python\nclass ServiceA:\n    def run(self): pass\n```',
    ]

    seen_hashes: set[str] = set()
    for i, text in enumerate(tests, 1):
        print(f"\n{'='*60}")
        print(f"TURN {i}: {text[:60].strip()!r}")
        print("="*60)
        result = await extractor.extract(text)
        print(f"Intent:    {result.intent}")
        print(f"Artifacts: {[a.label for a in result.artifacts]}")
        print(f"Entities:  {[e.text for e in result.entities]}")
        print(f"Relations: {[(r.subject, r.predicate, r.object) for r in result.relations]}")
        print(f"Candidates ({len(result.candidates)}):")
        for c in result.candidates:
            h = _candidate_hash(c.content)
            dup = "DUP" if h in seen_hashes else "NEW"
            seen_hashes.add(h)
            print(f"  [{dup}] [{c.confidence.value}] {c.key}")
        print(f"Referenced: {result.referenced_labels}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(_smoke_test())
