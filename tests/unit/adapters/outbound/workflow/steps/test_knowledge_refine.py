# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Refinement chain: each step's structural rule, and what it must not do."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from nlght.adapters.outbound.workflow.steps.knowledge import (
    KnowledgeAtomicityStep,
    KnowledgeCanonicalizeStep,
    KnowledgeCollapseStep,
    KnowledgeDedupStep,
    KnowledgeGraphQualityStep,
    KnowledgeIdentityMergeStep,
    KnowledgeMetaEnrichmentStep,
    KnowledgeNoiseFilterStep,
    KnowledgeNormalizeStep,
    KnowledgeQualityScoreStep,
)
from nlght.core.entry.context import RequestContext
from nlght.core.errors.errors import WorkflowConfigurationError
from nlght.core.knowledge import ExtractedItem
from nlght.core.protocol.protocol import ProtocolKind
from nlght.core.trigger.trigger import Trigger, TriggerKind
from nlght.core.workflow.step import WorkflowStepContext


class _Emitter:
    async def emit(self, signal: object) -> None: ...


def _ctx(items: tuple[ExtractedItem, ...]) -> WorkflowStepContext:
    context = RequestContext(
        correlation_id="run-1", request_id="rid", received_at=datetime(2026, 1, 1, tzinfo=UTC),
        path="/", method="POST", headers={}, query_params={}, client_host=None,
    )
    ctx = WorkflowStepContext(
        correlation_id="run-1",
        trigger=Trigger(
            kind=TriggerKind.INBOUND_EVENT, protocol=ProtocolKind.GENERIC_JSON,
            operation="ingest", payload={}, context=context,
        ),
        model="", messages=[], stream=False, emitter=_Emitter(),
    )
    ctx.metadata["knowledge.extracted"] = items
    return ctx


def _fact(subject="Spring", predicate="requires", obj="Java 17", *, confidence=0.6, **kw):
    return ExtractedItem(
        kind="fact",
        type=kw.pop("type", "dependency"),
        content={"subject": subject, "predicate": predicate, "object": obj},
        confidence=confidence,
        **kw,
    )


async def _run(step, items):
    ctx = _ctx(items)
    result = await step.run(ctx)
    return result.ctx.metadata["knowledge.extracted"]


# -- canonicalize ------------------------------------------------------------


async def test_canonicalize_makes_spacing_variants_structurally_equal() -> None:
    items = (_fact("  Spring   Boot ", "requires", "Java 17."),)

    out = await _run(KnowledgeCanonicalizeStep(config={}), items)

    assert out[0].content == {
        "subject": "Spring Boot", "predicate": "requires", "object": "Java 17"
    }


# -- normalize ---------------------------------------------------------------


async def test_normalize_marks_a_prose_predicate_as_narrative() -> None:
    long_predicate = "is generally considered to be required by most of the"
    out = await _run(KnowledgeNormalizeStep(config={}), (_fact(predicate=long_predicate),))

    assert out[0].metadata["form"] == "narrative"


async def test_normalize_marks_a_short_predicate_as_relational() -> None:
    out = await _run(KnowledgeNormalizeStep(config={}), (_fact(),))

    assert out[0].metadata["form"] == "relational"


# -- atomicity ---------------------------------------------------------------


async def test_atomicity_splits_a_conjunction_into_reviewable_claims() -> None:
    out = await _run(KnowledgeAtomicityStep(config={}), (_fact(obj="Java 17, Maven"),))

    assert sorted(item.content["object"] for item in out) == ["Java 17", "Maven"]


async def test_atomicity_leaves_prose_alone() -> None:
    # Long fragments are sentences, not list items; splitting them invents claims.
    prose = "the runtime and also whatever else the operator happens to configure"
    out = await _run(KnowledgeAtomicityStep(config={}), (_fact(obj=prose),))

    assert len(out) == 1


async def test_atomicity_gives_up_on_a_combinatorial_explosion() -> None:
    items = (_fact(subject="A, B, C", obj="D, E, F"),)

    out = await _run(KnowledgeAtomicityStep(config={"max_splits": 4}), items)

    assert len(out) == 1


# -- identity merge ----------------------------------------------------------


async def test_identity_merge_keeps_the_strongest_of_equal_claims() -> None:
    items = (_fact(confidence=0.4), _fact(confidence=0.8))

    out = await _run(KnowledgeIdentityMergeStep(config={}), items)

    assert len(out) == 1
    # Repetition inside one run is the same evidence twice, not corroboration,
    # so confidence is the maximum rather than a sum.
    assert out[0].confidence == 0.8


# -- collapse ----------------------------------------------------------------


async def test_collapse_merges_parallel_relations() -> None:
    items = (_fact(obj="A"), _fact(obj="B"))

    out = await _run(KnowledgeCollapseStep(config={}), items)

    assert len(out) == 1
    assert out[0].content["object"] == "A, B"


async def test_collapse_leaves_different_predicates_apart() -> None:
    items = (_fact(predicate="requires", obj="A"), _fact(predicate="supports", obj="B"))

    out = await _run(KnowledgeCollapseStep(config={}), items)

    assert len(out) == 2


# -- dedup -------------------------------------------------------------------


async def test_dedup_drops_case_variants_by_default() -> None:
    items = (_fact("Spring"), _fact("SPRING"))

    assert len(await _run(KnowledgeDedupStep(config={}), items)) == 1


async def test_dedup_can_be_made_case_sensitive() -> None:
    items = (_fact("Spring"), _fact("SPRING"))

    out = await _run(KnowledgeDedupStep(config={"case_sensitive": True}), items)

    assert len(out) == 2


async def test_dedup_keeps_two_n_ary_facts_that_differ_in_a_fourth_role() -> None:
    """The key had reached past the n-ary work in front of it.

    A fact was keyed on `(subject, predicate, object)`, so two claims differing
    only in a further role were one key and the second was dropped here —
    silently, before anything was persisted, while `Proposition` and the entity
    key had long since carried every role.
    """
    staging_to_archive = ExtractedItem(
        kind="fact", type="migration", confidence=0.9,
        content={"predicate": "transfers", "subject": "migration tool",
                 "object": "500 records", "source": "staging", "target": "archive"},
    )
    staging_to_backup = ExtractedItem(
        kind="fact", type="migration", confidence=0.9,
        content={"predicate": "transfers", "subject": "migration tool",
                 "object": "500 records", "source": "staging", "target": "backup"},
    )

    out = await _run(KnowledgeDedupStep(config={}), (staging_to_archive, staging_to_backup))

    assert len(out) == 2, "two different claims were folded into one"


async def test_dedup_keeps_two_rules_that_share_a_sentence_but_not_a_subject() -> None:
    """A rule was keyed on `rule_text` alone — and that is now often the sentence.

    Since the wording is observed rather than assembled (ADR-0048), `rule_text`
    commonly holds the source sentence verbatim. One sentence stating two
    requirements therefore produced two rules with identical `rule_text` and
    different subjects, and the second was dropped here: neither `subject` nor
    `rule_property` took part in the key.
    """
    sentence = "Access tokens must not be logged and sessions must expire after an hour."
    tokens = ExtractedItem(
        kind="rule", type="policy", confidence=0.9, observed_text=sentence,
        content={"subject": "access_token", "rule_property": "logging_prohibition",
                 "rule_text": sentence},
    )
    sessions = ExtractedItem(
        kind="rule", type="policy", confidence=0.9, observed_text=sentence,
        content={"subject": "session", "rule_property": "expiry",
                 "rule_text": sentence},
    )

    assert len(await _run(KnowledgeDedupStep(config={}), (tokens, sessions))) == 2


async def test_dedup_still_drops_the_same_claim_read_twice() -> None:
    # The point of the step, unchanged: field order is not a difference.
    first = ExtractedItem(
        kind="fact", type="dependency", confidence=0.9,
        content={"subject": "Spring", "predicate": "requires", "object": "Java 17"},
    )
    reordered = ExtractedItem(
        kind="fact", type="dependency", confidence=0.4,
        content={"object": "Java 17", "predicate": "requires", "subject": "Spring"},
    )

    assert len(await _run(KnowledgeDedupStep(config={}), (first, reordered))) == 1


async def test_two_claims_read_from_one_sentence_are_not_one_claim() -> None:
    """Atomicity, at the step that could most easily undo it.

    One sentence may state several assertions and they share their wording by
    design (ADR-0049), so `observed_text` must stay out of the key.
    """
    sentence = "Only the health endpoint is exposed, and shutdown is disabled."
    exposed = ExtractedItem(
        kind="fact", type="endpoint", confidence=0.9, observed_text=sentence,
        content={"predicate": "is_exposed", "subject": "health endpoint", "object": "true"},
    )
    disabled = ExtractedItem(
        kind="fact", type="endpoint", confidence=0.9, observed_text=sentence,
        content={"predicate": "is_disabled", "subject": "shutdown endpoint", "object": "true"},
    )

    out = await _run(KnowledgeDedupStep(config={}), (exposed, disabled))

    assert len(out) == 2
    assert {item.observed_text for item in out} == {sentence}


async def test_a_kind_is_part_of_the_key() -> None:
    # Two readings of one sentence that disagree about its kind are two claims,
    # not a duplicate: `kind` is part of the identity namespace (ADR-0049), so
    # folding them here would pick a winner nothing downstream could see.
    content = {"subject": "management server", "rule_property": "port_binding",
               "rule_text": "The management server binds to the application's port."}
    as_rule = ExtractedItem(kind="rule", type="config", confidence=0.9, content=dict(content))
    as_fact = ExtractedItem(kind="fact", type="config", confidence=0.9, content=dict(content))

    assert len(await _run(KnowledgeDedupStep(config={}), (as_rule, as_fact))) == 2


# -- noise -------------------------------------------------------------------


async def test_noise_filter_drops_placeholder_subjects() -> None:
    assert await _run(KnowledgeNoiseFilterStep(config={}), (_fact("<subject>"),)) == ()


async def test_noise_filter_drops_a_subject_echoing_its_section() -> None:
    items = (_fact("Installation", evidence={"section": "Installation"}),)

    assert await _run(KnowledgeNoiseFilterStep(config={}), items) == ()


async def test_noise_filter_drops_a_runaway_predicate() -> None:
    items = (_fact(predicate=" ".join(["word"] * 25)),)

    assert await _run(KnowledgeNoiseFilterStep(config={}), items) == ()


async def test_noise_filter_keeps_a_technical_assertion() -> None:
    assert len(await _run(KnowledgeNoiseFilterStep(config={}), (_fact(),))) == 1


# -- scoring -----------------------------------------------------------------


async def test_quality_score_rewards_a_configured_kind() -> None:
    out = await _run(
        KnowledgeQualityScoreStep(config={"weights": {"fact_bonus": 0.2}}),
        (_fact(confidence=0.5),),
    )

    assert out[0].confidence == pytest.approx(0.7)


async def test_a_relation_repeated_across_subjects_is_treated_as_boilerplate() -> None:
    items = tuple(_fact(subject=f"S{i}", confidence=0.8) for i in range(5))

    out = await _run(KnowledgeQualityScoreStep(config={"repetition_penalty": 0.1}), items)

    assert all(item.confidence < 0.8 for item in out)


async def test_confidence_never_leaves_the_unit_interval() -> None:
    out = await _run(
        KnowledgeQualityScoreStep(config={"weights": {"fact_bonus": 5.0}}),
        (_fact(confidence=0.9),),
    )

    assert out[0].confidence == 1.0


async def test_graph_quality_penalises_a_missing_object() -> None:
    out = await _run(
        KnowledgeGraphQualityStep(config={"penalties": {"missing_object": 0.3}}),
        (_fact(obj="", confidence=0.9),),
    )

    assert out[0].confidence == pytest.approx(0.6)


async def test_graph_quality_penalises_an_overlong_predicate() -> None:
    out = await _run(
        KnowledgeGraphQualityStep(
            config={"penalties": {"predicate_too_long": 0.2}, "limits": {"predicate_max_len": 5}}
        ),
        (_fact(confidence=0.9),),
    )

    assert out[0].confidence == pytest.approx(0.7)


# -- metadata ----------------------------------------------------------------


async def test_meta_enrichment_produces_keywords_and_no_assembled_text() -> None:
    """Terms for explicit lookups, and nothing that pretends to be language.

    It also used to emit `text_all`, a bag of every field value. The wording is
    indexed now, so the bag was a second copy of fields already indexed beside
    it — and it was never observed, so it could not be quoted.
    """
    out = await _run(KnowledgeMetaEnrichmentStep(config={}), (_fact(),))

    assert "spring" in out[0].metadata["keywords"]
    assert "text_all" not in out[0].metadata


async def test_meta_enrichment_drops_short_tokens_and_caps_the_list() -> None:
    long_object = " ".join(f"token{i}" for i in range(40)) + " a an"

    out = await _run(
        KnowledgeMetaEnrichmentStep(config={"max_keywords": 5}), (_fact(obj=long_object),)
    )

    keywords = out[0].metadata["keywords"]
    assert len(keywords) == 5
    assert "a" not in keywords


# -- shared contract ---------------------------------------------------------


@pytest.mark.parametrize(
    "step_cls",
    [
        KnowledgeCanonicalizeStep, KnowledgeNormalizeStep, KnowledgeAtomicityStep,
        KnowledgeIdentityMergeStep, KnowledgeCollapseStep, KnowledgeDedupStep,
        KnowledgeNoiseFilterStep, KnowledgeQualityScoreStep, KnowledgeGraphQualityStep,
        KnowledgeMetaEnrichmentStep,
    ],
)
async def test_every_refinement_needs_extracted_candidates(step_cls) -> None:
    ctx = _ctx(())
    del ctx.metadata["knowledge.extracted"]

    with pytest.raises(WorkflowConfigurationError, match="knowledge.extract"):
        await step_cls(config={}).run(ctx)


@pytest.mark.parametrize(
    "step_cls",
    [
        KnowledgeCanonicalizeStep, KnowledgeNormalizeStep, KnowledgeAtomicityStep,
        KnowledgeIdentityMergeStep, KnowledgeCollapseStep, KnowledgeDedupStep,
        KnowledgeNoiseFilterStep, KnowledgeQualityScoreStep, KnowledgeGraphQualityStep,
        KnowledgeMetaEnrichmentStep,
    ],
)
async def test_every_refinement_is_composable_in_any_order(step_cls) -> None:
    # Each writes back to the same key, so a workflow can chain them freely.
    out = await _run(step_cls(config={}), (_fact(),))

    assert isinstance(out, tuple)


# -- domain classification ---------------------------------------------------


async def test_domain_classify_labels_a_complete_assertion_technical() -> None:
    from nlght.adapters.outbound.workflow.steps.knowledge import KnowledgeDomainClassifyStep

    out = await _run(KnowledgeDomainClassifyStep(config={}), (_fact(confidence=0.9),))

    assert out[0].metadata["domain"] == "technical"
    assert out[0].metadata["penalties"] == []


async def test_domain_classify_penalises_and_explains_an_incomplete_fact() -> None:
    from nlght.adapters.outbound.workflow.steps.knowledge import KnowledgeDomainClassifyStep

    out = await _run(KnowledgeDomainClassifyStep(config={}), (_fact(obj="", confidence=0.9),))

    assert out[0].confidence < 0.9
    # The reasons are recorded so a review policy can act on them.
    assert "empty_object" in out[0].metadata["penalties"]
    assert "incomplete_fact" in out[0].metadata["penalties"]


async def test_domain_classify_marks_a_weak_candidate_unclear() -> None:
    from nlght.adapters.outbound.workflow.steps.knowledge import KnowledgeDomainClassifyStep

    out = await _run(KnowledgeDomainClassifyStep(config={}), (_fact(obj="", confidence=0.5),))

    assert out[0].metadata["domain"] == "unclear"


# -- validation --------------------------------------------------------------


def _rule(text="Never log secrets", *, confidence=0.9):
    return ExtractedItem(kind="rule", type="security",
                         content={"rule_text": text}, confidence=confidence)


async def test_validate_holds_rules_to_a_stricter_bar_than_facts() -> None:
    from nlght.adapters.outbound.workflow.steps.knowledge import KnowledgeValidateStep

    # A wrong normative statement is more damaging than a wrong descriptive one.
    step = KnowledgeValidateStep(
        config={"confidence_threshold_fact": 0.5, "confidence_threshold_rule": 0.8}
    )
    out = await _run(step, (_fact(confidence=0.6), _rule(confidence=0.6)))

    assert [item.kind for item in out] == ["fact"]


async def test_validate_drops_an_incomplete_candidate() -> None:
    from nlght.adapters.outbound.workflow.steps.knowledge import KnowledgeValidateStep

    assert await _run(KnowledgeValidateStep(config={}), (_fact(obj=""),)) == ()


async def test_validate_drops_placeholder_values() -> None:
    from nlght.adapters.outbound.workflow.steps.knowledge import KnowledgeValidateStep

    assert await _run(KnowledgeValidateStep(config={}), (_fact(obj="n/a"),)) == ()


async def test_validate_keeps_short_technical_tokens() -> None:
    from nlght.adapters.outbound.workflow.steps.knowledge import KnowledgeValidateStep

    # "-Xmx" is short but is exactly the kind of assertion worth keeping.
    step = KnowledgeValidateStep(config={"min_chars": 8, "min_tokens": 3})
    out = await _run(step, (_fact(subject="JVM", predicate="accepts", obj="-Xmx"),))

    assert len(out) == 1


async def test_validate_can_restrict_kinds_and_types() -> None:
    from nlght.adapters.outbound.workflow.steps.knowledge import KnowledgeValidateStep

    by_kind = await _run(
        KnowledgeValidateStep(config={"allowed_kinds": ["rule"]}), (_fact(), _rule())
    )
    assert [item.kind for item in by_kind] == ["rule"]

    by_type = await _run(
        KnowledgeValidateStep(config={"allowed_types": ["security"]}), (_fact(), _rule())
    )
    assert [item.type for item in by_type] == ["security"]


# -- what the chain says it did ----------------------------------------------

async def test_a_step_reports_how_many_candidates_it_rewrote(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The number that says who moved a key.

    A rerun over a corpus that was only reworded, coming back with an assertion
    more, has four steps between the model and the key that could have caused
    it. Counts cannot tell them apart: canonicalisation leaves `in` and `out`
    flat while rewriting the very field the key is built from. Without this the
    model carries the blame for whatever a rewrite rule did.
    """
    caplog.set_level("INFO")

    await _run(KnowledgeCanonicalizeStep(config={}), (_fact(obj="Java  17."), _fact()))

    line = next(m for m in caplog.messages if "knowledge.canonicalize.done" in m)
    # The messy one was rewritten onto the identity the clean one already had.
    # Counted against a set this would report nothing, because that identity was
    # handed in — and a claim quietly merging into another is precisely what a
    # rerun needs to see.
    assert "in=2 out=2 rewritten=1" in line


async def test_a_step_that_changes_nothing_reports_nothing_changed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("INFO")

    await _run(KnowledgeCanonicalizeStep(config={}), (_fact(), _fact(obj="Java 21")))

    line = next(m for m in caplog.messages if "knowledge.canonicalize.done" in m)
    assert "in=2 out=2 rewritten=0" in line


async def test_a_split_counts_both_halves_as_new_identities(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Atomicity invents claims rather than rewording one, and the number says so
    # — which is the difference between "a step moved a key" and "a step made
    # two assertions out of one".
    caplog.set_level("INFO")

    await _run(KnowledgeAtomicityStep(config={}), (_fact(obj="Java 17 and Maven"),))

    line = next(m for m in caplog.messages if "knowledge.atomicity.done" in m)
    assert "in=1 out=2 rewritten=2" in line
