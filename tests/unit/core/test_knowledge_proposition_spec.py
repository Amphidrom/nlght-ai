# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""What an atomic proposition is, before anything is built to hold one.

The gate on the fact model, and deliberately written against today's code rather
than against a design. A test that passes because the shape it describes does not
exist yet proves nothing; these are red where the current model cannot express
what a proposition needs, and each red one names what is missing.

**No data model is chosen here.** Not `roles: dict`, not fixed `subject`/`object`,
not triples, not event objects, not predicate schemas, not customer ontologies.
The earlier sketch — `FactPayload` as predicate plus a role map — is a hypothesis
and was demoted from a decision, because it moves the identity defect rather than
removing it: a model producing `data_key`/`origin` on one pass can produce
`key`/`property_source` on the next, so identity would follow invented role
*labels* instead of invented subjects.

Two decisions are settled and shape everything below.

**Argument roles are identity-relevant, and relations are directed by default.**
The platform may not assume symmetry, because no sentence says whether a relation
has it and guessing would need a registry nobody can maintain. A swap is a
different proposition until an explicit equivalence decision over the *whole*
propositions says otherwise — never a property attributed to a predicate.

**Retrieval may not project lossily.** A consumer that understands only
`subject`/`predicate`/`object` cannot represent a relation with more roles, and
must be told so as a typed state at the boundary rather than handed a quietly
shortened claim.

Four of these were `xfail(strict=True)` while the behaviour was missing, and
closing them was not a matter of making the old assertions pass. Three of the
four asserted a probe the design has since rejected: that a four-role claim makes
a legacy payload, and that two descriptions of one claim produce one *key*. A key
that folded a role rename would fold a role swap with it, and those two must end
opposite. Each now asserts what actually holds, and names where the end-to-end
version of it lives.

One thing stays open and worth keeping in view: `fact` may not be the data model
at all. It may be one *kind* of proposition, and if triples, n-ary relations and
events want different representations, that is an answer rather than a problem to
force back into one payload.
"""

from __future__ import annotations

from nlght.core.knowledge import ExtractedItem, Proposition, entity_key


def _fact(**content: str) -> ExtractedItem:
    return ExtractedItem(kind="fact", type="dependency", content=content, confidence=0.9)


# ---------------------------------------------------------------------------
# 1. One fact is one atomic proposition
# ---------------------------------------------------------------------------

async def test_a_distributive_list_may_be_split() -> None:
    """"provides access to key, value and origin" — three claims, if each holds.

    Splitting is allowed exactly when the parts can be true or false
    independently, which they are here: `SanitizableData` provides access to the
    key whether or not it provides access to the origin.
    """
    from tests.unit.adapters.outbound.workflow.steps.test_knowledge_refine import (  # noqa: PLC0415
        _ctx,
    )

    from nlght.adapters.outbound.workflow.steps.knowledge.refine import (
        KnowledgeAtomicityStep,
    )

    step = KnowledgeAtomicityStep(config={})
    ctx = _ctx((_fact(subject="SanitizableData", predicate="provides access to",
                      object="key, value and origin"),))

    result = await step.run(ctx)

    assert len(result.ctx.metadata["knowledge.extracted"]) == 3


def test_a_relation_with_more_than_three_roles_stays_one_proposition() -> None:
    """"Alice transfers CHF 500 to Bob" — one claim, three arguments.

    Splitting it destroys it: "Alice transfers CHF 500" and "Alice transfers to
    Bob" are not what the sentence said, and neither is true on its own. So it
    has to be held *as one*.

    While this was red it asserted `payload() is not None`, and that probe was
    the defect rather than a measure of it. The legacy payload takes a subject, a
    predicate and one object; a version of it that swallowed a fourth role would
    be the lossy projection invariant 7 forbids. So the claim is held whole and
    the old row is simply absent — and carried through the pipeline into
    retrieval by the gate in `tests/e2e/test_knowledge_corpus_lifecycle.py`.
    """
    item = _fact(subject="Alice", predicate="transfers", amount="CHF 500", recipient="Bob")

    item.well_formed()

    assert dict(item.proposition.fields) == {
        "subject": "Alice", "predicate": "transfers",
        "amount": "CHF 500", "recipient": "Bob",
    }
    assert item.payload() is None


def test_the_entity_key_already_accepts_more_than_three_roles() -> None:
    # The half that works, asserted so the failure above is read as the payload's
    # and not as the key's.
    key = entity_key(
        "fact", predicate="transfers", subject="Alice", amount="CHF 500", recipient="Bob"
    )

    assert key.key


# ---------------------------------------------------------------------------
# 2-3. Identity follows the proposition, not its description
# ---------------------------------------------------------------------------

def test_the_same_claim_said_actively_and_passively_is_one_claim() -> None:
    """"Spring requires Maven" and "Maven is required by Spring".

    One claim, two sentences — and it is *not* the key that says so. While this
    was red it asserted key equality, and closing it that way would have been the
    defect ADR-0043 removed: a key that folded a role swap would fold
    `transfers(Alice, Bob)` into `transfers(Bob, Alice)`, which are opposite
    claims. Nothing over the structure separates those two cases.

    So the keys differ, deliberately, and what joins the claims is a judgement
    over the whole propositions: `equivalent` expresses it, `continuation` acts
    on it, and end to end it is the renamed-role gate in
    `tests/e2e/test_knowledge_corpus_lifecycle.py`.
    """
    from nlght.core.knowledge.equivalence import (  # noqa: PLC0415
        SAME,
        AssertionObservation,
        equivalent,
        may_merge,
    )

    active = entity_key("fact", predicate="requires", subject="spring", object="maven")
    passive = entity_key("fact", predicate="required_by", subject="maven", object="spring")
    assert active != passive

    verdict = equivalent(
        AssertionObservation.of(Proposition(
            {"predicate": "requires", "subject": "spring", "object": "maven"})),
        AssertionObservation.of(Proposition(
            {"predicate": "required_by", "subject": "maven", "object": "spring"})),
        verdict=SAME,
    )

    assert may_merge(verdict)


def test_freely_generated_role_names_do_not_change_the_claim() -> None:
    """The defect ADR-0043 removed, in the shape it would come back in.

    A model producing `key`/`value`/`origin` on one pass and
    `data_key`/`data_value`/`property_source` on the next has described one claim
    twice. Identity following those labels is identity following the model's word
    choice, which is what the surrogate exists to prevent — and a key that folded
    them would be following them just as much, only in the other direction.

    The keys differ; the claim is one claim because a judge said so. Same
    mechanism as the active/passive case, and for the same reason: a role rename
    and a role swap are indistinguishable over the structure.
    """
    from nlght.core.knowledge.equivalence import (  # noqa: PLC0415
        SAME,
        AssertionObservation,
        equivalent,
        may_merge,
    )

    first = entity_key(
        "fact", predicate="provides access to", subject="SanitizableData",
        key="key", value="value", origin="PropertySource",
    )
    second = entity_key(
        "fact", predicate="provides access to", subject="SanitizableData",
        data_key="key", data_value="value", property_source="PropertySource",
    )
    assert first != second

    verdict = equivalent(
        AssertionObservation.of(Proposition(
            {"predicate": "provides access to", "subject": "SanitizableData",
             "key": "key", "value": "value", "origin": "PropertySource"})),
        AssertionObservation.of(Proposition(
            {"predicate": "provides access to", "subject": "SanitizableData",
             "data_key": "key", "data_value": "value",
             "property_source": "PropertySource"})),
        verdict=SAME,
    )

    assert may_merge(verdict)


# ---------------------------------------------------------------------------
# 4-6. What must stay apart
# ---------------------------------------------------------------------------

def test_a_moved_value_is_a_different_proposition() -> None:
    # Java 17 against Java 21. Whether the second supersedes the first is a
    # lineage question and never an identity one.
    assert entity_key("fact", predicate="requires", subject="spring", object="java 17") != \
           entity_key("fact", predicate="requires", subject="spring", object="java 21")


def test_swapping_the_roles_of_a_directed_relation_is_a_different_claim() -> None:
    """Directed by default, and this is that decision as a test.

    `transfers(Alice, Bob)` and `transfers(Bob, Alice)` are opposite claims. The
    platform may not decide from a sentence that a relation is symmetric — no
    sentence says so, and a predicate registry that claimed to know would be a
    vocabulary nobody can maintain. So a swap is a different proposition until an
    explicit equivalence decision over the whole propositions says otherwise.
    """
    assert entity_key("fact", predicate="transfers", sender="Alice", recipient="Bob") != \
           entity_key("fact", predicate="transfers", sender="Bob", recipient="Alice")


def test_two_claims_about_one_subject_are_both_live() -> None:
    """Simultaneously true, and never two states of one thing.

    "Spring requires Java" and "Spring requires Maven" are two claims. Reading
    several arguments as a sequence of states is how a corpus turns a list into a
    history it never had.
    """
    java = entity_key("fact", predicate="requires", subject="spring", object="java")
    maven = entity_key("fact", predicate="requires", subject="spring", object="maven")

    assert java != maven


def test_the_same_extraction_asked_twice_is_the_same_proposition() -> None:
    # The invariant everything above rests on. A key that moved between two
    # identical readings would make every other guarantee here unobservable.
    first = _fact(subject="spring", predicate="requires", object="java 17")
    second = _fact(subject="spring", predicate="requires", object="java 17")

    assert first.fingerprint == second.fingerprint
    assert entity_key("fact", predicate="requires", subject="spring", object="java 17") == \
           entity_key("fact", predicate="requires", subject="spring", object="java 17")


# ---------------------------------------------------------------------------
# 7. Nothing is delivered quietly shortened
# ---------------------------------------------------------------------------

def test_generic_retrieval_already_delivers_a_claim_whole() -> None:
    """Measured, not assumed — and it narrowed the gap it was meant to prove.

    "Retrieval loses n-ary propositions" was written down as a gap and used to
    justify an abstraction. The serialiser had never lost anything: it hands over
    whatever a claim carries and drops nothing for being unfamiliar. The
    measurement lives beside the serialiser, in
    `tests/unit/adapters/outbound/stores/test_knowledge_delivers_whole.py`.

    What was actually missing is narrower, and shape-free: a way to tell a
    consumer that asked for a particular *shape* that this claim does not fit in
    it. Which shapes exist is still undecided, so this asserts only that a
    refusal can be expressed — not what it refuses.
    """
    from nlght.core.knowledge import NotRepresentable  # noqa: PLC0415

    refusal = NotRepresentable(
        shape="whatever the caller asked for",
        reason="it does not fit",
        available=("amount", "predicate", "recipient", "subject"),
    )

    assert refusal.available == ("amount", "predicate", "recipient", "subject")
    assert "amount" in str(refusal)


def test_a_fact_that_cannot_name_what_it_claims_is_not_a_claim() -> None:
    """Two parts and nothing naming the relation between them.

    `{subject: Alice, object: Bob}` passed the gate while it read "at least two
    fields", and that approximation cost something concrete: no predicate means
    no entity key, no entity key means no lineage, no lineage means no revision —
    and a revision is where a fact's structure lives. It reached the graph as a
    row with no payload, no proposition and no revision, asserting nothing.

    So the gate asks the question the placement will ask. Not a new rule about
    what a claim must contain: `_fact_key` has always required a predicate and
    something standing in the relation, and this is that requirement asked early
    enough that the candidate can still be counted and reported.
    """
    import pytest  # noqa: PLC0415

    with pytest.raises(ValueError, match="name what it claims"):
        _fact(subject="Alice", object="Bob").well_formed()

    with pytest.raises(ValueError, match="name what it claims"):
        _fact(predicate="transfers").well_formed()


def test_a_fact_that_can_be_placed_passes_however_many_roles_it_has() -> None:
    # The guard above must not cost the thing this whole slice was for.
    _fact(subject="Alice", predicate="transfers", amount="CHF 500",
          recipient="Bob").well_formed()
    _fact(subject="spring", predicate="requires", object="maven").well_formed()
