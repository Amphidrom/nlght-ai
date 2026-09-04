# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""What kind a claim is, decided by what it says.

Every extracted assertion gets exactly one kind, and it is decided by the
**semantics of that assertion** — not by the shape of the fields it happens to
fill, not by which file it came from, and not by whatever the model tends to
prefer. Nothing downstream can repair a wrong one: `kind` is part of the identity
namespace, so a sentence read as a `rule` on one pass and a `decision` on the
next retires the claim and creates another (ADR-0048, ADR-0049).

The contract lived nowhere. The extraction prompt named the four kinds and gave
the field shape of each, which is a schema and not a criterion, so the model was
left to guess — and it guessed differently for one identical sentence in two
documents, producing two entity keys where the corpus expected one.

It lives here as data rather than as prose in a prompt string, so that the prompt
is assembled from it and a test can hold each part in place. A definition or a
contrast quietly deleted is a contract that stops binding while every string
still looks right.

**One prompt, not four.** Extraction and classification are one act: the same
pass produces the wording, the structured claim and the kind. A separate
classifier would be a second semantic authority that can disagree with the first.
"""

from __future__ import annotations

from dataclasses import dataclass

from nlght.core.knowledge.knowledge import DECISION, FACT, PATTERN, RULE


@dataclass(slots=True, frozen=True)
class KindDefinition:
    """One kind, and what a claim must be to earn it."""

    kind: str
    definition: str


@dataclass(slots=True, frozen=True)
class Contrast:
    """One sentence and the kind it is, used to draw a line rather than to fill a
    table: each is next to a neighbour it could plausibly be mistaken for."""

    sentence: str
    kind: str
    #: Why it is that kind and not the neighbouring one. Carried into the prompt,
    #: because the boundary is the part that has to survive.
    because: str


#: What each kind *is*. Deliberately short: a long definition invites the model
#: to match its phrasing rather than the claim's meaning.
DEFINITIONS: tuple[KindDefinition, ...] = (
    KindDefinition(
        FACT,
        "a descriptive assertion about what is, holds, or stands in a relation",
    ),
    KindDefinition(
        RULE,
        "a prescriptive or normative requirement: must / shall / may not / "
        "is required / if X then Y is demanded",
    ),
    KindDefinition(
        DECISION,
        "a choice or commitment the source explicitly records as taken",
    ),
    KindDefinition(
        PATTERN,
        "a solution or approach the source asserts as reusable",
    ),
)

#: The lines that are actually crossed. Each names a resemblance that is not
#: enough on its own, because every misclassification seen so far came from one
#: of these rather than from a claim nobody could place.
BOUNDARIES: tuple[str, ...] = (
    "a sentence is not a rule merely because it sounds technical",
    "a conditional is not a rule; a rule requires something of somebody",
    "a rule is not a decision merely because it recommends an action",
    "a decision is not a pattern merely because the choice would be reusable",
    "a pattern is not a fact merely because it describes what is often done",
    "useful architecture is not a pattern; a pattern is claimed as reusable",
)

#: One clear sentence per kind, placed together so each is read against the
#: others. These are the same sentences the live corpus classifies, so a prompt
#: that stops producing them is caught by a run and not only by a reading.
GATE_MATRIX: tuple[Contrast, ...] = (
    Contrast(
        "The management server binds to the same port as the application.",
        FACT,
        "it describes how the server behaves",
    ),
    Contrast(
        "Set management.server.port to use another port.",
        RULE,
        "it requires an action of the reader",
    ),
    Contrast(
        "We chose PostgreSQL instead of MySQL.",
        DECISION,
        "the source records a choice as taken",
    ),
    Contrast(
        "A common pattern is to expose management endpoints on a separate port.",
        PATTERN,
        "the source claims the approach is reusable",
    ),
)

#: The cases that look like one kind and are another. Every one of these was a
#: real disagreement rather than an invented difficulty.
AMBIGUITY_GATES: tuple[Contrast, ...] = (
    Contrast(
        "PostgreSQL can be used for persistence.",
        FACT,
        "a possibility is not a choice anybody made",
    ),
    Contrast(
        "You should use PostgreSQL.",
        RULE,
        "it tells the reader what to do; no choice is recorded as taken",
    ),
    Contrast(
        "We use PostgreSQL.",
        FACT,
        "a decision only where the source states the choice was made; "
        "bare present tense describes what is done",
    ),
    Contrast(
        "Using a separate management port is a common approach.",
        PATTERN,
        "it is offered as a reusable approach rather than as one system's trait",
    ),
    Contrast(
        "Metrics are exported to Prometheus when the registry is on the classpath.",
        FACT,
        "a described condition, not a demand",
    ),
    Contrast(
        "Schema scripts run before data scripts.",
        FACT,
        "a described order or guarantee, not a norm",
    ),
    Contrast(
        "A layered jar separates dependencies from application classes.",
        FACT,
        "it describes a mechanism; a pattern would claim it as reusable",
    ),
)

#: One sentence may say several things, and each is its own claim. Kept with the
#: contract because it is the same instruction: a kind belongs to an assertion,
#: never to a sentence, so a sentence that asserts two things gets two kinds.
ATOMICITY: tuple[str, ...] = (
    "One source sentence may contain several atomic assertions.",
    "Each assertion is extracted separately and classified separately.",
    "Several assertions may carry the same observed_text; that is correct, "
    "not a duplicate.",
)


def _contrasts(items: tuple[Contrast, ...]) -> str:
    return "\n".join(
        f'  "{item.sentence}"\n      -> {item.kind}  ({item.because})' for item in items
    )


def contract() -> str:
    """The contract as the extraction prompt carries it.

    Assembled rather than written out, so the prompt cannot drift from the
    definitions the tests and the corpus are held to. It is hashed into
    `extraction_version`, so changing any part of it re-extracts the corpus
    instead of leaving old classifications standing beside new ones.
    """
    definitions = "\n".join(
        f"  {item.kind:9}{item.definition}" for item in DEFINITIONS
    )
    boundaries = "\n".join(f"  - {line}" for line in BOUNDARIES)
    atomicity = "\n".join(f"  - {line}" for line in ATOMICITY)
    return (
        "Choose the kind from what the assertion means, not from which fields it\n"
        "would fill, not from the document it came from, and not from habit.\n"
        "Exactly one kind per assertion.\n\n"
        f"{definitions}\n\n"
        "Boundaries:\n"
        f"{boundaries}\n\n"
        "One of each, read against the others:\n"
        f"{_contrasts(GATE_MATRIX)}\n\n"
        "Cases that resemble the wrong kind:\n"
        f"{_contrasts(AMBIGUITY_GATES)}\n\n"
        "Atomicity:\n"
        f"{atomicity}"
    )
