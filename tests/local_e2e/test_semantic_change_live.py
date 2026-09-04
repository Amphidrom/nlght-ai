# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""The seven cases, put to a real model.

The unit suite proves the decision is asked for, honoured, and conservative when
the answer is doubt. It cannot prove that a model gets `must` becoming `must not`
right, because that is a property of the model — and the whole reason this one
judgement is delegated is that nothing in this codebase can settle it.

So the same seven cases are asked here of a live one. Needs a local Ollama and is
deselected by default:

    pytest tests/local_e2e/test_semantic_change_live.py -m local_e2e

Read a failure as a statement about the model, not about the pipeline. A model
that cannot separate `must not` from `shall` is not usable for this, and finding
that out costs a minute here rather than an audit later. `ambiguous` is not a
failure — it sends the revision to review, which is the safe direction — but a
model that answers it to everything is doing no work.
"""

from __future__ import annotations

import os

import httpx2
import pytest

from nlght.adapters.outbound.knowledge.semantic_classifier import (
    ModelPropositionClassifier,
)
from nlght.core.knowledge.semantics import (
    NORMATIVE,
    REWRITE,
    UNCHANGED,
    semantic_change,
)

pytestmark = pytest.mark.local_e2e

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
PREFER_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")

#: The specification, verbatim, with the answer the *model* must give where the
#: model is the one deciding. `None` marks a case settled before any call — those
#: exercise the deterministic half and say nothing about the model.
#:
#: Asserting the verdict rather than only the outcome is not pedantry. A dead
#: backend answers `ambiguous`, which resolves to `normative` — so a test checking
#: only the outcome would report every `normative` case as passing while nothing
#: was asked at all.
#:
#: The rewrite case reads "Expenses over CHF 500" and not "Anything over CHF 500",
#: which is what it said until a model called it normative and was right: *anything*
#: is broader than *expenses*, so the subject had moved and the rule with it. A
#: specification example that is not actually an example of what it specifies makes
#: every result unreadable.
CASES = [
    ("Requests must be signed.", "Requests must not be signed.", NORMATIVE, NORMATIVE),
    ("Clients may retry a failed request.", "Clients must retry a failed request.",
     NORMATIVE, NORMATIVE),
    ("Caching is enabled by default.", "Caching is disabled by default.", NORMATIVE, NORMATIVE),
    ("Requests must be signed.", "Requests shall be signed.", REWRITE, REWRITE),
    ("Expenses above CHF 500 require approval.",
     "Expenses over CHF 500 must be signed off.", REWRITE, REWRITE),
    ("Expenses above CHF 500 require approval.",
     "Expenses above CHF 550 require approval.", NORMATIVE, None),
    ("Lines are limited to 100 characters",
     "Lines  are   limited to 100 characters.", UNCHANGED, None),
]


class _Emitter:
    async def emit(self, signal: object) -> None: ...


def _local_model() -> str:
    """`OLLAMA_MODEL` if it is pulled, else whatever is — as the provider suite does."""
    try:
        response = httpx2.get(f"{OLLAMA_HOST}/api/tags", timeout=5.0)
        if response.status_code != 200:
            pytest.skip(f"Ollama at {OLLAMA_HOST} did not list its models.")
        names = [
            model["name"]
            for model in response.json().get("models", [])
            if not model.get("remote_model")
        ]
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"Ollama not reachable at {OLLAMA_HOST}: {exc}")
    for name in names:
        if name == PREFER_MODEL or name.startswith(f"{PREFER_MODEL}:"):
            return name
    if not names:
        pytest.skip(f"No model pulled in Ollama at {OLLAMA_HOST} — try `ollama pull qwen3:8b`.")
    return names[0]


@pytest.fixture
def classifier():  # noqa: ANN201
    from nlght.adapters.outbound.model import OllamaClient

    backend = OllamaClient(
        http_client=httpx2.AsyncClient(),
        base_url=OLLAMA_HOST,
        default_model=_local_model(),
    )
    return ModelPropositionClassifier(
        backend.bind(model=None, emitter=_Emitter(), stream=True)
    )


@pytest.mark.parametrize(("old", "new", "expected", "answer"), CASES)
async def test_the_specification_holds_against_a_real_model(
    classifier, old: str, new: str, expected: str, answer: str | None
) -> None:
    verdict = await classifier.classify(old, new)

    if answer is not None:
        # The model is what is under test here, so its answer is asserted rather
        # than only the outcome. `ambiguous` fails: it is the safe result and it
        # is not an answer, and a backend that never replied would otherwise look
        # like three passing cases.
        assert verdict == answer, (
            f"model answered {verdict!r}, expected {answer!r}:"
            f"\n  old: {old}\n  new: {new}"
        )
    assert semantic_change(old, new, verdict=verdict) == expected


async def test_the_answer_does_not_move_between_two_asks(classifier) -> None:
    """Pinned, or the corpus drifts on its own.

    An answer that varies run to run would retire and re-review claims nobody
    edited — the same drift the extraction skip and the resolver exist to remove,
    arriving through the mechanism added to close the last gap in them.
    """
    old, new = "Requests must be signed.", "Requests shall be signed."

    first = await classifier.classify(old, new)
    second = await classifier.classify(old, new)

    assert first == second
