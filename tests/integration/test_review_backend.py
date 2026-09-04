# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""The standalone review backend, over a real knowledge schema.

Deliberately end-to-end through the HTTP surface: the point of this service is
that a reviewer can work the queue without the runtime, so the contract worth
testing is the one they actually reach.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from review.backend import app as app_module
from sqlalchemy.ext.asyncio import create_async_engine

from nlght.adapters.outbound.persistence.knowledge_models import KnowledgeBase
from nlght.adapters.outbound.persistence.knowledge_repository import (
    SqlAlchemyKnowledgeRepository,
)
from nlght.core.knowledge import FactPayload, KnowledgeWrite, RulePayload


def _fact(
    identity: str,
    *,
    confidence: float = 0.4,
    review: bool = True,
    source_id: str = "handbook",
    run_id: str = "run-1",
) -> KnowledgeWrite:
    return KnowledgeWrite(
        identity=identity,
        payload=FactPayload(subject="Spring", predicate="requires", object="Java 17"),
        type="dependency",
        confidence=confidence,
        source_id=source_id,
        run_id=run_id,
        product="spring-boot",
        version="3.2",
        review_required=review,
        review_reason="low confidence" if review else None,
        evidence={"quote": "Spring Boot 3.2 requires Java 17"},
    )


@pytest.fixture
async def seeded():
    """A knowledge database with a queue in it, wired into the app."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(KnowledgeBase.metadata.create_all)
    repository = SqlAlchemyKnowledgeRepository(engine)

    await repository.upsert(_fact("unsure", confidence=0.2))
    await repository.upsert(
        KnowledgeWrite(
            identity="rule",
            payload=RulePayload(rule_text="Never log secrets"),
            type="security",
            confidence=0.6,
            source_id="handbook",
            run_id="run-1",
            review_required=True,
            review_reason="policy",
            evidence={"quote": "secrets must not be logged"},
        )
    )
    # A different document, so a merge actually moves evidence rather than
    # hitting the (identity, run_id, source_id) dedup.
    await repository.upsert(
        _fact("settled", confidence=0.95, review=False, source_id="release-notes", run_id="run-2")
    )

    app_module.app.state.repository = repository
    yield repository
    await engine.dispose()


@pytest.fixture
def client(seeded):
    # The app's own lifespan builds an engine from the environment; the fixture
    # has already supplied one, so it is deliberately not run here.
    return TestClient(app_module.app)


def test_the_queue_is_least_confident_first_and_omits_retrievable_knowledge(client) -> None:
    body = client.get("/api/queue").json()

    assert [item["identity"] for item in body] == ["unsure", "rule"]
    assert all(item["review_required"] for item in body)
    assert body[0]["payload"]["subject"] == "Spring"


def test_the_queue_pages(client) -> None:
    first = client.get("/api/queue", params={"limit": 1}).json()
    second = client.get("/api/queue", params={"limit": 1, "offset": 1}).json()

    assert [item["identity"] for item in first] == ["unsure"]
    assert [item["identity"] for item in second] == ["rule"]


def test_detail_carries_evidence_history_neighbours_and_findings(client) -> None:
    body = client.get("/api/assertions/unsure").json()

    assert body["assertion"]["identity"] == "unsure"
    assert [item["run_id"] for item in body["observations"]] == ["run-1"]
    assert body["observations"][0]["evidence"]["quote"].startswith("Spring Boot")
    assert body["reviews"] == []
    # Least confident first, so the neighbour ahead is the more confident one.
    assert (body["previous"], body["next"]) == (None, "rule")

    kinds = {finding["kind"] for finding in body["findings"]}
    assert "low confidence" in kinds       # 0.2 is well under the band
    assert "single observation" in kinds   # one sighting, not independent agreement
    # 'settled' is another fact of the same type and must be offered as context.
    assert [item["identity"] for item in body["similar"]] == ["settled"]


def test_an_unknown_identity_is_a_404(client) -> None:
    assert client.get("/api/assertions/never-extracted").status_code == 404


def test_approving_clears_the_queue_entry_and_records_the_reviewer(client) -> None:
    response = client.post(
        "/api/assertions/unsure/review",
        json={"decision": "approved", "comment": "checked the quote"},
        headers={"X-Reviewer-Id": "alice"},
    )

    assert response.status_code == 200
    assert response.json()["reviewer_id"] == "alice"
    assert [item["identity"] for item in client.get("/api/queue").json()] == ["rule"]
    assert client.get("/api/assertions/unsure").json()["reviews"][0]["comment"] == (
        "checked the quote"
    )


def test_rejecting_also_clears_the_queue_but_keeps_the_quarantine(client) -> None:
    client.post("/api/assertions/unsure/review", json={"decision": "rejected"})

    assert [item["identity"] for item in client.get("/api/queue").json()] == ["rule"]
    assert client.get("/api/assertions/unsure").json()["assertion"]["review_required"] is True


def test_editing_corrects_the_claim_and_records_what_changed(client) -> None:
    response = client.post(
        "/api/assertions/unsure/review",
        json={
            "decision": "edited",
            "comment": "singular subject",
            "payload": {"subject": "Spring Boot", "predicate": "requires", "object": "Java 17"},
            "confidence": 0.9,
        },
        headers={"X-Reviewer-Id": "bob"},
    )

    assert response.status_code == 200
    changes = response.json()["changes"]
    assert changes["payload"]["subject"] == {"from": "Spring", "to": "Spring Boot"}
    assert changes["confidence"]["to"] == 0.9

    detail = client.get("/api/assertions/unsure").json()
    assert detail["assertion"]["payload"]["subject"] == "Spring Boot"
    assert detail["assertion"]["confidence"] == 0.9
    # Editing lifts the quarantine, so the assertion becomes retrievable.
    assert detail["assertion"]["review_required"] is False


def test_a_correction_on_a_non_edit_decision_is_refused(client) -> None:
    # Otherwise content could change with no record of it as a change.
    response = client.post(
        "/api/assertions/unsure/review",
        json={
            "decision": "approved",
            "payload": {"subject": "x", "predicate": "y", "object": "z"},
        },
    )

    assert response.status_code == 422
    assert "only an 'edited' decision" in response.json()["detail"]


def test_an_empty_field_is_refused_the_same_way_an_extracted_one_would_be(client) -> None:
    response = client.post(
        "/api/assertions/unsure/review",
        json={
            "decision": "edited",
            "payload": {"subject": "  ", "predicate": "requires", "object": "Java 17"},
        },
    )

    assert response.status_code == 422
    assert "must not be empty" in response.json()["detail"]


def test_an_unknown_decision_names_the_ones_that_exist(client) -> None:
    response = client.post("/api/assertions/unsure/review", json={"decision": "maybe"})

    assert response.status_code == 422
    assert "approved" in response.json()["detail"]


def test_statistics_count_the_queue_as_the_queue_defines_it(client) -> None:
    before = client.get("/api/stats").json()
    assert (before["total"], before["pending"], before["retrievable"]) == (3, 2, 1)
    assert before["merged"] == 0

    client.post("/api/assertions/unsure/review", json={"decision": "rejected"})
    after = client.get("/api/stats").json()

    # Rejection leaves the quarantine but ends the review, so pending drops
    # while retrievable stays put.
    assert (after["pending"], after["retrievable"]) == (1, 1)


def test_merging_folds_one_assertion_into_another(client) -> None:
    # 'settled' is the other fact of the same type, so it is offered as a target.
    targets = [item["identity"] for item in client.get("/api/assertions/unsure").json()["similar"]]
    assert targets == ["settled"]

    response = client.post(
        "/api/assertions/unsure/review",
        json={"decision": "merged", "merge_into": "settled", "comment": "same claim"},
        headers={"X-Reviewer-Id": "carol"},
    )

    assert response.status_code == 200
    assert response.json()["changes"]["merged_into"]["target"] == "settled"

    # The source leaves the queue and stops being canonical, but keeps its trail.
    assert [item["identity"] for item in client.get("/api/queue").json()] == ["rule"]
    detail = client.get("/api/assertions/unsure").json()
    assert detail["assertion"]["merged_into"] == "settled"
    assert [review["decision"] for review in detail["reviews"]] == ["merged"]

    # Its evidence now belongs to the survivor.
    survivor = client.get("/api/assertions/settled").json()
    assert len(survivor["observations"]) == 2
    assert client.get("/api/stats").json()["merged"] == 1


def test_a_merge_without_a_target_is_refused(client) -> None:
    response = client.post("/api/assertions/unsure/review", json={"decision": "merged"})

    assert response.status_code == 422
    assert "requires a merge target" in response.json()["detail"]


def test_merging_into_something_that_does_not_exist_is_refused(client) -> None:
    response = client.post(
        "/api/assertions/unsure/review",
        json={"decision": "merged", "merge_into": "never-extracted"},
    )

    assert response.status_code == 422
    assert "unknown merge target" in response.json()["detail"]
