# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from nlght.adapters.outbound.ingestion.confluence_source import (
    ConfluenceDocumentSource,
    ConfluenceResponse,
    ConfluenceSourceSettings,
)
from nlght.adapters.outbound.ingestion.filesystem_source import (
    FilesystemDocumentSource,
    FilesystemRoot,
    FilesystemSourceSettings,
)


class _ConfluenceTransport:
    def __init__(self, responses: list[ConfluenceResponse | Exception]) -> None:
        self.responses = responses
        self.requests: list[tuple[Mapping[str, str | int], str]] = []

    async def search(
        self,
        *,
        base_url: str,
        params: Mapping[str, str | int],
        authorization: str,
        timeout_seconds: float,
    ) -> ConfluenceResponse:
        assert base_url == "https://docs.example/wiki"
        assert timeout_seconds == 3.0
        self.requests.append((params, authorization))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _confluence_settings(**overrides) -> ConfluenceSourceSettings:
    values = {
        "source_id": "engineering-wiki",
        "base_url": "https://docs.example/wiki",
        "space_keys": ("ENG",),
        "user_email": "reader@example.test",
        "api_token": "injected-test-token",
        "cql": "label=approved",
        "page_size": 2,
        "max_pages": 10,
        "timeout_seconds": 3.0,
        "retry_base_seconds": 0.0,
        "retry_max_seconds": 0.0,
    }
    values.update(overrides)
    return ConfluenceSourceSettings(**values)


@pytest.mark.asyncio
async def test_filesystem_source_applies_patterns_and_nested_ignore_files(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    nested = root / "nested"
    nested.mkdir(parents=True)
    (root / ".gitignore").write_text("ignored.py\n", encoding="utf-8")
    (nested / ".botignore").write_text("*.generated.py\n", encoding="utf-8")
    (root / "keep.py").write_text("def keep(): pass\n", encoding="utf-8")
    (root / "ignored.py").write_text("ignored\n", encoding="utf-8")
    (root / "excluded.py").write_text("excluded\n", encoding="utf-8")
    (root / "not-included.txt").write_text("text\n", encoding="utf-8")
    (nested / "keep.py").write_text("def nested(): pass\n", encoding="utf-8")
    (nested / "code.generated.py").write_text("generated\n", encoding="utf-8")
    (nested / "binary.py").write_bytes(b"code\0binary")

    source = FilesystemDocumentSource(
        FilesystemSourceSettings(
            source_id="source-a",
            roots=(FilesystemRoot(root, alias="repo"),),
            include=("**/*.py",),
            exclude=("excluded.py",),
        )
    )

    snapshot = await source.acquire()

    assert snapshot.complete
    assert [document.external_id for document in snapshot.documents] == [
        "fs:repo/keep.py",
        "fs:repo/nested/binary.py",
        "fs:repo/nested/keep.py",
    ]
    assert snapshot.documents[1].content == b"code\0binary"
    assert snapshot.documents[2].path == "repo/nested/keep.py"
    assert snapshot.observed_external_ids == tuple(document.external_id for document in snapshot.documents)


@pytest.mark.asyncio
async def test_filesystem_source_walks_directories_more_than_one_level_deep(
    tmp_path: Path,
) -> None:
    # Regression: the ignore-file chain rejoined the relative parents of a
    # directory without prefixing them with the root, so `relative_to(root)`
    # raised ValueError once the walk descended two levels (e.g. `a/b`).
    root = tmp_path / "repository"
    deep = root / "a" / "b"
    deep.mkdir(parents=True)
    (root / ".gitignore").write_text("skip.py\n", encoding="utf-8")
    (root / "top.py").write_text("def top(): pass\n", encoding="utf-8")
    (deep / "leaf.py").write_text("def leaf(): pass\n", encoding="utf-8")
    (deep / "skip.py").write_text("skipped\n", encoding="utf-8")

    source = FilesystemDocumentSource(
        FilesystemSourceSettings(
            source_id="source-deep",
            roots=(FilesystemRoot(root, alias="repo"),),
            include=("**/*.py",),
        )
    )

    snapshot = await source.acquire()

    assert snapshot.complete
    assert [document.external_id for document in snapshot.documents] == [
        "fs:repo/a/b/leaf.py",
        "fs:repo/top.py",
    ]


@pytest.mark.asyncio
async def test_filesystem_source_marks_missing_root_incomplete_and_rejects_alias_collision(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    source = FilesystemDocumentSource(FilesystemSourceSettings(source_id="source-a", roots=(FilesystemRoot(missing, alias="repo"),)))

    snapshot = await source.acquire()

    assert not snapshot.complete
    assert snapshot.documents == ()
    assert snapshot.diagnostics[0].code == "root-unavailable"
    with pytest.raises(ValueError, match="aliases must be unique"):
        FilesystemSourceSettings(
            source_id="source-a",
            roots=(FilesystemRoot(tmp_path, alias="same"), FilesystemRoot(tmp_path, alias="same")),
        )


@pytest.mark.asyncio
async def test_confluence_source_paginates_preserves_revisions_and_omits_archived_pages() -> None:
    transport = _ConfluenceTransport(
        [
            ConfluenceResponse(
                200,
                {},
                {
                    "results": [
                        {
                            "id": "10",
                            "title": "First",
                            "status": "current",
                            "body": {"storage": {"value": "<h1>Heading</h1><p>Hello <b>world</b></p>"}},
                            "version": {"number": 7},
                            "_links": {"webui": "/spaces/ENG/pages/10"},
                        },
                        {"id": "11", "status": "archived"},
                    ],
                    "_links": {"next": "/next"},
                },
            ),
            ConfluenceResponse(
                200,
                {},
                {
                    "results": [
                        {
                            "id": "12",
                            "title": "Second",
                            "body": {"storage": {"value": "<p>Safe</p><script>hidden()</script>"}},
                            "version": {"number": 2},
                            "_links": {},
                        }
                    ]
                },
            ),
        ]
    )
    source = ConfluenceDocumentSource(_confluence_settings(), transport)

    snapshot = await source.acquire()

    assert snapshot.complete
    assert snapshot.observed_external_ids == ("confluence:10", "confluence:12")
    assert [document.source_revision for document in snapshot.documents] == ["7", "2"]
    assert snapshot.documents[0].content.decode() == "Heading\nHello world"
    assert snapshot.documents[0].metadata["url"] == "https://docs.example/wiki/spaces/ENG/pages/10"
    assert snapshot.documents[1].content.decode() == "Safe"
    assert [request[0]["start"] for request in transport.requests] == [0, 2]
    assert transport.requests[0][0]["cql"] == "(space=ENG and type=page) and (label=approved)"
    assert all(request[1].startswith("Basic ") for request in transport.requests)


@pytest.mark.asyncio
async def test_confluence_source_retries_rate_limit_without_exposing_secret() -> None:
    delays: list[float] = []

    async def record_sleep(delay: float) -> None:
        delays.append(delay)

    transport = _ConfluenceTransport(
        [
            ConfluenceResponse(429, {"Retry-After": "1.5"}, {}),
            ConfluenceResponse(200, {}, {"results": []}),
        ]
    )
    source = ConfluenceDocumentSource(
        _confluence_settings(retry_max_seconds=2.0),
        transport,
        sleep=record_sleep,
    )

    snapshot = await source.acquire()

    assert snapshot.complete
    assert delays == [1.5]
    assert "injected-test-token" not in repr(snapshot)
    assert "injected-test-token" not in repr(source._settings)


@pytest.mark.asyncio
async def test_confluence_source_marks_permanent_failure_and_document_cap_incomplete() -> None:
    failed = await ConfluenceDocumentSource(
        _confluence_settings(),
        _ConfluenceTransport([ConfluenceResponse(401, {}, {})]),
    ).acquire()
    capped = await ConfluenceDocumentSource(
        _confluence_settings(max_pages=1),
        _ConfluenceTransport(
            [
                ConfluenceResponse(
                    200,
                    {},
                    {"results": [{"id": "1"}, {"id": "2"}], "_links": {"next": "/next"}},
                )
            ]
        ),
    ).acquire()
    malformed = await ConfluenceDocumentSource(
        _confluence_settings(),
        _ConfluenceTransport([ConfluenceResponse(200, {}, {"results": [{"title": "No id"}]})]),
    ).acquire()

    assert not failed.complete
    assert failed.diagnostics[0].detail == "HTTP 401"
    assert not capped.complete
    assert capped.observed_external_ids == ("confluence:1",)
    assert capped.diagnostics[0].code == "document-limit"
    assert not malformed.complete
    assert malformed.diagnostics[0].code == "missing-page-id"


@pytest.mark.asyncio
async def test_filesystem_source_reports_a_renamed_file_as_a_different_document(
    tmp_path: Path,
) -> None:
    """Identity is the path, so moving a file replaces one document with another.

    The snapshot must stay `complete` across the rename. Only a complete
    snapshot may tombstone what it does not mention, and a rename is exactly the
    case where something has to be tombstoned — the old path is gone and nothing
    else will ever report it again.
    """
    root = tmp_path / "repository"
    (root / "docs").mkdir(parents=True)
    (root / "readme.md").write_text("# Title\n\nUnchanged prose.\n", encoding="utf-8")

    settings = FilesystemSourceSettings(
        source_id="source-a",
        roots=(FilesystemRoot(root, alias="repo"),),
        include=("**/*.md",),
    )
    before = await FilesystemDocumentSource(settings).acquire()

    (root / "readme.md").rename(root / "docs" / "guide.md")
    after = await FilesystemDocumentSource(settings).acquire()

    assert [document.external_id for document in before.documents] == ["fs:repo/readme.md"]
    assert [document.external_id for document in after.documents] == ["fs:repo/docs/guide.md"]
    assert after.complete
    # The bytes never changed — only where they live. Whatever tells the two
    # documents apart, it cannot be the content.
    assert before.documents[0].content == after.documents[0].content
    assert before.documents[0].document_id != after.documents[0].document_id


@pytest.mark.asyncio
async def test_confluence_document_identity_survives_a_retitled_and_moved_page() -> None:
    """A document's identity has to be as stable as the slots that hang off it.

    Slot matching is scoped to `document_id`, so a document that changes
    identity takes every slot, assertion and approval underneath it: the old
    document is reported as deleted and the new one as never seen before, for a
    page somebody merely renamed.

    Confluence hands out a page id that survives a retitle, a move to another
    space and the URL change that follows. This pins that the id is what
    identity is built from — a version derived from the title or the URL would
    pass every other test in this file.
    """
    def _page(title: str, space: str) -> dict:
        return {
            "id": "827361",
            "title": title,
            "body": {"storage": {"value": "<p>Expenses above CHF 500 require approval.</p>"}},
            "version": {"number": 4},
            "_links": {"webui": f"/spaces/{space}/pages/827361"},
        }

    async def _acquire(title: str, space: str):
        transport = _ConfluenceTransport(
            [ConfluenceResponse(200, {}, {"results": [_page(title, space)]})]
        )
        source = ConfluenceDocumentSource(_confluence_settings(space_keys=(space,)), transport)
        return (await source.acquire()).documents[0]

    before = await _acquire("Travel approval", "ENG")
    after = await _acquire("Travel authorisation", "OPS")

    assert before.external_id == after.external_id == "confluence:827361"
    assert before.document_id == after.document_id
    # The things that did change are metadata, and the path follows the space —
    # neither may reach identity.
    assert before.metadata["title"] != after.metadata["title"]
    assert before.path != after.path
