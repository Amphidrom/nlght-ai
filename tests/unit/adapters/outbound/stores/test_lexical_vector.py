# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from nlght.adapters.outbound.stores import lexical, vector
from nlght.adapters.outbound.stores.lexical import LexicalStoreTool
from nlght.adapters.outbound.stores.vector import VectorStoreTool


class _FakeOpenSearch:
    instances: list[_FakeOpenSearch] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.responses: list[dict | Exception] = []
        self.calls: list[dict] = []
        _FakeOpenSearch.instances.append(self)

    def search(self, *, index: str, body: dict) -> dict:
        self.calls.append({"index": index, "body": body})
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class _FakeQdrant:
    instances: list[_FakeQdrant] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.responses: list[list[SimpleNamespace] | Exception] = []
        self.calls: list[dict] = []
        _FakeQdrant.instances.append(self)

    def search(self, **kwargs: object) -> list[SimpleNamespace]:
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _enable_lexical(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeOpenSearch.instances.clear()
    monkeypatch.setattr(lexical, "_HAS_OPENSEARCH", True)
    monkeypatch.setattr(lexical, "_OpenSearch", _FakeOpenSearch)


def _enable_vector(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeQdrant.instances.clear()
    monkeypatch.setattr(vector, "_HAS_QDRANT", True)
    monkeypatch.setattr(vector, "_QdrantClient", _FakeQdrant)


async def test_lexical_searches_content_symbols_and_deduplicates_multi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_lexical(monkeypatch)
    tool = LexicalStoreTool(
        name="lex",
        config={
            "url": "http://os",
            "collection": "code",
            "username": "u",
            "password": "p",
            "verify_certs": True,
            "timeout": 5,
        },
    )
    client = tool._client
    client.responses.extend([
        {"hits": {"hits": [{"_id": "1", "_score": 2.0, "_source": {"content": "body", "path": "a.py"}}]}},
        {"hits": {"hits": [{"_id": "sym", "_score": 3.0, "_source": {"content": "def", "symbols": ["Fn"]}}]}},
        {"hits": {"hits": [{"_id": "1", "_score": 2.0, "_source": {"content": "body"}}]}},
        {"hits": {"hits": [{"_id": "2", "_score": 1.0, "_source": {"content": "other"}}]}},
    ])

    content = await tool.search("Fn call", limit=4)
    symbols = await tool.search_symbols("Fn", limit=2)
    multi = await tool.search_multi(["Fn", "Other"], limit=1)

    assert content[0].metadata == {"path": "a.py"}
    assert symbols[0].id == "sym"
    assert [item.id for item in multi] == ["1", "2"]
    assert client.kwargs["http_auth"] == ("u", "p")
    assert client.calls[0]["index"] == "code"


async def test_lexical_tool_serializes_and_errors_return_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_lexical(monkeypatch)
    tool = LexicalStoreTool(name="lex", config={"url": "http://os"})
    tool._client.responses.extend([
        {"hits": {"hits": [{"_id": "1", "_score": 1.0, "_source": {"content": "body"}}]}},
        RuntimeError("os down"),
    ])

    serialized = await tool.search_content_tool(query="body", limit=1)

    assert json.loads(serialized)[0]["content"] == "body"
    assert await tool.search("broken", limit=1) == []


def test_lexical_requires_optional_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(lexical, "_HAS_OPENSEARCH", False)

    with pytest.raises(ImportError, match="opensearch-py"):
        LexicalStoreTool(name="lex", config={"url": "http://os"})


async def test_vector_searches_content_symbols_and_deduplicates_multi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _enable_vector(monkeypatch)
    tool = VectorStoreTool(
        name="vec",
        config={"url": "http://qdrant", "collection": "embeddings", "api_key": "secret", "timeout": 8},
    )
    client = tool._client
    client.responses.extend([
        [SimpleNamespace(id=1, score=0.9, payload={"content": "body", "path": "a.py"})],
        [
            SimpleNamespace(id="skip", score=0.8, payload={"content": "body"}),
            SimpleNamespace(id="sym", score=0.95, payload={"content": "def", "fqn": "pkg.fn"}),
        ],
        [SimpleNamespace(id=1, score=0.9, payload={"content": "body"})],
        [SimpleNamespace(id=2, score=0.8, payload={"content": "other"})],
    ])

    content = await tool.search([0.1], limit=3)
    symbols = await tool.search_symbols([0.2], limit=2)
    multi = await tool.search_multi([[0.1], [0.2]], limit=1)

    assert content[0].metadata == {"path": "a.py"}
    assert [item.id for item in symbols] == ["sym"]
    assert [item.id for item in multi] == ["1", "2"]
    assert client.kwargs["api_key"] == "secret"
    assert client.calls[0]["collection_name"] == "embeddings"


async def test_vector_tool_serializes_and_errors_return_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_vector(monkeypatch)
    tool = VectorStoreTool(name="vec", config={"url": "http://qdrant"})
    tool._client.responses.extend([
        [SimpleNamespace(id=1, score=1.0, payload={"content": "body"})],
        RuntimeError("qdrant down"),
    ])

    serialized = await tool.search_content_tool(query_vector=[0.1], limit=1)

    assert json.loads(serialized)[0]["content"] == "body"
    assert await tool.search([0.2], limit=1) == []


def test_vector_requires_optional_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vector, "_HAS_QDRANT", False)

    with pytest.raises(ImportError, match="qdrant-client"):
        VectorStoreTool(name="vec", config={"url": "http://qdrant"})
