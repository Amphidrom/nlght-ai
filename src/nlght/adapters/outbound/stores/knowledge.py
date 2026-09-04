# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Knowledge store retrieval tool.

One resource activation spanning two databases, as in the validated prototype
(``next-integration/old-jarvis-steps/resources/knowledge.py``): OpenSearch
answers "which assertions match this question", PostgreSQL then resolves those
identities into full objects with their payload and metadata.

Quarantine is enforced by the repository, which returns only reviewed knowledge
unless explicitly asked otherwise — so no query path here can leak an
unreviewed assertion.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, ClassVar

from nlght.adapters.outbound.persistence.knowledge_repository import (
    SqlAlchemyKnowledgeRepository,
)
from nlght.adapters.outbound.stores.connections import StoreConnections
from nlght.adapters.outbound.tools.builtin.action_semantics import READ_REQUEST
from nlght.core.knowledge import DECISION, FACT, PATTERN, RULE, KnowledgeObject
from nlght.core.tools.tool import ToolBase, ToolOption, ToolParameter, ToolSignature
from nlght.ports.outbound.knowledge_repository import KnowledgeRepository

logger = logging.getLogger(__name__)

try:  # pragma: no cover - import guard
    from opensearchpy import OpenSearch as _OpenSearch

    _HAS_OPENSEARCH = True
except ImportError:  # pragma: no cover - import guard
    _OpenSearch = None
    _HAS_OPENSEARCH = False

_TOKEN_RE = re.compile(r"[A-Za-z]{3,}")


def _serialize(items: list[KnowledgeObject]) -> str:
    return json.dumps(
        [
            {
                "identity": item.identity,
                "kind": item.kind,
                "type": item.type,
                "confidence": round(item.confidence, 4),
                "product": item.product,
                "version": item.version,
                **item.payload,
            }
            for item in items
        ],
        ensure_ascii=False,
    )


class KnowledgeStoreTool(ToolBase):
    """Typed retrieval over the knowledge graph.

    Config: ``os_url`` (required), optional ``os_username``, ``os_password``,
    ``os_verify_certs``, ``os_timeout``, and ``index_prefix``.

    Config: ``os_url`` and ``pg_url`` (both required), plus the optional
    ``os_username``, ``os_password``, ``os_verify_certs``, ``os_timeout``, and
    ``index_prefix``.

    ``pg_url`` points at the *knowledge* database, never the platform's. The
    resource row carries it, so a deployment can activate two knowledge stores
    against different databases, and the connection is visible where the rest of
    the activation is.
    """

    KIND: ClassVar[str] = "knowledge_store"
    PROVIDER: ClassVar[str] = "postgres+opensearch"

    def __init__(
        self,
        *,
        name: str,
        config: dict[str, Any],
        store_connections: StoreConnections | None = None,
        **kwargs: Any,  # noqa: ANN401 (forwarded into ToolBase.__init__ typed kw surface)
    ) -> None:
        super().__init__(name=name, config=config, **kwargs)
        # The activation's own pg_url, with no way around it. An injectable
        # repository would let something outside the activation decide which
        # database this store reads, which is the whole point of the resource.
        # The pool behind it is shared; which database is read is not.
        pg_url = str(config.get("pg_url", "")).strip()
        if not pg_url:
            raise ValueError(
                f"knowledge store '{name}' requires 'pg_url' in its resource config, "
                "pointing at the knowledge database (run 'nlght-ai migrate-knowledge')."
            )
        # The runtime's pools, or a private set when constructed outside it.
        self._connections = store_connections or StoreConnections()
        self._repository: KnowledgeRepository = SqlAlchemyKnowledgeRepository(
            self._connections.engine(pg_url, pool_pre_ping=True)
        )
        self._url = config["os_url"]
        self._index_prefix = str(config.get("index_prefix", ""))
        self._username = config.get("os_username")
        self._password = config.get("os_password")
        self._verify_certs = bool(config.get("os_verify_certs", True))
        self._timeout = int(config.get("os_timeout", 30))
        if not _HAS_OPENSEARCH:
            raise ImportError(
                "opensearch-py is required for KnowledgeStoreTool. "
                "Install it with: pip install 'nlght-ai[lexical-store]'"
            )

    @property
    def _client(self) -> Any:  # noqa: ANN401 (opensearch-py is untyped)
        http_auth = (self._username, self._password or "") if self._username else None
        return self._connections.client(
            ("opensearch", self._url, http_auth, self._verify_certs, self._timeout),
            lambda: _OpenSearch(
                hosts=[self._url],
                http_auth=http_auth,
                verify_certs=self._verify_certs,
                timeout=self._timeout,
            ),
            close=lambda client: client.close(),
        )

    @classmethod
    def options(cls) -> list[ToolOption]:
        return [
            ToolOption("pg_url", "string", "Knowledge database URL (async driver)",
                       required=True,
                       placeholder="postgresql+asyncpg://user:pass@host:5432/nlght_knowledge"),
            ToolOption("os_url", "string", "OpenSearch URL", required=True,
                       placeholder="http://localhost:9200"),
            ToolOption("index_prefix", "string", "Prefix for the per-kind indexes",
                       placeholder="tenant-a-"),
            ToolOption("os_username", "string", "OpenSearch user"),
            ToolOption("os_password", "string", "OpenSearch password"),
            ToolOption("os_verify_certs", "boolean", "Verify TLS certificates", default=True),
            ToolOption("os_timeout", "integer", "Request timeout in seconds", default=30),
        ]

    @classmethod
    def signatures(cls) -> list[ToolSignature]:
        query = ToolParameter("query", "string", "Natural-language question", required=False)
        type_ = ToolParameter("type", "string", "Restrict to one assertion type", required=False)
        min_confidence = ToolParameter(
            "min_confidence", "number", "Minimum confidence", required=False, default=0.0
        )
        limit = ToolParameter("limit", "number", "Max results", required=False, default=10)
        params = [query, type_, min_confidence, limit]
        return [
            ToolSignature(
                "knowledge-query-facts",
                "Search asserted facts (subject, predicate, object).",
                "query_facts_tool",
                params,
                action=READ_REQUEST,
            ),
            ToolSignature(
                "knowledge-query-rules",
                "Search normative rules and constraints.",
                "query_rules_tool",
                params,
                action=READ_REQUEST,
            ),
            ToolSignature(
                "knowledge-query-patterns",
                "Search architecture and design patterns.",
                "query_patterns_tool",
                params,
                action=READ_REQUEST,
            ),
            ToolSignature(
                "knowledge-query-decisions",
                "Search decisions and their consequences.",
                "query_decisions_tool",
                params,
                action=READ_REQUEST,
            ),
        ]

    # -- programmatic API ----------------------------------------------------

    async def query_facts(self, **kwargs: Any) -> list[KnowledgeObject]:  # noqa: ANN401
        return await self._query(FACT, **kwargs)

    async def query_rules(self, **kwargs: Any) -> list[KnowledgeObject]:  # noqa: ANN401
        return await self._query(RULE, **kwargs)

    async def query_patterns(self, **kwargs: Any) -> list[KnowledgeObject]:  # noqa: ANN401
        return await self._query(PATTERN, **kwargs)

    async def query_decisions(self, **kwargs: Any) -> list[KnowledgeObject]:  # noqa: ANN401
        return await self._query(DECISION, **kwargs)

    # -- LLM-facing wrappers -------------------------------------------------

    async def query_facts_tool(self, **kwargs: Any) -> str:  # noqa: ANN401
        return _serialize(await self.query_facts(**kwargs))

    async def query_rules_tool(self, **kwargs: Any) -> str:  # noqa: ANN401
        return _serialize(await self.query_rules(**kwargs))

    async def query_patterns_tool(self, **kwargs: Any) -> str:  # noqa: ANN401
        return _serialize(await self.query_patterns(**kwargs))

    async def query_decisions_tool(self, **kwargs: Any) -> str:  # noqa: ANN401
        return _serialize(await self.query_decisions(**kwargs))

    # -- internals -----------------------------------------------------------

    async def _query(
        self,
        kind: str,
        *,
        query: str | None = None,
        type: str | None = None,  # noqa: A002 (public tool parameter name)
        min_confidence: float | None = None,
        limit: int = 10,
    ) -> list[KnowledgeObject]:
        limit = max(1, int(limit))
        identities = self._search(
            kind=kind, query=query, type_=type, min_confidence=min_confidence, limit=limit
        )
        if not identities:
            return []
        # Resolution enforces the quarantine, so an unreviewed assertion that is
        # still present in the search index cannot be returned.
        return await self._repository.resolve(identities)

    def _search(
        self,
        *,
        kind: str,
        query: str | None,
        type_: str | None,
        min_confidence: float | None,
        limit: int,
    ) -> list[str]:
        index = f"{self._index_prefix}{kind}s"

        filters: list[dict[str, Any]] = []
        if type_:
            filters.append({"term": {"type": type_}})
        if min_confidence:
            filters.append({"range": {"confidence": {"gte": min_confidence}}})

        if not query:
            matcher: dict[str, Any] = {"bool": {"filter": filters}} if filters else {"match_all": {}}
        else:
            tokens = sorted(set(_TOKEN_RE.findall(query)))
            should: list[dict[str, Any]] = [
                {"term": {"keywords": {"value": token.lower(), "boost": 12}}} for token in tokens
            ]
            should += [
                {"match_phrase": {"subject": {"query": query, "boost": 10}}},
                {"match_phrase": {"pattern_name": {"query": query, "boost": 9}}},
                {"match_phrase": {"rule_text": {"query": query, "boost": 8}}},
                {"match_phrase": {"decision": {"query": query, "boost": 8}}},
                {"match_phrase": {"object": {"query": query, "boost": 7}}},
            ]
            should += [{"match": {"keywords": {"query": token, "boost": 5}}} for token in tokens]
            # What the source says, ranked as language. Above the structured
            # fields' single-term matches and below their phrase matches: a
            # question is usually a sentence, and the sentence the claim was
            # read from is the closest thing the index holds to an answer.
            #
            # It replaced a `text_all` bag assembled from every field value,
            # which carried no phrase, no grammar and no quotation — and for
            # several kinds was only a second copy of fields indexed beside it.
            should.append({"match_phrase": {"observed_text": {"query": query, "boost": 11}}})
            should.append({"match": {"observed_text": {"query": query, "boost": 6}}})
            matcher = {"bool": {"should": should, "minimum_should_match": 1}}
            if filters:
                matcher["bool"]["filter"] = filters

        body = {
            "size": limit,
            "query": matcher,
            "_source": ["id"],
            "track_total_hits": False,
        }
        try:
            response = self._client.search(index=index, body=body)
        except Exception:
            # A degraded search backend must not take the whole step down; an
            # empty result is the honest answer here.
            logger.exception("knowledge.search | index=%s", index)
            return []

        hits = ((response or {}).get("hits") or {}).get("hits") or []
        return [hit["_source"]["id"] for hit in hits if "id" in (hit.get("_source") or {})]
