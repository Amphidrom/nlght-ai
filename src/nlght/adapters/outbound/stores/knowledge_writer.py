# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Maintain the lexical projection of the knowledge graph.

PostgreSQL remains authoritative. This adapter maintains the OpenSearch projection.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from nlght.adapters.outbound.persistence.knowledge_repository import (
    SqlAlchemyKnowledgeRepository,
)
from nlght.adapters.outbound.stores.connections import StoreConnections
from nlght.core.knowledge import DECISION, FACT, PATTERN, RULE, KnowledgeObject
from nlght.core.tools.tool import ToolBase, ToolOption, ToolSignature

logger = logging.getLogger(__name__)

try:  # pragma: no cover - import guard
    from opensearchpy import OpenSearch as _OpenSearch

    _HAS_OPENSEARCH = True
except ImportError:  # pragma: no cover - import guard
    _OpenSearch = None
    _HAS_OPENSEARCH = False

# Stopwords are disabled on purpose: "not", "no", and "must" carry the meaning
# of a security rule, and a standard analyser would drop them.
_ANALYSIS = {
    "analysis": {"analyzer": {"knowledge_text": {"type": "standard", "stopwords": "_none_"}}}
}

_BASE_FIELDS: dict[str, Any] = {
    "id": {"type": "keyword"},
    "kind": {"type": "keyword"},
    "type": {"type": "keyword"},
    "confidence": {"type": "float"},
    "product": {"type": "keyword"},
    "version": {"type": "keyword"},
    "keywords": {"type": "keyword"},
    # What the source actually says. The field a lexical search should rank on:
    # real language keeps phrases, grammar and synonyms, and a claim found
    # through it can be quoted back to a reader.
    #
    # It replaced `text_all`, a bag assembled from every field value. That bag
    # was neither observed nor quotable. For a pattern it was literally
    # `description + pattern_name`, both already indexed beside it, and once
    # the wording is stored there is nothing it added but a second copy.
    "observed_text": {"type": "text", "analyzer": "knowledge_text"},  # _WORDING_FIELD
    "fulltext": {"type": "text", "analyzer": "knowledge_text"},
}

#: The field a lexical search ranks language on. Named once, because both the
#: mapping and the readiness check have to mean the same thing by it.
_WORDING_FIELD = "observed_text"

_ANALYSED = {"type": "text", "analyzer": "knowledge_text", "copy_to": "fulltext"}
_NAMED = {"type": "text", "fields": {"raw": {"type": "keyword"}}, "copy_to": "fulltext"}

_KIND_FIELDS: dict[str, dict[str, Any]] = {
    FACT: {"subject": _NAMED, "predicate": {"type": "keyword"}, "object": _ANALYSED},
    RULE: {"rule_text": _ANALYSED},
    PATTERN: {"pattern_name": _NAMED, "description": _ANALYSED},
    DECISION: {"decision": _NAMED, "effect": _ANALYSED},
}


class KnowledgeIndexWriterTool(ToolBase):
    """Maintains the lexical projection selected by knowledge publish steps.

    The persist and publish steps pass only readable assertions. This writer
    has a resource kind distinct from ``knowledge_store`` and declares no tool
    signatures, so model requests cannot reach its write operations.

    Config: ``os_url``, optional ``os_username``, ``os_password``,
    ``os_verify_certs``, ``os_timeout``, ``index_prefix``.
    """

    KIND: ClassVar[str] = "knowledge_index_writer"
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
        if not _HAS_OPENSEARCH:
            raise ImportError(
                "opensearch-py is required for KnowledgeIndexWriterTool. "
                "Install it with: pip install 'nlght-ai[lexical-store]'"
            )
        pg_url = str(config.get("pg_url", "")).strip()
        if not pg_url:
            raise ValueError(
                f"knowledge writer '{name}' requires 'pg_url' in its resource config, "
                "pointing at the knowledge database (run 'nlght-ai migrate-knowledge')."
            )
        # The runtime's pools, or a private set when constructed outside it.
        self._connections = store_connections or StoreConnections()
        # The writer owns both halves of one activation: PostgreSQL is the
        # authority, OpenSearch only makes it findable.
        self.repository = SqlAlchemyKnowledgeRepository(
            self._connections.engine(pg_url, pool_pre_ping=True)
        )
        self._url = config["os_url"]
        self._prefix = str(config.get("index_prefix", ""))
        self._username = config.get("os_username")
        self._password = config.get("os_password")
        self._verify_certs = bool(config.get("os_verify_certs", True))
        self._timeout = int(config.get("os_timeout", 30))

    @classmethod
    def options(cls) -> list[ToolOption]:
        return [
            ToolOption("pg_url", "string", "Knowledge database URL (async driver)",
                       required=True,
                       placeholder="postgresql+asyncpg://user:pass@host:5432/nlght_knowledge"),
            ToolOption("os_url", "string", "OpenSearch URL", required=True,
                       placeholder="http://localhost:9200"),
            ToolOption("index_prefix", "string", "Prefix for the per-kind indexes"),
            ToolOption("os_username", "string", "OpenSearch user"),
            ToolOption("os_password", "string", "OpenSearch password", secret=True),
            ToolOption("os_verify_certs", "boolean", "Verify TLS certificates", default=True),
            ToolOption("os_timeout", "integer", "Request timeout in seconds", default=30),
        ]

    @classmethod
    def signatures(cls) -> list[ToolSignature]:
        return []

    @property
    def _client(self) -> Any:  # noqa: ANN401 (opensearch-py is untyped)
        auth = (self._username, self._password or "") if self._username else None
        return self._connections.client(
            ("opensearch", self._url, auth, self._verify_certs, self._timeout),
            lambda: _OpenSearch(
                hosts=[self._url],
                http_auth=auth,
                verify_certs=self._verify_certs,
                timeout=self._timeout,
            ),
            close=lambda client: client.close(),
        )

    def index_for(self, kind: str) -> str:
        return f"{self._prefix}{kind}s"

    def ensure_ready(self) -> None:
        """Create per-kind indexes and mappings once per process.

        Side Effects:
            Creates OpenSearch indexes when missing. Backend failures propagate.
        """
        for kind, fields in _KIND_FIELDS.items():
            index = self.index_for(kind)
            if not self._connections.needs_bootstrap("opensearch", self._url, index):
                continue
            if not self._client.indices.exists(index=index):
                try:
                    self._client.indices.create(
                        index=index,
                        body={
                            "settings": _ANALYSIS,
                            # dynamic=False: an unexpected field is stored but
                            # never silently mapped, so a mapping stays a
                            # deliberate decision.
                            "mappings": {
                                "dynamic": False,
                                "properties": {**_BASE_FIELDS, **fields},
                            },
                        },
                    )
                except Exception:
                    if not self._client.indices.exists(index=index):
                        raise
                    logger.info("knowledge_writer.index.exists | name=%s", index)
                else:
                    logger.info("knowledge_writer.index.create | name=%s", index)
            self._report_schema(kind, index)
            self._connections.mark_bootstrapped("opensearch", self._url, index)

    def _report_schema(self, kind: str, index: str) -> None:
        """Log whether an index can search observed wording.

        Args:
            kind (str): Knowledge kind represented by the index.
            index (str): OpenSearch index name.

        Side Effects:
            Reads index mapping and logs its wording capability once.
        """
        if self.wording_indexed(kind):
            return
        logger.warning(
            "knowledge_writer.index.schema | name=%s schema=old %s=unavailable "
            "retrieval=degraded action=rebuild_required",
            index, _WORDING_FIELD,
        )

    def wording_indexed(self, kind: str) -> bool:
        """Check the live index mapping for observed wording.

        Args:
            kind (str): Knowledge kind to inspect.

        Returns:
            bool: True when the current mapping indexes observed wording.

        Side Effects:
            Reads the OpenSearch mapping and logs lookup failures.

        Notes:
            Live mapping state may differ from newly written document fields.
        """
        index = self.index_for(kind)
        try:
            mapping = self._client.indices.get_mapping(index=index)
        except Exception:
            logger.exception("knowledge_writer.index.mapping | name=%s", index)
            return False
        for body in (mapping or {}).values():
            properties = ((body or {}).get("mappings") or {}).get("properties") or {}
            if _WORDING_FIELD in properties:
                return True
        return False

    def schema_state(self) -> dict[str, str]:
        """Report mapping readiness for each knowledge kind.

        Returns:
            dict[str, str]: Each kind mapped to current or stale schema state.

        Side Effects:
            Reads OpenSearch mappings and may log lookup failures.
        """
        return {
            kind: "current" if self.wording_indexed(kind) else "stale"
            for kind in _KIND_FIELDS
        }

    async def index(self, item: KnowledgeObject, *, keywords: list[str] | None = None) -> None:
        """Upsert one assertion into its kind-specific search index.

        Args:
            item (KnowledgeObject): Persisted assertion selected for publication.
            keywords (list[str] | None): Optional search keywords.

        Side Effects:
            Writes OpenSearch and propagates backend failures.
        """
        body = {
            "id": item.identity,
            "kind": item.kind,
            "type": item.type,
            "confidence": item.confidence,
            "product": item.product,
            "version": item.version,
            "keywords": keywords or item.metadata.get("keywords", []),
            "observed_text": item.observed_text,
            **item.payload,
        }
        self._client.index(
            index=self.index_for(item.kind), id=item.identity, body=body, refresh=False
        )

    async def remove(self, *, identity: str, kind: str) -> None:
        """Remove one assertion from its search index.

        Args:
            identity (str): Assertion identity to withdraw.
            kind (str): Knowledge kind that selects the index.

        Side Effects:
            Deletes from OpenSearch and propagates backend failures.
        """
        self._client.delete(
            index=self.index_for(kind), id=identity, ignore=[404], refresh=False
        )
