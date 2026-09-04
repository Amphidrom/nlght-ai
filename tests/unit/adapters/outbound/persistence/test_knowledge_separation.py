# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""The knowledge base is a database of its own, not part of the platform's.

It holds the knowledge graph and the ingestion record of the corpus that graph
is derived from (ADR-0033). The platform database stays about the runtime:
workflows, resources, policies, executions.
"""

from __future__ import annotations

import configparser
import re
from pathlib import Path

# Imported for its side effect: the ingestion tables register themselves on
# KnowledgeBase when their module is loaded, and this file asserts what is on it.
# Without the import the assertion passed only when some other test module
# happened to load them first, which made the guard depend on collection order.
from nlght.adapters.outbound.persistence import ingestion_models as _ingestion_models  # noqa: F401
from nlght.adapters.outbound.persistence.knowledge_models import KnowledgeBase
from nlght.adapters.outbound.persistence.models import Base

_PACKAGE = Path(__file__).resolve().parents[5] / "src" / "nlght"

_KNOWLEDGE_TABLES = {
    "knowledge",
    "knowledge_facts",
    "knowledge_rules",
    "knowledge_patterns",
    "knowledge_decisions",
    "knowledge_observations",
    "knowledge_reviews",
    "knowledge_metadata",
    "knowledge_extraction_state",
    "knowledge_slots",
    "knowledge_slot_anchors",
    "knowledge_slot_observations",
    "knowledge_propositions",
    "knowledge_assertion_observations",
    "knowledge_equivalence_assessments",
    "knowledge_assertions",
    "knowledge_variants",
    "knowledge_revisions",
    "knowledge_evidence",
    "knowledge_document_support",
    "knowledge_assertion_lineage",
}

_INGESTION_TABLES = {
    "ingestion_sources",
    "ingestion_snapshot_runs",
    "ingestion_documents",
    "ingestion_document_revisions",
    "ingestion_snapshot_observations",
    "ingestion_index_state",
    # Which incarnation of a target's indexes the state above describes.
    "ingestion_index_generation",
}


def test_the_platform_schema_contains_no_knowledge_or_ingestion_table() -> None:
    # `nlght-ai migrate` must never create either in the runtime database.
    assert set(Base.metadata.tables) & (_KNOWLEDGE_TABLES | _INGESTION_TABLES) == set()


def test_the_knowledge_schema_holds_the_graph_and_the_corpus_record() -> None:
    assert set(KnowledgeBase.metadata.tables) == _KNOWLEDGE_TABLES | _INGESTION_TABLES


def test_the_two_schemas_share_no_table() -> None:
    assert set(Base.metadata.tables).isdisjoint(KnowledgeBase.metadata.tables)


def test_the_knowledge_chain_reads_no_configured_url_at_all() -> None:
    platform = (_PACKAGE / "migrations" / "env.py").read_text(encoding="utf-8")
    knowledge = (_PACKAGE / "knowledge_migrations" / "env.py").read_text(encoding="utf-8")

    # The runtime's own database is named in platform.yaml; it is the runtime's.
    assert '.get("workflows", {})' in platform

    # The knowledge database is not. It is reached only through the resource
    # activations that use it, so there is no platform-wide URL for a migration
    # to inherit — and none for an activation to be silently overridden by.
    assert "yaml.safe_load" not in knowledge
    assert "NLGHT_CONFIG" not in knowledge
    assert '.get("knowledge", {})' not in knowledge
    assert '_x_args.get("url")' in knowledge


def test_each_chain_targets_its_own_metadata() -> None:
    platform = (_PACKAGE / "migrations" / "env.py").read_text(encoding="utf-8")
    knowledge = (_PACKAGE / "knowledge_migrations" / "env.py").read_text(encoding="utf-8")

    assert "target_metadata = Base.metadata" in platform
    assert "target_metadata = KnowledgeBase.metadata" in knowledge


def test_the_chains_have_separate_alembic_configs_and_directories() -> None:
    def script_location(name: str) -> str:
        parser = configparser.ConfigParser(interpolation=None)
        parser.read(_PACKAGE / name)
        return parser["alembic"]["script_location"]

    assert script_location("alembic.ini").endswith("/migrations")
    assert script_location("knowledge_alembic.ini").endswith("/knowledge_migrations")


def test_the_knowledge_chain_starts_from_its_own_root() -> None:
    versions = sorted((_PACKAGE / "knowledge_migrations" / "versions").glob("*.py"))
    revisions = {}
    for path in versions:
        text = path.read_text(encoding="utf-8")
        revision = re.search(r'^revision: str = "(\w+)"', text, re.M)
        down = re.search(r"^down_revision: str \| None = (.+)$", text, re.M)
        assert revision and down
        revisions[revision.group(1)] = down.group(1).strip()

    # A down_revision pointing into the platform chain would couple them.
    assert revisions["0001"] == "None"
    assert revisions == {
        "0001": "None", "0002": '"0001"', "0003": '"0002"', "0004": '"0003"',
        "0005": '"0004"', "0006": '"0005"', "0007": '"0006"', "0008": '"0007"',
        "0009": '"0008"', "0010": '"0009"', "0011": '"0010"', "0012": '"0011"',
        "0013": '"0012"', "0014": '"0013"', "0015": '"0014"', "0016": '"0015"', "0017": '"0016"',
        "0018": '"0017"', "0019": '"0018"', "0020": '"0019"', "0021": '"0020"', "0022": '"0021"',
    }


def test_no_platform_migration_creates_an_ingestion_table() -> None:
    # The ingestion tables moved to the knowledge chain; `nlght-ai migrate` must
    # not recreate them in the runtime database.
    for path in (_PACKAGE / "migrations" / "versions").glob("*.py"):
        assert "ingestion_" not in path.read_text(encoding="utf-8")


def test_the_knowledge_chain_ships_with_the_package() -> None:
    pyproject = (_PACKAGE.parents[1] / "pyproject.toml").read_text(encoding="utf-8")

    assert "knowledge_alembic.ini" in pyproject
    assert "knowledge_migrations/versions/*.py" in pyproject


def test_the_cli_exposes_a_separate_migration_command() -> None:
    from nlght.main import _USAGE, migrate, migrate_knowledge

    assert "migrate-knowledge" in _USAGE
    assert migrate is not migrate_knowledge


def test_both_chains_accept_a_url_override_on_the_command_line() -> None:
    # A CI job or one-off setup must be able to migrate without a platform.yaml.
    for name in ("migrations", "knowledge_migrations"):
        env = (_PACKAGE / name / "env.py").read_text(encoding="utf-8")
        assert "get_x_argument" in env
        assert '_x_args.get("url")' in env
