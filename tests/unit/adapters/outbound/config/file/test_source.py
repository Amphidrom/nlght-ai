# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

import asyncio
from pathlib import Path

import pytest

from nlght.adapters.outbound.config.file.source import FileConfigurationSource


def test_load_returns_empty_snapshot_when_file_missing(tmp_path: Path) -> None:
    source = FileConfigurationSource(str(tmp_path / "missing.yaml"))

    snapshot = asyncio.run(source.load())

    assert snapshot.protocol_adapters == []
    assert snapshot.gateways == []
    assert snapshot.os_runtime is None
    assert snapshot.integrations.model_providers == []
    assert snapshot.embedding.device == "auto"
    assert snapshot.embedding.cache_dir == ""
    assert snapshot.embedding.offline == "auto"


def test_load_maps_full_yaml_structure(tmp_path: Path) -> None:
    path = tmp_path / "platform.yaml"
    path.write_text(
        """
catalogs:
  playbooks:
    definitions_path: .config/playbooks
protocol_adapters:
  - name: openai-http
    kind: openai
    enabled: true
    config:
      base_path: /v1
gateways:
  - name: gateway
    kind: http
    enabled: true
    config:
      host: 0.0.0.0
      port: 8100
os_runtime:
  kind: local
  enabled: true
  config:
    workdir: .
    workspace_path: ./.workspaces
integrations:
  model_providers:
    - name: ollama-main
      kind: ollama
      enabled: true
      config:
        base_url: http://localhost:11434
        default_model: llama3
  persistence:
    workflows:
      backend: postgres
      url: postgresql://runtime
    hive_mind:
      provider:
        name: hive
        kind: postgres
        enabled: true
        config:
          url: postgresql://hive
execution:
  role: worker
  stream:
    transport: postgres
    url: postgresql+asyncpg://stream
    channel: nlght_stream_a
  worker:
    instance_name: ingestion-a
    capabilities: [parser:pdf, projection:qdrant]
    concurrency: 4
    poll_interval_seconds: 0.25
    lease_seconds: 40
    heartbeat_seconds: 8
    retry_base_seconds: 2
    retry_max_seconds: 30
embedding:
  provider: sentence-transformers
  model: all-MiniLM-L6-v2
  batch_size: 32
  normalize: false
  device: cuda:1
  cache_dir: /var/cache/models
  offline: always
        """.strip()
    )

    source = FileConfigurationSource(str(path))
    snapshot = asyncio.run(source.load())

    assert snapshot.catalogs.playbooks.definitions_path == ".config/playbooks"
    assert snapshot.protocol_adapters[0].kind == "openai"
    assert snapshot.gateways[0].config["port"] == 8100
    assert snapshot.os_runtime is not None
    assert snapshot.os_runtime.kind == "local"
    assert snapshot.integrations.model_providers[0].name == "ollama-main"
    assert snapshot.integrations.persistence.workflows.url == "postgresql://runtime"
    assert snapshot.integrations.persistence.hive_mind.provider is not None
    assert snapshot.integrations.persistence.hive_mind.provider.name == "hive"
    assert snapshot.execution.role == "worker"
    assert snapshot.execution.stream.transport == "postgres"
    assert snapshot.execution.stream.url == "postgresql+asyncpg://stream"
    assert snapshot.execution.stream.channel == "nlght_stream_a"
    assert snapshot.execution.worker.instance_name == "ingestion-a"
    assert snapshot.execution.worker.capabilities == ["parser:pdf", "projection:qdrant"]
    assert snapshot.execution.worker.concurrency == 4
    assert snapshot.execution.worker.poll_interval_seconds == 0.25
    assert snapshot.execution.worker.lease_seconds == 40
    assert snapshot.execution.worker.heartbeat_seconds == 8
    assert snapshot.embedding.provider == "sentence-transformers"
    assert snapshot.embedding.model == "all-MiniLM-L6-v2"
    assert snapshot.embedding.batch_size == 32
    assert snapshot.embedding.normalize is False
    assert snapshot.embedding.device == "cuda:1"
    assert snapshot.embedding.cache_dir == "/var/cache/models"
    assert snapshot.embedding.offline == "always"


def test_load_rejects_multiple_hive_mind_providers(tmp_path: Path) -> None:
    path = tmp_path / "platform.yaml"
    path.write_text(
        """
integrations:
  persistence:
    hive_mind:
      providers:
        - name: one
          kind: postgres
        - name: two
          kind: postgres
        """.strip()
    )

    source = FileConfigurationSource(str(path))

    with pytest.raises(ValueError, match="supports only one provider"):
        asyncio.run(source.load())
