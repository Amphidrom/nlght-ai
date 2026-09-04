# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Ingestion workflow steps.

Each step is an ordinary ``StepBase`` — ingestion is a workflow like any other,
executed over the platform's durable dispatch, with no pipeline engine of its
own. A pipeline is therefore a workflow version, and recombining its building
blocks is a configuration change rather than a code change.

Sources are bound through the step's own ``config``, mirroring how the
prototype's ``SourceConfig`` declared roots, includes, and credentials.
"""

from nlght.adapters.outbound.workflow.steps.ingestion.embed import IngestionEmbedStep
from nlght.adapters.outbound.workflow.steps.ingestion.fan_out import IngestionFanOutStep
from nlght.adapters.outbound.workflow.steps.ingestion.process import IngestionProcessStep
from nlght.adapters.outbound.workflow.steps.ingestion.reconcile import IngestionReconcileStep
from nlght.adapters.outbound.workflow.steps.ingestion.source import IngestionSourceStep
from nlght.adapters.outbound.workflow.steps.ingestion.write import IngestionWriteStep

__all__ = [
    "IngestionEmbedStep",
    "IngestionFanOutStep",
    "IngestionProcessStep",
    "IngestionReconcileStep",
    "IngestionSourceStep",
    "IngestionWriteStep",
]
