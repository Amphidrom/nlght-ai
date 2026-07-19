# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from nlght.ports.outbound.metering import MeteringPort

if TYPE_CHECKING:
    from nlght.core.entry.context import RequestContext

logger = logging.getLogger(__name__)


class PrometheusMeteringAdapter(MeteringPort):
    """MeteringPort implementation backed by prometheus_client.

    Requires ``pip install 'nlght-ai[metrics]'``.

    Registers six metric families on init:
    - nlght_tokens_input_total   (Counter)  labels: session, workflow, step, model, provider
    - nlght_tokens_output_total  (Counter)  labels: session, workflow, step, model, provider
    - nlght_requests_total       (Counter)  labels: workflow, status
    - nlght_request_duration_seconds (Histogram)  labels: workflow
    - nlght_steps_total          (Counter)  labels: workflow, step, status
    - nlght_step_duration_seconds (Histogram)  labels: workflow, step

    ``emit()`` registers a per-name Counter on first call (labels must be
    consistent across calls for the same metric name).
    """

    def __init__(self, *, endpoint: str = "/metrics") -> None:
        try:
            import prometheus_client as _prom  # noqa: PLC0415 (lazy import: optional extra or deliberate startup-cost/cycle avoidance)
        except ImportError as exc:
            raise ImportError(
                "prometheus-client is required for Prometheus metrics. "
                "Install with: pip install 'nlght-ai[metrics]'"
            ) from exc

        self.endpoint = endpoint
        self._prom = _prom
        reg = _prom.REGISTRY

        self._tokens_in = _prom.Counter(
            "nlght_tokens_input_total",
            "Total LLM input tokens consumed",
            ["session", "workflow", "step", "model", "provider"],
            registry=reg,
        )
        self._tokens_out = _prom.Counter(
            "nlght_tokens_output_total",
            "Total LLM output tokens generated",
            ["session", "workflow", "step", "model", "provider"],
            registry=reg,
        )
        self._requests = _prom.Counter(
            "nlght_requests_total",
            "Total workflow requests processed",
            ["workflow", "status"],
            registry=reg,
        )
        self._request_duration = _prom.Histogram(
            "nlght_request_duration_seconds",
            "Workflow request end-to-end duration",
            ["workflow"],
            registry=reg,
        )
        self._steps = _prom.Counter(
            "nlght_steps_total",
            "Total workflow steps executed",
            ["workflow", "step", "status"],
            registry=reg,
        )
        self._step_duration = _prom.Histogram(
            "nlght_step_duration_seconds",
            "Workflow step execution duration",
            ["workflow", "step"],
            registry=reg,
        )
        self._custom: dict[tuple[str, tuple[str, ...]], Any] = {}
        self._registry = reg

    def make_asgi_app(self) -> object:
        """Return an ASGI app that serves the /metrics scrape endpoint."""
        return self._prom.make_asgi_app(registry=self._registry)

    async def record_tokens(
        self,
        *,
        caller: RequestContext,
        session_key: str | None,
        workflow: str,
        step: str,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        session = session_key or "ephemeral"
        labels = [session, workflow, step, model, provider]
        self._tokens_in.labels(*labels).inc(input_tokens)
        self._tokens_out.labels(*labels).inc(output_tokens)

    async def record_request(
        self,
        *,
        caller: RequestContext,
        session_key: str | None,
        workflow: str,
        status: str,
        duration_ms: float,
    ) -> None:
        self._requests.labels(workflow, status).inc()
        self._request_duration.labels(workflow).observe(duration_ms / 1000.0)

    async def record_step(
        self,
        *,
        workflow: str,
        step: str,
        status: str,
        duration_ms: float,
    ) -> None:
        self._steps.labels(workflow, step, status).inc()
        self._step_duration.labels(workflow, step).observe(duration_ms / 1000.0)

    async def emit(
        self,
        *,
        name: str,
        value: float,
        labels: dict[str, str] | None = None,
    ) -> None:
        resolved = labels or {}
        label_names = tuple(sorted(resolved.keys()))
        cache_key = (name, label_names)

        if cache_key not in self._custom:
            self._custom[cache_key] = self._prom.Counter(
                name,
                f"Custom step metric: {name}",
                list(label_names),
                registry=self._registry,
            )

        counter = self._custom[cache_key]
        label_values = [resolved[k] for k in label_names]
        if label_values:
            counter.labels(*label_values).inc(value)
        else:
            counter.inc(value)
