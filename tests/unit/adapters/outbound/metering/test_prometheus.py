# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from types import SimpleNamespace

from nlght.adapters.outbound.metering.prometheus import PrometheusMeteringAdapter


class _Metric:
    def __init__(self, name: str, label_names: list[str] | None = None, **_: object) -> None:
        self.name = name
        self.label_names = label_names or []
        self.calls: list[tuple[str, tuple[object, ...], float]] = []

    def labels(self, *values: object) -> _Metric:
        self.calls.append(("labels", values, 0))
        return self

    def inc(self, value: float = 1) -> None:
        self.calls.append(("inc", (), value))

    def observe(self, value: float) -> None:
        self.calls.append(("observe", (), value))


class _Prom:
    REGISTRY = object()

    def __init__(self) -> None:
        self.metrics: dict[str, _Metric] = {}
        self.Counter = self._counter
        self.Histogram = self._histogram

    def _counter(self, name: str, *_: object, **kwargs: object) -> _Metric:
        if name in self.metrics:
            # Matches real prometheus_client: a CollectorRegistry rejects a
            # second metric registered under a name it already knows about,
            # even with a different label shape.
            raise ValueError(f"Duplicated timeseries in CollectorRegistry: {{'{name}'}}")
        metric = _Metric(name, kwargs.get("labelnames") or (kwargs.get("labelnames", [])))
        self.metrics[name] = metric
        return metric

    def _histogram(self, name: str, *_: object, **kwargs: object) -> _Metric:
        metric = _Metric(name, kwargs.get("labelnames") or [])
        self.metrics[name] = metric
        return metric

    def make_asgi_app(self, *, registry: object) -> SimpleNamespace:
        return SimpleNamespace(registry=registry)


def test_prometheus_adapter_records_standard_metrics(monkeypatch) -> None:
    prom = _Prom()
    monkeypatch.setitem(__import__("sys").modules, "prometheus_client", prom)
    adapter = PrometheusMeteringAdapter(endpoint="/custom-metrics")

    assert adapter.endpoint == "/custom-metrics"
    assert adapter.make_asgi_app().registry is prom.REGISTRY

    import asyncio

    asyncio.run(
        adapter.record_tokens(
            caller=SimpleNamespace(),
            session_key=None,
            workflow="wf",
            step="step",
            provider="provider",
            model="model",
            input_tokens=11,
            output_tokens=7,
        )
    )
    asyncio.run(
        adapter.record_request(
            caller=SimpleNamespace(),
            session_key="session",
            workflow="wf",
            status="ok",
            duration_ms=250,
        )
    )
    asyncio.run(adapter.record_step(workflow="wf", step="step", status="ok", duration_ms=50))

    assert ("labels", ("ephemeral", "wf", "step", "model", "provider"), 0) in prom.metrics[
        "nlght_tokens_input_total"
    ].calls
    assert ("inc", (), 11) in prom.metrics["nlght_tokens_input_total"].calls
    assert ("inc", (), 7) in prom.metrics["nlght_tokens_output_total"].calls
    assert ("labels", ("wf", "ok"), 0) in prom.metrics["nlght_requests_total"].calls
    assert ("inc", (), 1) in prom.metrics["nlght_requests_total"].calls
    assert ("labels", ("wf", "step", "ok"), 0) in prom.metrics["nlght_steps_total"].calls
    assert ("inc", (), 1) in prom.metrics["nlght_steps_total"].calls
    assert ("observe", (), 0.25) in prom.metrics["nlght_request_duration_seconds"].calls
    assert ("observe", (), 0.05) in prom.metrics["nlght_step_duration_seconds"].calls


def test_prometheus_adapter_raises_clear_error_without_dependency(monkeypatch) -> None:
    import builtins

    real_import = builtins.__import__

    def _fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "prometheus_client":
            raise ImportError("No module named 'prometheus_client'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    try:
        PrometheusMeteringAdapter()
        raise AssertionError("expected ImportError")
    except ImportError as exc:
        assert "nlght-ai[metrics]" in str(exc)


def test_prometheus_adapter_caches_custom_metrics_by_label_shape(monkeypatch) -> None:
    prom = _Prom()
    monkeypatch.setitem(__import__("sys").modules, "prometheus_client", prom)
    adapter = PrometheusMeteringAdapter()

    import asyncio

    asyncio.run(adapter.emit(name="custom_total", value=2, labels={"b": "2", "a": "1"}))
    asyncio.run(adapter.emit(name="custom_total", value=3, labels={"a": "1", "b": "2"}))
    asyncio.run(adapter.emit(name="custom_unlabelled_total", value=4))

    assert len(adapter._custom) == 2
    labelled = adapter._custom[("custom_total", ("a", "b"))]
    assert ("labels", ("1", "2"), 0) in labelled.calls
    assert ("inc", (), 3) in labelled.calls
    assert ("inc", (), 4) in adapter._custom[("custom_unlabelled_total", ())].calls


def test_prometheus_adapter_emit_same_name_different_label_shape_raises(monkeypatch) -> None:
    # A metric name is registered once per label-key shape (cache key is
    # (name, sorted label keys)) -- calling emit() with the SAME name but a
    # DIFFERENT set of label keys tries to register a second Counter under
    # a name the registry already knows, which Prometheus itself rejects.
    # This documents that emit() does not silently swallow or merge
    # incompatible label shapes for the same metric name -- it surfaces the
    # registry's error to the caller.
    prom = _Prom()
    monkeypatch.setitem(__import__("sys").modules, "prometheus_client", prom)
    adapter = PrometheusMeteringAdapter()

    import asyncio

    asyncio.run(adapter.emit(name="conflict_total", value=1, labels={"a": "1"}))

    import pytest

    with pytest.raises(ValueError, match="Duplicated timeseries"):
        asyncio.run(adapter.emit(name="conflict_total", value=1, labels={"b": "2"}))
