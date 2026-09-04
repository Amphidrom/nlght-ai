# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Local embedding client: batching, ordering, provenance, and failure modes."""

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

import pytest

from nlght.adapters.outbound.embedding.sentence_transformers import (
    PROVIDER,
    SentenceTransformersEmbeddingClient,
)
from nlght.ports.outbound.embedding_client import EmbeddingClient, EmbeddingResult


class _FakeModel:
    """Encodes deterministically so order and batching stay observable."""

    def __init__(self, name: str, *, dimension: int = 3) -> None:
        self.name = name
        self.dimension = dimension
        self.batches: list[list[str]] = []
        self.max_concurrent = 0
        self._active = 0

    def encode(self, texts: list[str], *, normalize_embeddings: bool = True) -> list[list[float]]:
        self._active += 1
        self.max_concurrent = max(self.max_concurrent, self._active)
        try:
            self.batches.append(list(texts))
            return [[float(len(text))] * self.dimension for text in texts]
        finally:
            self._active -= 1


def _client(model: _FakeModel, **kwargs: object) -> SentenceTransformersEmbeddingClient:
    return SentenceTransformersEmbeddingClient(
        model=model.name, loader=lambda _name: model, **kwargs
    )


async def test_returns_one_vector_per_text_in_input_order() -> None:
    model = _FakeModel("test-model")
    client = _client(model)

    result = await client.embed(["a", "bb", "ccc"])

    assert result.vectors == ((1.0, 1.0, 1.0), (2.0, 2.0, 2.0), (3.0, 3.0, 3.0))
    assert result.dimension == 3
    assert result.provider == PROVIDER
    assert result.model == "test-model"


async def test_large_input_is_split_into_bounded_batches() -> None:
    model = _FakeModel("test-model")
    client = _client(model, batch_size=2)

    result = await client.embed(["a", "b", "c", "d", "e"])

    assert model.batches == [["a", "b"], ["c", "d"], ["e"]]
    assert len(result.vectors) == 5


async def test_concurrent_calls_are_serialised() -> None:
    # sentence-transformers models are not safe to call concurrently.
    model = _FakeModel("test-model")
    client = _client(model)

    await asyncio.gather(*(client.embed([f"text-{i}"]) for i in range(8)))

    assert model.max_concurrent == 1


async def test_embedding_an_empty_batch_is_rejected() -> None:
    client = _client(_FakeModel("test-model"))

    with pytest.raises(ValueError, match="at least one text"):
        await client.embed([])


async def test_a_changed_vector_dimension_fails_loudly() -> None:
    model = _FakeModel("test-model")
    client = _client(model)
    await client.embed(["a"])

    model.dimension = 5
    with pytest.raises(RuntimeError, match="dimension"):
        await client.embed(["b"])


def test_construction_rejects_an_empty_model_or_batch_size() -> None:
    model = _FakeModel("test-model")
    with pytest.raises(ValueError, match="model must not be empty"):
        SentenceTransformersEmbeddingClient(model="  ", loader=lambda _n: model)
    with pytest.raises(ValueError, match="batch_size must be positive"):
        SentenceTransformersEmbeddingClient(model="m", batch_size=0, loader=lambda _n: model)


def test_the_client_satisfies_the_embedding_port() -> None:
    assert isinstance(_client(_FakeModel("test-model")), EmbeddingClient)


def test_result_rejects_inconsistent_provenance() -> None:
    with pytest.raises(ValueError, match="dimension must be positive"):
        EmbeddingResult(vectors=(), provider="p", model="m", dimension=0)
    with pytest.raises(ValueError, match="must not be empty"):
        EmbeddingResult(vectors=(), provider=" ", model="m", dimension=3)
    with pytest.raises(ValueError, match="declared dimension"):
        EmbeddingResult(vectors=((1.0, 2.0),), provider="p", model="m", dimension=3)


# -- loading: where it runs, and whether it touches the network --------------


class _RecordingLoader:
    """A loader that accepts the real keywords, so what we pass is observable."""

    def __init__(self, *, fail_local: bool = False) -> None:
        self.calls: list[dict] = []
        self._fail_local = fail_local

    def __call__(self, name: str, **kwargs: object) -> _FakeModel:
        self.calls.append({"name": name, **kwargs})
        if self._fail_local and kwargs.get("local_files_only"):
            raise OSError("model is not fully cached")
        return _FakeModel(name)


def _fake_torch(
    *, cuda: bool, mps: bool, built_cuda: str | None = "12.4", devices: int = 1
) -> object:
    return SimpleNamespace(
        __version__="2.13.0" + ("" if built_cuda else "+cpu"),
        cuda=SimpleNamespace(
            is_available=lambda: cuda,
            device_count=lambda: devices if cuda else 0,
        ),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
        version=SimpleNamespace(cuda=built_cuda),
    )


def test_an_explicit_device_is_passed_through_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Naming cuda:1 is how a deployment keeps the other card free; the client
    # must not second-guess it.
    #
    # What torch reports is *stated* here rather than inherited from whatever
    # wheel the machine happens to have. Left to the machine, this exact
    # assertion passed on a GPU box and failed on a laptop — the client refuses a
    # CUDA device it cannot provide, correctly — which is a test of the hardware
    # wearing the name of a test of the pass-through.
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=True, mps=False, devices=2))
    loader = _RecordingLoader()
    SentenceTransformersEmbeddingClient(model="m", device="cuda:1", loader=loader)

    assert loader.calls[0]["device"] == "cuda:1"


def test_a_device_index_that_does_not_exist_fails_loudly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # `is_available()` answers "is there a GPU", not "is there *that* GPU", so a
    # single-card machine accepted `cuda:1` and failed deep inside model loading
    # — past the point this check exists to fail at. Off by one is the ordinary
    # way to get an index wrong.
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=True, mps=False, devices=1))

    with pytest.raises(RuntimeError, match="1 CUDA device"):
        SentenceTransformersEmbeddingClient(model="m", device="cuda:1", loader=_RecordingLoader())


def test_the_first_card_is_accepted_on_a_single_gpu_machine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=True, mps=False, devices=1))
    loader = _RecordingLoader()

    SentenceTransformersEmbeddingClient(model="m", device="cuda:0", loader=loader)

    assert loader.calls[0]["device"] == "cuda:0"


def test_a_bare_cuda_device_is_not_index_checked(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # No index named, nothing to be wrong about: torch picks the current device.
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=True, mps=False, devices=1))
    loader = _RecordingLoader()

    SentenceTransformersEmbeddingClient(model="m", device="cuda", loader=loader)

    assert loader.calls[0]["device"] == "cuda"


@pytest.mark.parametrize(
    ("cuda", "mps", "expected"),
    [(True, False, "cuda"), (True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")],
)
def test_auto_takes_the_accelerator_that_is_actually_usable(
    monkeypatch: pytest.MonkeyPatch, cuda: bool, mps: bool, expected: str
) -> None:
    # The reported symptom was "No device provided, using cpu" on a machine that
    # has a GPU. auto must ask what is usable, and prefer CUDA over MPS.
    from nlght.adapters.outbound.embedding.sentence_transformers import resolve_device

    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=cuda, mps=mps))
    assert resolve_device("auto").device == expected


def test_an_explicit_device_never_consults_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    from nlght.adapters.outbound.embedding.sentence_transformers import resolve_device

    def _explode() -> bool:
        raise AssertionError("torch must not be asked when the device is named")

    monkeypatch.setitem(
        sys.modules, "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=_explode), backends=SimpleNamespace()),
    )
    assert resolve_device("cpu").device == "cpu"
    assert resolve_device("CUDA").device == "cuda"   # case and spacing do not matter


def test_the_cache_directory_reaches_the_loader() -> None:
    # It is what lets several workers, or a container, share one download.
    loader = _RecordingLoader()
    SentenceTransformersEmbeddingClient(model="m", cache_dir="/models", loader=loader)

    assert loader.calls[0]["cache_folder"] == "/models"


def test_loading_prefers_the_cache_and_only_then_the_hub() -> None:
    # The reported problem: two dozen revalidation requests before the first
    # vector, on every worker start, for files that were already downloaded.
    loader = _RecordingLoader()
    SentenceTransformersEmbeddingClient(model="m", offline="auto", loader=loader)

    assert [call["local_files_only"] for call in loader.calls] == [True]


def test_an_incomplete_cache_falls_back_to_the_hub_once() -> None:
    loader = _RecordingLoader(fail_local=True)
    SentenceTransformersEmbeddingClient(model="m", offline="auto", loader=loader)

    assert [call["local_files_only"] for call in loader.calls] == [True, False]


def test_always_offline_refuses_to_reach_the_network() -> None:
    # An air-gapped deployment wants the failure, not a silent download.
    loader = _RecordingLoader(fail_local=True)

    with pytest.raises(OSError, match="not fully cached"):
        SentenceTransformersEmbeddingClient(model="m", offline="always", loader=loader)
    assert [call["local_files_only"] for call in loader.calls] == [True]


def test_never_offline_keeps_the_previous_behaviour() -> None:
    loader = _RecordingLoader()
    SentenceTransformersEmbeddingClient(model="m", offline="never", loader=loader)

    assert [call["local_files_only"] for call in loader.calls] == [False]


def test_an_unknown_offline_mode_is_refused() -> None:
    with pytest.raises(ValueError, match="offline must be one of"):
        SentenceTransformersEmbeddingClient(model="m", offline="sometimes", loader=_RecordingLoader())


def test_a_loader_without_the_keywords_still_works() -> None:
    # The plain one-argument seam the rest of these tests use, and any
    # sentence-transformers old enough to predate the keywords.
    model = _FakeModel("m")
    client = SentenceTransformersEmbeddingClient(model="m", loader=lambda _name: model)

    assert client.model == "m"


# -- what an older sentence-transformers may and may not cost you -------------
#
# The keywords are negotiated against the installed signature rather than
# discovered by being refused. Catching TypeError from the call and retrying as
# `factory(name)` meant one unsupported keyword dropped *all* of them — so a
# named CUDA device became the CPU, a cache directory became the default one,
# and `offline: always` reached the network on an air-gapped machine. Each is
# the failure its own setting exists to prevent, arriving through a
# compatibility shim.


class _OlderLoader:
    """A sentence-transformers predating `local_files_only`, signature and all."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(
        self, name: str, *, device: str | None = None, cache_folder: str | None = None
    ) -> _FakeModel:
        self.calls.append({"name": name, "device": device, "cache_folder": cache_folder})
        return _FakeModel(name)


class _NameOnlyLoader:
    """One that takes nothing but the model name."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, name: str) -> _FakeModel:
        self.calls.append(name)
        return _FakeModel(name)


def test_an_older_library_still_gets_every_option_it_does_take() -> None:
    loader = _OlderLoader()

    SentenceTransformersEmbeddingClient(
        model="m", device="cpu", cache_dir="/models", offline="auto", loader=loader
    )

    assert loader.calls == [{"name": "m", "device": "cpu", "cache_folder": "/models"}]


def test_a_configured_option_the_library_cannot_take_is_refused_not_dropped() -> None:
    # Silently dropping it is how `device: cuda` becomes a CPU nobody ordered.
    with pytest.raises(RuntimeError, match="does not accept 'device'"):
        SentenceTransformersEmbeddingClient(
            model="m", device="cpu", loader=_NameOnlyLoader()
        )
    with pytest.raises(RuntimeError, match="does not accept 'cache_folder'"):
        SentenceTransformersEmbeddingClient(
            model="m", cache_dir="/models", loader=_NameOnlyLoader()
        )


def test_always_offline_on_a_library_that_cannot_promise_it_refuses() -> None:
    # An air-gapped deployment asked for no network. A library that cannot be
    # told that must not be handed the request anyway.
    with pytest.raises(RuntimeError, match="loading cannot be kept off the network"):
        SentenceTransformersEmbeddingClient(
            model="m", offline="always", loader=_OlderLoader()
        )


def test_auto_offline_on_such_a_library_loads_once_and_not_twice() -> None:
    # The cache-first attempt cannot be expressed, so it is skipped rather than
    # run as a duplicate of the online one.
    loader = _OlderLoader()

    SentenceTransformersEmbeddingClient(model="m", offline="auto", loader=loader)

    assert len(loader.calls) == 1


def test_a_device_chosen_by_auto_is_allowed_to_be_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Nobody asked for this device; the library picking its own is not a broken
    # promise. Only a *configured* one may not be silently discarded.
    monkeypatch.setitem(sys.modules, "torch", _fake_torch(cuda=True, mps=False))
    loader = _NameOnlyLoader()

    SentenceTransformersEmbeddingClient(model="m", device="auto", loader=loader)

    assert loader.calls == ["m"]


# -- saying why it is not on a GPU ------------------------------------------


def test_a_cpu_only_torch_build_is_named_as_the_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The reported case: device=cpu on a machine bought for its GPU, with nothing
    # saying that the installed wheel simply has no CUDA in it. No driver and no
    # card can change that outcome, so the message has to name the wheel.
    from nlght.adapters.outbound.embedding.sentence_transformers import resolve_device

    monkeypatch.setitem(
        sys.modules, "torch", _fake_torch(cuda=False, mps=False, built_cuda=None)
    )
    choice = resolve_device("auto")

    assert choice.device == "cpu"
    assert "no CUDA support" in choice.reason
    # It must name the way this project installs it, not a raw pip incantation.
    assert "--extra embedding-cuda" in choice.reason


def test_a_cuda_build_with_no_visible_device_reads_differently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The other cause, and it needs a different action — the machine, not the wheel.
    from nlght.adapters.outbound.embedding.sentence_transformers import resolve_device

    monkeypatch.setitem(
        sys.modules, "torch", _fake_torch(cuda=False, mps=False, built_cuda="12.4")
    )
    choice = resolve_device("auto")

    assert choice.device == "cpu"
    assert "sees no" in choice.reason and "driver" in choice.reason
    assert "no CUDA support" not in choice.reason


def test_falling_back_to_the_cpu_is_warned_about_not_swallowed(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setitem(
        sys.modules, "torch", _fake_torch(cuda=False, mps=False, built_cuda=None)
    )
    with caplog.at_level("WARNING"):
        SentenceTransformersEmbeddingClient(model="m", device="auto", loader=_RecordingLoader())

    assert any("no_accelerator" in record.message for record in caplog.records)


def test_a_configured_cuda_device_that_is_unusable_fails_loudly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Asking for a GPU and silently getting a CPU is the failure the setting
    # exists to prevent, so this must not quietly degrade.
    monkeypatch.setitem(
        sys.modules, "torch", _fake_torch(cuda=False, mps=False, built_cuda=None)
    )
    with pytest.raises(RuntimeError, match="no CUDA device is usable"):
        SentenceTransformersEmbeddingClient(model="m", device="cuda", loader=_RecordingLoader())


def test_a_configured_cpu_is_never_second_guessed(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Opting out of the GPU deliberately must not produce a warning about it.
    monkeypatch.setitem(
        sys.modules, "torch", _fake_torch(cuda=False, mps=False, built_cuda=None)
    )
    with caplog.at_level("WARNING"):
        SentenceTransformersEmbeddingClient(model="m", device="cpu", loader=_RecordingLoader())

    assert not [r for r in caplog.records if "no_accelerator" in r.message]


def test_the_packaging_offers_a_torch_build_per_accelerator() -> None:
    """The property, not the CUDA release it currently points at.

    "Install the embedding extra" must not leave which torch build you get up to
    the platform's default wheel — that is what the two extras are for, and each
    has to reach its own explicit index.

    Which CUDA line that index carries is deliberately *not* asserted. pyproject
    says moving to a newer driver is "a change to this URL and nothing else", and
    a test naming `pytorch-cu124` made that false: bumping the URL would have
    failed a test about packaging structure, for a reason that has nothing to do
    with the structure. Names are read out of the sources rather than typed in.
    """
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve().parents[5]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    extras = project["project"]["optional-dependencies"]
    uv = project["tool"]["uv"]

    assert {"embedding-cuda", "embedding-cpu"} <= set(extras)
    for name in ("embedding-cuda", "embedding-cpu"):
        # torch has to be a declared dependency of the extra, or the per-extra
        # index below never applies to it.
        assert any(dep.startswith("torch") for dep in extras[name]), name

    sources = {source["extra"]: source["index"] for source in uv["sources"]["torch"]}
    assert set(sources) == {"embedding-cuda", "embedding-cpu"}
    # Two extras pointing at one index would make the choice decorative.
    assert len(set(sources.values())) == 2

    indexes = {index["name"]: index for index in uv["index"]}
    for extra, index in sources.items():
        assert index in indexes, f"{extra} names an index that is not declared"
        # Not explicit means the index joins general resolution, and every other
        # dependency could start coming from PyTorch's mirror.
        assert indexes[index]["explicit"] is True, index
        # The point of the whole arrangement: this wheel comes from PyTorch and
        # not from PyPI's platform default. The CUDA line in the path is free to
        # move.
        assert "download.pytorch.org" in indexes[index]["url"], index

    # One machine has one torch build; without this the lock cannot be solved.
    assert [{"extra": "embedding-cuda"}, {"extra": "embedding-cpu"}] in uv["conflicts"]

    # An accelerator build is a property of the machine, not a capability, so it
    # must stay out of the everything-extra — which would otherwise drag a
    # multi-gigabyte CUDA wheel onto every machine without a GPU.
    assert "embedding-cuda" not in extras["all"][0]
    assert "embedding-cpu" not in extras["all"][0]
