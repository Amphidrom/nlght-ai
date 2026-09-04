# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Local sentence-transformers embedding client.

The model runs in-process, so encoding is compute-bound rather than a network
call. It is therefore executed on a worker thread and serialised behind a
lock: sentence-transformers models are not safe to call concurrently, and one
oversized batch would otherwise stall every other caller.

Two things about *loading* it are configuration rather than accident.

**Where it runs.** Left alone, sentence-transformers picks the CPU and says so
in a line easy to miss. A machine with a usable GPU should use it, and one whose
GPU belongs to something else should be able to say so — hence `device`, with
`auto` meaning "take an accelerator if there is one".

**Whether loading touches the network.** Even with every file already
downloaded, the hub is asked to revalidate each one: roughly two dozen requests
before the first vector, several seconds of startup, and a worker that cannot
begin when the hub is unreachable. `offline: auto` loads from the cache when the
model is fully there and only reaches out when it is not.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from nlght.ports.outbound.embedding_client import EmbeddingResult

logger = logging.getLogger(__name__)

try:  # pragma: no cover - import guard
    from sentence_transformers import SentenceTransformer as _SentenceTransformer

    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:  # pragma: no cover - import guard
    _SentenceTransformer = None
    _HAS_SENTENCE_TRANSFORMERS = False

PROVIDER = "sentence-transformers"

OFFLINE_MODES = ("auto", "always", "never")
"""How loading may use the network. See the module docstring."""


def _source(missing: set[str]) -> str:
    """Where a load that was not cache-first got its files.

    "hub" claims the load was told to go there, which is only true when the
    keyword could be passed at all.
    """
    return "library default" if "local_files_only" in missing else "hub"


@dataclass(slots=True, frozen=True)
class DeviceChoice:
    """Which device was chosen, and why it was not a faster one.

    The reason is the whole point. "Running on the CPU" on a machine bought for
    its GPU looks like a bug, and the two causes need different actions: a torch
    built without CUDA is fixed by installing a different wheel, a torch with
    CUDA and no visible device is fixed on the machine. Reporting only the
    outcome leaves an operator to guess which.
    """

    device: str | None
    reason: str = ""


def resolve_device(requested: str) -> DeviceChoice:
    """Turn a configured device into one sentence-transformers understands.

    ``auto`` asks torch what is actually usable — CUDA first, then Apple's MPS,
    then the CPU — rather than assuming a GPU exists because the machine has
    one. Anything else is passed through: naming ``cuda:1`` is how a deployment
    keeps the other card free, and naming ``cpu`` is how it opts out entirely.

    ``device`` is ``None`` when torch is not importable, which leaves the choice
    to sentence-transformers exactly as before.
    """
    requested = (requested or "auto").strip().lower()
    if requested and requested != "auto":
        return DeviceChoice(requested, "configured explicitly")
    try:
        import torch  # noqa: PLC0415 (lazy import: optional extra or deliberate startup-cost/cycle avoidance)
    except ImportError:  # pragma: no cover - torch ships with the extra
        return DeviceChoice(None, "torch is not installed")
    if torch.cuda.is_available():
        return DeviceChoice("cuda", "CUDA is available")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return DeviceChoice("mps", "Apple MPS is available")
    if getattr(torch.version, "cuda", None) is None:
        # The common one, and invisible without saying it: the installed wheel
        # simply has no CUDA in it, so no driver or card can change the outcome.
        return DeviceChoice(
            "cpu",
            f"this torch build has no CUDA support (torch {torch.__version__}); "
            f"install the CUDA variant with: uv sync --extra embedding "
            f"--extra embedding-cuda",
        )
    return DeviceChoice(
        "cpu",
        f"torch {torch.__version__} was built for CUDA {torch.version.cuda} but sees no "
        f"device — check the driver and that a GPU is visible to this process",
    )


class SentenceTransformersEmbeddingClient:
    """Embeds text with a locally loaded sentence-transformers model.

    ``batch_size`` bounds how many texts are handed to the model at once, so
    memory stays predictable for a large document. Vectors are normalised,
    which is what the cosine distance configured on the vector backend expects.
    """

    def __init__(
        self,
        *,
        model: str,
        batch_size: int = 64,
        normalize: bool = True,
        device: str = "auto",
        cache_dir: str = "",
        offline: str = "auto",
        loader: Any | None = None,  # noqa: ANN401 (injection seam for tests)
    ) -> None:
        if not model.strip():
            raise ValueError("embedding model must not be empty")
        if batch_size < 1:
            raise ValueError("embedding batch_size must be positive")
        if offline not in OFFLINE_MODES:
            raise ValueError(
                f"embedding offline must be one of {list(OFFLINE_MODES)}, got '{offline}'"
            )

        factory = loader
        if factory is None:
            if not _HAS_SENTENCE_TRANSFORMERS:
                raise ImportError(
                    "sentence-transformers is required for SentenceTransformersEmbeddingClient. "
                    "Install it with: pip install 'nlght-ai[embedding]'"
                )
            factory = _SentenceTransformer

        self._model_name = model
        self._batch_size = batch_size
        self._normalize = normalize
        self._model = self._load(factory, device=device, cache_dir=cache_dir, offline=offline)
        self._lock = asyncio.Lock()
        self._dimension: int | None = None

    @staticmethod
    def _require_cuda(device: str) -> None:
        """Refuse a configured CUDA device the installed torch cannot provide."""
        try:
            import torch  # noqa: PLC0415 (lazy import: optional extra or deliberate startup-cost/cycle avoidance)
        except ImportError:  # pragma: no cover - torch ships with the extra
            return
        if torch.cuda.is_available():
            SentenceTransformersEmbeddingClient._require_cuda_index(torch, device)
            return
        detail = (
            f"this torch build has no CUDA support (torch {torch.__version__}); install the "
            f"CUDA variant with: uv sync --extra embedding --extra embedding-cuda"
            if getattr(torch.version, "cuda", None) is None
            else f"torch {torch.__version__} was built for CUDA {torch.version.cuda} but sees "
                 f"no device — check the driver and that a GPU is visible to this process"
        )
        raise RuntimeError(
            f"embedding device '{device}' was configured, but no CUDA device is usable: "
            f"{detail}. Set embedding.device to 'auto' to fall back to the CPU deliberately."
        )

    @staticmethod
    def _require_cuda_index(torch: Any, device: str) -> None:  # noqa: ANN401 (the torch module)
        """Refuse `cuda:N` where there is no card N.

        `is_available()` answers "is there a GPU", not "is there *that* GPU", so a
        machine with one card accepted `cuda:1` here and failed deep inside model
        loading instead — which is the one thing this check exists to prevent.
        Naming an index is how a deployment keeps a card free, so getting it wrong
        by one is the ordinary mistake, not an exotic one.
        """
        _, _, index = device.partition(":")
        if not index.isdigit():
            return
        count = getattr(torch.cuda, "device_count", None)
        if count is None:  # pragma: no cover - a torch too old to be asked
            return
        available = count()
        if int(index) < available:
            return
        raise RuntimeError(
            f"embedding device '{device}' was configured, but this machine has "
            f"{available} CUDA device(s), numbered 0 to {available - 1}. Name one that "
            f"exists, or set embedding.device to 'auto' to let torch choose."
        )

    @staticmethod
    def _unsupported(factory: Any, wanted: Sequence[str]) -> set[str]:  # noqa: ANN401 (the model class, or a test double)
        """Which of these keywords the installed sentence-transformers will not take.

        Asked of the signature rather than discovered by being refused. The
        previous version caught `TypeError` from the call and fell back to
        `factory(name)` — so *one* unsupported keyword silently dropped **all**
        of them: a named CUDA device became the CPU, a cache directory became
        the default one, and `offline: always` quietly reached the network on an
        air-gapped machine. Each is the exact failure its setting exists to
        prevent, arriving through a compatibility shim.

        An empty answer where the callable cannot be introspected — a C
        extension, an odd wrapper — because "cannot ask" is not "does not
        support", and refusing on that would be the opposite mistake.
        """
        try:
            parameters = inspect.signature(factory).parameters
        except (TypeError, ValueError):  # pragma: no cover - an uninspectable callable
            return set()
        if any(p.kind is p.VAR_KEYWORD for p in parameters.values()):
            return set()
        return {name for name in wanted if name not in parameters}

    def _load(
        self,
        factory: Any,  # noqa: ANN401 (the SentenceTransformer class, or a test double)
        *,
        device: str,
        cache_dir: str,
        offline: str,
    ) -> Any:  # noqa: ANN401 (sentence-transformers is untyped)
        options: dict[str, Any] = {}
        choice = resolve_device(device)
        if choice.device is not None:
            options["device"] = choice.device
        if cache_dir.strip():
            options["cache_folder"] = cache_dir.strip()

        # Asking for a GPU and silently getting a CPU is the failure this whole
        # setting exists to prevent, so a named CUDA device that torch cannot
        # provide fails here rather than deep inside model loading.
        if choice.device is not None and choice.device.startswith("cuda"):
            self._require_cuda(choice.device)
        elif choice.device == "cpu" and (device or "auto").strip().lower() in ("", "auto"):
            # Chosen, not defaulted into — and the reason is what tells an
            # operator whether to change the wheel or the machine.
            logger.warning("embedding.device.no_accelerator | %s", choice.reason)

        # 'auto' tries the cache first and only reaches for the hub when the
        # model is not fully there. That is the difference between a worker that
        # starts in milliseconds offline and one that revalidates two dozen
        # files against the network before its first vector.
        attempts: list[bool] = {"auto": [True, False], "always": [True], "never": [False]}[offline]

        # What this installation cannot be told, decided once and acted on
        # according to what each option *is*. A version floor would be the other
        # way to do this and a worse one: it would refuse to run on a
        # sentence-transformers that is merely older, rather than on one that
        # cannot do the thing actually being asked for.
        missing = self._unsupported(factory, (*options, "local_files_only"))
        configured = {
            "device": device.strip().lower() not in ("", "auto"),
            "cache_folder": bool(cache_dir.strip()),
        }
        for name in sorted(missing & set(options)):
            if configured[name]:
                # Explicitly asked for and impossible to deliver. Silently
                # dropping it is how "device: cuda" becomes a CPU nobody
                # ordered.
                raise RuntimeError(
                    f"embedding {name.removesuffix('_folder')} was configured, but the "
                    f"installed sentence-transformers does not accept '{name}'. Upgrade "
                    f"it, or remove the setting."
                )
            logger.info(
                "embedding.model.option_unsupported | %s is not accepted by this "
                "sentence-transformers; leaving the choice to the library", name,
            )
            options.pop(name)
        if "local_files_only" in missing:
            if offline == "always":
                raise RuntimeError(
                    "embedding offline 'always' was configured, but the installed "
                    "sentence-transformers does not accept 'local_files_only', so "
                    "loading cannot be kept off the network. Upgrade it, or set "
                    "offline to 'never' to accept that it reaches the hub."
                )
            logger.info(
                "embedding.model.option_unsupported | local_files_only is not accepted "
                "by this sentence-transformers; the cache-first attempt is skipped",
            )

        last_error: Exception | None = None
        for local_only in attempts:
            keywords = (
                {} if "local_files_only" in missing else {"local_files_only": local_only}
            )
            if not keywords and local_only:
                # The cache-first attempt cannot be expressed, so it is skipped
                # rather than run as a duplicate of the online one.
                continue
            try:
                loaded = factory(self._model_name, **options, **keywords)
            except Exception as exc:  # noqa: BLE001 - any load failure is worth one retry online
                last_error = exc
                if local_only and offline == "auto":
                    logger.info(
                        "embedding.model.cache_miss | model=%s — not complete in the cache, "
                        "fetching from the hub once",
                        self._model_name,
                    )
                    continue
                raise
            logger.info(
                "embedding.model.loaded | model=%s device=%s source=%s (%s)",
                self._model_name, choice.device or "(library default)",
                "cache" if local_only else _source(missing), choice.reason,
            )
            return loaded
        raise RuntimeError(  # pragma: no cover - the loop above either returns or raises
            f"embedding model '{self._model_name}' could not be loaded: {last_error}"
        )

    @property
    def model(self) -> str:
        return self._model_name

    async def embed(self, texts: Sequence[str]) -> EmbeddingResult:
        if not texts:
            raise ValueError("embed() requires at least one text")

        vectors: list[tuple[float, ...]] = []
        async with self._lock:
            for start in range(0, len(texts), self._batch_size):
                batch = list(texts[start : start + self._batch_size])
                encoded = await asyncio.to_thread(self._encode, batch)
                vectors.extend(encoded)

        dimension = len(vectors[0])
        if self._dimension is None:
            self._dimension = dimension
        elif self._dimension != dimension:
            # A silently changed model would poison an index with incomparable
            # vectors, so fail loudly instead.
            raise RuntimeError(
                f"embedding model '{self._model_name}' returned dimension {dimension}, "
                f"but previously returned {self._dimension}"
            )

        return EmbeddingResult(
            vectors=tuple(vectors),
            provider=PROVIDER,
            model=self._model_name,
            dimension=dimension,
        )

    def _encode(self, batch: list[str]) -> list[tuple[float, ...]]:
        encoded = self._model.encode(batch, normalize_embeddings=self._normalize)
        return [tuple(float(value) for value in vector) for vector in encoded]
