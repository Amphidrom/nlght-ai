# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from nlght.core.model.budget import TokenBudget
    from nlght.core.model.model_info import ModelInfo, RunningModelInfo
    from nlght.ports.outbound.model_client import ModelClient
    from nlght.ports.outbound.signal_emitter import SignalEmitter


class ModelProviderBackend(Protocol):
    """Unbound provider backend — factory for bound ModelClients and model catalogue.

    Implemented by all provider backend classes (OllamaModelClient,
    AnthropicModelClient, OpenAICloudModelClient, GoogleModelClient).
    Inbound protocol adapters receive this type instead of a raw base URL.
    """

    async def list_models(self) -> list[ModelInfo]:
        """Return the models available from this provider."""
        raise NotImplementedError

    def bind(
        self,
        *,
        model: str | None,
        emitter: SignalEmitter,
        stream: bool,
        token_budget: TokenBudget | None = None,
    ) -> ModelClient:
        """Return a request-scoped BoundModelClient."""
        raise NotImplementedError

    async def token_budget(
        self, model: str | None = None, *, client_max_tokens: int | None = None
    ) -> TokenBudget:
        """Return a TokenBudget for the given model."""
        raise NotImplementedError


@runtime_checkable
class RunningModelsCapable(Protocol):
    """Optional capability — provider can report which models are currently loaded.

    Only ``OllamaModelClient`` implements this. The ``OllamaHttpProtocolAdapter``
    checks ``isinstance(backend, RunningModelsCapable)`` to decide whether to
    serve ``GET /api/ps`` with live data or return an empty list.
    """

    async def list_running_models(self) -> list[RunningModelInfo]:
        raise NotImplementedError


@runtime_checkable
class NativeProxyCapable(Protocol):
    """Optional capability — provider exposes a native HTTP endpoint for passthrough.

    Used by ``OllamaHttpProtocolAdapter`` for Ollama-only endpoints
    (``/api/generate``, ``/api/show``, ``/api/version``) that have no
    provider-agnostic equivalent.  Only ``OllamaModelClient`` implements this.
    """

    @property
    def native_base_url(self) -> str:
        raise NotImplementedError
