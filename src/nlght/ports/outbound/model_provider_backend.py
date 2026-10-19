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
    """Unbound provider backend for model discovery and bound clients.

    Implemented by all provider backend classes (OllamaModelClient,
    AnthropicModelClient, OpenAICloudModelClient, GoogleModelClient).
    Inbound protocol adapters receive this type instead of a raw base URL.
    """

    async def list_models(self) -> list[ModelInfo]:
        """Discover models available from this provider.

        Returns:
            list[ModelInfo]: Available model information records.

        Side Effects:
            May query the provider endpoint.
        """
        raise NotImplementedError

    def bind(
        self,
        *,
        model: str | None,
        emitter: SignalEmitter,
        stream: bool,
        token_budget: TokenBudget | None = None,
    ) -> ModelClient:
        """Bind a provider to one model request.

        Args:
            model (str | None): Model selected for this request.
            emitter (SignalEmitter): Signal emitter for streamed output.
            stream (bool): Whether the request streams output.
            token_budget (TokenBudget | None): Optional budget override.

        Returns:
            ModelClient: Bound model client.

        Side Effects:
            May initialize request-scoped client state.
        """
        raise NotImplementedError

    async def token_budget(
        self, model: str | None = None, *, client_max_tokens: int | None = None
    ) -> TokenBudget:
        """Resolve the model's usable context budget.

        Args:
            model (str | None): Optional model name.
            client_max_tokens (int | None): Optional output-token cap.

        Returns:
            TokenBudget: Budget for the selected model.

        Side Effects:
            May query or cache provider metadata.
        """
        raise NotImplementedError


@runtime_checkable
class RunningModelsCapable(Protocol):
    """Optional capability for reporting currently loaded models.

    Only ``OllamaModelClient`` implements this. The ``OllamaHttpProtocolAdapter``
    checks ``isinstance(backend, RunningModelsCapable)`` to decide whether to
    serve ``GET /api/ps`` with live data or return an empty list.
    """

    async def list_running_models(self) -> list[RunningModelInfo]:
        raise NotImplementedError


@runtime_checkable
class NativeProxyCapable(Protocol):
    """Optional capability for native HTTP passthrough.

    Used by ``OllamaHttpProtocolAdapter`` for Ollama-only endpoints
    (``/api/generate``, ``/api/show``, ``/api/version``) that have no
    provider-agnostic equivalent.  Only ``OllamaModelClient`` implements this.
    """

    @property
    def native_base_url(self) -> str:
        raise NotImplementedError
