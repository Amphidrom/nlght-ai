# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import httpx2

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fallback table for the local Ollama provider — used when /api/show fails or
# returns no context_length (e.g. cloud-passthrough models not present locally).
# Source: model documentation / Ollama defaults. Treat as last resort.
# ---------------------------------------------------------------------------
_FALLBACK_CONTEXT_WINDOWS: dict[str, int] = {
    "qwen3-coder:480b": 262144,
    "qwen3-coder:30b": 262144,
    "qwen3-coder:8b": 32768,
    "qwen2.5-coder:32b": 32768,
    "qwen2.5-coder:7b": 32768,
    "llama3.1:8b": 131072,
    "llama3.1:70b": 131072,
    "llama3.2:3b": 131072,
    "mistral:7b": 32768,
    "deepseek-coder-v2:16b": 163840,
}

_DEFAULT_CONTEXT_WINDOW = 32768

# Per-process cache — /api/show is called at most once per model per process.
_context_window_cache: dict[str, int] = {}


async def _fetch_context_window(
    model: str,
    *,
    base_url: str,
    http_client: httpx2.AsyncClient,
    headers: dict[str, str] | None = None,
) -> int:
    if model in _context_window_cache:
        return _context_window_cache[model]

    try:
        resp = await http_client.post(
            f"{base_url}/api/show",
            json={"name": model},
            timeout=5.0,
            headers=headers or {},
        )
        resp.raise_for_status()
        data = resp.json()

        model_info: dict[str, Any] = data.get("model_info") or data.get("modelinfo") or {}
        arch: str = model_info.get("general.architecture", "")
        if arch:
            context_length = model_info.get(f"{arch}.context_length")
            if isinstance(context_length, (int, float)) and context_length > 0:
                window = int(context_length)
                logger.info(
                    "token_budget.window | model=%s window=%d source=ollama",
                    model, window,
                )
                _context_window_cache[model] = window
                return window

        logger.warning(
            "token_budget.window_missing | model=%s falling_back=static_table",
            model,
        )

    except Exception as exc:
        logger.warning(
            "token_budget.fetch_failed | model=%s error=%s falling_back=static_table",
            model, exc,
        )

    window = _FALLBACK_CONTEXT_WINDOWS.get(model, _DEFAULT_CONTEXT_WINDOW)
    logger.info(
        "token_budget.window | model=%s window=%d source=fallback",
        model, window,
    )
    _context_window_cache[model] = window
    return window


class TokenBudget:
    """Token budget for a single LLM call.

    Partitioning (per ADR-0012) — all values in tokens:

      store_budget   (40% of M)  -- Store / Hive-Mind injection
      prompt_budget  (20% of M)  -- system prompt + directives
      output_budget  (20% of M)  -- LLM response reserve
      _safety_margin (20% of M)  -- overhead / buffer

    The ``OllamaModelClient`` uses ``get_available_for_input()`` as the
    hard-cap limit. All other layers (Store, SystemPromptBuilder) get
    their slice as a constraint — they trim it themselves.

    Always instantiate via ``TokenBudget.for_model()`` (async) in
    production. The sync constructor is reserved for unit tests.
    """

    def __init__(
        self,
        context_window: int,
        *,
        client_max_tokens: int | None = None,
    ) -> None:
        M = context_window if context_window > 0 else 1024
        self.total = M
        self.store_budget = int(0.40 * M)
        self.prompt_budget = int(0.20 * M)
        self._safety_margin = int(0.20 * M)

        raw_output = int(0.20 * M)
        self.output_budget = (
            min(client_max_tokens, raw_output)
            if client_max_tokens and client_max_tokens < raw_output
            else raw_output
        )

        logger.debug(
            "token_budget.init | total=%d store=%d prompt=%d output=%d safety=%d",
            M, self.store_budget, self.prompt_budget, self.output_budget, self._safety_margin,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_available_for_input(self, system_prompt_tokens: int = 0) -> int:
        """Tokens available for combined user input + store injection.

        Used by the LLM client hard cap to detect overflow.
        Can also be used by the Store as a constraint basis.
        """
        reserved = self.output_budget + self._safety_margin + system_prompt_tokens
        return max(0, self.total - reserved)

    # ------------------------------------------------------------------
    # Async factory
    # ------------------------------------------------------------------

    @staticmethod
    async def for_model(
        model: str,
        *,
        base_url: str,
        http_client: httpx2.AsyncClient,
        client_max_tokens: int | None = None,
        headers: dict[str, str] | None = None,
    ) -> TokenBudget:
        """Creates a TokenBudget by querying the real context window via
        Ollama ``/api/show``.

        Results are cached per process — ``/api/show`` is called at most
        once per model.
        """
        context_window = await _fetch_context_window(
            model, base_url=base_url, http_client=http_client, headers=headers
        )
        return TokenBudget(context_window, client_max_tokens=client_max_tokens)
