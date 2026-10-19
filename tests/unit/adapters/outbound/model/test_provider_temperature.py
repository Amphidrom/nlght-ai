# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Send configured sampling temperature through model provider calls.

An unconfigured provider leaves the value unset so the provider default applies."""
from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx2
import pytest

from nlght.adapters.outbound.model.ollama_cloud import OllamaCloudClient
from nlght.adapters.outbound.model.ollama_local import OllamaClient
from nlght.adapters.outbound.signals.buffering import BufferingSignalEmitter
from nlght.core.model.messages import UserMessage


def _openai(temperature: float | None) -> tuple[Any, AsyncMock]:
    pytest.importorskip("openai")
    from nlght.adapters.outbound.model.openai_cloud import OpenAICloudModelClient

    client = OpenAICloudModelClient(api_key="test-key", default_model="gpt-4o", temperature=temperature)
    create = AsyncMock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))], usage=None,
    ))
    client._client = MagicMock()
    client._client.chat.completions.create = create
    return client, create


async def test_openai_sends_the_configured_temperature() -> None:
    client, create = _openai(0.3)

    await client.bind(model=None, emitter=BufferingSignalEmitter(), stream=False).call([UserMessage("hi")])

    assert create.await_args.kwargs["temperature"] == 0.3


async def test_openai_sends_no_temperature_when_unconfigured() -> None:
    client, create = _openai(None)

    await client.bind(model=None, emitter=BufferingSignalEmitter(), stream=False).call([UserMessage("hi")])

    assert "temperature" not in create.await_args.kwargs


async def test_google_sends_the_configured_temperature(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("google.genai")
    from nlght.adapters.outbound.model.google import GoogleModelClient

    # Records what the client builds its request config from, independent of
    # whichever google.genai module object sys.modules holds at this point.
    configs: list[dict[str, Any]] = []
    fake_types = SimpleNamespace(GenerateContentConfig=lambda **kw: configs.append(kw) or kw)
    monkeypatch.setitem(sys.modules, "google.genai.types", fake_types)
    monkeypatch.setattr(sys.modules["google.genai"], "types", fake_types, raising=False)

    # Built without __init__ so the request runs against the recording types
    # above. The constructor's own wiring is asserted in the next test.
    client = GoogleModelClient.__new__(GoogleModelClient)
    client._client = MagicMock()
    client._client.aio.models.generate_content = AsyncMock(return_value=SimpleNamespace(text="ok"))
    client._default_model = "gemini-3.6-flash"
    client._temperature = 0.3

    await client.bind(model=None, emitter=BufferingSignalEmitter(), stream=False).call([UserMessage("hi")])

    assert configs[-1]["temperature"] == 0.3


def test_google_constructor_takes_the_temperature() -> None:
    pytest.importorskip("google.genai")
    from nlght.adapters.outbound.model.google import GoogleModelClient

    client = GoogleModelClient(api_key="test-key", default_model="gemini-3.6-flash", temperature=0.4)

    assert client._temperature == 0.4


@pytest.mark.parametrize("client_cls", [OllamaClient, OllamaCloudClient])
@pytest.mark.parametrize("temperature", [0.0, 0.7, None])
async def test_ollama_sends_exactly_the_configured_temperature(
    client_cls: type[OllamaClient] | type[OllamaCloudClient], temperature: float | None,
) -> None:
    client = client_cls(
        http_client=httpx2.AsyncClient(),
        base_url="http://ollama.test",
        default_model="llama3",
        temperature=temperature,
    )
    complete = AsyncMock()
    client._http.complete = complete

    await client.bind(model=None, emitter=BufferingSignalEmitter(), stream=False).call([UserMessage("hi")])

    body = complete.await_args.args[1]
    if temperature is None:
        assert "temperature" not in body
    else:
        assert body["temperature"] == temperature


def test_anthropic_offers_no_temperature() -> None:
    pytest.importorskip("anthropic")
    from nlght.adapters.outbound.model.anthropic import AnthropicModelClient

    with pytest.raises(TypeError):
        AnthropicModelClient(api_key="test-key", default_model="claude-sonnet-4-6", temperature=0.0)  # type: ignore[call-arg]


async def test_anthropic_never_sends_a_temperature() -> None:
    pytest.importorskip("anthropic")
    from nlght.adapters.outbound.model.anthropic import AnthropicModelClient

    client = AnthropicModelClient(api_key="test-key", default_model="claude-sonnet-4-6")
    create = AsyncMock(return_value=SimpleNamespace(content=[SimpleNamespace(text="ok")], usage=None))
    client._client = MagicMock()
    client._client.messages.create = create

    await client.bind(model=None, emitter=BufferingSignalEmitter(), stream=False).call([UserMessage("hi")])

    assert "temperature" not in create.await_args.kwargs
    assert "extra_body" not in create.await_args.kwargs


@pytest.mark.parametrize("method", ["call", "stream"])
def test_the_port_takes_no_sampling_argument(method: str) -> None:
    import inspect

    from nlght.ports.outbound.model_client import ModelClient

    assert "temperature" not in inspect.signature(getattr(ModelClient, method)).parameters
