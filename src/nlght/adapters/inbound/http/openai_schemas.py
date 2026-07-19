# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, field_validator


class ChatMessage(BaseModel):
    role: str
    content: str | list[Any]

    @field_validator("role")
    @classmethod
    def role_must_be_known(cls, v: str) -> str:
        allowed = {"system", "user", "assistant", "tool", "function"}
        if v not in allowed:
            raise ValueError(f"role must be one of {sorted(allowed)}, got {v!r}")
        return v


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    stream: bool = False

    @field_validator("messages")
    @classmethod
    def messages_must_not_be_empty(cls, v: list[ChatMessage]) -> list[ChatMessage]:
        if not v:
            raise ValueError("messages must contain at least one entry")
        return v

    model_config = {"extra": "allow"}  # forward unknown fields to the model provider
