# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from enum import Enum


class KnownDirective(Enum):
    """All known behavioral directives with their system-prompt templates.

    Each directive has a ``key`` (used in DirectiveStore) and a
    ``template`` string with a ``{value}`` placeholder.

    Usage::

        rendered = KnownDirective.CONVERSATION_LANGUAGE.render("de")
        # → "Respond in the language specified by ISO 639-1 code: de."

    Unknown keys (not in this enum) are rendered as plain
    ``key: value`` lines by SystemPromptBuilder.
    """

    CONVERSATION_LANGUAGE = (
        "conversation_language",
        "Respond in the language specified by ISO 639-1 code: {value}.",
    )
    TIMEZONE = (
        "timezone",
        "The user's local timezone is {value} (IANA format). "
        "Use this when interpreting or formatting dates and times.",
    )
    TONE = (
        "tone",
        "Use a {value} tone throughout your response.",
    )

    def __init__(self, key: str, template: str) -> None:
        self.key      = key
        self.template = template

    def render(self, value: str) -> str:
        return self.template.format(value=value)

    @classmethod
    def for_key(cls, key: str) -> KnownDirective | None:
        for member in cls:
            if member.key == key:
                return member
        return None
