# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Track source provenance and content trust across model context."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from enum import StrEnum


class ContentTrust(StrEnum):
    TRUSTED = "trusted"
    UNTRUSTED = "untrusted"


#: Reserve these names for platform origins so integrations can add distinct kinds.
SOURCE_DATA = "data_source"
SOURCE_KNOWLEDGE = "knowledge_source"
SOURCE_WEB = "web"
SOURCE_GIT = "git"
SOURCE_RUNTIME = "runtime"
SOURCE_CALLER = "caller"
SOURCE_CALLER_ATTACHMENT = "caller_attachment"
SOURCE_CALLER_HISTORY = "caller_history"
SOURCE_PLAYBOOK = "playbook"
SOURCE_PLATFORM = "platform"
SOURCE_UNANNOTATED = "unannotated"


@dataclass(frozen=True, order=True)
class SourceRef:
    """One source of content, unique by ``(kind, resource_id, source_id)``.

    ``resource_id`` scopes the source: ``confluence-handbook`` in two stores is
    two sources. ``name`` is for display only and never part of the identity.
    """

    kind: str
    resource_id: str
    source_id: str
    trust: ContentTrust
    name: str = ""

    @property
    def identity(self) -> tuple[str, str, str]:
        return (self.kind, self.resource_id, self.source_id)

    @property
    def is_trusted(self) -> bool:
        return self.trust is ContentTrust.TRUSTED


UNANNOTATED_SOURCE = SourceRef(SOURCE_UNANNOTATED, "", "", ContentTrust.UNTRUSTED)
PLATFORM_SOURCE = SourceRef(SOURCE_PLATFORM, "", "", ContentTrust.TRUSTED)


def merge_sources(refs: Iterable[SourceRef]) -> tuple[SourceRef, ...]:
    """Merge source references by identity without raising trust.

    Args:
        refs (Iterable[SourceRef]): Source references in any order.

    Returns:
        tuple[SourceRef, ...]: Canonically sorted references with untrusted precedence.
    """
    by_identity: dict[tuple[str, str, str], SourceRef] = {}
    for ref in refs:
        current = by_identity.get(ref.identity)
        if current is None:
            by_identity[ref.identity] = ref
            continue
        trust = (
            ContentTrust.TRUSTED
            if current.is_trusted and ref.is_trusted
            else ContentTrust.UNTRUSTED
        )
        names = [name for name in (current.name, ref.name) if name]
        by_identity[ref.identity] = replace(
            current, trust=trust, name=min(names) if names else "",
        )
    return tuple(sorted(by_identity.values()))


@dataclass(frozen=True)
class ContentOrigin:
    """The sources of **all** content that went into something, trusted or not.

    Stored and derived content keeps its origin. The decision gets an exposure
    computed from it. A summary, a remembered fact or a model answer inherits the
    origin of what it was made from.
    """

    sources: tuple[SourceRef, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "sources", merge_sources(self.sources))

    @classmethod
    def of(cls, *sources: SourceRef) -> ContentOrigin:
        return cls(tuple(sources))

    @classmethod
    def combine(cls, *origins: ContentOrigin) -> ContentOrigin:
        return cls(tuple(ref for origin in origins for ref in origin.effective_sources))

    @property
    def effective_sources(self) -> tuple[SourceRef, ...]:
        """Represent empty provenance as unannotated content.

        Returns:
            tuple[SourceRef, ...]: Recorded sources or the unannotated source.
        """
        return self.sources or (UNANNOTATED_SOURCE,)

    @property
    def trust(self) -> ContentTrust:
        """Evaluate trust across every recorded source.

        Returns:
            ContentTrust: Trusted only for a nonempty all-trusted source set.
        """
        if self.sources and all(ref.is_trusted for ref in self.sources):
            return ContentTrust.TRUSTED
        return ContentTrust.UNTRUSTED

    @property
    def untrusted_sources(self) -> tuple[SourceRef, ...]:
        return tuple(ref for ref in self.effective_sources if not ref.is_trusted)


UNANNOTATED = ContentOrigin()
PLATFORM_TEXT = ContentOrigin.of(PLATFORM_SOURCE)


@dataclass(frozen=True)
class ContentExposure:
    """What a decision needs to know: the untrusted sources the model has seen.

    Derived from an origin, never maintained on its own. So is the bool.
    """

    untrusted_sources: tuple[SourceRef, ...]

    def __post_init__(self) -> None:
        refs = merge_sources(self.untrusted_sources)
        if any(ref.is_trusted for ref in refs):
            raise ValueError("ContentExposure holds untrusted sources only")
        object.__setattr__(self, "untrusted_sources", refs)

    @classmethod
    def of(cls, origin: ContentOrigin) -> ContentExposure:
        return cls(origin.untrusted_sources)

    @property
    def has_external_untrusted_input(self) -> bool:
        return bool(self.untrusted_sources)


@dataclass
class ContentLedger:
    """Everything a running workflow's model context has taken in.

    Monotonic: it only grows. Trimming or dropping messages later does not
    remove what was already read, and a decision is always made on the ledger,
    not on whatever messages happen to be left.
    """

    _origin: ContentOrigin = field(default_factory=ContentOrigin)

    @property
    def origin(self) -> ContentOrigin:
        return self._origin

    @property
    def exposure(self) -> ContentExposure:
        """Report untrusted sources read by this workflow.

        Returns:
            ContentExposure: Exposure from recorded untrusted sources.
        """
        return ContentExposure(tuple(ref for ref in self._origin.sources if not ref.is_trusted))

    def record(self, *origins: ContentOrigin) -> ContentOrigin:
        """Add read content origins to the workflow ledger.

        Args:
            *origins (ContentOrigin): Origins of content read by this workflow.

        Returns:
            ContentOrigin: The ledger's combined origin.

        Side Effects:
            Mutates the ledger's origin when inputs exist.
        """
        if origins:
            self._origin = ContentOrigin(
                self._origin.sources
                + tuple(ref for origin in origins for ref in origin.effective_sources)
            )
        return self._origin


def origin_to_json(origin: ContentOrigin | None) -> list[dict[str, str]] | None:
    """Serialize an origin without inventing absent provenance.

    Args:
        origin (ContentOrigin | None): A content origin or None.

    Returns:
        list[dict[str, str]] | None: JSON-compatible source records or None.
    """
    if origin is None:
        return None
    return [
        {
            "kind": ref.kind,
            "resource_id": ref.resource_id,
            "source_id": ref.source_id,
            "trust": ref.trust.value,
            "name": ref.name,
        }
        for ref in origin.sources
    ]


def origin_from_json(data: object) -> ContentOrigin | None:
    """Parse origin records while rejecting unreadable provenance.

    Args:
        data (object): An arbitrary decoded JSON value.

    Returns:
        ContentOrigin | None: A content origin or None for invalid records.
    """
    if not isinstance(data, list):
        return None
    refs: list[SourceRef] = []
    for item in data:
        if not isinstance(item, dict):
            return None
        try:
            refs.append(SourceRef(
                kind=str(item["kind"]),
                resource_id=str(item.get("resource_id", "")),
                source_id=str(item.get("source_id", "")),
                trust=ContentTrust(item["trust"]),
                name=str(item.get("name", "")),
            ))
        except (KeyError, ValueError):
            return None
    return ContentOrigin(tuple(refs))
