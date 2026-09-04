# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import asyncio
import base64
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any, Protocol

import httpx2

from nlght.core.ingestion import AcquisitionDiagnostic, SourceDocument, SourceSnapshot


@dataclass(slots=True, frozen=True)
class ConfluenceResponse:
    status_code: int
    headers: dict[str, str]
    payload: object


class ConfluenceTransport(Protocol):
    async def search(
        self,
        *,
        base_url: str,
        params: Mapping[str, str | int],
        authorization: str,
        timeout_seconds: float,
    ) -> ConfluenceResponse: ...


class HttpxConfluenceTransport:
    def __init__(self, client: httpx2.AsyncClient) -> None:
        self._client = client

    async def search(
        self,
        *,
        base_url: str,
        params: Mapping[str, str | int],
        authorization: str,
        timeout_seconds: float,
    ) -> ConfluenceResponse:
        response = await self._client.get(
            f"{base_url.rstrip('/')}/rest/api/content/search",
            params=params,
            headers={"Authorization": authorization, "Accept": "application/json"},
            timeout=timeout_seconds,
        )
        try:
            payload = response.json()
        except ValueError:
            payload = None
        return ConfluenceResponse(
            status_code=response.status_code,
            headers=dict(response.headers),
            payload=payload,
        )


@dataclass(slots=True, frozen=True)
class ConfluenceSourceSettings:
    source_id: str
    base_url: str
    space_keys: tuple[str, ...]
    user_email: str
    api_token: str = field(repr=False)
    cql: str | None = None
    max_pages: int = 1000
    page_size: int = 50
    timeout_seconds: float = 30.0
    max_attempts: int = 3
    retry_base_seconds: float = 0.25
    retry_max_seconds: float = 5.0

    def __post_init__(self) -> None:
        if not self.source_id.strip() or not self.base_url.strip() or not self.space_keys:
            raise ValueError("Confluence source_id, base_url, and space_keys must not be empty")
        if any(not key.strip() for key in self.space_keys):
            raise ValueError("Confluence space keys must not be empty")
        if not self.user_email or not self.api_token:
            raise ValueError("Confluence credentials must not be empty")
        if self.max_pages < 1 or not 1 <= self.page_size <= 100 or self.max_attempts < 1:
            raise ValueError("Confluence limits and max_attempts must be positive")
        if self.timeout_seconds <= 0 or self.retry_base_seconds < 0 or self.retry_max_seconds < self.retry_base_seconds:
            raise ValueError("Confluence timeout and retry settings are invalid")


class ConfluenceDocumentSource:
    def __init__(
        self,
        settings: ConfluenceSourceSettings,
        transport: ConfluenceTransport,
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._settings = settings
        self._transport = transport
        self._sleep = sleep
        encoded = base64.b64encode(f"{settings.user_email}:{settings.api_token}".encode()).decode("ascii")
        self._authorization = f"Basic {encoded}"

    @property
    def source_id(self) -> str:
        return self._settings.source_id

    async def acquire(self) -> SourceSnapshot:
        documents: dict[str, SourceDocument] = {}
        observed: set[str] = set()
        diagnostics: list[AcquisitionDiagnostic] = []
        complete = True

        for space_key in self._settings.space_keys:
            start = 0
            fetched = 0
            space_complete = False
            while fetched < self._settings.max_pages:
                try:
                    payload = await self._fetch(space_key=space_key, start=start)
                except _ConfluenceRequestFailed as exc:
                    diagnostics.append(AcquisitionDiagnostic("request-failed", space_key, exc.reason))
                    complete = False
                    break

                results = payload.get("results")
                if not isinstance(results, list):
                    diagnostics.append(AcquisitionDiagnostic("invalid-response", space_key, "results is not a list"))
                    complete = False
                    break
                if not results:
                    space_complete = True
                    break

                remaining = self._settings.max_pages - fetched
                selected = results[:remaining]
                for raw_page in selected:
                    if not isinstance(raw_page, dict):
                        diagnostics.append(AcquisitionDiagnostic("invalid-page", space_key, "page is not an object"))
                        complete = False
                        continue
                    if not self._add_page(raw_page, space_key, documents, observed, diagnostics):
                        complete = False

                count = len(selected)
                fetched += count
                start += count
                has_more = len(results) >= self._settings.page_size or bool(payload.get("_links", {}).get("next"))
                if count < len(results) or (fetched >= self._settings.max_pages and has_more):
                    diagnostics.append(AcquisitionDiagnostic("document-limit", space_key, "configured max_pages reached before snapshot completed"))
                    complete = False
                    break
                if not has_more:
                    space_complete = True
                    break
            if not space_complete and fetched >= self._settings.max_pages:
                complete = False

        ordered_documents = tuple(documents[key] for key in sorted(documents))
        return SourceSnapshot(
            source_id=self.source_id,
            documents=ordered_documents,
            observed_external_ids=tuple(sorted(observed)),
            complete=complete,
            diagnostics=tuple(diagnostics),
        )

    async def _fetch(self, *, space_key: str, start: int) -> dict[str, Any]:
        cql = f"space={space_key} and type=page"
        if self._settings.cql:
            cql = f"({cql}) and ({self._settings.cql})"
        params: dict[str, str | int] = {
            "cql": cql,
            "limit": self._settings.page_size,
            "start": start,
            "expand": "body.storage,version,space,status,_links",
        }
        reason = "request failed"
        for attempt in range(1, self._settings.max_attempts + 1):
            try:
                response = await self._transport.search(
                    base_url=self._settings.base_url,
                    params=params,
                    authorization=self._authorization,
                    timeout_seconds=self._settings.timeout_seconds,
                )
                if response.status_code < 400:
                    payload = response.payload
                    if isinstance(payload, dict):
                        return payload
                    raise _ConfluenceRequestFailed("response is not an object")
                reason = f"HTTP {response.status_code}"
                if response.status_code not in {408, 425, 429, 500, 502, 503, 504}:
                    raise _ConfluenceRequestFailed(reason)
                retry_after = _retry_after_seconds(response.headers)
            except _ConfluenceRequestFailed:
                raise
            except Exception as exc:
                reason = type(exc).__name__
                retry_after = None
            if attempt < self._settings.max_attempts:
                delay = (
                    min(self._settings.retry_max_seconds, retry_after)
                    if retry_after is not None
                    else min(
                        self._settings.retry_max_seconds,
                        self._settings.retry_base_seconds * (2 ** (attempt - 1)),
                    )
                )
                await self._sleep(delay)
        raise _ConfluenceRequestFailed(reason)

    def _add_page(
        self,
        page: dict[str, Any],
        space_key: str,
        documents: dict[str, SourceDocument],
        observed: set[str],
        diagnostics: list[AcquisitionDiagnostic],
    ) -> bool:
        page_id = str(page.get("id") or "").strip()
        if not page_id:
            diagnostics.append(AcquisitionDiagnostic("missing-page-id", space_key, "page omitted id"))
            return False
        if str(page.get("status") or "current").lower() in {"archived", "trashed", "deleted"}:
            return True
        external_id = f"confluence:{page_id}"
        observed.add(external_id)
        body = page.get("body") or {}
        storage = body.get("storage") if isinstance(body, dict) else {}
        html = str((storage or {}).get("value") or "") if isinstance(storage, dict) else ""
        text = strip_html(html)
        version_data = page.get("version") or {}
        version = str(version_data.get("number") or 0) if isinstance(version_data, dict) else "0"
        links = page.get("_links") or {}
        webui = links.get("webui") if isinstance(links, dict) else None
        location = f"{self._settings.base_url.rstrip('/')}{webui}" if webui else None
        title = str(page.get("title") or "Untitled")
        documents[external_id] = SourceDocument(
            source=self.source_id,
            external_id=external_id,
            path=f"{space_key}/{page_id}.html",
            # Text for the index, markup for the parser. Storing only the text
            # is what made every page produce zero knowledge units; storing only
            # the markup would change what the index contains, which is a
            # separate decision needing a reindex.
            content=text.encode("utf-8"),
            source_form=html.encode("utf-8"),
            source_revision=version,
            metadata={
                "source_type": "confluence",
                "space_key": space_key,
                "page_id": page_id,
                "title": title,
                "url": location,
                "version": version,
            },
        )
        return True


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if tag in {"script", "style"}:
            self._ignored_depth += 1
        elif not self._ignored_depth and tag in {"br", "p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._ignored_depth:
            self._ignored_depth -= 1
        elif not self._ignored_depth and tag in {"p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth and data.strip():
            self._parts.append(data)

    def text(self) -> str:
        return "\n".join(line.strip() for line in "".join(self._parts).splitlines() if line.strip())


def strip_html(value: str) -> str:
    parser = _TextExtractor()
    parser.feed(value)
    parser.close()
    return parser.text()


def _retry_after_seconds(headers: Mapping[str, str]) -> float | None:
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None


class _ConfluenceRequestFailed(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason
