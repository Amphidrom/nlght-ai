# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Tests for session key resolver adapters."""
from __future__ import annotations

import asyncio
import json

from nlght.adapters.outbound.session.body import BodyParameterSessionKeyResolver
from nlght.adapters.outbound.session.composite import CompositeSessionKeyResolver
from nlght.adapters.outbound.session.header import HeaderSessionKeyResolver
from nlght.adapters.outbound.session.query import QueryParamSessionKeyResolver

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_JSON_CT = {"content-type": "application/json"}


def _run(coro):
    return asyncio.run(coro)


def _resolve(resolver, *, headers=None, query_params=None, raw_body=b""):
    return _run(
        resolver.resolve(
            path="/test",
            method="POST",
            headers=headers or {},
            query_params=query_params or {},
            raw_body=raw_body,
        )
    )


# ---------------------------------------------------------------------------
# HeaderSessionKeyResolver
# ---------------------------------------------------------------------------


def test_header_resolver_returns_value() -> None:
    resolver = HeaderSessionKeyResolver("x-session-id")
    assert _resolve(resolver, headers={"x-session-id": "sess-abc"}) == "sess-abc"


def test_header_resolver_lowercases_header_name() -> None:
    resolver = HeaderSessionKeyResolver("X-Session-ID")
    assert _resolve(resolver, headers={"x-session-id": "sess-abc"}) == "sess-abc"


def test_header_resolver_returns_none_if_missing() -> None:
    resolver = HeaderSessionKeyResolver("x-session-id")
    assert _resolve(resolver, headers={"x-other": "val"}) is None


def test_header_resolver_returns_none_for_empty_header() -> None:
    resolver = HeaderSessionKeyResolver("x-session-id")
    assert _resolve(resolver, headers={"x-session-id": ""}) is None


# ---------------------------------------------------------------------------
# BodyParameterSessionKeyResolver
# ---------------------------------------------------------------------------


def test_body_resolver_extracts_top_level_key() -> None:
    resolver = BodyParameterSessionKeyResolver("session_id")
    body = json.dumps({"session_id": "body-sess"}).encode()
    assert _resolve(resolver, headers=_JSON_CT, raw_body=body) == "body-sess"


def test_body_resolver_extracts_nested_key() -> None:
    resolver = BodyParameterSessionKeyResolver("meta.session_id")
    body = json.dumps({"meta": {"session_id": "nested-sess"}}).encode()
    assert _resolve(resolver, headers=_JSON_CT, raw_body=body) == "nested-sess"


def test_body_resolver_returns_none_non_json_content_type() -> None:
    resolver = BodyParameterSessionKeyResolver("session_id")
    body = json.dumps({"session_id": "x"}).encode()
    assert _resolve(resolver, headers={"content-type": "text/plain"}, raw_body=body) is None


def test_body_resolver_returns_none_empty_body() -> None:
    resolver = BodyParameterSessionKeyResolver("session_id")
    assert _resolve(resolver, headers=_JSON_CT, raw_body=b"") is None


def test_body_resolver_returns_none_missing_key() -> None:
    resolver = BodyParameterSessionKeyResolver("session_id")
    body = json.dumps({"other": "value"}).encode()
    assert _resolve(resolver, headers=_JSON_CT, raw_body=body) is None


def test_body_resolver_returns_none_invalid_json() -> None:
    resolver = BodyParameterSessionKeyResolver("session_id")
    assert _resolve(resolver, headers=_JSON_CT, raw_body=b"not-json") is None


def test_body_resolver_returns_none_for_whitespace_only_value() -> None:
    resolver = BodyParameterSessionKeyResolver("session_id")
    body = json.dumps({"session_id": "   "}).encode()
    assert _resolve(resolver, headers=_JSON_CT, raw_body=body) is None


# ---------------------------------------------------------------------------
# QueryParamSessionKeyResolver
# ---------------------------------------------------------------------------


def test_query_resolver_returns_value() -> None:
    resolver = QueryParamSessionKeyResolver("session_id")
    assert _resolve(resolver, query_params={"session_id": "qsess"}) == "qsess"


def test_query_resolver_returns_none_if_missing() -> None:
    resolver = QueryParamSessionKeyResolver("session_id")
    assert _resolve(resolver, query_params={"other": "x"}) is None


def test_query_resolver_returns_none_for_empty_param() -> None:
    resolver = QueryParamSessionKeyResolver("session_id")
    assert _resolve(resolver, query_params={"session_id": ""}) is None


# ---------------------------------------------------------------------------
# CompositeSessionKeyResolver
# ---------------------------------------------------------------------------


def test_composite_returns_first_match() -> None:
    resolver = CompositeSessionKeyResolver(
        [
            HeaderSessionKeyResolver("x-session-id"),
            QueryParamSessionKeyResolver("session_id"),
        ]
    )
    result = _resolve(
        resolver,
        headers={"x-session-id": "from-header"},
        query_params={"session_id": "from-query"},
    )
    assert result == "from-header"


def test_composite_falls_through_to_second() -> None:
    resolver = CompositeSessionKeyResolver(
        [
            HeaderSessionKeyResolver("x-session-id"),
            QueryParamSessionKeyResolver("session_id"),
        ]
    )
    result = _resolve(resolver, query_params={"session_id": "from-query"})
    assert result == "from-query"


def test_composite_returns_none_if_all_none() -> None:
    resolver = CompositeSessionKeyResolver(
        [
            HeaderSessionKeyResolver("x-session-id"),
            QueryParamSessionKeyResolver("session_id"),
        ]
    )
    assert _resolve(resolver) is None


def test_composite_stops_at_first_match() -> None:
    """Second resolver must never be called once first returns a value."""
    called = []

    class _TrackingResolver:
        def __init__(self, name: str, returns: str | None) -> None:
            self._name = name
            self._returns = returns

        async def resolve(self, path, method, headers, query_params, raw_body):
            called.append(self._name)
            return self._returns

    resolver = CompositeSessionKeyResolver(
        [_TrackingResolver("first", "sess-x"), _TrackingResolver("second", "sess-y")]
    )
    result = _run(
        resolver.resolve(
            path="/test", method="POST", headers={}, query_params={}, raw_body=b""
        )
    )
    assert result == "sess-x"
    assert called == ["first"]
