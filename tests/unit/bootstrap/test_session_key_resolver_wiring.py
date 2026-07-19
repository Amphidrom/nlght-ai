# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Tests for _build_single_resolver in wiring.py."""
from __future__ import annotations

from nlght.adapters.outbound.session.body import BodyParameterSessionKeyResolver
from nlght.adapters.outbound.session.composite import CompositeSessionKeyResolver
from nlght.adapters.outbound.session.header import HeaderSessionKeyResolver
from nlght.adapters.outbound.session.query import QueryParamSessionKeyResolver
from nlght.bootstrap.wiring import _build_single_resolver

# ---------------------------------------------------------------------------
# _build_single_resolver
# ---------------------------------------------------------------------------


def test_builds_header_resolver() -> None:
    r = _build_single_resolver({"resolver": "header", "key": "x-session-id"})
    assert isinstance(r, HeaderSessionKeyResolver)


def test_builds_query_resolver() -> None:
    r = _build_single_resolver({"resolver": "query", "parameter": "session_id"})
    assert isinstance(r, QueryParamSessionKeyResolver)


def test_builds_body_resolver() -> None:
    r = _build_single_resolver({"resolver": "body", "key": "meta.session_id"})
    assert isinstance(r, BodyParameterSessionKeyResolver)


def test_builds_composite_resolver() -> None:
    r = _build_single_resolver(
        {
            "resolver": "composite",
            "resolvers": [
                {"resolver": "header", "key": "x-session-id"},
                {"resolver": "query", "parameter": "session_id"},
            ],
        }
    )
    assert isinstance(r, CompositeSessionKeyResolver)


def test_unknown_kind_returns_none() -> None:
    assert _build_single_resolver({"resolver": "magic"}) is None


def test_header_missing_key_returns_none() -> None:
    assert _build_single_resolver({"resolver": "header"}) is None


def test_query_missing_parameter_returns_none() -> None:
    assert _build_single_resolver({"resolver": "query"}) is None


def test_body_missing_key_returns_none() -> None:
    assert _build_single_resolver({"resolver": "body"}) is None


def test_composite_empty_resolvers_returns_none() -> None:
    assert _build_single_resolver({"resolver": "composite", "resolvers": []}) is None


def test_composite_all_invalid_children_returns_none() -> None:
    assert (
        _build_single_resolver(
            {
                "resolver": "composite",
                "resolvers": [{"resolver": "unknown"}],
            }
        )
        is None
    )