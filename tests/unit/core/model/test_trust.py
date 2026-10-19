# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from itertools import permutations

import pytest

from nlght.core.model.trust import (
    PLATFORM_TEXT,
    UNANNOTATED,
    UNANNOTATED_SOURCE,
    ContentExposure,
    ContentLedger,
    ContentOrigin,
    ContentTrust,
    SourceRef,
    merge_sources,
)

TRUSTED = ContentTrust.TRUSTED
UNTRUSTED = ContentTrust.UNTRUSTED


def _ref(trust: ContentTrust, *, name: str = "", resource: str = "store-a") -> SourceRef:
    return SourceRef("data_source", resource, "handbook", trust, name)


def test_the_same_source_seen_untrusted_once_is_untrusted_in_every_order() -> None:
    refs = [_ref(TRUSTED, name="Handbook"), _ref(UNTRUSTED, name="handbook (web)"), _ref(TRUSTED)]

    results = {merge_sources(order) for order in permutations(refs)}

    assert len(results) == 1
    (merged,) = results.pop()
    assert merged.trust is UNTRUSTED
    assert merged.name == "Handbook"


def test_display_name_is_not_part_of_the_identity() -> None:
    merged = merge_sources([_ref(TRUSTED, name="A"), _ref(TRUSTED, name="B")])

    assert len(merged) == 1
    assert merged[0].trust is TRUSTED


def test_one_source_id_in_two_stores_is_two_sources() -> None:
    merged = merge_sources([_ref(TRUSTED, resource="store-a"), _ref(UNTRUSTED, resource="store-b")])

    assert [(ref.resource_id, ref.trust) for ref in merged] == [
        ("store-a", TRUSTED),
        ("store-b", UNTRUSTED),
    ]


def test_an_empty_origin_is_unannotated_and_untrusted() -> None:
    assert ContentOrigin().trust is UNTRUSTED
    assert UNANNOTATED.trust is UNTRUSTED
    assert UNANNOTATED.effective_sources == (UNANNOTATED_SOURCE,)
    assert ContentExposure.of(ContentOrigin()).has_external_untrusted_input


def test_an_origin_is_trusted_only_when_every_source_is() -> None:
    trusted = ContentOrigin.of(_ref(TRUSTED))
    mixed = ContentOrigin.combine(trusted, ContentOrigin.of(SourceRef("web", "", "x", UNTRUSTED)))

    assert trusted.trust is TRUSTED
    assert mixed.trust is UNTRUSTED


def test_combine_keeps_trusted_sources_next_to_untrusted_ones() -> None:
    web = SourceRef("web", "", "https://example.org", UNTRUSTED)

    combined = ContentOrigin.combine(ContentOrigin.of(_ref(TRUSTED)), ContentOrigin.of(web))

    assert set(combined.sources) == {_ref(TRUSTED), web}
    assert ContentExposure.of(combined).untrusted_sources == (web,)


def test_combining_an_unannotated_origin_makes_the_result_untrusted() -> None:
    combined = ContentOrigin.combine(PLATFORM_TEXT, UNANNOTATED)

    assert combined.trust is UNTRUSTED
    assert UNANNOTATED_SOURCE in combined.sources


def test_exposure_holds_untrusted_sources_only() -> None:
    with pytest.raises(ValueError):
        ContentExposure((_ref(TRUSTED),))


def test_ledger_only_grows() -> None:
    ledger = ContentLedger()
    assert not ledger.exposure.has_external_untrusted_input

    web = ContentOrigin.of(SourceRef("web", "", "https://example.org", UNTRUSTED))
    ledger.record(web)
    ledger.record(ContentOrigin.of(_ref(TRUSTED)))

    assert ledger.exposure.untrusted_sources == web.sources
    assert set(ledger.origin.sources) == {*web.sources, _ref(TRUSTED)}


def test_a_later_trusted_sighting_does_not_clear_an_untrusted_one() -> None:
    ledger = ContentLedger()
    ledger.record(ContentOrigin.of(_ref(UNTRUSTED)))
    ledger.record(ContentOrigin.of(_ref(TRUSTED)))

    assert ledger.exposure.untrusted_sources == (_ref(UNTRUSTED),)
