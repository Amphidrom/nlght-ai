# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

from datetime import UTC
from pathlib import Path

import pytest

from nlght.adapters.outbound.licensing import keygen_license_adapter as keygen
from nlght.adapters.outbound.licensing.keygen_license_adapter import (
    KeygenLicenseAdapter,
    _machine_fingerprint,
    _parse_expiry,
)
from nlght.core.licensing.license import Tier


class _Response:
    def __init__(self, body: dict, status_code: int = 200, text: str = "") -> None:
        self._body = body
        self.status_code = status_code
        self.text = text

    def json(self) -> dict:
        return self._body


class _Client:
    def __init__(self) -> None:
        self.posts: list[dict] = []
        self.gets: list[dict] = []
        self.post_responses: list[_Response | Exception] = []
        self.get_responses: list[_Response | Exception] = []

    def __enter__(self) -> _Client:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def post(self, url: str, **kwargs: object) -> _Response:
        self.posts.append({"url": url, **kwargs})
        response = self.post_responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def get(self, url: str, **kwargs: object) -> _Response:
        self.gets.append({"url": url, **kwargs})
        response = self.get_responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _adapter() -> KeygenLicenseAdapter:
    return KeygenLicenseAdapter.__new__(KeygenLicenseAdapter)


def _valid_body() -> dict:
    return {
        "meta": {"valid": True},
        "data": {
            "id": "license-1",
            "attributes": {
                "maxMachines": 3,
                "metadata": {"tier": "commercial"},
            },
            "relationships": {
                "entitlements": {"links": {"related": "/v1/licenses/license-1/entitlements"}}
            },
        },
    }


def test_parse_expiry_handles_zulu_and_invalid_values() -> None:
    parsed = _parse_expiry("2026-01-01T12:00:00Z")

    assert parsed is not None
    assert parsed.tzinfo == UTC
    assert _parse_expiry(None) is None
    assert _parse_expiry("not-a-date") is None


def test_machine_fingerprint_uses_host_platform_and_mac_addresses(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(keygen.socket, "gethostname", lambda: "host-a")
    monkeypatch.setattr(keygen.platform, "system", lambda: "Windows")
    monkeypatch.setattr(keygen.platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(keygen.platform, "processor", lambda: "cpu")
    monkeypatch.setattr(keygen, "_get_mac_addresses", lambda: {"eth0": "aa:bb"})

    first = _machine_fingerprint()
    second = _machine_fingerprint()

    assert first == second
    assert len(first) == 64


def test_validate_posts_license_key_and_wraps_transport_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(keygen, "_machine_fingerprint", lambda: "fp")
    client = _Client()
    client.post_responses.append(_Response({"meta": {"valid": True}}))

    body = _adapter()._validate(client, "https://license.example/", "key-1")

    assert body["meta"]["valid"] is True
    assert client.posts[0]["url"] == "https://license.example/v1/licenses/actions/validate-key"
    assert client.posts[0]["json"] == {
        "meta": {"key": "key-1", "scope": {"fingerprint": "fp"}}
    }

    client.post_responses.append(RuntimeError("offline"))
    with pytest.raises(RuntimeError, match="License server unreachable"):
        _adapter()._validate(client, "https://license.example", "key-1")


def test_fetch_related_and_single_return_safe_defaults_on_errors() -> None:
    client = _Client()
    client.get_responses.extend([
        _Response({"data": {"id": "one"}}),
        RuntimeError("broken"),
        _Response({"data": [{"id": "a"}]}),
        RuntimeError("broken"),
    ])

    adapter = _adapter()

    assert adapter._fetch_single(client, "https://license.example", "/one", "key") == {"id": "one"}
    assert adapter._fetch_single(client, "https://license.example", "/broken", "key") is None
    assert adapter._fetch_related(client, "https://license.example", "/many", "key") == [{"id": "a"}]
    assert adapter._fetch_related(client, "https://license.example", "/broken", "key") == []
    assert client.gets[2]["url"].endswith("/many?limit=100")


def test_activate_accepts_existing_fingerprint_but_raises_other_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(keygen, "_machine_fingerprint", lambda: "fingerprint")
    monkeypatch.setattr(keygen.socket, "gethostname", lambda: "host")
    monkeypatch.setattr(keygen.platform, "system", lambda: "Windows")
    monkeypatch.setattr(keygen.platform, "processor", lambda: "cpu")
    monkeypatch.setattr(keygen.platform, "machine", lambda: "AMD64")

    client = _Client()
    client.post_responses.extend([
        _Response({"data": {"id": "machine"}}, status_code=201),
        _Response({"errors": [{"code": "FINGERPRINT_TAKEN"}]}, status_code=422),
        _Response({"errors": [{"code": "NO_MACHINE"}]}, status_code=422),
    ])
    adapter = _adapter()

    adapter._activate(client, "https://license.example", "lic-1", "key")
    adapter._activate(client, "https://license.example", "lic-1", "key")
    with pytest.raises(RuntimeError, match="Machine activation failed"):
        adapter._activate(client, "https://license.example", "lic-1", "key")


def test_fetch_builds_license_and_activates_no_machine(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _Client()
    client_options: list[dict[str, object]] = []
    client.post_responses.extend([
        _Response({"meta": {"valid": False, "code": "NO_MACHINE"}, "data": {"id": "license-1"}}),
        _Response({"data": {"id": "machine"}}, status_code=201),
        _Response(_valid_body()),
    ])
    client.get_responses.append(
        _Response({"data": [{"attributes": {"code": "playbook-a"}}, {"attributes": {"code": ""}}]})
    )
    def _client_factory(**kwargs: object) -> _Client:
        client_options.append(kwargs)
        return client

    monkeypatch.setattr(keygen.httpx2, "Client", _client_factory)
    monkeypatch.setattr(keygen, "_machine_fingerprint", lambda: "fingerprint")

    license_ = _adapter()._fetch("https://license.example", "key")

    assert license_.customer_id == "license-1"
    assert license_.tier is Tier.COMMERCIAL
    assert license_.entitlements == frozenset({"playbook-a"})
    assert client_options == [{}]
    # max_machines population was intentionally dropped in "Update Licensing";
    # limits is now always empty until the tier rework lands.
    assert license_.limits == {}


def test_fetch_rejects_invalid_license_without_activation(monkeypatch: pytest.MonkeyPatch) -> None:
    client = _Client()
    client.post_responses.append(
        _Response({"meta": {"valid": False, "code": "SUSPENDED", "detail": "blocked"}})
    )
    monkeypatch.setattr(keygen.httpx2, "Client", lambda **_: client)

    with pytest.raises(RuntimeError, match="License invalid: code=SUSPENDED"):
        _adapter()._fetch("https://license.example", "key")


def test_constructor_pins_origin_and_uses_signed_lease_only_for_transport_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    online = keygen.License("license-1", Tier.FREE)

    def fetch_online(self: KeygenLicenseAdapter, origin: str, key: str) -> keygen.License:
        calls.extend([origin, key])
        return online

    monkeypatch.setattr(KeygenLicenseAdapter, "_fetch", fetch_online)
    adapter = KeygenLicenseAdapter("secret-key")
    assert adapter._license is online
    assert calls == ["https://license.nlght.ai", "secret-key"]

    monkeypatch.setattr(
        KeygenLicenseAdapter,
        "_fetch",
        lambda *_: (_ for _ in ()).throw(keygen.LicenseServerUnavailable("offline")),
    )
    monkeypatch.setattr(KeygenLicenseAdapter, "_load_lease", lambda *_: online)
    assert KeygenLicenseAdapter("secret-key")._license is online

    monkeypatch.setattr(
        KeygenLicenseAdapter,
        "_fetch",
        lambda *_: (_ for _ in ()).throw(RuntimeError("License invalid: SUSPENDED")),
    )
    with pytest.raises(RuntimeError, match="SUSPENDED"):
        KeygenLicenseAdapter("secret-key")


def test_checkout_requests_pinned_signed_machine_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    client = _Client()
    client.get_responses.append(_Response({}, text="signed-certificate"))
    adapter = _adapter()
    adapter._lease_path = tmp_path / "machine.lic"
    monkeypatch.setattr(keygen, "_machine_fingerprint", lambda: "fingerprint")
    monkeypatch.setattr(keygen, "verify_machine_lease", lambda *_args, **_kwargs: {})

    adapter._checkout_machine_lease(client, "https://license.nlght.ai", "license-key")

    request = client.gets[0]
    assert request["url"].endswith(
        "/v1/accounts/bedd4f90-4c78-49e0-87a1-c2abf26eb364/"
        "machines/fingerprint/actions/check-out"
    )
    assert request["params"] == {
        "ttl": 259200,
        "include": "license",
        "algorithm": "base64+ed25519",
    }
    assert adapter._lease_path.read_text(encoding="utf-8") == "signed-certificate"
