# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import base64
import json
from datetime import UTC, datetime, timedelta

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from nlght.adapters.outbound.licensing import machine_lease
from nlght.adapters.outbound.licensing.machine_lease import (
    InvalidMachineLease,
    verify_machine_lease,
)


def _certificate(
    private_key: Ed25519PrivateKey,
    *,
    fingerprint: str = "machine-a",
    ttl: int = machine_lease.KEYGEN_LEASE_TTL_SECONDS,
    expiry_offset_seconds: int | None = None,
    now: datetime | None = None,
) -> str:
    current = now or datetime.now(UTC)
    expiry_offset = ttl if expiry_offset_seconds is None else expiry_offset_seconds
    document = {
        "meta": {
            "issued": current.isoformat(),
            "expiry": (current + timedelta(seconds=expiry_offset)).isoformat(),
            "ttl": ttl,
        },
        "data": {"id": "machine-1", "type": "machines", "attributes": {"fingerprint": fingerprint}},
        "included": [
            {
                "id": "license-1",
                "type": "licenses",
                "attributes": {
                    "suspended": False,
                    "expiry": (current + timedelta(days=30)).isoformat(),
                    "metadata": {"tier": "commercial"},
                },
            }
        ],
    }
    enc = base64.b64encode(json.dumps(document).encode()).decode()
    sig = base64.b64encode(private_key.sign(f"machine/{enc}".encode())).decode()
    outer = base64.b64encode(
        json.dumps({"alg": "base64+ed25519", "enc": enc, "sig": sig}).encode()
    ).decode()
    return f"-----BEGIN MACHINE FILE-----\n{outer}\n-----END MACHINE FILE-----\n"


@pytest.fixture
def signing_key(monkeypatch: pytest.MonkeyPatch) -> Ed25519PrivateKey:
    private_key = Ed25519PrivateKey.generate()
    public = private_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    monkeypatch.setattr(machine_lease, "KEYGEN_ED25519_PUBLIC_KEY", public.hex())
    return private_key


def test_verifies_keygen_signed_machine_lease(signing_key: Ed25519PrivateKey) -> None:
    document = verify_machine_lease(_certificate(signing_key), expected_fingerprint="machine-a")
    assert document["data"]["id"] == "machine-1"


def test_rejects_tampering_and_wrong_machine(signing_key: Ed25519PrivateKey) -> None:
    certificate = _certificate(signing_key)
    with pytest.raises(InvalidMachineLease, match="signature"):
        verify_machine_lease(
            _certificate(Ed25519PrivateKey.generate()),
            expected_fingerprint="machine-a",
        )
    with pytest.raises(InvalidMachineLease, match="another machine"):
        verify_machine_lease(certificate, expected_fingerprint="machine-b")


def test_rejects_expired_or_overlong_lease(signing_key: Ed25519PrivateKey) -> None:
    past = datetime.now(UTC) - timedelta(days=4)
    with pytest.raises(InvalidMachineLease, match="has expired"):
        verify_machine_lease(_certificate(signing_key, now=past), expected_fingerprint="machine-a")
    with pytest.raises(InvalidMachineLease, match="TTL"):
        verify_machine_lease(
            _certificate(signing_key, ttl=machine_lease.KEYGEN_LEASE_TTL_SECONDS + 1),
            expected_fingerprint="machine-a",
        )
    with pytest.raises(InvalidMachineLease, match="signed duration") as exc_info:
        verify_machine_lease(
            _certificate(
                signing_key,
                expiry_offset_seconds=machine_lease.KEYGEN_LEASE_TTL_SECONDS + 1,
            ),
            expected_fingerprint="machine-a",
        )
    assert f"max_seconds={machine_lease.KEYGEN_LEASE_TTL_SECONDS}" in str(exc_info.value)


def test_accepts_issued_timestamp_within_clock_skew(signing_key: Ed25519PrivateKey) -> None:
    current = datetime.now(UTC)
    issued = current + machine_lease.KEYGEN_LEASE_CLOCK_SKEW

    document = verify_machine_lease(
        _certificate(signing_key, now=issued),
        expected_fingerprint="machine-a",
        now=current,
    )

    assert document["meta"]["issued"] == issued.isoformat()


def test_rejects_issued_timestamp_beyond_clock_skew_with_details(signing_key: Ed25519PrivateKey) -> None:
    current = datetime.now(UTC)
    issued = current + machine_lease.KEYGEN_LEASE_CLOCK_SKEW + timedelta(microseconds=1)

    with pytest.raises(InvalidMachineLease) as exc_info:
        verify_machine_lease(
            _certificate(signing_key, now=issued),
            expected_fingerprint="machine-a",
            now=current,
        )

    message = str(exc_info.value)
    assert "issued timestamp is too far in the future" in message
    assert f"issued={issued.isoformat()}" in message
    assert f"current={current.isoformat()}" in message
    assert "max_clock_skew_seconds=300" in message
