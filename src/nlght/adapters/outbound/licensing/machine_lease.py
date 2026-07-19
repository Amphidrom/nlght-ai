# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import base64
import json
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

KEYGEN_ACCOUNT_ID = "bedd4f90-4c78-49e0-87a1-c2abf26eb364"
KEYGEN_ED25519_PUBLIC_KEY = "1cf32413680790602a3467a8c48574515ae3868e764691fce07ee677328c774d"
KEYGEN_LEASE_ALGORITHM = "base64+ed25519"
KEYGEN_LEASE_TTL_SECONDS = 72 * 60 * 60
KEYGEN_LEASE_CLOCK_SKEW = timedelta(minutes=5)

_BEGIN = "-----BEGIN MACHINE FILE-----"
_END = "-----END MACHINE FILE-----"


class InvalidMachineLease(RuntimeError):
    """Raised when a cached machine file cannot be trusted."""


def _timestamp(value: object, field: str) -> datetime:
    if not isinstance(value, str):
        raise InvalidMachineLease(f"Machine lease has no valid {field}")
    try:
        normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            raise ValueError("timezone required")
        return parsed.astimezone(UTC)
    except ValueError as exc:
        raise InvalidMachineLease(f"Machine lease has no valid {field}") from exc


def _decode_certificate(certificate: str) -> dict[str, Any]:
    value = certificate.strip()
    if not value.startswith(_BEGIN) or not value.endswith(_END):
        raise InvalidMachineLease("Machine lease certificate framing is invalid")
    encoded = value.removeprefix(_BEGIN).removesuffix(_END).strip().replace("\n", "")
    try:
        outer = json.loads(base64.b64decode(encoded, validate=True))
    except (ValueError, json.JSONDecodeError) as exc:
        raise InvalidMachineLease("Machine lease certificate is malformed") from exc
    if not isinstance(outer, dict):
        raise InvalidMachineLease("Machine lease certificate is not an object")
    return cast(dict[str, Any], outer)


def verify_machine_lease(
    certificate: str,
    *,
    expected_fingerprint: str,
    now: datetime | None = None,
) -> dict[str, Any]:
    outer = _decode_certificate(certificate)
    if outer.get("alg") != KEYGEN_LEASE_ALGORITHM:
        raise InvalidMachineLease("Machine lease algorithm is not trusted")
    enc = outer.get("enc")
    sig = outer.get("sig")
    if not isinstance(enc, str) or not isinstance(sig, str):
        raise InvalidMachineLease("Machine lease signature fields are missing")
    try:
        signature = base64.b64decode(sig, validate=True)
        public_key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(KEYGEN_ED25519_PUBLIC_KEY))
        public_key.verify(signature, f"machine/{enc}".encode())
    except (ValueError, InvalidSignature) as exc:
        raise InvalidMachineLease("Machine lease signature is invalid") from exc
    try:
        document = json.loads(base64.b64decode(enc, validate=True))
    except (ValueError, json.JSONDecodeError) as exc:
        raise InvalidMachineLease("Machine lease payload is malformed") from exc
    if not isinstance(document, dict):
        raise InvalidMachineLease("Machine lease payload is not an object")
    payload = cast(dict[str, Any], document)
    meta = payload.get("meta")
    if not isinstance(meta, dict):
        raise InvalidMachineLease("Machine lease metadata is missing")
    ttl = meta.get("ttl")
    if not isinstance(ttl, int) or not 0 < ttl <= KEYGEN_LEASE_TTL_SECONDS:
        raise InvalidMachineLease("Machine lease TTL exceeds the trusted maximum")
    issued = _timestamp(meta.get("issued"), "issued timestamp")
    expiry = _timestamp(meta.get("expiry"), "expiry timestamp")
    current = (now or datetime.now(UTC)).astimezone(UTC)
    if issued > current + KEYGEN_LEASE_CLOCK_SKEW:
        raise InvalidMachineLease(
            "Machine lease issued timestamp is too far in the future: "
            f"issued={issued.isoformat()} current={current.isoformat()} "
            f"max_clock_skew_seconds={int(KEYGEN_LEASE_CLOCK_SKEW.total_seconds())}"
        )
    if expiry <= current:
        raise InvalidMachineLease(
            "Machine lease has expired: "
            f"expiry={expiry.isoformat()} current={current.isoformat()}"
        )
    signed_duration = (expiry - issued).total_seconds()
    if signed_duration > KEYGEN_LEASE_TTL_SECONDS:
        raise InvalidMachineLease(
            "Machine lease signed duration exceeds the trusted maximum: "
            f"duration_seconds={signed_duration:g} max_seconds={KEYGEN_LEASE_TTL_SECONDS}"
        )
    data = payload.get("data")
    if not isinstance(data, dict) or data.get("type") != "machines":
        raise InvalidMachineLease("Machine lease does not contain a machine")
    attrs = data.get("attributes")
    if not isinstance(attrs, dict) or attrs.get("fingerprint") != expected_fingerprint:
        raise InvalidMachineLease("Machine lease belongs to another machine")
    included = payload.get("included")
    if not isinstance(included, list):
        raise InvalidMachineLease("Machine lease has no signed license snapshot")
    license_data = next(
        (item for item in included if isinstance(item, dict) and item.get("type") == "licenses"),
        None,
    )
    if not isinstance(license_data, dict):
        raise InvalidMachineLease("Machine lease has no signed license snapshot")
    license_attrs = license_data.get("attributes")
    if not isinstance(license_attrs, dict) or license_attrs.get("suspended") is True:
        raise InvalidMachineLease("Machine lease contains an unusable license")
    license_expiry = license_attrs.get("expiry")
    if license_expiry is not None and _timestamp(license_expiry, "license expiry") <= current:
        raise InvalidMachineLease("Machine lease contains an expired license")
    return payload


def included_license(document: dict[str, Any]) -> dict[str, Any]:
    included = cast(list[object], document["included"])
    return cast(dict[str, Any], next(item for item in included if isinstance(item, dict) and item.get("type") == "licenses"))
