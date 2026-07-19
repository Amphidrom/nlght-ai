# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

from __future__ import annotations

import hashlib
import logging
import os
import platform
import socket
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import httpx2

from nlght.adapters.outbound.licensing.machine_lease import (
    KEYGEN_ACCOUNT_ID,
    KEYGEN_LEASE_ALGORITHM,
    KEYGEN_LEASE_TTL_SECONDS,
    InvalidMachineLease,
    included_license,
    verify_machine_lease,
)
from nlght.core.licensing.license import License, Tier
from nlght.core.licensing.service import LicenseService

_log = logging.getLogger(__name__)
_KEYGEN_ORIGIN = "https://license.nlght.ai"


class LicenseServerUnavailable(RuntimeError):
    """Raised only for technical failures where a signed lease may be used."""


def _parse_expiry(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value).astimezone(UTC)
    except Exception:
        _log.warning("license.keygen_bad_expiry | value=%s", value)
        return None


def _get_mac_addresses() -> dict[str, str]:
    try:
        import ifaddr  # noqa: PLC0415 (lazy import: optional extra or deliberate startup-cost/cycle avoidance)
        result = {}
        for adapter in ifaddr.get_adapters():
            for ip in adapter.ips:
                if isinstance(ip.ip, str) and ":" in ip.ip:
                    result[adapter.name] = ip.ip
        return result
    except ImportError:
        return {}


def _machine_fingerprint() -> str:
    parts = [
        socket.gethostname(),
        platform.system(),
        platform.machine(),
        platform.processor(),
    ]
    for iface, mac in _get_mac_addresses().items():
        parts.append(f"{iface}:{mac}")
    raw = "|".join(p for p in parts if p)
    return hashlib.sha256(raw.encode()).hexdigest()


class KeygenLicenseAdapter(LicenseService):
    def __init__(self, license_key: str, *, lease_path: Path | None = None) -> None:
        configured_path = os.environ.get("NLGHT_LICENSE_LEASE_PATH")
        self._lease_path = lease_path or (
            Path(configured_path) if configured_path else Path.home() / ".nlght" / "machine.lic"
        )
        try:
            license_ = self._fetch(_KEYGEN_ORIGIN, license_key)
        except LicenseServerUnavailable as exc:
            _log.warning("license.keygen_offline | using_signed_machine_lease error=%s", exc)
            license_ = self._load_lease()
        super().__init__(license_)

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/vnd.api+json",
            "Accept": "application/vnd.api+json",
        }

    def _validate(self, client: httpx2.Client, server_url: str, license_key: str) -> dict[str, Any]:
        fingerprint = _machine_fingerprint()
        url = f"{server_url.rstrip('/')}/v1/licenses/actions/validate-key"
        error: Exception | None = None
        for _attempt in range(2):
            try:
                resp = client.post(
                    url,
                    json={"meta": {"key": license_key, "scope": {"fingerprint": fingerprint}}},
                    headers=self._headers(),
                    timeout=5.0,
                )
                return cast(dict[str, Any], resp.json())
            except Exception as exc:
                error = exc
        raise LicenseServerUnavailable(f"License server unreachable: {error}") from error

    def _fetch_single(
        self,
        client: httpx2.Client,
        server_url: str,
        related_path: str,
        license_key: str,
    ) -> dict[str, Any] | None:
        url = f"{server_url.rstrip('/')}{related_path}"
        try:
            resp = client.get(
                url,
                headers={
                    **self._headers(),
                    "Authorization": f"License {license_key}",
                },
                timeout=10.0,
            )
            data = resp.json().get("data")
            return data if isinstance(data, dict) else None
        except Exception as exc:
            _log.warning("license.fetch_single_failed | path=%s error=%s", related_path, exc)
            return None

    def _fetch_related(
        self,
        client: httpx2.Client,
        server_url: str,
        related_path: str,
        license_key: str,
    ) -> list[Any]:
        url = f"{server_url.rstrip('/')}{related_path}?limit=100"
        try:
            resp = client.get(
                url,
                headers={
                    **self._headers(),
                    "Authorization": f"License {license_key}",
                },
                timeout=10.0,
            )
            data = resp.json().get("data", [])
            return data if isinstance(data, list) else []
        except Exception as exc:
            _log.warning("license.fetch_related_failed | path=%s error=%s", related_path, exc)
            return []

    def _activate(
        self,
        client: httpx2.Client,
        server_url: str,
        license_id: str,
        license_key: str,
    ) -> None:
        fingerprint = _machine_fingerprint()
        url = f"{server_url.rstrip('/')}/v1/machines"
        _log.info("license.machine_activate | fingerprint=%s...", fingerprint[:16])
        try:
            resp = client.post(
                url,
                json={
                    "data": {
                        "type": "machines",
                        "attributes": {
                            "fingerprint": fingerprint,
                            "name":        socket.gethostname(),
                            "platform":    platform.system(),
                            "metadata": {
                                "processor": platform.processor(),
                                "machine":   platform.machine(),
                            },
                        },
                        "relationships": {
                            "license": {
                                "data": {"type": "licenses", "id": license_id}
                            }
                        },
                    }
                },
                headers={
                    **self._headers(),
                    "Authorization": f"License {license_key}",
                },
                timeout=10.0,
            )
            body = resp.json()
        except Exception as exc:
            raise LicenseServerUnavailable(f"Machine activation unreachable: {exc}") from exc
        if resp.status_code not in (200, 201):
            errors = body.get("errors", [])
            if any(e.get("code") == "FINGERPRINT_TAKEN" for e in errors):
                _log.info("license.machine_already_activated")
                return
            raise RuntimeError(f"Machine activation failed: {body}")
        _log.info("license.machine_activated")

    def _checkout_machine_lease(
        self,
        client: httpx2.Client,
        server_url: str,
        license_key: str,
    ) -> None:
        fingerprint = _machine_fingerprint()
        url = (
            f"{server_url}/v1/accounts/{KEYGEN_ACCOUNT_ID}/machines/"
            f"{fingerprint}/actions/check-out"
        )
        try:
            response = client.get(
                url,
                params={
                    "ttl": KEYGEN_LEASE_TTL_SECONDS,
                    "include": "license",
                    "algorithm": KEYGEN_LEASE_ALGORITHM,
                },
                headers={
                    "Accept": "application/vnd.api+json",
                    "Authorization": f"License {license_key}",
                },
                timeout=10.0,
            )
            if response.status_code != 200:
                raise RuntimeError(f"checkout returned HTTP {response.status_code}")
            certificate = response.text
            verify_machine_lease(certificate, expected_fingerprint=fingerprint)
            self._lease_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self._lease_path.with_suffix(".tmp")
            temporary.write_text(certificate, encoding="utf-8")
            temporary.chmod(0o600)
            temporary.replace(self._lease_path)
            _log.info("license.machine_lease_refreshed | ttl_seconds=%d", KEYGEN_LEASE_TTL_SECONDS)
        except Exception as exc:
            _log.warning("license.machine_lease_refresh_failed | error=%s", exc)

    def _load_lease(self) -> License:
        try:
            certificate = self._lease_path.read_text(encoding="utf-8")
            document = verify_machine_lease(
                certificate,
                expected_fingerprint=_machine_fingerprint(),
            )
            data = included_license(document)
            attrs = cast(dict[str, Any], data.get("attributes", {}))
            return License(
                customer_id=str(data["id"]),
                tier=Tier.from_str(str((attrs.get("metadata") or {}).get("tier", "free"))),
                entitlements=frozenset(),
                limits={},
            )
        except (OSError, InvalidMachineLease, KeyError, StopIteration, TypeError) as exc:
            raise RuntimeError(f"No valid signed machine lease available: {exc}") from exc

    def _fetch(self, server_url: str, license_key: str) -> License:
        with httpx2.Client() as client:
            body = self._validate(client, server_url, license_key)
            meta = body.get("meta", {})

            if not meta.get("valid", False):
                code = meta.get("code")
                if code in {"NO_MACHINE", "NO_MACHINES"}:
                    license_id = body.get("data", {}).get("id")
                    if not license_id:
                        raise RuntimeError(
                            "License invalid: NO_MACHINE and no license ID in response"
                        )
                    self._activate(client, server_url, license_id, license_key)
                    body = self._validate(client, server_url, license_key)
                    meta = body.get("meta", {})
                    if not meta.get("valid", False):
                        raise RuntimeError(
                            f"License invalid after activation: code={meta.get('code')} detail={meta.get('detail')}"
                        )
                else:
                    raise RuntimeError(
                        f"License invalid: code={code} detail={meta.get('detail')}"
                    )

            data = body.get("data", {})
            rels = data.get("relationships", {})

            # Entitlements via related link
            entitlements_path = rels.get("entitlements", {}).get("links", {}).get("related", "")
            entitlement_items = self._fetch_related(client, server_url, entitlements_path, license_key)
            entitlements = frozenset(
                item["attributes"]["code"]
                for item in entitlement_items
                if item.get("attributes", {}).get("code")
            )

            limits: dict[str, int] = {}
            attrs = data.get("attributes", {})

            tier  = Tier.from_str(
                str((attrs.get("metadata") or {}).get("tier", "free"))
            )

            license_ = License(
                customer_id=data["id"],
                #tier ausbauen
                tier=tier,
                entitlements=entitlements,
                limits=limits,
            )
            self._checkout_machine_lease(client, server_url, license_key)
            return license_
