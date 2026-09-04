# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""The dependency gate, and the three rules that keep an exception honest.

An exception names one advisory, says why this product is not exposed, and
carries a date. Each of those is load-bearing: a wildcard would silence a
package rather than a finding, a missing reason would record nothing, and a
missing expiry would make a postponed decision a permanent one.
"""

from __future__ import annotations

import datetime as dt
import importlib.util
import json
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[3] / "scripts" / "security"


def _load(name: str):  # noqa: ANN202
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


audit_gate = _load("audit_gate")

TODAY = dt.date(2026, 9, 4)
FINDING = {
    "dependencies": [
        {
            "name": "cryptography",
            "version": "49.0.0",
            "vulns": [{"id": "PYSEC-2026-3552", "aliases": ["CVE-2026-69247"]}],
        }
    ]
}
CLEAN = {"dependencies": [{"name": "cryptography", "version": "50.0.1", "vulns": []}]}


def _exception(**overrides):  # noqa: ANN202
    entry = {
        "advisory": "PYSEC-2026-3552",
        "package": "cryptography",
        "reason": "The affected PKCS#7 API is not reachable in this product.",
        "expires": dt.date(2026, 10, 1),
        "owner": "security",
    }
    entry.update(overrides)
    return {"exceptions": [entry]}


def _run(report, raw, today=TODAY):  # noqa: ANN001, ANN202
    exceptions, problems = audit_gate.load_exceptions(raw)
    failures, notes = audit_gate.evaluate(
        audit_gate.load_findings(report), exceptions, today,
    )
    return problems + failures, notes


def test_a_clean_report_passes() -> None:
    failures, _ = _run(CLEAN, {"exceptions": []})
    assert failures == []


def test_a_finding_without_an_exception_fails() -> None:
    failures, _ = _run(FINDING, {"exceptions": []})
    assert len(failures) == 1
    assert "unreviewed finding" in failures[0]


def test_a_reviewed_unexpired_exception_lets_the_build_through() -> None:
    failures, notes = _run(FINDING, _exception())
    assert failures == []
    assert any("excepted" in note for note in notes)


def test_an_exception_may_name_the_alias_instead_of_the_id() -> None:
    # Scanners disagree about which identifier is primary; a reviewer should not
    # have to guess which one the report will use.
    failures, _ = _run(FINDING, _exception(advisory="CVE-2026-69247"))
    assert failures == []


def test_an_expired_exception_turns_the_gate_red_again() -> None:
    failures, _ = _run(FINDING, _exception(), today=dt.date(2026, 10, 2))
    assert any("expired exception" in f for f in failures)
    # And the finding it used to cover is unreviewed again, not silently allowed.
    assert any("unreviewed finding" in f for f in failures)


def test_an_exception_for_another_package_does_not_cover_this_finding() -> None:
    failures, _ = _run(FINDING, _exception(package="h2"))
    assert any("unreviewed finding" in f for f in failures)


def test_wildcards_are_refused() -> None:
    for entry in ({"advisory": "*"}, {"package": "*"}, {"package": "crypt*"}):
        failures, _ = _run(FINDING, _exception(**entry))
        assert any("wildcard" in f for f in failures), entry


def test_every_field_is_required() -> None:
    for field in ("advisory", "package", "reason", "expires", "owner"):
        failures, _ = _run(FINDING, _exception(**{field: ""}))
        assert any("missing" in f for f in failures), field


def test_an_expiry_that_is_not_a_date_is_refused() -> None:
    failures, _ = _run(FINDING, _exception(expires="soon"))
    assert any("must be a date" in f for f in failures)


def test_an_exception_matching_nothing_is_reported_but_does_not_fail() -> None:
    # Dead configuration is worth saying out loud; it is not a security failure.
    failures, notes = _run(CLEAN, _exception())
    assert failures == []
    assert any("unused exception" in note for note in notes)


def test_the_shipped_exception_list_is_empty_and_parses() -> None:
    import yaml

    raw = yaml.safe_load((_SCRIPTS / "advisory-exceptions.yaml").read_text(encoding="utf-8"))
    exceptions, problems = audit_gate.load_exceptions(raw)

    assert problems == []
    assert exceptions == [], "the baseline is zero; an entry here needs a reviewer, not a default"


def test_main_exits_non_zero_on_an_unreviewed_finding(tmp_path: Path) -> None:
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps(FINDING), encoding="utf-8")
    empty = tmp_path / "exceptions.yaml"
    empty.write_text("exceptions: []\n", encoding="utf-8")

    code = audit_gate.main([
        "--audit", str(audit), "--exceptions", str(empty), "--today", TODAY.isoformat(),
    ])

    assert code == 1
