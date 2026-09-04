# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Turn a pip-audit report into a build decision.

Scanner output establishes which dependency is affected. It cannot establish
whether this product is exposed — that judgement is made by a person, written
down, and given a date by which it must be made again (ADR-0070).

    python scripts/security/audit_gate.py \\
        --audit results/security/pip-audit.json \\
        --exceptions scripts/security/advisory-exceptions.yaml

Exit codes: 0 clean or fully excepted, 1 a finding or a stale exception.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_REQUIRED = ("advisory", "package", "reason", "expires", "owner")


@dataclass(frozen=True)
class Exception_:
    advisory: str
    package: str
    reason: str
    expires: dt.date
    owner: str


@dataclass(frozen=True)
class Finding:
    package: str
    version: str
    advisory: str
    aliases: tuple[str, ...]

    @property
    def ids(self) -> set[str]:
        return {self.advisory, *self.aliases}


def load_findings(report: dict[str, Any]) -> list[Finding]:
    findings: list[Finding] = []
    for dependency in report.get("dependencies") or []:
        for vuln in dependency.get("vulns") or []:
            findings.append(Finding(
                package=str(dependency.get("name", "")),
                version=str(dependency.get("version", "")),
                advisory=str(vuln.get("id", "")),
                aliases=tuple(str(a) for a in (vuln.get("aliases") or [])),
            ))
    return findings


def load_exceptions(raw: dict[str, Any] | None) -> tuple[list[Exception_], list[str]]:
    """Parse the exception list, refusing anything that is not a decision."""
    problems: list[str] = []
    parsed: list[Exception_] = []
    entries = (raw or {}).get("exceptions") or []
    if not isinstance(entries, list):
        return [], ["`exceptions` must be a list"]

    for index, entry in enumerate(entries):
        where = f"exception #{index + 1}"
        if not isinstance(entry, dict):
            problems.append(f"{where}: not a mapping")
            continue
        missing = [field for field in _REQUIRED if not entry.get(field)]
        if missing:
            problems.append(f"{where}: missing {', '.join(missing)}")
            continue
        advisory, package = str(entry["advisory"]), str(entry["package"])
        if "*" in advisory or "*" in package:
            # A wildcard turns "we looked at this finding" into "we stopped
            # looking at this package", which is the thing a gate exists to
            # prevent.
            problems.append(f"{where}: wildcards are not allowed ({advisory} / {package})")
            continue
        expires = entry["expires"]
        if isinstance(expires, dt.datetime):
            expires = expires.date()
        if not isinstance(expires, dt.date):
            problems.append(f"{where}: `expires` must be a date (YYYY-MM-DD)")
            continue
        parsed.append(Exception_(
            advisory=advisory,
            package=package,
            reason=str(entry["reason"]).strip(),
            expires=expires,
            owner=str(entry["owner"]),
        ))
    return parsed, problems


def evaluate(
    findings: list[Finding],
    exceptions: list[Exception_],
    today: dt.date,
) -> tuple[list[str], list[str]]:
    """Return (failures, notes)."""
    failures: list[str] = []
    notes: list[str] = []

    for exception in exceptions:
        if exception.expires <= today:
            failures.append(
                f"expired exception: {exception.advisory} ({exception.package}) "
                f"lapsed {exception.expires}; owner {exception.owner}. "
                "Re-review it or fix the dependency."
            )

    live = [e for e in exceptions if e.expires > today]
    for finding in findings:
        covered = next(
            (e for e in live if e.package == finding.package and e.advisory in finding.ids),
            None,
        )
        if covered is None:
            failures.append(
                f"unreviewed finding: {finding.advisory} in "
                f"{finding.package} {finding.version}"
            )
        else:
            notes.append(
                f"excepted: {finding.advisory} in {finding.package} "
                f"until {covered.expires} ({covered.owner})"
            )

    used = {(e.package, e.advisory) for e in live if any(
        e.package == f.package and e.advisory in f.ids for f in findings
    )}
    for exception in live:
        if (exception.package, exception.advisory) not in used:
            notes.append(
                f"unused exception: {exception.advisory} ({exception.package}) "
                "matches no current finding and can be removed"
            )
    return failures, notes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True, type=Path)
    parser.add_argument("--exceptions", required=True, type=Path)
    parser.add_argument("--today", default=None, help="override for tests")
    args = parser.parse_args(argv)

    today = dt.date.fromisoformat(args.today) if args.today else dt.date.today()
    report = json.loads(args.audit.read_text(encoding="utf-8"))
    raw = yaml.safe_load(args.exceptions.read_text(encoding="utf-8"))

    exceptions, problems = load_exceptions(raw)
    findings = load_findings(report)
    failures, notes = evaluate(findings, exceptions, today)
    failures = problems + failures

    for note in notes:
        print(f"  note: {note}")
    for failure in failures:
        print(f"  FAIL: {failure}", file=sys.stderr)

    if failures:
        print(
            f"\n{len(failures)} unresolved security finding(s). "
            "Fix the dependency, or record a reviewed, time-bounded exception in "
            f"{args.exceptions}.",
            file=sys.stderr,
        )
        return 1
    print(f"dependency gate: {len(findings)} finding(s), all accounted for")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
