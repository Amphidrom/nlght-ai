# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Tests for scripts/release/release_badges.py.

Verifies:
  - summarize_junit maps success/skipped/total junit results to badge value/color
    (green/yellow/red, both <testsuite> root and <testsuites> wrapper)
  - inject replaces the README marker block with immutable public release
    badge URLs and refuses READMEs without markers
  - generate writes an SVG
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[3] / "scripts" / "release"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


release_badges = _load("release_badges")


def _junit(tmp_path: Path, body: str, name: str = "junit.xml") -> Path:
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# summarize_junit
# ---------------------------------------------------------------------------


def test_summarize_all_passed_is_green(tmp_path) -> None:
    junit = _junit(tmp_path, '<testsuite tests="10" failures="0" errors="0" skipped="0"/>')
    assert release_badges.summarize_junit([junit]) == ("10/0/10", "green")


def test_summarize_skips_are_yellow(tmp_path) -> None:
    junit = _junit(tmp_path, '<testsuite tests="10" failures="0" errors="0" skipped="2"/>')
    assert release_badges.summarize_junit([junit]) == ("8/2/10", "yellow")


def test_summarize_failures_are_red(tmp_path) -> None:
    junit = _junit(tmp_path, '<testsuite tests="10" failures="1" errors="1" skipped="0"/>')
    assert release_badges.summarize_junit([junit]) == ("8/0/10", "red")


def test_summarize_sums_across_testsuites_wrapper(tmp_path) -> None:
    junit = _junit(
        tmp_path,
        '<testsuites><testsuite tests="4" failures="0" errors="0" skipped="0"/><testsuite tests="6" failures="0" errors="0" skipped="0"/></testsuites>',
    )
    assert release_badges.summarize_junit([junit]) == ("10/0/10", "green")


def test_summarize_sums_across_separate_junit_files(tmp_path) -> None:
    unit = _junit(tmp_path, '<testsuite tests="4" failures="0" errors="0" skipped="0"/>', name="unit.xml")
    integration = _junit(tmp_path, '<testsuite tests="6" failures="0" errors="0" skipped="1"/>', name="integration.xml")
    assert release_badges.summarize_junit([unit, integration]) == ("9/1/10", "yellow")


# ---------------------------------------------------------------------------
# inject
# ---------------------------------------------------------------------------


def test_inject_replaces_marker_block(tmp_path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text(
        "# Title\n\n<!-- release-badges:start -->\n<!-- release-badges:end -->\n\nBody\n",
        encoding="utf-8",
    )

    release_badges.inject(readme, public_ref="v0.2.0")

    text = readme.read_text(encoding="utf-8")
    assert "raw.githubusercontent.com/Amphidrom/nlght-ai/v0.2.0/.github/badges/tests.svg" in text
    assert "raw.githubusercontent.com/Amphidrom/nlght-ai/v0.2.0/.github/badges/coverage.svg" in text
    assert "raw.githubusercontent.com/Amphidrom/nlght-ai/v0.2.0/.github/badges/pypi.svg" in text
    assert "/main/.github/badges/" not in text
    assert "](.github/badges/" not in text
    assert "img.shields.io" not in text
    assert text.startswith("# Title\n\n<!-- release-badges:start -->\n[![Tests]")
    assert text.endswith("<!-- release-badges:end -->\n\nBody\n")


def test_inject_is_idempotent(tmp_path) -> None:
    """Release injection can safely run in the build checkout and export."""
    readme = tmp_path / "README.md"
    readme.write_text(
        "# Title\n\n<!-- release-badges:start -->\n<!-- release-badges:end -->\n\nBody\n",
        encoding="utf-8",
    )

    release_badges.inject(readme, public_ref="v0.2.0-RC1")
    release_badges.inject(readme, public_ref="v0.2.0-RC1")

    text = readme.read_text(encoding="utf-8")
    assert text.count(release_badges.START_MARKER) == 1
    assert text.count(release_badges.END_MARKER) == 1
    assert "raw.githubusercontent.com/Amphidrom/nlght-ai/v0.2.0-RC1/.github/badges/tests.svg" in text


@pytest.mark.parametrize("public_ref", ["", "main", "feature/badges", "v0.2.0?x"])
def test_public_badge_markdown_rejects_mutable_or_unsafe_refs(public_ref) -> None:
    with pytest.raises(ValueError, match="immutable public ref"):
        release_badges.public_badge_markdown(public_ref)


def test_inject_fails_without_markers(tmp_path) -> None:
    readme = tmp_path / "README.md"
    readme.write_text("# Title\n\nBody\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="marker"):
        release_badges.inject(readme, public_ref="v0.2.0")


def test_repo_readme_carries_marker_block() -> None:
    readme = Path(__file__).resolve().parents[3] / "README.md"
    text = readme.read_text(encoding="utf-8")
    assert release_badges.START_MARKER in text
    assert release_badges.END_MARKER in text
    assert "https://raw.githubusercontent.com/Amphidrom/nlght-ai/" not in text
    for _, filename in release_badges.BADGES:
        assert f"](.github/badges/{filename})" in text


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


def test_generate_writes_svg(tmp_path) -> None:
    junit = _junit(tmp_path, '<testsuite tests="3" failures="0" errors="0" skipped="0"/>')
    out = tmp_path / "badges" / "tests.svg"

    release_badges.generate([junit], out, label="tests", value_prefix="unit")

    assert out.exists()
    assert "<svg" in out.read_text(encoding="utf-8")
    assert "tests" in out.read_text(encoding="utf-8")
    assert "unit 3/0/3" in out.read_text(encoding="utf-8")


def test_generate_sums_across_multiple_junit_files(tmp_path) -> None:
    unit = _junit(tmp_path, '<testsuite tests="3" failures="0" errors="0" skipped="0"/>', name="unit.xml")
    integration = _junit(tmp_path, '<testsuite tests="2" failures="0" errors="0" skipped="0"/>', name="integration.xml")
    out = tmp_path / "badges" / "tests.svg"

    release_badges.generate([unit, integration], out, label="tests", value_prefix="combined")

    assert out.exists()
    assert "5/0/5" in out.read_text(encoding="utf-8")


def test_generate_test_family_writes_one_connected_svg(tmp_path) -> None:
    reports = {
        "unit": _junit(tmp_path, '<testsuite tests="3" skipped="0"/>', "unit.xml"),
        "arch": _junit(tmp_path, '<testsuite tests="2" skipped="0"/>', "arch.xml"),
        "integration": _junit(tmp_path, '<testsuite tests="4" skipped="1"/>', "integration.xml"),
        "e2e": _junit(tmp_path, '<testsuite tests="5" failures="1"/>', "e2e.xml"),
    }
    out = tmp_path / "tests.svg"
    release_badges.generate_test_family(tuple(reports.items()), out)
    svg = out.read_text(encoding="utf-8")
    assert svg.count("<svg") == 1
    assert svg.count('rx="3"') == 1
    for expected in (
        "tests (success/skip/total)",
        "unit 3/0/3",
        "arch 2/0/2",
        "integration 3/1/4",
        "e2e 4/0/5",
    ):
        assert expected in svg
    for color in release_badges.STATUS_COLORS.values():
        assert color in svg


# ---------------------------------------------------------------------------
# summarize_coverage / generate_coverage
# ---------------------------------------------------------------------------


def _coverage_xml(tmp_path: Path, line_rate: str) -> Path:
    path = tmp_path / "coverage.xml"
    path.write_text(f'<coverage line-rate="{line_rate}"></coverage>', encoding="utf-8")
    return path


def test_summarize_coverage_high_is_green(tmp_path) -> None:
    xml = _coverage_xml(tmp_path, "0.92")
    assert release_badges.summarize_coverage(xml) == ("92%", "green")


def test_summarize_coverage_mid_is_yellow(tmp_path) -> None:
    xml = _coverage_xml(tmp_path, "0.6")
    assert release_badges.summarize_coverage(xml) == ("60%", "yellow")


def test_summarize_coverage_low_is_red(tmp_path) -> None:
    xml = _coverage_xml(tmp_path, "0.3")
    assert release_badges.summarize_coverage(xml) == ("30%", "red")


def test_generate_coverage_writes_svg(tmp_path) -> None:
    xml = _coverage_xml(tmp_path, "0.73")
    out = tmp_path / "badges" / "coverage.svg"

    release_badges.generate_coverage(xml, out)

    assert out.exists()
    assert "73%" in out.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# generate_pypi
# ---------------------------------------------------------------------------


def test_generate_pypi_writes_svg(tmp_path) -> None:
    out = tmp_path / "badges" / "pypi.svg"

    release_badges.generate_pypi("0.2.0", out)

    assert out.exists()
    svg = out.read_text(encoding="utf-8")
    assert "<svg" in svg
    assert "pypi" in svg
    assert "v0.2.0" in svg
