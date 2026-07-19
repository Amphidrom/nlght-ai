# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Tests for the release helper scripts under scripts/release/.

Verifies:
  - set_version.normalize accepts vX.Y.Z / X.Y.Z / RC variants and rejects junk
  - set_version.apply patches exactly the [project] version line
  - set_version.check passes/fails against the committed version
  - rotate_change_notes.rotate moves the unreleased section into <version>.md,
    resets main.md, and refuses empty sections and overwrites
  - prepare_release.main combines version bump, note rotation, and pypi badge
    regeneration into one pre-tag step
"""
from __future__ import annotations

import datetime
import importlib.util
import sys
import tomllib
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parents[3] / "scripts" / "release"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


set_version = _load("set_version")
rotate_change_notes = _load("rotate_change_notes")
release_badges = _load("release_badges")
prepare_release = _load("prepare_release")


# ---------------------------------------------------------------------------
# set_version.normalize
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("tag", "expected"),
    [
        ("v0.2.0", "0.2.0"),
        ("0.2.0", "0.2.0"),
        ("v1.10.3", "1.10.3"),
        ("v0.2.0-RC1", "0.2.0rc1"),
        ("v0.2.0-rc2", "0.2.0rc2"),
        ("v0.2.0rc3", "0.2.0rc3"),
        ("v0.2.0.RC10", "0.2.0rc10"),
    ],
)
def test_normalize_accepts_release_tags(tag: str, expected: str) -> None:
    assert set_version.normalize(tag) == expected


@pytest.mark.parametrize("tag", ["main", "v1.2", "v1.2.3.4", "v1.2.3-beta1", "release-1.2.3", ""])
def test_normalize_rejects_non_release_tags(tag: str) -> None:
    with pytest.raises(ValueError):
        set_version.normalize(tag)


# ---------------------------------------------------------------------------
# set_version.apply
# ---------------------------------------------------------------------------


def test_apply_patches_version_line(tmp_path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[project]\nname = "nlght-ai"\nversion = "0.1.0"\nrequires-python = ">=3.11"\n',
        encoding="utf-8",
    )

    set_version.apply("0.2.0rc1", pyproject)

    text = pyproject.read_text(encoding="utf-8")
    assert 'version = "0.2.0rc1"' in text
    assert 'version = "0.1.0"' not in text
    assert 'name = "nlght-ai"' in text


def test_apply_fails_without_version_line(tmp_path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "nlght-ai"\n', encoding="utf-8")

    with pytest.raises(ValueError, match="no 'version"):
        set_version.apply("0.2.0", pyproject)


# ---------------------------------------------------------------------------
# set_version.check
# ---------------------------------------------------------------------------


def test_check_passes_when_versions_match(tmp_path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "nlght-ai"\nversion = "0.2.0"\n', encoding="utf-8")

    set_version.check("0.2.0", pyproject)  # no raise


def test_check_fails_when_versions_differ(tmp_path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "nlght-ai"\nversion = "0.1.0"\n', encoding="utf-8")

    with pytest.raises(ValueError, match="tag resolves to version"):
        set_version.check("0.2.0", pyproject)


# ---------------------------------------------------------------------------
# rotate_change_notes.rotate
# ---------------------------------------------------------------------------


def _notes_dir(tmp_path, body: str) -> Path:
    notes = tmp_path / "change-notes"
    notes.mkdir()
    (notes / "main.md").write_text(
        f"# Change notes\n\n## main (unreleased)\n{body}", encoding="utf-8"
    )
    return notes


def test_rotate_moves_entries_and_resets_main(tmp_path) -> None:
    notes = _notes_dir(tmp_path, "\n- area: did a thing. (#1)\n- other: fixed a bug.\n")

    target = rotate_change_notes.rotate("0.2.0", notes, today=datetime.date(2026, 7, 17))

    assert target == notes / "0.2.0.md"
    released = target.read_text(encoding="utf-8")
    assert released.startswith("# Change notes — v0.2.0\n\nReleased: 2026-07-17\n")
    assert "- area: did a thing. (#1)" in released
    assert "- other: fixed a bug." in released

    main_text = (notes / "main.md").read_text(encoding="utf-8")
    assert main_text == "# Change notes\n\n## main (unreleased)\n"


def test_rotate_refuses_empty_section(tmp_path) -> None:
    notes = _notes_dir(tmp_path, "\n\n")

    with pytest.raises(ValueError, match="empty"):
        rotate_change_notes.rotate("0.2.0", notes)


def test_rotate_allow_empty_overrides(tmp_path) -> None:
    notes = _notes_dir(tmp_path, "\n\n")

    target = rotate_change_notes.rotate("0.2.0", notes, allow_empty=True)

    assert target.exists()


def test_rotate_refuses_overwrite(tmp_path) -> None:
    notes = _notes_dir(tmp_path, "\n- entry\n")
    (notes / "0.2.0.md").write_text("existing", encoding="utf-8")

    with pytest.raises(ValueError, match="already exists"):
        rotate_change_notes.rotate("0.2.0", notes)


def test_rotate_fails_without_main_section(tmp_path) -> None:
    notes = tmp_path / "change-notes"
    notes.mkdir()
    (notes / "main.md").write_text("# Change notes\n\n## 0.1.0\n- old\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no '## main"):
        rotate_change_notes.rotate("0.2.0", notes)


# ---------------------------------------------------------------------------
# prepare_release.main
# ---------------------------------------------------------------------------


def test_prepare_release_bumps_version_and_rotates_notes(tmp_path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "nlght-ai"\nversion = "0.1.0"\n', encoding="utf-8")
    notes = _notes_dir(tmp_path, "\n- area: did a thing.\n")
    badge = tmp_path / "badges" / "pypi.svg"

    rc = prepare_release.main([
        "v0.2.0", "--pyproject", str(pyproject), "--notes-dir", str(notes), "--badge", str(badge),
    ])

    assert rc == 0
    assert 'version = "0.2.0"' in pyproject.read_text(encoding="utf-8")
    assert (notes / "0.2.0.md").exists()
    assert "- area: did a thing." in (notes / "0.2.0.md").read_text(encoding="utf-8")
    assert set_version.committed_version(pyproject) == "0.2.0"
    assert badge.exists()
    assert "v0.2.0" in badge.read_text(encoding="utf-8")


def test_prepare_release_fails_on_bad_tag(tmp_path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nversion = "0.1.0"\n', encoding="utf-8")
    notes = _notes_dir(tmp_path, "\n- area: did a thing.\n")
    badge = tmp_path / "badges" / "pypi.svg"

    rc = prepare_release.main([
        "not-a-tag", "--pyproject", str(pyproject), "--notes-dir", str(notes), "--badge", str(badge),
    ])

    assert rc == 2
    assert 'version = "0.1.0"' in pyproject.read_text(encoding="utf-8")
    assert not badge.exists()


# ---------------------------------------------------------------------------
# Repository release contract
# ---------------------------------------------------------------------------


def test_project_metadata_targets_public_release_surfaces() -> None:
    root = Path(__file__).resolve().parents[3]
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    project = data["project"]

    assert project["description"]
    assert project["readme"] == "README.md"
    assert project["urls"]["Issues"].endswith("/issues")
    assert project["urls"]["Changelog"].endswith("/releases")
    assert "Typing :: Typed" in project["classifiers"]


def test_release_workflow_creates_public_release_after_mirror() -> None:
    root = Path(__file__).resolve().parents[3]
    workflow = (root / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

    mirror = workflow.index("- name: Mirror public snapshot")
    public_release = workflow.index("- name: Create public GitHub release")
    assert public_release > mirror
    assert 'NOTES="docs/change-notes/${VERSION}.md"' in workflow
    assert 'gh release create "$GITHUB_REF_NAME"' in workflow
    assert "--repo Amphidrom/nlght-ai" in workflow
    assert "--verify-tag" in workflow
    assert "--prerelease" in workflow
