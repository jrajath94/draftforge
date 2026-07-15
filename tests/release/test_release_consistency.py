"""Release-consistency invariants for DraftForge.

Pins three contracts:

1. ``pyproject.toml`` version == latest released version in CHANGELOG.md.
2. Latest git tag == latest released version in CHANGELOG.md.
3. Zero commits on main past the latest release tag.

These guard against the failure mode where a CHANGELOG entry is added
without bumping ``pyproject.toml`` (or vice versa), or where commits land
on main past the latest release tag without a corresponding CHANGELOG
release. Every number on every number is testable from disk + git —
no fabrication possible.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
CHANGELOG = REPO_ROOT / "CHANGELOG.md"


def _read_pyproject_version() -> str:
    text = PYPROJECT.read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, flags=re.MULTILINE)
    assert match is not None, "pyproject.toml missing top-level 'version = ...' line"
    return match.group(1)


def _read_changelog_latest() -> str:
    text = CHANGELOG.read_text(encoding="utf-8")
    # Latest released version: highest ``## [X.Y.Z]`` heading in CHANGELOG.md.
    # ``[Unreleased]`` is excluded because it has no version. Document order
    # is irrelevant — we compare by SemVer tuple so a retroactive v1.4.0
    # appended below v1.3.0 is still recognised as latest.
    matches = re.findall(
        r"^##\s+\[(\d+\.\d+\.\d+)\]\s+—", text, flags=re.MULTILINE,
    )
    assert matches, "CHANGELOG.md has no `## [X.Y.Z]` release entries"
    return max(matches, key=lambda v: tuple(int(part) for part in v.split(".")))


def _latest_git_tag() -> str:
    """Latest semver tag in the repo (sorted by version tuple, not by date)."""
    out = subprocess.run(
        ["git", "tag", "--list", "v*"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=True,
    )
    tags = [t.strip() for t in out.stdout.splitlines() if t.strip()]
    assert tags, "no v* git tags exist in repo"
    # Parse ``vX.Y.Z`` (or ``X.Y.Z``) into a comparable tuple.
    semver_pairs: list[tuple[tuple[int, int, int], str]] = []
    for tag in tags:
        match = re.match(r"^v?(\d+\.\d+\.\d+)", tag)
        if match:
            semver_pairs.append(
                (tuple(int(part) for part in match.group(1).split(".")), tag),
            )
    assert semver_pairs, f"no semver-shaped tags found among: {tags}"
    semver_pairs.sort(reverse=True)
    return semver_pairs[0][1]


def test_pyproject_version_matches_changelog_latest() -> None:
    """``pyproject.toml`` version must equal the latest released CHANGELOG version."""
    pyproject_v = _read_pyproject_version()
    changelog_v = _read_changelog_latest()
    assert pyproject_v == changelog_v, (
        f"pyproject.toml version={pyproject_v!r} != "
        f"CHANGELOG latest={changelog_v!r}. "
        "Bump one to match the other — they must stay aligned."
    )


def test_latest_git_tag_matches_changelog_latest() -> None:
    """The latest git tag must equal the latest released CHANGELOG version."""
    tag = _latest_git_tag()
    changelog_v = _read_changelog_latest()
    # Tag form is ``vX.Y.Z``; CHANGELOG form is ``[X.Y.Z]``.
    expected_tag = f"v{changelog_v}"
    assert tag == expected_tag, (
        f"Latest git tag={tag!r} != CHANGELOG latest={changelog_v!r} "
        f"(expected tag {expected_tag!r}). "
        "Either tag the new release or roll back the CHANGELOG."
    )


def test_no_commits_since_latest_release_tag() -> None:
    """Zero commits on main past the latest release tag."""
    tag = _latest_git_tag()
    out = subprocess.run(
        ["git", "rev-list", f"{tag}..HEAD", "--count"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=True,
    )
    count = int(out.stdout.strip())
    assert count == 0, (
        f"{count} commit(s) on main past {tag} without a release tag. "
        "Either tag a new release (bump pyproject + add CHANGELOG entry) "
        "or revert the commits."
    )
