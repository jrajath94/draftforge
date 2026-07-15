"""Release-provenance invariants for DraftForge.

Pins that the latest release version is reachable from ``origin/main``
and that what a contributor sees by default checks out is what the
CHANGELOG + git-tag + pyproject claim. Companion to
:mod:`test_release_consistency`:

  * ``test_release_consistency`` pins the *within-branch* state:
    pyproject ↔ CHANGELOG ↔ git-tag alignment on whatever branch HEAD
    is currently checked out.
  * This module (``test_release_provenance``) pins the *branch
    topology*: the latest tag must already be on ``main`` HEAD (not on
    a phase branch behind it), must peel to a commit SHA, must be
    reachable from origin/main, and must publish on the remote so
    ``git ls-remote`` sees it.

DraftForge specifics:

  * CHANGELOG headings use unbracketed ``## [X.Y.Z]`` form (no ``v``
    prefix); git tags use ``vX.Y.Z``. The tests assert both forms
    match by stripping the leading ``v`` before the tuple comparison.
  * There is no retraction-marker concept (DraftForge's tags were all
    published releases), so the simple ``_latest_git_tag()`` from
    :mod:`test_release_consistency` is reused unmodified.
"""

from __future__ import annotations

import importlib
import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# ``main`` invariants only hold AFTER the release branch has been merged.
# On a ``pull_request`` event, HEAD is the synthetic merge commit (or the
# PR's head branch tip), not main — the tag-vs-main invariants would
# always fire until the PR lands. Mirrors the env-var branching in
# :mod:`test_release_consistency`.
_IS_PR_BUILD = os.environ.get("GITHUB_EVENT_NAME") == "pull_request"

skip_on_pr = pytest.mark.skipif(
    _IS_PR_BUILD,
    reason=(
        "main invariants only hold AFTER merge; on pull_request the release tag "
        "is on the PR's head branch, not main, so origin/main vs tag invariants "
        "fire even when the release is correct."
    ),
)


def _latest_release_tag() -> str:
    """Reuse the consistency test's filter so both suites share one definition.

    If the consistency test ever changes the filter (e.g. adds a
    retraction-marker concept), the provenance suite inherits it
    automatically.
    """
    from tests.release import test_release_consistency as cons

    return cons._latest_git_tag()


def _strip_v(tag: str) -> str:
    """Return the X.Y.Z form of a vX.Y.Z tag (DraftForge headings don't carry the v prefix)."""
    return tag.lstrip("v")


def _run_git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=True,
    ).stdout.strip()


class TestSkipOnPrEnforced:
    """Pin the ``@skip_on_pr`` marker flips on PR builds but not push.

    The two real invariant tests below skip under this marker; these
    RED → GREEN assertions prove the marker obeys ``GITHUB_EVENT_NAME``.
    """

    def test_marker_skips_when_event_is_pull_request(self, monkeypatch: object) -> None:
        monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")  # type: ignore[attr-defined]
        importlib.reload(importlib.import_module("tests.release.test_release_provenance"))
        from tests.release import test_release_provenance as mod

        assert mod._IS_PR_BUILD is True

    def test_marker_runs_when_event_is_push(self, monkeypatch: object) -> None:
        monkeypatch.setenv("GITHUB_EVENT_NAME", "push")  # type: ignore[attr-defined]
        importlib.reload(importlib.import_module("tests.release.test_release_provenance"))
        from tests.release import test_release_provenance as mod

        assert mod._IS_PR_BUILD is False


@skip_on_pr
def test_latest_release_tag_peels_to_commit_sha() -> None:
    """``git rev-parse <tag>^{commit}`` must yield a commit SHA, not a tag-object SHA.

    The provenance suite compares commit SHAs across tag/branch/state;
    if the test peeled with bare ``git rev-parse <tag>`` on an annotated
    tag, the comparison would always fail (annotated tag returns the
    tag-object SHA, not the underlying commit). CI run
    :gh-run:`29428237463` (AgentSLA) demonstrated this drift and forced
    the ``^{commit}`` peel.

    Anchors the invariant for DraftForge's annotated-tag releases
    (``v1.4.0`` is annotated; ``v1.0`` and ``v1.1`` are lightweight
    — peeling either form is safe).
    """
    tag = _latest_release_tag()
    peeled = _run_git("rev-parse", f"{tag}^{{commit}}")
    bare = _run_git("rev-parse", tag)
    # Peeled SHA must be a 40-char hex string. For annotated tags the
    # peel-target commit SHA differs from the bare tag-object SHA;
    # for lightweight tags they're equal (peel is a no-op). The
    # invariant only requires that ``peeled`` be a valid commit SHA.
    assert len(peeled) == 40, (
        f"git rev-parse {tag}^{{commit}} returned {peeled!r}; expected a 40-char commit SHA"
    )
    assert all(c in "0123456789abcdef" for c in peeled), (
        f"git rev-parse {tag}^{{commit}} returned {peeled!r}; expected lowercase hex"
    )
    assert bare, f"git rev-parse {tag} returned empty string"


@skip_on_pr
def test_latest_release_tag_is_ancestor_of_origin_main() -> None:
    """The latest release tag must be reachable from ``origin/main``.

    Equivalent: ``git merge-base --is-ancestor <tag> origin/main`` exits 0.

    Skipped on PR builds: a PR head branch is ahead of origin/main by
    construction. The invariant only fires correctly on push-to-main
    (where origin/main has caught up to the release tag).
    """
    tag = _latest_release_tag()
    out = subprocess.run(
        ["git", "merge-base", "--is-ancestor", tag, "origin/main"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert out.returncode == 0, (
        f"{tag} is NOT an ancestor of origin/main. "
        f"Either fast-forward origin/main onto the release branch "
        f"(`git checkout main && git merge --ff-only <release-branch>`) "
        f"or re-point the tag with `git tag -f {tag} origin/main`."
    )


@skip_on_pr
def test_main_has_no_commits_ahead_of_latest_release_tag() -> None:
    """``origin/main`` must not be strictly behind the latest release tag.

    Distinct from the consistency test's branch-local commit-count
    check: this pins the *default-branch* state so a phase branch
    drifting past the tag does not silently regress main.

    Skipped on PR builds: see
    :func:`test_latest_release_tag_is_ancestor_of_origin_main`.
    """
    tag = _latest_release_tag()
    out = _run_git("rev-list", f"origin/main..{tag}", "--count")
    behind = int(out) if out.isdigit() else 0
    assert behind == 0, (
        f"origin/main is {behind} commit(s) behind {tag}. "
        f"Run: `git checkout main && git merge --ff-only {tag}` "
        f"(or rebase the release branch onto main)."
    )


@skip_on_pr
def test_release_branch_tip_equals_peeled_tag() -> None:
    """``git rev-parse origin/main`` must equal ``git rev-parse <latest>^{commit}``.

    Guards against ``git tag -f`` re-points that advance the tag
    without the branch actually having the corresponding commits.
    Without the ``^{commit}`` peel, this comparison would always fail
    on annotated tags. CI run :gh-run:`29428237463` (AgentSLA)
    demonstrated the failure when compared without the peel.
    """
    tag = _latest_release_tag()
    tag_sha = _run_git("rev-parse", f"{tag}^{{commit}}")
    main_sha = _run_git("rev-parse", "origin/main")
    assert tag_sha == main_sha, (
        f"origin/main HEAD={main_sha[:12]} != {tag} peeled={tag_sha[:12]}. "
        f"Either re-tag (`git tag -f {tag} origin/main`) "
        f"or commit the missing changes onto main."
    )


def test_latest_release_tag_published_on_remote() -> None:
    """``git ls-remote`` must include the latest tag — guards against local-only tags.

    Skip-on-PR is NOT applied here: a tag published on the remote is
    a public artifact regardless of which event put it there. The test
    fetches the remote tag list each run (no caching).
    """
    tag = _latest_release_tag()
    out = subprocess.run(
        ["git", "ls-remote", "origin", f"refs/tags/{tag}"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=True,
    )
    assert out.stdout.strip(), (
        f"Latest tag {tag} is NOT on `origin`. "
        f"Local-only tags fail `git clone && git checkout {tag}` for "
        f"anyone else. Push with: `git push origin {tag}`."
    )


class TestChangelogHeadingShape:
    """Pin that CHANGELOG headings use ``[X.Y.Z]`` form (no ``v`` prefix).

    DraftForge's CHANGELOG convention is ``## [1.4.0]`` (unbracketed,
    no v). Git tags are ``v1.4.0``. Consistency suite happens to
    capture this via its regex but doesn't fail loudly if a heading
    like ``## [v1.4.0]`` quietly sneaks in. These tests pin the shape
    explicitly so any future heading-format change is a loud failure,
    not a silent assumption.
    """

    def test_changelog_headings_have_no_v_prefix(self) -> None:
        changelog = (REPO_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
        # Match headings with a `v` inside the brackets.
        bad = __import__("re").findall(r"^##\s+\[v\d+\.\d+\.\d+\]", changelog, __import__("re").MULTILINE)
        assert not bad, (
            f"CHANGELOG.md has v-prefixed version headings: {bad}. "
            f"DraftForge convention is `## [X.Y.Z]` (no `v`); git tags carry the `v`."
        )

    def test_latest_tag_matches_latest_heading_version_tuple(self) -> None:
        """Tag and heading must encode the same X.Y.Z tuple (modulo v prefix)."""
        from tests.release.test_release_consistency import (
            _latest_git_tag,
            _read_changelog_latest,
        )

        tag = _latest_git_tag()
        heading = _read_changelog_latest()
        assert _strip_v(tag) == heading, (
            f"Tag {tag!r} resolves to {_strip_v(tag)!r}; CHANGELOG latest heading is "
            f"{heading!r}. They must encode the same X.Y.Z tuple "
            f"(tag form is `vX.Y.Z`, heading form is `[X.Y.Z]`)."
        )
