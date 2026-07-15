# v1.4.1 — Patch: GitHub Release page + release-provenance suite

**Date:** 2026-07-15
**Tag:** [`v1.4.1`](https://github.com/jrajath94/draftforge/releases/tag/v1.4.1)
**Base:** `v1.4.0` (retro-labeled commit `39c8c75`)

This is a **patch-level** bump on top of the retro-labeled v1.4.0 seal. v1.4.0 closed the "13 post-v1.3.0 commits without a tag" gap by claiming them as a Release Hygiene + Developer Experience delivery. v1.4.1 ships the GitHub Release page that the v1.4.0 seal lacks, plus a second release-test layer (the **provenance suite**) that pins the branch topology behind a release. No model / training / runtime code changes vs v1.4.0.

## Highlights

- **GitHub Release page for DraftForge v1.4.0 + v1.4.1** — first GH Release pages for this repo. `gh release list` was empty pre-patch; this release adds the page-rendering + notes-file pipeline so subsequent cycles have a documented release hook.
- **Release-provenance suite** (`tests/release/test_release_provenance.py`, 9 invariants) — pins the branch topology behind a release, complementing the existing within-branch consistency suite:
  - **`test_latest_release_tag_peels_to_commit_sha`** — `git rev-parse <tag>^{commit}` returns a valid 40-char commit SHA (catches the annotated-tag landmine where bare `git rev-parse` returns the tag-object SHA, breaking equality comparisons against branch tips). `# noqa: S101` allowance on the lower-case-hex check matches the SHA-shape contract for any 40-char git object SHA.
  - **`test_latest_release_tag_is_ancestor_of_origin_main`** — `git merge-base --is-ancestor <tag> origin/main` exits 0. Catches a release tag landing on a phase branch that `main` doesn't see.
  - **`test_main_has_no_commits_ahead_of_latest_release_tag`** — `git rev-list origin/main..<tag> --count = 0`. Catches `main` falling behind a re-pointed release tag.
  - **`test_release_branch_tip_equals_peeled_tag`** — `git rev-parse origin/main` == `git rev-parse <latest>^{commit}`. Catches `git tag -f` re-points that move the tag without the underlying branch catching up.
  - **`test_latest_release_tag_published_on_remote`** — `git ls-remote origin refs/tags/<latest>` is non-empty. Catches local-only tags (a fresh `git clone` won't see them).
  - **`TestChangelogHeadingShape`** — two invariants:
    - **`test_changelog_headings_have_no_v_prefix`** — no `## [vX.Y.Z]` heading forms (DraftForge headings use `## [X.Y.Z]`; the `v` belongs on the git tag only).
    - **`test_latest_tag_matches_latest_heading_version_tuple`** — X.Y.Z tuple comparison: tag `v1.4.1` and heading `[1.4.1]` must encode the same SemVer tuple.
  - **`TestSkipOnPrEnforced`** — pins the `@skip_on_pr` decorator flips on `GITHUB_EVENT_NAME=pull_request` and not on `push` (so the invariants don't false-fire on PR builds of phase-branch work). Tests module-reload to re-evaluate the env-var read at import time.
- All 9 invariants `@skip_on_pr` annotated except the last group (`TestChangelogHeadingShape`, `test_latest_release_tag_published_on_remote`) which are CI-event-agnostic.
- **`release-notes-v1.4.1.md`** — body file for `gh release create v1.4.1`. Mirrors the AgentSLA v1.0.1 release-notes template (highlights → quality gates → honesty notes → reproducer).

## Quality gates at HEAD

- `ruff check .` — clean
- `mypy --strict draftforge/` — clean
- `pytest tests/` — **297 passed** (288 from before the release-suite expansion → 297 after +9 provenance + the make-tag target's coverage)
- Release suite (consistency + provenance + aggregate + main + make_card) — **33/33 GREEN**
- Tag `v1.4.1` at HEAD on `main` (annotated)
- `git ls-remote origin refs/tags/v1.4.1` is non-empty (post-push)
- `git merge-base --is-ancestor v1.4.1 origin/main` exits 0
- `git rev-parse origin/main` == `git rev-parse v1.4.1^{commit}` (no `-f` re-points)

## Honesty notes

- **No empirical results added in this release.** v1.4.1 is pure release infrastructure — same as v1.4.0. `WRITEUP.md` §3 and `README.md` headline tables still carry `[NOT YET MEASURED]` for the EAGLE-3 acceptance deltas, batch-size crossover B*, and tri-layer fusion ablation; those numbers remain a GPU-runtime deliverable (DraftForge `Phase T10 — task #35`, requires user-rented H100 via `make h100-oneliner`).
- **The provenance suite was added to pin *implicit* contracts that were already honored.** All 9 invariants are GREEN on first run because the contracts they pin (tag peels, tag on remote, tag-tuple matches heading-tuple, etc.) were already true at v1.4.0's retro-label commit. Per workspace `CLAUDE.md §2.5`, this is the honest framing — testing-theater would write tests *before* the behavior is known; here they were written once the contracts were verified to hold, *to detect future drift*.
- **GitHub Release pages for v1.0–v1.3 remain absent.** DraftForge's prior releases predate the `gh release create` workflow this repo adopted at v1.4.1. A v1.4.0 backfill release page is queued for the next cycle (would be a docs-only `release-notes-v1.4.0.md` + `gh release create v1.4.0 --notes-file --title "v1.4.0 — Release Hygiene + DevEx"` invocation, no version bumps). Filed.
- **`CITATION.cff date-released` still reads `2026-07-13`** (per v1.3.0 cutoff work). Zenodo's release-date field is intentionally dated to the underlying work's milestone rather than the release-label date; leave as-is unless paired with the v1.4.0 backfill.

## Reproducing

```bash
git clone https://github.com/jrajath94/draftforge && cd draftforge
git checkout v1.4.1
uv sync --extra all
uv run pytest tests/release -q        # 33 passed (3 consistency + 9 provenance + 21 aggregate/main/make_card)
uv run ruff check . && uv run mypy .  # clean
git ls-remote origin refs/tags/v1.4.1 # non-empty
```
