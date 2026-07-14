# Branch Protection — DraftForge

Required GitHub branch-protection rules for `main`. Apply via
repo Settings → Code and automation → Rules → Branches.

## Rule on `main`

| Setting | Value |
|---|---|
| Branch name pattern | `main` |
| Restrict pushes that create new branches | enabled |
| Require a pull request before merging | enabled |
| Require approvals | 1 (the maintainer, `@jrajath94`) |
| Dismiss stale pull request approvals when new commits are pushed | enabled |
| Require review from Code Owners | enabled (honors `CODEOWNERS`) |
| Allow specified actors to bypass | none |
| Require status checks to pass before merging | enabled |
| Required status checks | `audit (ruff + mypy + pytest) (3.12)`, `coverage (>=75% core modules)`, `conventional-commits`, `Analyze (python)` |
| Require branches to be up to date before merging | enabled |
| Require conversation resolution before merging | enabled |
| Require signed commits | optional (GPG key required) |
| Require linear history | enabled (rebase or squash) |
| Allow force pushes | **disabled** |
| Allow deletions | **disabled** |

## Auxiliary rules

### Tag protection

GitHub offers no native tag-protection UI. Conventions:

1. **Release tags** (`vX.Y.Z`) — force-push forbidden.
   `git tag -d vX.Y.Z && git push origin :refs/tags/vX.Y.Z` only in an
   emergency; bump to `vX.Y.Z-fix1` instead.
2. **Pre-release tags** (`vX.Y.Z-rc.N`) — mutable; fast-forward allowed.
3. Use `make tag VERSION=X.Y.Z` for all release tagging — see the Makefile.

### Auto-merge via CI

Disabled by default. PRs that pass all required checks can be auto-merged
by `@jrajath94` only.

## Operational notes

- **Single-maintainer assumption.** The `1 approval` requirement relies on
  `CODEOWNERS` to make `@jrajath94` the reviewer of record for every path.
  If maintainer count grows to 2+, raise approvals to 2 and update
  `CODEOWNERS` to spread ownership.
- **Release tags are independent of branch rules.** A merged release PR
  does NOT auto-tag; `make tag` is a separate human action.
- **No force-push.** If history must be rewritten (e.g., leaking a
  secret), follow the `git filter-repo` playbook in SECURITY.md.

## Verification

To check current settings:

```bash
gh api repos/jrajath94/draftforge/branches/main/protection | jq '.'
```

For a comparison to what we think is configured, see this file. Drift is
expected if GitHub UI is used directly; recomment after any settings
change.
