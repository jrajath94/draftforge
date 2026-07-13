## Problem

What gap or bug this PR closes. Reference the issue number with `Closes #N`
or explain why no issue exists.

## Approach

What you changed and why. Keep it tight: bullet points beat prose for the
mechanics, prose is fine for the tradeoffs.

## Evidence

How a reviewer verifies this works. Paste:

- `make audit` output (or the failing-before / passing-after diff)
- `pytest` summary for new or modified tests
- Any new figure, table, or artifact under `results/`

If the PR introduces a measured number, cite the artifact it lives in
(`results/train/<seed>/loss_curve.csv`, `results/eval/acceptance_grid.csv`,
etc.). Unmeasured numbers stay `[NOT YET MEASURED]`.

## Tradeoffs

What you chose not to do, and why. Call out anything a reviewer might push
back on.

## Out of scope

What this PR does not address. Be explicit so scope creep stays out of the
review.

## Checklist

- [ ] `make audit` passes locally (ruff + mypy + pytest).
- [ ] New tests cover the change; coverage on touched modules is at least 75%.
- [ ] Commit subjects follow Conventional Commits (≤72 chars, no period).
- [ ] No fabricated numbers; every measured value traces to a committed artifact.
- [ ] No secrets committed (`.env`, weights, `.safetensors`, `*.pt`).
- [ ] Branch is up to date with `main` (`git rebase main` if needed).
- [ ] PR title matches `<type>(<scope>): <subject>` (Conventional Commits).