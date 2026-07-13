---
name: Bug report
about: Report something broken, wrong, or undocumented
title: "[bug] "
labels: ["bug"]
assignees: []
---

## Summary

One or two sentences. What broke, what you expected instead.

## Reproduction

Minimal steps that trigger the bug. Include the exact command or snippet.

```bash
# paste the command here
```

If the bug needs data, attach the smallest fixture that reproduces it (or
point to a public dataset / cached JSONL).

## Environment

- DraftForge version (commit SHA or `pip show draftforge | grep Version`):
- Python version (`python --version`):
- OS (`uname -a`):
- GPU model and driver (`nvidia-smi`), if relevant:
- Target model id (e.g. `Qwen/Qwen3-4B-Instruct-2507`):
- vLLM or SGLang version, if the bug is in the integration layer:

## Actual behavior

What happened. Paste exact output, including stack traces.

```text
paste here
```

## Expected behavior

What you expected.

## Logs / artifacts

Attach or link:

- `results/manifest.json` (if a run completed)
- `results/train/<seed>/loss_curve.csv` (if training-related)
- Any `*.nsys-rep` traces (Nsight Systems)
- `pytest` output for failing tests

## Severity

How bad is it? Pick one:

- [ ] Blocker (cannot run `make audit`, `make demo`, or any default target)
- [ ] High (default pipeline produces wrong output, no workaround)
- [ ] Medium (workaround exists, or only affects non-default paths)
- [ ] Low (cosmetic, docs, or non-default path)

## Checklist

- [ ] I searched existing issues and did not find a duplicate.
- [ ] I can reproduce on a clean checkout (no local edits).
- [ ] I have not modified the commit message convention or CI workflow
      unless the bug is in those files.