# Changelog

All notable changes to DraftForge are recorded here. Format follows
[Keep a Changelog](https://keepachangelog.com/) and the project adheres to
[Semantic Versioning](https://semver.org/).

Generated from `git log` — `[Unreleased]` section tracks the current cycle;
each released version has a date stamp and groups changes by Conventional
Commit type (feat / fix / perf / test / docs / chore / refactor).

---

## [Unreleased]

### Added
- End-to-end local demo pipeline via `make demo` (no GPU, no HF, no network) — `scripts/run_demo.py`
- Sample finance Q&A fixture: `data/fixtures/sample_finance.jsonl` (30 synthetic rows)
- Local-only data config for demo: `data/demo_config.yaml`
- Pipeline-shape regression tests: `tests/test_demo_pipeline.py` (2 `@slow` tests)

### Changed
- Makefile: added `demo` target, listed in `make help`

---

## [0.1.0] — 2026-07-09 — Milestone v0.1 (CODE-READY)

First complete release. All 6 phases shipped; 147 tests pass, 82.9% aggregate coverage,
ruff + mypy clean across 32 source files. GPU bench numbers `[NOT YET MEASURED]` —
user-runtime.

### Added
- **Data pipeline (Phase 1)**: ShareGPT / OpenHermes / finance loaders, dedup (exact + MinHash),
  stratified split with SHA256 reproducibility log, tokenization, domain-distribution plot
- **Training (Phase 2)**: `EAGLE3Head` (tri-layer fusion [8, 20, 32] → projection → decoder
  blocks → LM head), bf16 training driver with training-time-test, DeepSpeed ZeRO-2 config,
  ≥3-seed run-script, seed-determinism contract (4 `@slow` tests)
- **Ablation (Phase 3)**: 4 layer-fusion presets (tri_layer, final_layer, low_only,
  mid_only), ≥3-seed comparison runner, ValueError paths for missing headers /
  malformed rows
- **Integration + Profile (Phase 4)**: vLLM + SGLang invocation builders, Nsight wrapper
  with kernel-attribution classification (draft-bound / balanced / verify-bound), bench
  shell wrapper, cross-project pod safety (refuse to start if other project holds
  >50% GPU memory)
- **Acceptance Analysis (Phase 5)**: geometric-mean EAL, batch-size crossover model,
  full acceptance-grid CLI, batch-size sweep, plotting helpers
- **Release (Phase 6)**: HF model-card template, `release.aggregate` JSON manifest,
  `release.make_card` template renderer, typer CLI (`python -m release.__main__`)
- **Writeup template**: 8-section NeurIPS-style structure (468 lines, 84 `[NOT YET MEASURED]`
  placeholders to be filled by real bench)
- **CI**: 3-job GitHub Actions gate (audit / coverage / conventional-commits), Makefile
  targets (`make audit`, `make bench`, `make onboard`, `make demo`)
- **Docs**: README headline result table + Limitations + Citation, CONTRIBUTING.md, badges
- **Post-milestone coverage gap closure**: 53 new tests across 4 rounds, lifting aggregate
  coverage 67.6% → 82.9%
- **Real production bug fix**: `data/sources/finance.py:_results_path_check` had a dead
  branch (`_FIXTURE_PATH in path.parents` — `_FIXTURE_PATH` is a file path; `path.parents`
  only contains directories). Discovered by writing the negative-path test.

### Security
- `chore(security): harden gitignore + scrub breach paths from history` (`115f79d`)
- Preflight refuses to start training if another project holds >50% GPU memory
- HF auth preflight (`onboard_pod.sh`) gates the pipeline
- Finance fixture guard: refuses to load synthetic test fixtures into `results/`

### Notes
- Per parent spec integrity baseline: **no fabricated numbers**. Every figure in the
  writeup must trace to a `make bench` invocation. Until training runs on H100, all
  bench/acceptance/writeup numbers stay `[NOT YET MEASURED]`.
- Repo pushes: 15 commits on `origin/main` at `115f79d`. Working tree clean at audit time.

---

## Contributing

See `CONTRIBUTING.md`. Changes are recorded here; the latest released version
appears at the top, `feat:` entries appear under "Added", `fix:` under "Fixed",
etc. Commits outside the Conventional Commits format are grouped under "Other".