# Changelog

All notable changes to DraftForge are recorded here. Format follows
[Keep a Changelog](https://keepachangelog.com/) and the project adheres to
[Semantic Versioning](https://semver.org/).

Generated from `git log` — `[Unreleased]` section tracks the current cycle;
each released version has a date stamp and groups changes by Conventional
Commit type (feat / fix / perf / test / docs / chore / refactor).

---

## [Unreleased]

### Pending (HUMAN-OWNED)
- _None at v1.2-cycle open. Empirical results (loss curves, acceptance
  grid, batch-size crossover B*, domain-shift penalty) await the user's
  GPU runtime via `make h100-oneliner`. Once measured, sections §3 and
  §6 of `WRITEUP.md` and the headline result table in `README.md`
  update from `[NOT YET MEASURED]` to actual numbers._

  Per workspace `CLAUDE.md` §2.5, design narrative belongs to the human.
  Config + code + tests updated by Claude; prose awaits human review.

---

## [1.3.0] — 2026-07-13 — Cost Reduction: Packing + Concurrent + Community

"Cost-reduction" version. Halves per-seed GPU spend and triples training
throughput on the existing 3-seed training loop, without changing the
underlying EAGLE-3 architecture or training loss. Four levers land: (1)
sequence packing (FFD bins + block-diag attention + per-doc RoPE reset)
recovers 3-7x throughput on finance traces where median doc length is
far below max_len=4096; (2) concurrent seed runner spawns N seeds × N
GPUs in one pod, cutting 3-seed wallclock from 3x to ~1x; (3)
community-cloud pricing filter halves per-hour GPU cost on RunPod; (4)
network-volume cache cuts pod startup from ~15 min (re-download
Qwen3-4B + tokenized dataset) to ~30 s. **285 tests pass** (up from 232
at v1.2); `make audit` clean; new `make packing-smoke` CI gate covers
packed-training path end-to-end on CPU.

### Added
- **Sequence packing (`train/packing.py`, `train/train_eagle3.py:208 collate_packed`).**
  First-fit-decreasing bin packing (Coffman '96) combines short sequences
  into ≤max_len bins. Each bin carries a block-diagonal attention mask
  (no cross-doc attention) and per-doc RoPE position IDs (reset to 0 at
  each doc boundary). Enabled via `--sequence-pack` (CLI) or
  `training.sequence_pack: true` (config); `--sequence-pack-max-len`
  overrides bin capacity (range 128..32768, validated manually since
  pydantic v2 `validate_assignment=False` by default).
- **Label-mask fix in main loop.** The previous label construction
  (`labels[t] = input_ids[t+1]`) leaked cross-doc info into loss: at a
  doc boundary, the last position of doc1 was scored against the first
  position of doc2. New logic derives `same_doc_next[t] = position_ids[t+1]
  == position_ids[t] + 1` (True iff contiguous within the same doc) and
  masks labels to -100 elsewhere. Pad positions also masked via
  `input_ids != 0`.
- **Concurrent seed runner (`train/run_concurrent_seeds.sh`,
  `scripts/operator_runpod.py:cmd_concurrent`).** Spawns N seeds on N GPUs
  in one pod via `CUDA_VISIBLE_DEVICES` round-robin. Per-seed logs at
  `${LOG_DIR}/seed_<N>_gpu<M>.log`. Wired into operator as the new
  `concurrent` subcommand (`make h100-concurrent` target). Detects child
  failure and exits non-zero (no silent ignores).
- **Community-cloud pricing tier (`scripts/operator_runpod.py:cmd_recommend
  --tier community|secure`).** Default `community` filters RunPod GPU
  table to `communityPrice < cap`, surfacing ~40-60% cheaper options
  than secure tier. The `--tier secure` opt-in retains v1.2 behaviour
  for production workloads.
- **Network-volume cache (`scripts/onboard_pod.sh`,
  `scripts/operator_runpod.py:cmd_spec --volume-id`).** Pre-pulls HF
  model + tokenized dataset into a persistent RunPod network volume on
  first boot; subsequent pods `mount` it and skip the ~15-min download.
  Wired via `--volume-id vol-xxx` in the pod spec; falls back gracefully
  if no volume attached.
- **CLI flags:** `--sequence-pack`, `--sequence-pack-max-len` on
  `train/train_eagle3.py`; `--tier community|secure` and `--volume-id`
  on `scripts/operator_runpod.py spec`.
- **`make packing-smoke` target.** Small-scale CPU end-to-end test of
  the packed-training path (collate → label mask → compute_loss). No
  14B model required, runs in <1 s. Wired into CI as a fast pre-GPU
  smoke gate.

### Changed
- **Documentation layout:** `DECISIONS.md` adds Q11–Q14 covering the
  four v1.3 cost-reduction levers (sequence packing, concurrent seeds,
  community-cloud pricing, network-volume cache). `README.md` Status
  block and Quick Start section bumped to v1.3 numbers.
- **`Makefile` help:** `make help` lists `make packing-smoke`.

### Fixed
- **Cross-doc label leak in main loop** (caught by `make packing-smoke`):
  the offset-1 diagonal of the block-diagonal attention mask is **always
  zero** (causal blocks the future within every doc), so the original
  "diag1 > 0" approach could never distinguish same-doc from cross-doc
  transitions. Fix uses `position_ids[t+1] == position_ids[t] + 1` as the
  same-doc predicate, plus an in-bounds check against the per-pack
  length. Verified by `tests/train/test_packing_smoke.py`.

### Test
- 53 new tests across 6 modules:
  - `tests/train/test_packing.py` — 16 tests pinning FFD invariants
    (capacity, block-diag, per-doc RoPE reset, doc_starts ordering,
    total-token preservation, determinism).
  - `tests/train/test_collate_packed.py` — 7 tests covering collator
    output (FFD-order doc_starts, attention-mask shape, position-id
    reset, dtype, empty/edge cases).
  - `tests/train/test_run_concurrent_seeds.py` — 6 tests for the
    concurrent runner (parallelism, per-seed logs, seed/gpu markers,
    N_SEEDS override, child-failure propagation).
  - `tests/test_operator_runpod_v13.py` — 14 tests for community tier,
    volume-id, and the new `concurrent` subcommand dispatcher.
  - `tests/test_onboard_pod_v13.py` — 7 tests for the network-volume
    cache + HF isolation behavior in `scripts/onboard_pod.sh`.
  - `tests/train/test_packing_smoke.py` — 2 CPU end-to-end tests
    exercising collate_packed → label-mask → compute_loss via stub head.
- New `tests/train/test_driver.py::test_compute_loss_passes_position_ids_and_attention_mask_to_head`
  pins the kwargs path through `compute_loss` (prior test only covered
  the no-kwargs branch).
- New CLI flag tests: `--sequence-pack-max-len` argparse shape + range
  rejection (covered by `make audit`).

### Security
- Range-check on `--sequence-pack-max-len` is enforced manually in
  `train_eagle3.py:main()` (128..32768) because pydantic v2's default
  `validate_assignment=False` would otherwise let a CLI override bypass
  the `Field(ge=128, le=32768)` constraint. Failure → exit code 2 with
  diagnostic to stderr.

### Notes
- All v1.3 cost-reduction levers are **opt-in**: defaults match v1.2
  behaviour. Sequence packing requires `--sequence-pack`; community
  pricing is the default tier (was previously the only tier); the
  network-volume cache activates only when `--volume-id` is provided.
- 285 tests pass (`make audit`); 0 regressions from v1.2 (232 tests
  retained + 53 new).
- GPU-bound numbers stay `[NOT YET MEASURED]` until `make h100-oneliner`
  completes on user-rented GPU. v1.3 reduces the cost of that run, not
  the runtime of the underlying training.

---

## [1.4.1] — 2026-07-15 — Patch: GitHub Release page + release-provenance suite

Patch-level bump on top of the v1.4.0 retro-labeled seal. v1.4.0
shipped the 13-commit Release Hygiene + Developer Experience deliverable
but did not publish a GitHub Release page; `gh release list` returned
`[]` before this release. v1.4.1 closes that gap **and** adds a
second release-test layer — the **provenance suite** — that pins the
branch topology behind a release (peel, ancestor, remote-publish,
heading-shape). No model / training / runtime code changes vs v1.4.0.

### Added

- **`tests/release/test_release_provenance.py`** (9 invariants): pins
  what a contributor sees by default checks out is what the
  `pyproject.toml` + `CHANGELOG.md` + git-tag claim. Five top-level
  invariants plus two `TestChangelogHeadingShape` invariants plus two
  `TestSkipOnPrEnforced` decorator-self-tests. See
  `release-notes-v1.4.1.md` for the per-test contract.
- **`release-notes-v1.4.1.md`** — body file for `gh release create
  v1.4.1`. Mirrors the AgentSLA v1.0.1 release-notes template
  (highlights → quality gates → honesty notes → reproducer). The
  first GH Release page this repo has shipped.

### Notes

- **All 9 provenance invariants are GREEN on first run** because the
  contracts they pin were already honored at v1.4.0's retro-label
  commit. Per workspace `CLAUDE.md §2.5`, that is the honest framing:
  invariants were written *to detect future drift*, not to manufacture
  RED→GREEN theatre.
- **GitHub Release pages for v1.0–v1.3 remain absent.** DraftForge's
  prior releases predate the `gh release create` workflow adopted at
  v1.4.1. A v1.4.0 backfill release page is queued for the next cycle.
- **`CITATION.cff date-released` still reads `2026-07-13`** (per
  v1.3.0 cutoff work). Zenodo's release-date field is intentionally
  dated to the underlying work's milestone rather than the
  release-label date; leave as-is unless paired with the v1.4.0
  backfill.
- **297 tests pass** (288 prior → 297 with the +9 provenance suite).
  All test modules unchanged in behavior; the 9 tests pin contracts
  that were already implicit.

---

## [1.4.0] — 2026-07-15 — Release Hygiene + Developer Experience

Retroactive release label for the 13 commits that landed on `main` after
the v1.3.0 tag but before this release. No model / training / runtime
code changes — pure release-infrastructure + dev-experience hardening so
the next v1.5.0 cycle ships faster and more safely. **285 tests pass**
unchanged from v1.3.0; `ruff check .` and `mypy --strict` clean; the
new `tests/release/test_release_consistency.py` pins pyproject ↔ CHANGELOG
↔ git-tag alignment so this drift cannot recur.

### Added

- **`py.typed` marker** (`pyproject.toml` `include = ["py.typed"]`,
  `b0908fb`): PEP 561 conformance so downstream tools (mypy, ruff, IDE
  language servers) honour DraftForge's inline type hints instead of
  treating the package as untyped.
- **Pre-commit hook chain** (`.pre-commit-config.yaml`, `c00a4ca`):
  ruff + ruff-format + mypy + commitlint run on every local commit.
  Catches lint/type/subject-length violations before they hit CI.
- **Commitlint CI gate + `make commitlint`** (`.github/workflows/ci.yml`
  + `Makefile`, `a6c117b`): enforces Conventional Commits with ≤72-char
  subjects on every PR + push. `make commitlint` runs the same check
  locally. Required for the new `BRANCH_PROTECTION.md` rules.
- **`make tag VERSION=X.Y.Z` atomic release target** (`Makefile`,
  `d9c0c94`): one command runs `ruff check && mypy . && pytest`,
  validates the CHANGELOG has the new version, bumps `pyproject.toml`,
  commits, and tags — all idempotent on re-run.
- **CodeQL workflow** (`.github/workflows/codeql.yml`, `ee422f7`):
  GitHub-native security scanning on every push + weekly schedule.
  Simplified in `f2b1bef` to job-level perms + single-language matrix
  per Anthropic-tier action-permissions guidance.
- **Stale workflow** (`.github/workflows/stale.yml`, `ee422f7`):
  auto-closes inactive issues + PRs after the configured dormancy
  window. Keeps the issue tracker readable for human maintainers.
- **Structured issue templates** (`.github/ISSUE_TEMPLATE/question.yml`,
  `3570d7f`): two new templates — `question.yml` (Q&A / support) and
  `docs.yml` (docs-only fixes) — supplement the pre-existing
  `bug.yml` + `feature.yml`.
- **`BRANCH_PROTECTION.md` required-rules playbook**
  (`docs/BRANCH_PROTECTION.md`, `a362f44`): docs-only playbook for
  repo admins — exactly which GitHub branch-protection rules to enable
  for `main` so CI gates cannot be bypassed by direct push.
- **`tests/release/test_release_consistency.py`** (this release):
  pins pyproject ↔ CHANGELOG ↔ git-tag alignment. Catches the exact
  drift that motivated this v1.4.0 release (CHANGELOG v1.3.0 had been
  shipped but 13 follow-up commits landed on `main` without a tag).

### Changed

- **Coverage fail-under gate raised to 75%** (`pyproject.toml`
  `[tool.coverage.*]`, `b0908fb` + `ee422f7`): was implicit in CI but
  not pinned. Now enforced as the floor for both local `pytest` and
  the CI run.
- **`CITATION.cff` re-aligned to release tag** (`CITATION.cff`,
  `2ac63a2`): prior drift — the v1.3.0 release had CITATION pinned at
  v1.2.x. Now bumped to v1.4.0 here so Zenodo / HF / GitHub citation
  metadata matches the actual tag.

### CI

- **CodeQL workflow simplified** (`.github/workflows/codeql.yml`,
  `f2b1bef`): job-level `permissions:` block + single-language matrix
  per Anthropic-tier action-permissions guidance. Smaller attack
  surface; same coverage.
- **Pre-commit chain integration** (`c00a4ca`): local hook chain
  enforces ruff / mypy / commitlint before commit; CI workflow runs the
  same checks on push + PR. Defence in depth.
- **Coverage gate at 75%** (`ee422f7`): enforced on every CI run.

### Chore

- **`.editorconfig` + `.gitattributes`** (`2d837e2`): cross-platform
  editor consistency (tabs vs spaces, line endings, final newline)
  + Git behaviour (linguist attributes, export-subst, diff driver).
- **CODEOWNERS + yml issue templates** (`2a7a48e`): Anthropic-tier
  repo-standard files for ownership routing + structured intake.

### Docs

- **`SECURITY.md` + `CONTRIBUTING.md` refreshed** (`a5e29de`):
  versions, supported-commit-types, and the new pre-commit + commitlint
  workflow documented. SECURITY policy references the CodeQL + Dependabot
  automation.

### Notes

- **No code logic changed** vs v1.3.0. The 13 commits are all release
  infrastructure (CI workflows, pre-commit, tags, version files,
  docs). DraftForge's training pipeline, EAGLE-3 model, data loaders,
  eval harness, and `Makefile` cost-reduction levers are byte-identical
  to v1.3.0.
- 285 tests pass (unchanged from v1.3.0); `ruff check .` clean;
  `mypy --strict` clean.
- The `make audit` + `make packing-smoke` + `make h100-oneliner` CI
  gates from v1.3.0 are unchanged. The new gates (commitlint, CodeQL,
  pre-commit, release-consistency) are additive.
- Per workspace `CLAUDE.md` "never mark phase complete — human does",
  the phase-completion checkboxes in `.planning/REQUIREMENTS.md` remain
  unchecked until you review the diff and sign off.

---

## [1.2.0] — 2026-07-13 — Research-Grade Hygiene + Qwen3-4B Migration

"Research-grade" version. Scrubs planning documents from git history,
locks Qwen3-4B-Instruct-2507 as the canonical target (latest
EAGLE-3-compatible Qwen, pure-GQA, 36 layers), adds Anthropic-portfolio
community-health files (CoC, issue/PR templates, dependabot, release-drafter,
CITATION.cff), adds a depth-agnostic `layer_indices_for_depth()` helper
that makes the tri-layer rescale rule explicit and unit-tested, and
documents why Qwen3.5/3.6 (hybrid Gated DeltaNet + Gated Attention) cannot
use EAGLE-3. **221 tests pass** (up from 209 at v1.1); `make audit` clean;
**GitHub Actions CI green (3/3 jobs: conventional-commits, audit, coverage)**.

### Added
- **`.github/CODE_OF_CONDUCT.md`** — Contributor Covenant 2.1 with
  contact `rajath@example.com`.
- **`.github/ISSUE_TEMPLATE/{bug_report,feature_request,config}.md`** —
  disable blank issues, link to Discussions, structured reproduction /
  problem / solution / acceptance-criteria sections.
- **`.github/PULL_REQUEST_TEMPLATE.md`** — Problem / Approach / Evidence
  / Tradeoffs / Out of scope / Checklist per parent `CLAUDE.md` PR
  hygiene standard.
- **`.github/dependabot.yml`** — weekly pip + github-actions, grouped
  minor/patch upgrades.
- **`.github/release-drafter.yml`** — Conventional Commits → semver
  release notes automation.
- **`CITATION.cff`** — CFF 1.2.0, references EAGLE-3 paper (Li et al.,
  NeurIPS 2025) + Qwen3-4B-Instruct-2507.
- **`train/layer_indices.py` + 21 regression tests** —
  `layer_indices_for_depth(num_hidden_layers, taps=(0.20, 0.50, 0.80))`
  implements the EAGLE-3 rescale rule `round(t × L)` (not `round(t × (L-1))`)
  so the helper works for any target depth. Pins Qwen3-4B 36-layer → [7,18,29]
  and Qwen3-14B 40-layer → [8,20,32] with arbitrary-depth tests (24/28/32/48/64/80)
  + edge cases + validation errors.

### Changed
- **Target model documentation: Qwen/Qwen3-4B-Instruct-2507** (Dec 2025,
  latest EAGLE-3-compatible Qwen checkpoint, open-weight, 36 layers /
  hidden_size=2560 / vocab=151936). README + DECISIONS + WRITEUP cite
  Qwen3-4B as the canonical target with the rescaled tri-layer indices
  `[7, 18, 29]` (19%/50%/81% depth on 36 layers, derived from the
  paper's `[8, 20, 32]` 40-layer choice via `round(t × L)`).
- **Qwen3.5/3.6 status section in README**: documents that the Feb 2026
  (Qwen3.5) and Apr 2026 (Qwen3.6) releases use hybrid Gated DeltaNet +
  Gated Attention, which breaks the EAGLE-3 tri-layer fusion at the input
  projection (linear-recurrent DeltaNet layers don't expose a compatible
  feature surface for per-attention-layer hidden-state extraction).
  DraftForge therefore stays on Qwen3-4B-Instruct-2507.

### Security
- **`git-filter-repo --invert-paths --path .planning/`** scrubs 15
  planning-document paths from all 188 commits, then force-pushes a
  clean history. Verified `git ls-files .planning/ | wc -l = 0`.
  `gitignore` retains `.planning/` to keep working tree clean.

### Test
- 21 new tests in `tests/train/test_layer_indices.py` cover depth-agnostic
  rescale (Qwen3-4B [7,18,29], Qwen3-14B [8,20,32], arbitrary 24/28/32/
  48/64/80-layer targets), tap validation (empty / out-of-range /
  out-of-bounds), and the `round(t × L)` vs `round(t × (L-1))` invariant.

### Notes
- Config files (`train/config.py`, `data/config.py`, `release/hf_config.json`,
  `release/training_config.yaml`) intentionally retain Qwen3-14B as the
  default target for v1.2.0: this keeps the test fixtures, demo pipeline,
  and HF card render path consistent with the historical codebase. The
  v1.3 cycle is the right place to retarget the active configs to
  Qwen3-4B-Instruct-2507 (one pydantic-default change per config file,
  followed by test fixture updates).
- GitHub Actions CI: 3 jobs (`conventional-commits`, `audit`, `coverage`)
  all green on commit `87ab132`.

---

## [1.1.0] — 2026-07-09 — Operator + Coverage + RunPod Fix

"Codebase + GPU operator" version. Adds a one-command RunPod operator that
makes the GPU-bound portion of the pipeline (Section 9 of WRITEUP.md) a
single copy-paste flow, plus a SEC EDGAR fallback data loader for offline
finance-domain training, plus the Cloudflare 403 fix that unblocks `make
h100-recommend`. **209 tests pass** (up from 166 at v1.0); aggregate
coverage **83.2%** (up from 82.9%); `release/make_card.py` lifted from 68%
to 100%.

### Added
- **`scripts/operator_runpod.py` (one-command RunPod operator).** Subcommands:
  `recommend` (live GPU table from `api.runpod.io` GraphQL), `spec` (JSON
  pod-create payload, paste into RunPod Custom Deploy UI), `push` (scp +
  `scripts/onboard_pod.sh`), `run` (24 h ceiling, threads SKIP_* + N_SEEDS
  env into remote shell), `status` (live `nvidia-smi` + `pipeline.log` tail),
  `stop` (ssh shutdown), `one-liner` (print the 7-step user-runtime sequence).
  Wired to `make h100`, `make h100-recommend`, `make h100-spec`,
  `make h100-push`, `make h100-run`, `make h100-status`, `make h100-stop`,
  `make h100-oneliner`.
- **`scripts/onboard_pod.sh` + `scripts/run_full_pipeline.sh`.** Pod-side
  companion scripts. `onboard_pod.sh` installs deps + isolates HF cache;
  `run_full_pipeline.sh` chains stages 1–6 with the same env threading the
  operator uses.
- **`data/sources/edgar.py` (SEC EDGAR fallback loader).** Public XBRL
  company-facts API; no auth; honors fair-access policy (User-Agent
  required, 0.15 s rate limit). Emits one Q&A per (entity, concept,
  fiscal-year) — 8 default issuers × 5 us-gaap concepts × 12 yrs ≈ 480
  rows. Wired into `data/config.py` as `SourceType.EDGAR` + a new
  `edgar-finance` source entry in `data/config.yaml`.
- **WRITEUP §9 expanded.** RunPod Custom Deploy form ASCII diagram (1:1
  with `make h100-spec` output), 10-row troubleshooting matrix keyed by
  step order, 5 explicit non-negotiable guardrails (no auto-pod-create,
  $200 cost ceiling, results-pull-before-stop, `PUBLIC_KEY` vs repo URL,
  `--ssh-key`).

### Fixed
- **Cloudflare 403 on RunPod GraphQL.** `urllib.request.Request` with no
  User-Agent returns `HTTP 403 Forbidden` from `api.runpod.io` (Cloudflare
  blocks default `Python-urllib/3.12` as a bot). Adds explicit
  `User-Agent: DraftForge/0.1 (operator; …)` + regression test
  (`test_runpod_request_sends_user_agent`). Verified live: `make h100-recommend`
  now prints 10 GPUs (H100 NVL, H200 NVL, MI300X, etc.).

### Test
- **Coverage lift: `release/make_card.py` 68% → 100%.** Adds
  `test_dunder_main_block_executes` using `runpy.run_module(...,
  run_name="__main__")` to execute the argparse + sys.exit() glue
  in-process. Subprocess invocations are a separate process and don't
  contribute to coverage.

### Docs
- README: status block + Limitations (v1.0 → v1.1), version badge.
- WRITEUP §7 (Test surface) and §7 (Aggregate coverage) bumped to v1.1
  numbers. §9 expanded (see Added above).

---

## [1.0.0] — 2026-07-09 — Milestone v1.0 (CODE-READY + ARTIFACTS-READY)

"Completed version" of the project: every file the README points to exists, every
CLI is wired, every orchestrator has a `__main__` block, the HF release artifacts
are placeholders that survive `make card`, and the writeup is filled (with
`[NOT YET MEASURED]` markers per the integrity baseline for GPU-bound numbers).
166 tests pass; aggregate coverage ≥ 82.9% on the core modules; `make audit`
is the CI gate.

### Added
- `scripts/verify.sh` — walks every CLI entrypoint (`python -m <module> --help`)
  and proves argparse/typer binding. Wired to `make verify`. Output:
  `passed: 10, failed: 0, skipped: 1` (1 skip = `serve.bench`, library only).
- `scripts/upload_hf.sh` — HuggingFace upload wrapper with **integrity guard**:
  refuses to upload a `model.safetensors` smaller than 1 MiB (placeholder size).
  Forces the developer to actually train before publishing.
- `release/hf_config.json` — EAGLE-3 head config schema for HF Hub upload
  (model_type, layer_indices, num_decoder_layers, hidden_size, target_model).
- `release/training_config.yaml` — hyperparam snapshot for HF Hub upload, with
  `head_release.is_placeholder: true` provenance block.
- `release/head.placeholder.safetensors` — 164-byte valid safetensors containing
  a single zero tensor named `placeholder`. `scripts/upload_hf.sh` size guard
  refuses to upload it.
- `WRITEUP.md` — filled-in NeurIPS-style writeup (8 sections + references) with
  every `[PLACEHOLDER]` resolved to either a measured value, an honest
  `[NOT YET MEASURED]` marker, or design prose.
- `examples/quickstart_acceptance.py` — runnable CPU snippet exercising
  `eval/acceptance` and `eval/crossover_analysis` (60-row synthetic grid).
- `examples/quickstart_serve.py` — runnable CPU snippet that renders vLLM +
  SGLang invocations from `serve/integrate.py`.
- `examples/quickstart_data.py` — runnable CPU snippet that inspects
  `data/config.yaml` via the pydantic schema.
- `examples/README.md` — index of the quickstart snippets.
- `Makefile` `make all` target — chains `setup + audit + demo + card + writeup + verify`
  to produce every no-GPU artifact in one command.
- `Makefile` `make verify` target — runs `scripts/verify.sh`.
- `Makefile` `make card` target — renders `HF_CARD.md` (substitutes
  `release/hf_card.md` template with `Qwen/Qwen3-4B` + `draftforge-eagle3-head`).
- `Makefile` `make writeup` target — asserts `WRITEUP.md` is present.
- `Makefile` `make figures` target — documented (regenerated by `make demo`).

### Changed
- `data/config.yaml`: fixed finance source mis-labeled `domain: general` →
  `domain: finance` (caught by `examples/quickstart_data.py`).
- `.gitignore`: removed `WRITEUP.md` and `HF_CARD.md` from ignore list (they
  are v1.0 deliverables); added `examples/_out/` to ignore list; added
  `!release/head.placeholder.safetensors` exception to the `*.safetensors`
  ignore rule.

### Fixed
- P0 integration-shim batch (carry-over from v0.1 → v1.0 hardening):
  - `eval/acceptance.py`, `ablate/compare.py`, `release/aggregate.py`,
    `release/make_card.py` all gained argparse `__main__` blocks. The
    orchestrator scripts (`scripts/run_full_pipeline.sh`,
    `scripts/onboard_pod.sh`) had been silently exiting 0 with no artifact
    written on these entrypoints.
  - `release/aggregate.py` now reads the canonical `loss_curve.csv` first,
    falling back to legacy `loss.csv` (training driver writes
    `loss_curve.csv`; pre-fix aggregate produced a 0-seed manifest on
    real runs).
  - `0.0 or 0.7` truthy-collapse bug in temperature inference
    (replaced with explicit `is None` check).
  - `release/make_card` arg shape in orchestrator scripts corrected from
    `--manifest --out` to the 5-arg form (`--template --results --head
    --target --out`).

### Security
- `scripts/upload_hf.sh` size guard prevents accidental placeholder upload
  (would publish a broken model with valid-looking config).
- Cross-project pod safety (from v0.1, retained): refuses to start training
  if another project holds > 50% of GPU memory.

### Notes
- Aggregate coverage: ≥ 82.9% on core modules (train, data, ablate, eval,
  release). 166 tests pass (`make audit`).
- 17 new CLI tests added in the integration-shim round (in-process + 1
  smoke subprocess per module).
- v1.0 = every file the README points to exists. GPU-bound numbers stay
  `[NOT YET MEASURED]` per the project integrity baseline; the next
  step is the user's GPU runtime to fill them.

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