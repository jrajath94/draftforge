# DraftForge Makefile
# Every figure in the writeup must trace to a `make bench` invocation.
# Per parent CLAUDE.md README Standard: "Benchmarks: exact commands to reproduce every number"
#
# Standard targets (CPU-safe unless noted):
#   make setup       — venv + install
#   make lint        — ruff
#   make types       — mypy
#   make test        — pytest
#   make coverage    — pytest + coverage report
#   make audit       — lint + types + test (CI gate, run before every commit)
#   make verify      — bash scripts/verify.sh (walks every CLI)
#   make demo        — local CPU pipeline (scripts/run_demo.py)
#   make figures     — regenerate plots (called by demo)
#   make card        — render HF_CARD.md (release.make_card)
#   make writeup     — validate WRITEUP.md is current (no-op if file present)
#   make all         — setup + audit + demo + card + verify  (no-GPU full artifact set)
#   make onboard     — run scripts/onboard_pod.sh (pod setup)
#   make bench       — full pipeline (scripts/run_full_pipeline.sh) — needs GPU
#   make h100        — RunPod operator (recommend / spec / push / run / status / stop)
#   make clean       — remove pyc + caches
#
# Env knobs:
#   PYTHON     python interpreter (default: .venv/bin/python)
#   N_SEEDS    seeds for training (default: 3)
#   SKIP_*     skip pipeline stage (default: 0). See scripts/run_full_pipeline.sh
#   GPU_ID     RunPod GPU type id (for `make h100-spec`)

PYTHON ?= .venv/bin/python
N_SEEDS ?= 3

.PHONY: setup test lint types coverage audit verify demo figures card writeup commitlint all onboard bench clean tag help h100 h100-recommend h100-spec h100-push h100-run h100-status h100-stop h100-oneliner

help:
	@echo "DraftForge Makefile"
	@echo "  make setup     venv + install"
	@echo "  make test      pytest"
	@echo "  make lint      ruff check"
	@echo "  make types     mypy"
	@echo "  make coverage  pytest + coverage report"
	@echo "  make audit     lint + types + test (CI gate)"
	@echo "  make commitlint  validate HEAD commit subject against commitlint.config.js"
	@echo "  make packing-smoke  small-scale CPU smoke for sequence packing"
	@echo "  make verify    bash scripts/verify.sh (every CLI binds)"
	@echo "  make demo      local CPU pipeline (scripts/run_demo.py)"
	@echo "  make figures   regenerate plots (eval/plot.py via demo)"
	@echo "  make card      render HF_CARD.md (release.make_card)"
	@echo "  make writeup   validate WRITEUP.md is committed"
	@echo "  make all       setup + audit + demo + card + verify (no-GPU)"
	@echo "  make tag VERSION=X.Y.Z  atomic release tag (bump + audit + tag)"
	@echo "  make onboard   pod onboarding (scripts/onboard_pod.sh)"
	@echo "  make bench     full pipeline (scripts/run_full_pipeline.sh) — needs GPU"
	@echo "  make h100-*    RunPod operator (recommend/spec/push/run/status/stop)"
	@echo "  make clean     remove caches"

setup:
	@# Idempotent: only create venv if missing or wrong Python version.
	@# Prevents clobbering a working Python 3.12 venv when system `python3`
	@# resolves to 3.14 (no torch wheel for 3.14 yet as of 2026-07).
	@if [ ! -d .venv ] || [ ! -x .venv/bin/python ]; then \
		echo "[setup] creating .venv with python3.12 (pinned — torch wheel compatible)"; \
		python3.12 -m venv .venv || python3 -m venv .venv; \
		.venv/bin/pip install --quiet --upgrade pip; \
		.venv/bin/pip install --quiet -e ".[train,dev]"; \
	else \
		echo "[setup] .venv exists with $$(.venv/bin/python -c 'import sys; print(sys.version.split()[0])') — skipping (delete .venv to force recreate)"; \
	fi

test:
	$(PYTHON) -m pytest -q --no-header

lint:
	ruff check .

types:
	$(PYTHON) -m mypy train ablate serve eval release data

coverage:
	$(PYTHON) -m pytest --cov=train --cov=ablate --cov=serve --cov=eval --cov=release --cov=data --cov-report=term-missing --cov-report=html:htmlcov --cov-report=xml:coverage.xml -q

audit: lint types test
	@echo "audit OK — safe to commit"

# commitlint validates the most recent commit subject against the
# @commitlint/config-conventional rules in commitlint.config.js
# (72-char max, lower-case type, no trailing period, conventional type).
# Pre-commit: install husky + commitlint npm packages, wire commit-msg hook.
# Until then, run `make commitlint` manually before each commit.
commitlint:
	@if command -v commitlint >/dev/null 2>&1; then \
		commitlint --from=HEAD~1 --to=HEAD --config commitlint.config.js || \
			(echo "[commitlint] HEAD commit violates rules in commitlint.config.js"; exit 1); \
	else \
		echo "[commitlint] npx commitlint not installed — skipping (run: npm i -D @commitlint/cli @commitlint/config-conventional)"; \
	fi

verify:
	bash scripts/verify.sh

# demo runs the full local CPU pipeline: data → train-shape → ablate → eval → release.
# It writes synthetic artifacts to results/demo/ (gitignored) so reviewers can
# verify pipeline shape end-to-end without GPU.
demo:
	$(PYTHON) scripts/run_demo.py

# figures is a sub-target of demo; exposed separately so plot regeneration is
# cheap and doesn't re-run the full pipeline.
figures:
	@echo "[figures] regenerated by 'make demo' (no separate script — see scripts/run_demo.py)"
	@test -f results/demo/eval/acceptance_by_batch.png && echo "[figures] OK — see results/demo/eval/" || echo "[figures] NOTE — run 'make demo' first to materialize results/demo/"

# card renders HF_CARD.md at the repo root. Input: release/hf_card.md template
# + results/manifest.json. If manifest.json is missing, falls back to {} (card
# still renders with placeholders substituted).
card:
	@mkdir -p results
	@test -f results/manifest.json || $(PYTHON) -c "import json, pathlib; pathlib.Path('results').mkdir(exist_ok=True); pathlib.Path('results/manifest.json').write_text(json.dumps({'root':'results','note':'placeholder for v1.0 card render'}, indent=2, sort_keys=True))"
	$(PYTHON) -m release.make_card \
		--template release/hf_card.md \
		--results results \
		--head draftforge-eagle3-head \
		--target Qwen/Qwen3-4B-Instruct-2507 \
		--out HF_CARD.md

# packing-smoke: small-scale CPU smoke for sequence packing. Exercises the
# full packed-training path (collate → label mask → compute_loss) with a
# stub head. Confirms block-diag mask, per-doc RoPE reset, and label-mask
# leak fix work end-to-end before any GPU run.
packing-smoke:
	$(PYTHON) -m pytest tests/train/test_packing_smoke.py -v --no-header

# writeup just validates that WRITEUP.md is present (it's committed, not generated).
writeup:
	@test -f WRITEUP.md && echo "[writeup] OK — WRITEUP.md is committed" || (echo "[writeup] ERROR — WRITEUP.md missing"; exit 1)

# all = the no-GPU full artifact set. Anyone with a laptop + a venv can run
# `make all` and reproduce every artifact except the trained weights.
all: setup audit demo card writeup verify
	@echo ""
	@echo "[all] complete. Artifacts produced:"
	@echo "  - results/demo/                   CPU pipeline artifacts"
	@echo "  - HF_CARD.md                      rendered HF model card"
	@echo "  - WRITEUP.md                      NeurIPS-style writeup"
	@echo "  - release/head.placeholder.safetensors  (164 bytes; upload_hf.sh refuses)"
	@echo ""
	@echo "[all] GPU-bound (NOT in 'all'): run 'make bench' on a rented H100"
	@echo "      to produce: trained weights, real loss curves, ITL tables, crossover B*."

onboard:
	bash scripts/onboard_pod.sh

bench:
	N_SEEDS=$(N_SEEDS) bash scripts/run_full_pipeline.sh

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov coverage.xml .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf examples/_out

# tag: atomic release tag (refuses if dirty tree or missing CHANGELOG entry).
# Usage: make tag VERSION=1.4.0
# Pre-flight: bumps pyproject.toml + CITATION.cff + CHANGELOG header,
# asserts a `## [VERSION]` entry exists, runs `make audit`, then commits
# + annotated-tags + pushes. The release commit is itself a tiny bump
# commit so the tag points at a verified state.
tag:
	@if [ -z "$(VERSION)" ]; then echo "ERROR: set VERSION=X.Y.Z"; exit 1; fi
	@if [ -n "$$(git status --porcelain)" ]; then echo "ERROR: working tree dirty"; git status --short; exit 1; fi
	@grep -q "^## \[$(VERSION)\]" CHANGELOG.md || (echo "ERROR: CHANGELOG.md missing ## [$(VERSION)] entry"; exit 1)
	@grep -q "^version = \"$(VERSION)\"" pyproject.toml || (echo "ERROR: pyproject.toml version != $(VERSION)"; exit 1)
	@grep -q "^version: $(VERSION)" CITATION.cff || (echo "ERROR: CITATION.cff version != $(VERSION)"; exit 1)
	@make audit
	@git add pyproject.toml CITATION.cff CHANGELOG.md
	@git diff --cached --quiet || git commit -m "chore(release): prepare $(VERSION)"
	@git tag -a "v$(VERSION)" -m "Release v$(VERSION)"
	@echo "tagged v$(VERSION) — push with: git push origin main v$(VERSION)"

# ── RunPod one-command GPU operator (scripts/operator_runpod.py) ──────────────
# Lets the user go from "no GPU" → "trained head" with: spec → runpod UI → push → run.
# Does NOT auto-create a pod (cost gate); user pastes `make h100-spec` output into
# RunPod UI to deploy, then runs `make h100-push POD_ID=...` etc.

GPU_ID ?= NVIDIA H100 80GB HBM3
SSH_HOST ?= pod-XYZ.runpod.io
SSH_PORT ?= 22
SSH_KEY ?= ~/.ssh/id_rsa
POD_ID ?= my-pod-id

h100:
	@echo "RunPod one-command operator for DraftForge."
	@echo ""
	@echo "Quick start:"
	@echo "  make h100-recommend           # live RunPod GPU table"
	@echo "  make h100-spec                # pod-create JSON (paste into RunPod UI)"
	@echo "  ... user pastes JSON into RunPod UI, deploys, notes host+port"
	@echo "  make h100-push SSH_HOST=...   # SCP repo + onboard_pod.sh"
	@echo "  make h100-run SSH_HOST=...    # full pipeline (24h ceiling)"
	@echo "  make h100-status SSH_HOST=... # nvidia-smi + tail pipeline.log"
	@echo "  make h100-stop SSH_HOST=...   # shutdown -h now"
	@echo ""
	@echo "End-to-end: make h100-oneliner"
	@echo "Sub-tool:   python scripts/operator_runpod.py --help"

h100-recommend:
	$(PYTHON) scripts/operator_runpod.py recommend

h100-spec:
	$(PYTHON) scripts/operator_runpod.py spec --gpu "$(GPU_ID)"

h100-push:
	@test -n "$(SSH_HOST)" || (echo "ERROR: set SSH_HOST=<pod>.runpod.io"; exit 1)
	$(PYTHON) scripts/operator_runpod.py push "$(POD_ID)" \
		--ssh-host "$(SSH_HOST)" --ssh-port $(SSH_PORT) --ssh-key "$(SSH_KEY)"

h100-run:
	@test -n "$(SSH_HOST)" || (echo "ERROR: set SSH_HOST=<pod>.runpod.io"; exit 1)
	$(PYTHON) scripts/operator_runpod.py run "$(POD_ID)" \
		--ssh-host "$(SSH_HOST)" --ssh-port $(SSH_PORT) --ssh-key "$(SSH_KEY)" \
		--n-seeds $(N_SEEDS)

h100-status:
	@test -n "$(SSH_HOST)" || (echo "ERROR: set SSH_HOST=<pod>.runpod.io"; exit 1)
	$(PYTHON) scripts/operator_runpod.py status "$(POD_ID)" \
		--ssh-host "$(SSH_HOST)" --ssh-port $(SSH_PORT) --ssh-key "$(SSH_KEY)"

h100-stop:
	@test -n "$(SSH_HOST)" || (echo "ERROR: set SSH_HOST=<pod>.runpod.io"; exit 1)
	$(PYTHON) scripts/operator_runpod.py stop "$(POD_ID)" \
		--ssh-host "$(SSH_HOST)" --ssh-port $(SSH_PORT) --ssh-key "$(SSH_KEY)"

h100-oneliner:
	$(PYTHON) scripts/operator_runpod.py one-liner
