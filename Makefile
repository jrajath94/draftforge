# DraftForge Makefile
# Every figure in the writeup must trace to a `make bench` invocation.
# Per parent CLAUDE.md README Standard: "Benchmarks: exact commands to reproduce every number"
#
# Standard targets:
#   make setup     — venv + install
#   make test      — pytest
#   make lint      — ruff
#   make types     — mypy
#   make coverage  — pytest + coverage
#   make audit     — ruff + mypy + pytest (the gate before every commit)
#   make onboard   — run scripts/onboard_pod.sh (pod setup)
#   make bench     — full pipeline (scripts/run_full_pipeline.sh) — needs GPU
#   make clean     — remove pyc + caches
#
# Env knobs:
#   PYTHON    python interpreter (default: .venv/bin/python)
#   N_SEEDS   seeds for training (default: 3)
#   SKIP_*    skip pipeline stage (default: 0). See scripts/run_full_pipeline.sh

PYTHON ?= .venv/bin/python
N_SEEDS ?= 3

.PHONY: setup test lint types coverage audit onboard bench demo clean help

help:
	@echo "DraftForge Makefile"
	@echo "  make setup     venv + install"
	@echo "  make test      pytest"
	@echo "  make lint      ruff check"
	@echo "  make types     mypy"
	@echo "  make coverage  pytest + coverage report"
	@echo "  make audit     lint + types + test (CI gate)"
	@echo "  make onboard   pod onboarding (scripts/onboard_pod.sh)"
	@echo "  make bench     full pipeline (scripts/run_full_pipeline.sh) — needs GPU"
	@echo "  make demo      local CPU pipeline (scripts/run_demo.py) — no GPU"
	@echo "  make clean     remove caches"

setup:
	python3 -m venv .venv 2>/dev/null || python3.12 -m venv .venv
	.venv/bin/pip install --quiet --upgrade pip
	.venv/bin/pip install --quiet -e ".[train,dev]"

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

onboard:
	bash scripts/onboard_pod.sh

bench:
	N_SEEDS=$(N_SEEDS) bash scripts/run_full_pipeline.sh

demo:
	$(PYTHON) scripts/run_demo.py

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov coverage.xml .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true