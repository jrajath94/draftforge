# Contributing to DraftForge

## Commit Standards

All commits must follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <subject>
```

**Types:**
- `feat`: New feature or phase
- `fix`: Bug fix
- `perf`: Performance improvement
- `test`: Test additions or fixes
- `docs`: Documentation or phase notes
- `refactor`: Code refactoring without behavioral change
- `bench`: Benchmark changes
- `ci`: CI/CD pipeline changes

**Subject:**
- Imperative mood ("add" not "added")
- No period at end
- ≤72 characters
- Lowercase

## Code Quality

All code must pass:

```bash
# Formatting
ruff format .

# Linting
ruff check .

# Type checking
mypy train ablate serve eval release data

# Testing (≥60% coverage)
pytest tests/ --cov-fail-under=60
```

## Pull Request Process

1. **Branch:** Create from `main` with name `phase-N/description` (e.g., `phase-2/data-pipeline`)
2. **Commits:** Atomic, conventional format
3. **Tests:** All new code must have tests
4. **Coverage:** Do not decrease coverage
5. **CI:** All workflows must pass (lint, type, test, conventional-commits)
6. **Description:** Clear problem, approach, tradeoffs

## Code Style

- Python 3.12+
- Type hints required on public APIs
- Minimal docstrings (one-line unless complex)
- No `TODO` comments without issue reference
- Pydantic v2 for configs

## Testing

Tests should cover:
- Core data pipeline logic
- Training configuration validation
- Integration points (vLLM/SGLang)
- Ablation setup

Use `pytest` fixtures for common setup. GPU-intensive tests marked with `@pytest.mark.slow`.

## Hardware

- Local development: CPU tests pass without GPU
- Full training/eval: RunPod H100 spot ($30-50/hr)
- All results committed to `results/` with hardware metadata

## Release

Releases follow [semantic versioning](https://semver.org/). Tag on `main` after phase completion.

## Phases

Follow the 6-week phase plan in project CLAUDE.md:
- W1: Data pipeline
- W2-3: Training
- W4-5: Integration + profiling
- W6: Analysis + HF release

## Questions?

Open an issue or discussion on GitHub.
