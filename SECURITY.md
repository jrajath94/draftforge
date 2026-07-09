# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 0.1.x   | :white_check_mark: |
| < 0.1   | :x:                |

DraftForge v0.1 is the first tagged release. Pre-0.1 development commits are
not security-supported — they live on `main` for reproducibility and may be
force-pushed.

## Reporting a Vulnerability

**Please do not file a public GitHub issue for security vulnerabilities.**

Report privately via one of the following channels:

- **Email:** `rajath@example.com` (PGP key on request)
- **GitHub:** Open a private security advisory at
  `https://github.com/anthropic-research/draftforge/security/advisories/new`

Include the following:

1. A clear description of the vulnerability and its impact
2. Reproduction steps (commands, config snippets, or a minimal harness)
3. Affected version(s) and commit hash(es)
4. Any known mitigations or workarounds you have already tried

## Response Timeline

- **Acknowledgement:** within 3 business days
- **Triage + severity assessment:** within 7 business days
- **Patch for critical issues:** within 30 days
- **Patch for high/medium:** within 90 days

We follow a 90-day disclosure window. If a fix is not ready by day 90 we will
coordinate a public disclosure date with the reporter.

## Scope

The following are in scope:

- Code that executes on a user machine or pod (data pipeline, training driver,
  release CLI)
- Shell scripts shipped under `scripts/` (especially `onboard_pod.sh` and
  `run_full_pipeline.sh` — these run on rented compute with network access)
- HuggingFace model card generation logic
- Sample fixtures under `data/fixtures/` — these must NEVER contain real
  secrets or real user data; we publish only synthetic content

Out of scope:

- Issues in upstream dependencies (`transformers`, `datasets`, `pydantic`,
  `vllm`, `sglang`, etc.) — please report those to the relevant upstream
  project
- Theoretical attacks that require physical access to the user's machine
- Social engineering

## Hardening Already in Place

- Cross-project GPU preflight refuses to start training if another project
  holds >50% of GPU memory (`scripts/onboard_pod.sh`)
- HF auth preflight gates the pod onboarding script
- `_results_path_check` guard in `data/sources/finance.py` refuses to load
  synthetic test fixtures into `results/` directories (domain-shift bias
  prevention)
- `.gitignore` excludes `.env`, `results/*`, `artifacts/`, model weights
  (`*.pt`, `*.safetensors`, `*.bin`, `*.pkl`)
- All commits signed via conventional-commits lint in CI; no surprise
  `chore:` that may hide a credential

## Recognition

We credit reporters in the release notes (unless you prefer to remain
anonymous). Thank you for keeping DraftForge and its users safe.