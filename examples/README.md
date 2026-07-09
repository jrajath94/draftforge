# DraftForge examples

Runnable quickstart snippets that exercise the public API. Each runs on CPU
in a few seconds. Use them as a starting point for your own experiments.

| File | What it does | CPU | GPU | HF auth |
|------|--------------|-----|-----|---------|
| `quickstart_acceptance.py` | Geometric EAL + crossover calculation from a synthetic grid | ✓ | - | - |
| `quickstart_serve.py` | Render a vLLM invocation for a trained head | ✓ | - | - |
| `quickstart_data.py` | Inspect the data config without running the pipeline | ✓ | - | - |

## Usage

```bash
# All three work on CPU
.venv/bin/python examples/quickstart_acceptance.py
.venv/bin/python examples/quickstart_serve.py
.venv/bin/python examples/quickstart_data.py
```

## Why these examples

- **`quickstart_acceptance.py`** is the most useful. It walks through the
  geometric acceptance model and the crossover batch-size calculation with
  numbers you can adjust. Use it to sanity-check expected values before
  spending GPU time on a real grid sweep.
- **`quickstart_serve.py`** renders the vLLM command DraftForge would
  launch for a checkpoint. Useful for verifying `--speculative-config` JSON
  shape before going to a GPU pod.
- **`quickstart_data.py`** shows the data config schema. Run it after
  editing `data/config.yaml` to confirm the YAML is well-formed.

## Not an example

These snippets are intentionally not included:

- **Training a head.** That requires a GPU and the EAGLE-3 training
  pipeline (`train/train_eagle3.py`). See `train/README.md` and
  `make bench`.
- **Running a vLLM/SGLang bench.** That requires a GPU with the runtime
  installed. See `serve/README.md` and `serve/bench.sh`.
- **Uploading to HuggingFace.** See `scripts/upload_hf.sh` and
  `release/README.md`.
