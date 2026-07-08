# serve/ — vLLM + SGLang integration

Phase 4 deliverable: ship both runtimes loading the trained draft head.

## Quickstart (user-rented GPU)

```bash
# 1. Build per-runtime serve invocation
python -m serve.integrate \
    --target Qwen/Qwen3-14B \
    --draft results/train/tri_layer/42/checkpoint-2000 \
    --runtime vllm \
    --num-spec 4 \
    --out results/serve/vllm.sh

python -m serve.integrate \
    --target Qwen/Qwen3-14B \
    --draft results/train/tri_layer/42/checkpoint-2000 \
    --runtime sglang \
    --num-spec 4 \
    --out results/serve/sglang.sh

# 2. Run benchmarks (baseline first, then with spec)
bash results/serve/vllm.sh &
vllm bench latency --model Qwen/Qwen3-14B --save-json results/serve/vllm/baseline.json

# 3. Capture Nsight trace
nsys profile --trace=cuda,nvtx --output=results/serve/nsys/spec \
    -- python -m sglang.launch_server --model-path ... --speculative-algorithm EAGLE3 ...
```

## CLI flags fixed at run time

- vLLM `--speculative-config` JSON requires `method: "eagle3"` (not "eagle")
- SGLang `--speculative-algorithm EAGLE3` (uppercase) and
  `--speculative-draft-model-path` (no --draft flag)
- vLLM ≥ 0.10.0 required (draft models not supported before)
- SGLang `--speculative-num-steps` (= draft tokens per step)

See `tests/serve/test_integration.py` for runtime invocation shape.
