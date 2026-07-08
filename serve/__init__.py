"""vLLM + SGLang integration with the trained EAGLE-3 draft head.

Phase 4 deliverable: load `results/train/tri_layer/<seed>/checkpoint-*` head
into vLLM >= 0.10 (--speculative-config) and SGLang (--speculative-algorithm),
measure baseline vs speculative ITL, capture Nsight Systems traces.
"""
