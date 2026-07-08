"""Data pipeline for DraftForge.

Pipeline: raw traces → dedup → stratified split → tokenize (Qwen3 BPE).
Designed for offline reproducibility: same seed + same config = same SHA256 splits.
"""
