"""Release: HuggingFace card, bench wrapper, writeup scaffolding, aggregation.

Phase 6 deliverable. Reads results from train/, ablate/, serve/, eval/ and
emits the artifacts needed to publish the draft head:

- HuggingFace model card (`hf_card.md` template, populated by `make_card.py`)
- Bench shell driver (`bench.sh` — one-command reproduction)
- Writeup skeleton (`writeup_template.md` — 1.5K-word analysis framework)
- Aggregator (`aggregate.py` — collect loss curves, ablations, accept grid)

Pure CPU. All numbers must come from prior committed GPU runs.
"""
