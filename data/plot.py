"""Domain distribution plot: stacked bar per split, general vs finance."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_domain_distribution(
    train_counts: dict[str, int],
    val_counts: dict[str, int],
    test_counts: dict[str, int],
    out_path: Path,
) -> None:
    """Bar plot: x = split, stacked by domain. Saves PNG."""
    splits = ["train", "val", "test"]
    series = [train_counts, val_counts, test_counts]
    domains = sorted({d for s in series for d in s})
    if not domains:
        domains = ["general"]

    x = np.arange(len(splits))
    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = np.zeros(len(splits))
    width = 0.6
    for dom in domains:
        vals = np.array([s.get(dom, 0) for s in series], dtype=float)
        ax.bar(x, vals, width, label=dom, bottom=bottom)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(splits)
    ax.set_ylabel("count")
    ax.set_title("DraftForge: domain distribution per split")
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
