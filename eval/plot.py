"""Acceptance-length curves and ITL plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_acceptance_by_batch(
    rows: list[dict], out_path: Path, title: str = ""
) -> None:
    """Plot acceptance length vs batch size, hue=domain, subplots=temperature."""
    domains = sorted({r["domain"] for r in rows})
    temps = sorted({r["temperature"] for r in rows})

    fig, axes = plt.subplots(1, len(temps), figsize=(5 * len(temps), 4), sharey=True)
    if len(temps) == 1:
        axes = [axes]
    for ax, t in zip(axes, temps, strict=True):
        for d in domains:
            xs = [
                float(r["batch_size"])
                for r in rows
                if r["domain"] == d and float(r["temperature"]) == t
            ]
            ys = [
                float(r["eal"])
                for r in rows
                if r["domain"] == d and float(r["temperature"]) == t
            ]
            ax.plot(xs, ys, "-o", label=d)
        ax.set_title(f"T={t}")
        ax.set_xlabel("batch size")
        ax.set_ylabel("expected acceptance length (tokens)")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_itl_reduction(
    base_itl: dict[tuple[str, str, int], float],
    spec_itl: dict[tuple[str, str, int], float],
    out_path: Path,
) -> None:
    """Bar plot of ITL reduction % vs batch size; hue=domain x temperature."""
    keys = sorted(set(base_itl) & set(spec_itl))
    if not keys:
        return
    bs = sorted({k[2] for k in keys})
    fig, ax = plt.subplots(figsize=(8, 4))
    width = 0.18
    offsets_per_label: dict[str, int] = {}
    next_offset = [0]
    for k in keys:
        d, t, b = k
        base = base_itl[k]
        spec = spec_itl[k]
        pct = (1.0 - spec / base) * 100.0 if base > 0 else 0.0
        label = f"{d}/{t}"
        if label not in offsets_per_label:
            offsets_per_label[label] = next_offset[0]
            next_offset[0] += 1
        x = bs.index(b) + offsets_per_label[label] * width
        ax.bar(x, pct, width=width, label=label)
    ax.set_xticks([i + 0.5 for i in range(len(bs))])
    ax.set_xticklabels([str(b) for b in bs])
    ax.set_xlabel("batch size")
    ax.set_ylabel("ITL reduction (%)")
    ax.set_title("Spec-decode ITL reduction vs baseline")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
