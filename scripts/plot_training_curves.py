"""Training curves for the three students, from the run logs and the paper_docs extracts.

    python scripts/plot_training_curves.py [OUT.png]

Three small multiples (three different measures never share an axis): held-out loss in nats per paper
(comparable across tokenizers), validation ROUGE-L, and training loss per target token. The x axis is
papers seen, logarithmic, because everything interesting happens in the first few thousand.
Colours are the first three slots of the validated categorical palette (blue, orange, aqua); aqua is
below 3:1 on the surface, so every line carries a direct end label as well as the legend.
"""
from __future__ import annotations

import csv
import json
import statistics as st
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, NullLocator  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
COLOR = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"}
NAME = {"A": "A  Qwen3-4B, 70k papers", "B": "B  K2-Horizon-7B, 70k papers", "C": "C  Qwen3.5-9B, 20k papers"}
RUN = {"A": "main", "B": "k2", "C": "q35"}
# 50k is left unlabelled: on a log axis it sits a hair from 70k and the two labels run together
XTICKS = [500, 1000, 2500, 5000, 10000, 20000, 70000]
XLAB = ["500", "1k", "2.5k", "5k", "10k", "20k", "70k"]
MILESTONES = {500, 1000, 2500, 5000, 10000, 20000, 50000, 70000}


def held_out() -> dict[str, list[tuple[int, float]]]:
    out: dict[str, list[tuple[int, float]]] = {k: [] for k in COLOR}
    with open("paper_docs/data/val_loss.csv") as f:
        for r in csv.DictReader(f):
            n = int(r["examples"])
            if n > 0:
                out[r["student"]].append((n, float(r["nats_per_paper"])))
    return {k: sorted(v) for k, v in out.items()}


def rouge() -> dict[str, list[tuple[int, float]]]:
    out: dict[str, list[tuple[int, float]]] = {k: [] for k in COLOR}
    with open("paper_docs/data/evaluations.csv") as f:
        for r in csv.DictReader(f):
            n = int(r["papers_seen"] or 0)
            if r["split"] == "val" and n > 0:
                out[r["student"]].append((n, float(r["all_rougeL"])))
    return {k: sorted(v) for k, v in out.items()}


def train_loss(start: int = 400, bins: int = 20) -> dict[str, list[tuple[float, float]]]:
    out = {}
    for k, run in RUN.items():
        with open(f"runs/{run}/metrics.jsonl") as f:
            pts = [(r["examples"], r["loss"]) for r in map(json.loads, f) if r.get("kind") == "train"]
        top = max(e for e, _ in pts)
        edges = [start * (top / start) ** (i / bins) for i in range(bins + 1)]
        row = []
        for a, b in zip(edges, edges[1:], strict=False):
            sel = [(e, v) for e, v in pts if a < e <= b]
            if sel:
                row.append((st.fmean(e for e, _ in sel), st.fmean(v for _, v in sel)))
        out[k] = row
    return out


def panel(ax, title, sub, series, fmt, ylim, markers=True, marker_at=None):
    ax.set_facecolor(SURFACE)
    for k, pts in series.items():
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        ax.plot(xs, ys, color=COLOR[k], lw=2.0, solid_capstyle="round", solid_joinstyle="round", zorder=3)
        if markers:   # at the checkpoints only: held-out loss is logged more often than that
            keep = [i for i, x in enumerate(xs[:-1]) if marker_at is None or x in marker_at]
            ax.plot([xs[i] for i in keep], [ys[i] for i in keep], "o", color=COLOR[k], ms=5.5, mec=SURFACE, mew=1.5, zorder=4)
        ax.plot(xs[-1], ys[-1], "o", color=COLOR[k], ms=8, mec=SURFACE, mew=2, zorder=5)
        ax.annotate(f"{k}  {fmt.format(ys[-1])}", (xs[-1], ys[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=9.5, color=INK2, annotation_clip=False)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator(XTICKS))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticks(XTICKS, XLAB)
    ax.set_xlim(380, 120000)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", color=GRID, lw=1, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=0, pad=6)
    ax.set_title(title, loc="left", fontsize=12, color=INK, fontweight="semibold", pad=24)
    ax.text(0, 1.045, sub, transform=ax.transAxes, fontsize=9.5, color=INK2, va="bottom")
    ax.set_xlabel("papers seen in training", fontsize=9.5, color=INK2, labelpad=8)


def main(out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4), facecolor=SURFACE)
    panel(axes[0], "Held-out loss", "nats per paper, 200 fixed validation papers; lower is better",
          held_out(), "{:.0f}", (420, 740), marker_at=MILESTONES)
    panel(axes[1], "Validation ROUGE-L", "against the teacher, 1,000 fixed papers; higher is better",
          rouge(), "{:.3f}", (0.405, 0.487))
    panel(axes[2], "Training loss", "per target token, from 400 papers on; compare shapes, not levels",
          train_loss(), "{:.2f}", (0.43, 0.84), markers=False)
    fig.legend([Line2D([0], [0], color=COLOR[k], lw=2.0, marker="o", ms=7, mec=SURFACE, mew=1.5) for k in COLOR],
               [NAME[k] for k in COLOR], loc="upper center", ncol=3, frameon=False, fontsize=10.5,
               labelcolor=INK, bbox_to_anchor=(0.5, 1.0), handlelength=2.2, columnspacing=3)
    fig.text(0.01, 0.012, "One seed per run. The final point of each run also carries the learning-rate decay. "
             "Untrained held-out loss: A 1,323 · B 644 · C 700 (not shown).",
             fontsize=9, color=INK2, ha="left")
    fig.subplots_adjust(left=0.05, right=0.955, top=0.76, bottom=0.17, wspace=0.34)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main(Path(sys.argv[1] if len(sys.argv) > 1 else "reports/curves/training_curves_ABC.png"))
