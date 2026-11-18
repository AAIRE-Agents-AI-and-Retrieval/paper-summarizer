"""Learning curves of every run: one figure per measure with all students overlaid, and one page per run.

    python scripts/analysis/learning_curves.py RUN [RUN ...] [--runs-dir runs] [--out-dir reports/curves] \\
        [--label RUN=TEXT ...]

Reads ``runs/<run>/metrics.jsonl`` and the milestone evaluations ``runs/<run>/eval/val-ex<N>-n<M>.json``.
SFT runs (``scisum.train``) give: training loss per target token (geometric bins over papers seen),
held-out loss in nats per paper (comparable across tokenizers, unlike per-token loss), validation ROUGE-L
and schema validity at each milestone, training throughput (tokens/s per GPU) and peak memory.
Preference runs (``scisum.pref_train``) give: DPO loss, likelihood (NLL) term, reward margin, preference
accuracy and chosen-minus-rejected length per step, and the held-out loss on teacher summaries.
The x axis of SFT curves is papers seen, logarithmic, because most of the change happens in the first few
thousand papers. Each measure has its own figure: measures never share an axis.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_lib import INK2, PALETTE, read_json, read_jsonl, save_figure, setup_matplotlib  # noqa: E402

SUFFIX = re.compile(r"-(sft\d+k?|prefcc(-s\d+)?)$")


def student_of(run: str) -> str:
    return SUFFIX.sub("", run)


def binned(pts: list[tuple[float, float]], bins: int = 24, start: float | None = None) -> list[tuple[float, float]]:
    """Means in geometric bins of x (training loss is logged every few steps and is noisy)."""
    pts = [(x, y) for x, y in pts if x and x > 0 and y is not None]
    if len(pts) < 3:
        return pts
    lo, hi = start or min(x for x, _ in pts), max(x for x, _ in pts)
    if hi <= lo:
        return pts
    edges = [lo * (hi / lo) ** (i / bins) for i in range(bins + 1)]
    out = []
    for a, b in zip(edges, edges[1:], strict=False):
        sel = [(x, y) for x, y in pts if a <= x <= b]
        if sel:
            out.append((st.fmean(x for x, _ in sel), st.fmean(y for _, y in sel)))
    return out


def milestone_evals(run_dir: Path, split: str = "val") -> list[dict]:
    rows = []
    for p in sorted((run_dir / "eval").glob(f"{split}-*-n*.json")):
        if p.name.endswith((".gen.json",)):
            continue
        m = re.match(rf"{split}-(ex(\d+)|base)-n(\d+)\.json$", p.name)
        if not m:
            continue
        d = read_json(p) or {}
        met = d.get("metrics") or {}
        rows.append({"examples": int(m.group(2)) if m.group(2) else 0, "rougeL": met.get("all_rougeL"),
                     "schema_ok": met.get("schema_ok"), "bertscore": met.get("bertscore_mean"),
                     "length_ratio": met.get("length_ratio")})
    return sorted(rows, key=lambda r: r["examples"])


def read_run(run_dir: Path) -> dict:
    rec = read_jsonl(run_dir / "metrics.jsonl")
    train = [r for r in rec if r.get("kind") == "train"]
    val = [r for r in rec if r.get("kind") == "val_loss"]
    kind = "preference" if any("dpo" in r for r in train) else "sft"
    return {"run": run_dir.name, "kind": kind, "train": train, "val": val,
            "evals": milestone_evals(run_dir) if kind == "sft" else []}


def colours(runs: list[str]) -> dict[str, str]:
    """One colour per student (a student's SFT and preference runs share it), in order of appearance."""
    studs: list[str] = []
    for r in runs:
        s = student_of(r)
        if s not in studs:
            studs.append(s)
    return {r: PALETTE[studs.index(student_of(r)) % len(PALETTE)] for r in runs}


def _line(ax, xs, ys, color, label, markers=False):
    if not xs:
        return
    ax.plot(xs, ys, color=color, lw=2.0, solid_capstyle="round", label=label, zorder=3)
    if markers:
        ax.plot(xs, ys, "o", color=color, ms=5, mec="#fcfcfb", mew=1.2, zorder=4)
    ax.annotate(label, (xs[-1], ys[-1]), xytext=(5, 0), textcoords="offset points", fontsize=7.5, color=INK2,
                va="center")


SFT_PANELS = {
    "train_loss": ("Training loss per target token", "loss (nats / token)"),
    "heldout_loss": ("Held-out loss on teacher summaries", "nats per paper"),
    "val_rougeL": ("Validation ROUGE-L at each milestone", "ROUGE-L"),
    "val_schema": ("Valid schema at each milestone", "share of summaries"),
    "throughput": ("Training throughput", "tokens/s per GPU"),
    "vram": ("Peak memory (rank 0, PyTorch allocated)", "GB"),
}
PREF_PANELS = {
    "dpo": ("DPO loss", "loss"), "nll": ("Likelihood term (chosen summary)", "nats / token"),
    "margin": ("Reward margin (chosen − rejected)", "margin"), "pref_acc": ("Preference accuracy", "share of pairs"),
    "len_diff": ("Chosen − rejected length in a batch", "tokens"),
    "heldout_loss": ("Held-out loss on teacher summaries", "nats per paper"),
}


def series(d: dict, key: str) -> tuple[list[float], list[float]]:
    """x/y of one measure for one run (SFT: x = papers seen; preference: x = optimiser step)."""
    if d["kind"] == "sft":
        if key == "train_loss":
            pts = binned([(r["examples"], r["loss"]) for r in d["train"]], start=max(1, min(
                (r["examples"] for r in d["train"]), default=1)))
            return [p[0] for p in pts], [p[1] for p in pts]
        if key in ("throughput", "vram"):
            src = "tokens_per_s" if key == "throughput" else "vram_gb"
            pts = binned([(r["examples"], r.get(src)) for r in d["train"] if r.get(src)])
            return [p[0] for p in pts], [p[1] for p in pts]
        if key in ("val_rougeL", "val_schema"):
            k = "rougeL" if key == "val_rougeL" else "schema_ok"
            pts = [(max(e["examples"], 1), e[k]) for e in d["evals"] if e.get(k) is not None and e["examples"] > 0]
            return [p[0] for p in pts], [p[1] for p in pts]
    else:
        if key in PREF_PANELS and key != "heldout_loss":
            pts = [(r["step"], r[key]) for r in d["train"] if r.get(key) is not None]
            return [p[0] for p in pts], [p[1] for p in pts]
    if key == "heldout_loss":
        xk = "examples" if d["kind"] == "sft" else "step"
        pts = [((r.get(xk) or 0), r["loss"] * r["target_tokens"] / r["papers"]) for r in d["val"]
               if r.get("papers") and r.get("target_tokens")]
        if d["kind"] == "sft":
            pts = [(max(x, 1), y) for x, y in pts if x > 0]
        return [p[0] for p in pts], [p[1] for p in pts]
    return [], []


def overview(plt, data: list[dict], key: str, panels: dict, cols: dict, labels: dict, out_dir: Path, kind: str):
    title, ylab = panels[key]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    drawn = 0
    for d in data:
        xs, ys = series(d, key)
        if xs:
            _line(ax, xs, ys, cols[d["run"]], labels.get(d["run"], d["run"]), markers=key.startswith("val_"))
            drawn += 1
    if not drawn:
        plt.close(fig)
        return []
    if kind == "sft":
        ax.set_xscale("log")
        ax.set_xlabel("papers seen (log scale)")
    else:
        ax.set_xlabel("optimiser step")
    ax.set_ylabel(ylab)
    ax.set_title(title, loc="left")
    if drawn > 1:
        ax.legend(loc="best", fontsize=7.5)
    fig.tight_layout()
    paths = save_figure(fig, out_dir, f"curves_{kind}_{key}")
    plt.close(fig)
    return paths


def page(plt, d: dict, color: str, label: str, out_dir: Path):
    panels = SFT_PANELS if d["kind"] == "sft" else PREF_PANELS
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.4))
    for ax, key in zip(axes.flat, panels, strict=False):
        xs, ys = series(d, key)
        t, ylab = panels[key]
        ax.set_title(t, loc="left", fontsize=9.5)
        ax.set_ylabel(ylab, fontsize=8.5)
        if not xs:
            ax.text(0.5, 0.5, "not logged", ha="center", va="center", transform=ax.transAxes, color=INK2)
            continue
        ax.plot(xs, ys, color=color, lw=2.0)
        if key.startswith("val_") or key == "heldout_loss":
            ax.plot(xs, ys, "o", color=color, ms=4.5, mec="#fcfcfb", mew=1.0)
        if d["kind"] == "sft":
            ax.set_xscale("log")
    fig.suptitle(f"{label} — {'SFT' if d['kind'] == 'sft' else 'preference'} run", x=0.01, ha="left",
                 fontweight="bold", fontsize=11)
    fig.supxlabel("papers seen (log scale)" if d["kind"] == "sft" else "optimiser step", fontsize=9)
    fig.tight_layout()
    paths = save_figure(fig, out_dir, f"run_{d['run']}")
    plt.close(fig)
    return paths


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--out-dir", default="reports/curves")
    ap.add_argument("--label", action="append", default=[], help="RUN=TEXT (repeatable)")
    a = ap.parse_args(argv)
    labels = dict(x.split("=", 1) for x in a.label)
    data = [read_run(Path(a.runs_dir) / r) for r in a.runs]
    data = [d for d in data if d["train"] or d["val"]]
    if not data:
        raise SystemExit("no metrics found for any run")
    cols = colours([d["run"] for d in data])
    plt = setup_matplotlib()
    out = Path(a.out_dir)
    written = []
    sft = [d for d in data if d["kind"] == "sft"]
    pref = [d for d in data if d["kind"] == "preference"]
    for key in SFT_PANELS:
        written += overview(plt, sft, key, SFT_PANELS, cols, labels, out, "sft") if sft else []
    for key in PREF_PANELS:
        written += overview(plt, pref, key, PREF_PANELS, cols, labels, out, "pref") if pref else []
    for d in data:
        written += page(plt, d, cols[d["run"]], labels.get(d["run"], d["run"]), out)
    (out / "index.json").write_text(json.dumps({"runs": [d["run"] for d in data],
                                                "figures": [p.name for p in written]}, indent=1))
    for p in written:
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
