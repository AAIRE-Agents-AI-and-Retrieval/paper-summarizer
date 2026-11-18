"""The learning curves, drawn from the logs (``scisum curves``).

Four figures, each saved as png and svg under ``reports/curves/<run>/``:

* ``loss`` — training loss (raw and exponentially smoothed) and held-out loss against examples
  seen, with the milestones marked. The one plot that shows whether the run is healthy.
* ``quality`` — the generation metrics at every milestone: ROUGE-L, BERTScore, NLI faithfulness,
  keyword F1. The x axis is examples seen, so this *is* the data-scaling curve the milestones were
  chosen to measure; the base model sits at x = 0 (plotted at the left edge on a log axis).
* ``schema`` — JSON validity, schema compliance, the schema's count ranges and the share of
  generations that stopped on EOS instead of the token limit.
* ``throughput`` — tokens per second, peak VRAM, learning rate and gradient norm, for diagnosing
  the run rather than the model.

The curves are redrawn after every evaluation (the evaluation worker calls this), so the figures on
disk always match the logs.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

# /home is read-only on the compute nodes and not writable from here either; matplotlib must not
# try to build its font cache there.
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent.parent / ".cache" / "matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scisum.config import Settings  # noqa: E402

log = logging.getLogger(__name__)

plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 150, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.constrained_layout.use": True})


def read_metrics(s: Settings, run: str) -> list[dict]:
    p = s.run_dir(run) / "metrics.jsonl"
    if not p.exists():
        return []
    out = []
    with p.open() as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:  # a line half-written when the job was killed
                continue
    return out


def read_evals(s: Settings, run: str, split: str = "val") -> list[dict]:
    """Every evaluation result for this run plus the base model, ordered by examples seen."""
    results = []
    for directory in ({s.run_dir(run) / "eval", s.run_dir("base") / "eval"}):
        for p in sorted(directory.glob(f"{split}-*.json")):
            if p.name.endswith(".gen.json"):
                continue
            r = json.loads(p.read_text())
            judge = Path(str(p).replace(".json", ".judge.jsonl.gz"))
            if judge.exists():  # the judge runs after scoring, so it is merged in on reading
                from scisum.judge import summarise

                agg = summarise(judge, r.get("name", ""), run, r.get("ckpt", ""), split, r.get("papers", 0))
                r["judge"] = agg
                r["metrics"].update({k: v for k, v in agg.items() if k.startswith("judge_")})
            ck = r.get("ckpt", "")
            r["examples"] = (r.get("checkpoint") or {}).get("examples") or (
                int(ck[2:]) if ck.startswith("ex") and ck[2:].isdigit() else 0)
            results.append(r)
    results.sort(key=lambda r: (r["examples"], r.get("papers", 0)))
    return results


def _ema(ys: list[float], alpha: float = 0.05) -> list[float]:
    out: list[float] = []
    acc = None
    for y in ys:
        acc = y if acc is None else (1 - alpha) * acc + alpha * y
        out.append(acc)
    return out


def _save(fig, out_dir: Path, name: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "svg"):
        p = out_dir / f"{name}.{ext}"
        fig.savefig(p)
        paths.append(p)
    plt.close(fig)
    return paths


def _milestones(ax, s: Settings, limit: int | None = None) -> None:
    for m in s.milestones:
        if limit is None or m <= limit:
            ax.axvline(m, color="0.75", lw=0.7, ls=":", zorder=0)


def draw_loss(s: Settings, run: str, out_dir: Path) -> list[Path]:
    rows = read_metrics(s, run)
    tr = [r for r in rows if r["kind"] == "train"]
    va = [r for r in rows if r["kind"] == "val_loss"]
    if not tr and not va:
        return []
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    if tr:
        x = [r["examples"] for r in tr]
        y = [r["loss"] for r in tr]
        ax.plot(x, y, lw=0.6, color="#9ecae1", label="training loss (per 5 steps)")
        ax.plot(x, _ema(y), lw=1.8, color="#1f6fb4", label="training loss (smoothed)")
    if va:
        ax.plot([r["examples"] for r in va], [r["loss"] for r in va], "o-", ms=4, lw=1.6,
                color="#d1495b", label=f"held-out loss ({va[-1]['papers']} papers)")
    _milestones(ax, s, max([r["examples"] for r in tr] or [0]) or None)
    ax.set_xlabel("training papers seen")
    ax.set_ylabel("cross-entropy per target token")
    ax.set_title(f"Learning curve — {run} ({s.base_model.split('/')[-1]}, LoRA r{s.lora_r})")
    ax.legend(frameon=False)
    return _save(fig, out_dir, "loss")


_QUALITY = [("all_rougeL", "ROUGE-L (whole summary)"), ("bertscore_mean", "BERTScore F1 (mean of fields)"),
            ("faithfulness", "NLI faithfulness (entailed claims)"), ("keywords_f1", "keyword set F1")]


def _base_x(evals: list[dict]) -> float:
    """Where to draw the untrained model on a log axis: half the first milestone."""
    trained = [e["examples"] for e in evals if e["examples"] > 0]
    return (min(trained) / 2) if trained else 1.0


def draw_quality(s: Settings, run: str, out_dir: Path, split: str = "val") -> list[Path]:
    evals = read_evals(s, run, split)
    if not evals:
        return []
    bx = _base_x(evals)
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.6), sharex=True)
    for ax, (key, title) in zip(axes.ravel(), _QUALITY, strict=True):
        pts = [(e["examples"], e["metrics"][key]) for e in evals if key in e["metrics"]]
        trained = [(x, y) for x, y in pts if x > 0]
        base = [y for x, y in pts if x == 0]
        if not pts:
            ax.set_visible(False)
            continue
        if trained:
            ax.plot([x for x, _ in trained], [y for _, y in trained], "o-", ms=4, color="#1f6fb4")
        if base:
            ax.axhline(base[0], color="#d1495b", ls="--", lw=1, label="base model, no training")
            ax.plot([bx], base[:1], "s", ms=5, mfc="none", color="#d1495b")
            ax.legend(frameon=False, fontsize=7)
        ax.set_xscale("log")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("training papers seen")
    fig.suptitle(f"Quality against training data — {run}, {split} split "
                 f"({evals[-1]['papers']} papers per point)", fontsize=10)
    return _save(fig, out_dir, "quality")


_SCHEMA = [("json_ok", "valid JSON"), ("schema_ok", "all 8 fields, right types"),
           ("keywords_in_range", "5-10 keywords"), ("contributions_in_range", "2-5 contributions"),
           ("stopped_on_eos", "stopped on EOS"), ("repaired", "needed repair")]


def draw_schema(s: Settings, run: str, out_dir: Path, split: str = "val") -> list[Path]:
    evals = read_evals(s, run, split)
    if not evals:
        return []
    bx = _base_x(evals)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for key, label in _SCHEMA:
        pts = [(e["examples"] or bx, 100 * e["metrics"][key]) for e in evals if key in e["metrics"]]
        if pts:
            line, = ax.plot([x for x, _ in pts if x > bx], [y for x, y in pts if x > bx],
                            "o-", ms=3.5, lw=1.3, label=label)
            base = [(x, y) for x, y in pts if x == bx]
            if base:  # the untrained model, drawn at half the first milestone
                ax.plot([base[0][0]], [base[0][1]], "s", ms=5, mfc="none", color=line.get_color())
    ax.set_xscale("log")
    ax.set_ylim(-2, 102)
    ax.set_xlabel("training papers seen (hollow square: the untrained base model)")
    ax.set_ylabel("% of generated summaries")
    ax.set_title(f"Structured-output compliance — {run}, {split} split")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    return _save(fig, out_dir, "schema")


def draw_throughput(s: Settings, run: str, out_dir: Path) -> list[Path]:
    tr = [r for r in read_metrics(s, run) if r["kind"] == "train"]
    if not tr:
        return []
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.2), sharex=True)
    x = [r["examples"] for r in tr]
    for ax, (key, label) in zip(axes.ravel(), [("tokens_per_s", "tokens / second"),
                                               ("vram_gb", "peak VRAM (GiB)"),
                                               ("lr", "learning rate"),
                                               ("grad_norm", "gradient norm")], strict=True):
        ys = [r.get(key) for r in tr]
        ax.plot(x, [y if y is not None else float("nan") for y in ys], lw=0.9, color="#1f6fb4")
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("training papers seen")
        if key == "grad_norm":
            ax.set_yscale("log")
    fig.suptitle(f"Run health — {run}", fontsize=10)
    return _save(fig, out_dir, "throughput")


def draw_all(s: Settings, run: str, out_dir: Path | None = None, split: str = "val") -> list[Path]:
    out = Path(out_dir) if out_dir else (s.reports_dir / "curves" / run)
    paths: list[Path] = []
    for fn in (draw_loss, draw_throughput):
        paths += fn(s, run, out)
    for fn in (draw_quality, draw_schema):
        paths += fn(s, run, out, split)
    log.info("curves: %d files in %s", len(paths), out)
    return paths
