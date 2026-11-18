"""Pareto-front plots of quality against cost, and a decision table, from ``collect_results.py``'s output.

    python scripts/analysis/pareto.py reports/analysis/results.json --out-dir reports/analysis/pareto \\
        [--recall-floor -0.02] [--require-criteria] [--max-unsupported 0.05] [--max-e2e-p95 120] \\
        [--load 100000] [--utilisation 0.7]

Figures (PNG and SVG), one per cost axis, for each quality measure:

* quality: fact recall (higher is better) and unsupported share (lower is better);
* cost: inference GPU-seconds per paper, p95 single-request latency, serving memory (weights + the KV cache
  of a p95 paper), parameters, and training GPU-hours (SFT + one preference run).

A point is **Pareto-optimal** (ringed, joined by the dashed step line) when no other system is at least as
good on the quality measure and at least as cheap, and strictly better on one. Each student has its own
colour and marker and a direct label; its SFT model is hollow, its preference-tuned model filled, joined by
an arrow (what the prefcc stage bought and cost). Anchors (the old students, e.g. C and prefcc) are grey.
Vertical bars are 95% intervals. Fronts use means: points whose intervals overlap the front are not
meaningfully worse, which the decision table states.

The decision table applies an explicit, adjustable rule; the choice stays with the team:

1. **Quality floor**: recall - teacher >= ``--recall-floor`` (default -0.02); with ``--require-criteria``, a
   preference-tuned system must also pass the four criteria against its own SFT model; with
   ``--max-unsupported``, an unsupported share at most that.
2. **Latency cap** (optional): p95 end-to-end latency per paper <= ``--max-e2e-p95`` seconds.
3. **Cost at production load**: GPUs needed = load x GPU-seconds per paper / 86,400 / utilisation, for
   ``--load`` papers per day. Eligible systems are ranked by it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_lib import (  # noqa: E402
    INK,
    INK2,
    MUTED,
    SURFACE,
    fmt,
    md_table,
    pareto_front,
    save_figure,
    setup_matplotlib,
    student_style,
)

COSTS = {
    "gpu_s_per_paper": "inference GPU-seconds per paper (full batching)",
    "e2e_p95_s": "p95 latency per paper, single request (s)",
    "serve_mem_gb": "serving memory: weights + KV cache of a p95 paper (GB)",
    "params_b": "parameters (billions)",
    "train_gpu_hours": "training GPU-hours (SFT + one preference run)",
}
QUALITY = {"recall": ("fact recall", True), "unsupported_share": ("unsupported share of statements", False)}


def points_from(res: dict) -> list[dict]:
    pts = []
    for s in res["systems"]:
        f = dict(s["flat"])
        w, kv = f.get("serve_weights_gb"), f.get("kv_gb_per_paper_p95")
        f["serve_mem_gb"] = (w + (kv or 0)) if w else None
        pts.append(f)
    return pts


def _students(pts: list[dict]) -> list[str]:
    seen: list[str] = []
    for p in pts:
        if not p.get("anchor") and p["student"] not in seen:
            seen.append(p["student"])
    return seen


def plot_panel(plt, pts: list[dict], quality: str, cost: str, teacher: float | None, title_note: str = ""):
    label, higher = QUALITY[quality]
    have = [p for p in pts if p.get(quality) is not None and p.get(cost) is not None]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    if not have:
        ax.text(0.5, 0.5, f"no system has both {quality} and {cost}", ha="center", va="center",
                transform=ax.transAxes, color=INK2)
        ax.set_axis_off()
        return fig
    order = _students(pts)
    xs = [p[cost] for p in have]
    if min(xs) > 0 and max(xs) / min(xs) > 12:
        ax.set_xscale("log")
    # SFT -> preference arrows, per student
    by_student: dict[str, dict[str, dict]] = {}
    for p in have:
        by_student.setdefault(p["student"], {})[p.get("stage") or ""] = p
    for stu, stages in by_student.items():
        a, b = stages.get("sft"), stages.get("prefcc")
        if a and b:
            col = MUTED if a.get("anchor") else student_style(stu, order)["color"]
            ax.annotate("", xy=(b[cost], b[quality]), xytext=(a[cost], a[quality]),
                        arrowprops={"arrowstyle": "-|>", "color": col, "lw": 1.2, "shrinkA": 6, "shrinkB": 6,
                                    "alpha": 0.8}, zorder=2)
    # intervals and points
    lo_k, hi_k = ("recall_lo", "recall_hi") if quality == "recall" else ("unsupported_lo", "unsupported_hi")
    for p in have:
        sty = {"color": MUTED, "marker": "o"} if p.get("anchor") else student_style(p["student"], order)
        if p.get(lo_k) is not None and p.get(hi_k) is not None:
            ax.plot([p[cost]] * 2, [p[lo_k], p[hi_k]], color=sty["color"], lw=1.2, alpha=0.6, zorder=2)
        filled = p.get("stage") != "sft"
        ax.plot(p[cost], p[quality], sty["marker"], ms=8.5, mew=1.8, color=sty["color"],
                mfc=sty["color"] if filled else SURFACE, mec=sty["color"], zorder=4)
    # Pareto front
    front = pareto_front(have, quality, cost, higher_is_better=higher)
    fx, fy = [have[i][cost] for i in front], [have[i][quality] for i in front]
    if len(front) > 1:
        ax.step(fx, fy, where="post", color=INK2, lw=1.2, ls="--", zorder=1)
    ax.plot(fx, fy, "o", ms=15, mfc="none", mec=INK, mew=1.2, zorder=3, label="Pareto-optimal")
    # labels
    for p in have:
        ax.annotate(p["name"], (p[cost], p[quality]), xytext=(7, 5), textcoords="offset points", fontsize=8,
                    color=INK2 if not p.get("anchor") else MUTED, zorder=5)
    if teacher is not None:
        ax.axhline(teacher, color=INK2, lw=1.0, ls=":", zorder=1)
        ax.annotate(f"teacher {teacher:.3f}", (1.0, teacher), xycoords=("axes fraction", "data"),
                    xytext=(-4, 4), textcoords="offset points", ha="right", fontsize=8, color=INK2)
    better = "higher is better" if higher else "lower is better"
    ax.set_ylabel(f"{label} ({better})")
    ax.set_xlabel(COSTS[cost] + " — lower is better")
    ax.set_title(f"{label.capitalize()} against {COSTS[cost].split(' (')[0]}", loc="left")
    # legend: stage encoding + front (identity is carried by the direct labels)
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], marker="o", ls="", mfc=SURFACE, mec=INK2, mew=1.6, ms=7, label="SFT"),
               Line2D([], [], marker="o", ls="", mfc=INK2, mec=INK2, ms=7, label="preference-tuned"),
               Line2D([], [], marker="o", ls="", mfc="none", mec=INK, ms=12, label="Pareto-optimal")]
    ax.legend(handles=handles, loc="lower right" if higher else "upper right", fontsize=8)
    if title_note:
        fig.text(0.01, -0.02, title_note, fontsize=7.5, color=INK2, ha="left", va="top", wrap=True)
    fig.tight_layout()
    return fig


def gpus_needed(gpu_s_per_paper: float | None, load: float, utilisation: float) -> float | None:
    if gpu_s_per_paper is None:
        return None
    return load * gpu_s_per_paper / 86400 / utilisation


def decide(pts: list[dict], recall_floor: float, require_criteria: bool, max_unsupported: float | None,
           max_e2e_p95: float | None, load: float, utilisation: float) -> tuple[list[dict], dict | None]:
    rows = []
    for p in pts:
        checks = {}
        rvt = p.get("recall_vs_teacher")
        checks["recall floor"] = None if rvt is None else rvt >= recall_floor
        if require_criteria and p.get("stage") not in ("sft", "", None):
            checks["four criteria"] = p.get("criteria_all_four")
        if max_unsupported is not None:
            u = p.get("unsupported_share")
            checks["unsupported cap"] = None if u is None else u <= max_unsupported
        if max_e2e_p95 is not None:
            e = p.get("e2e_p95_s")
            checks["latency cap"] = None if e is None else e <= max_e2e_p95
        g = gpus_needed(p.get("gpu_s_per_paper"), load, utilisation)
        eligible = all(v is True for v in checks.values()) and g is not None and not p.get("anchor")
        rows.append({**p, "checks": checks, "gpus_at_load": g, "eligible": eligible})
    ranked = sorted([r for r in rows if r["eligible"]], key=lambda r: (r["gpus_at_load"], -(r.get("recall") or 0)))
    return rows, (ranked[0] if ranked else None)


def decision_markdown(rows: list[dict], pick: dict | None, args: argparse.Namespace, front_names: list[str]) -> str:
    def yn(v):
        return "–" if v is None else ("yes" if v else "no")
    checks = sorted({k for r in rows for k in r["checks"]})
    body = []
    for r in sorted(rows, key=lambda r: (not r["eligible"], r["gpus_at_load"] if r["gpus_at_load"] is not None else math.inf)):
        body.append([r["name"], fmt(r.get("recall"), 4), fmt(r.get("recall_vs_teacher"), 4, signed=True),
                     fmt(r.get("unsupported_share"), 4), fmt(r.get("gpu_s_per_paper"), 2), fmt(r.get("e2e_p95_s"), 1),
                     fmt(r["gpus_at_load"], 2), *[yn(r["checks"].get(c)) for c in checks],
                     "yes" if r["name"] in front_names else "", "**yes**" if r["eligible"] else "no"])
    rule = [f"- quality floor: recall − teacher ≥ {args.recall_floor:+.3f}"]
    if args.require_criteria:
        rule.append("- a preference-tuned system must pass the four criteria against its own SFT model")
    if args.max_unsupported is not None:
        rule.append(f"- unsupported share ≤ {args.max_unsupported}")
    if args.max_e2e_p95 is not None:
        rule.append(f"- p95 latency per paper ≤ {args.max_e2e_p95} s")
    rule.append(f"- then the fewest GPUs at {args.load:,.0f} papers/day and {args.utilisation:.0%} utilisation "
                "(GPUs = load × GPU-s per paper / 86,400 / utilisation); anchors are never picked")
    out = ["# Decision table", "", "Rule (adjustable on the command line):", "", *rule, "",
           md_table(["system", "recall", "− teacher", "unsupported", "GPU-s / paper", "p95 latency s", "GPUs at load",
                     *checks, "Pareto (recall, GPU-s)", "eligible"], body), ""]
    if pick:
        out.append(f"**Cheapest system that meets the rule: {pick['name']}** "
                   f"({fmt(pick['gpus_at_load'], 2)} GPUs at the stated load, recall {fmt(pick.get('recall'), 4)}).")
    else:
        out.append("**No system meets the rule** (or the measurements it needs are missing).")
    out += ["", "This table applies a rule to measurements; it is an input to the team's decision, not the "
            "decision. Check the intervals: a system whose recall interval overlaps the pick's is not "
            "measurably worse, and serving costs are only comparable when measured on the same hardware "
            "(see the cost-source column of results.md)."]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", help="results.json from collect_results.py")
    ap.add_argument("--out-dir", default="reports/analysis/pareto")
    ap.add_argument("--costs", default=",".join(COSTS), help="comma list of cost axes")
    ap.add_argument("--recall-floor", type=float, default=-0.02, help="minimum recall - teacher")
    ap.add_argument("--require-criteria", action="store_true")
    ap.add_argument("--max-unsupported", type=float, default=None)
    ap.add_argument("--max-e2e-p95", type=float, default=None)
    ap.add_argument("--load", type=float, default=100_000, help="production load, papers per day")
    ap.add_argument("--utilisation", type=float, default=0.7)
    ap.add_argument("--note", default="", help="text printed under every figure (e.g. hardware caveats)")
    a = ap.parse_args(argv)
    res = json.loads(Path(a.results).read_text())
    pts = points_from(res)
    teacher = ((res.get("teacher") or {}).get("recall") or {}).get("mean_ci", [None])[0]
    tu = ((res.get("teacher") or {}).get("unsupported_share") or {}).get("mean_ci", [None])[0]
    gpu_types = sorted({str(p.get("serve_gpu_type")) for p in pts if p.get("serve_gpu_type")})
    note = a.note or (f"Serving costs measured on different GPUs ({', '.join(gpu_types)}): not like-for-like."
                      if len(gpu_types) > 1 else "")
    plt = setup_matplotlib()
    out_dir = Path(a.out_dir)
    written = []
    for cost in [c for c in a.costs.split(",") if c]:
        if cost not in COSTS:
            raise SystemExit(f"unknown cost axis {cost!r}; choose from {', '.join(COSTS)}")
        for q in QUALITY:
            fig = plot_panel(plt, pts, q, cost, teacher if q == "recall" else tu, note)
            written += save_figure(fig, out_dir, f"pareto_{q}_vs_{cost}")
            plt.close(fig)
    front = pareto_front(pts, "recall", "gpu_s_per_paper")
    rows, pick = decide(pts, a.recall_floor, a.require_criteria, a.max_unsupported, a.max_e2e_p95, a.load,
                        a.utilisation)
    md = decision_markdown(rows, pick, a, [pts[i]["name"] for i in front])
    (out_dir / "decision.md").write_text(md)
    (out_dir / "decision.json").write_text(json.dumps({"pick": pick["name"] if pick else None,
                                                       "front_recall_vs_gpu_s": [pts[i]["name"] for i in front],
                                                       "rows": rows,
                                                       "rule": {k: v for k, v in vars(a).items()
                                                                if k not in ("results", "out_dir")}},
                                                      indent=1, default=str))
    written += [out_dir / "decision.md", out_dir / "decision.json"]
    for p in written:
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
