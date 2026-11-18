"""``scisum report``: every number in one markdown file and one json file.

``reports/REPORT.md`` is written to be read by a person — what was trained, on what, how it scored
at each milestone, how much GPU time it cost — and ``reports/results.json`` holds the same content
for anything downstream. Both are regenerated from the logs, so they can never drift from the run.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from scisum.config import Settings, scrub
from scisum.curves import read_evals, read_metrics
from scisum.dataset import SPLITS
from scisum.datastats import format_table

log = logging.getLogger(__name__)

_CURVE_COLS = [("val loss", "val_loss", "{:.4f}"), ("JSON ok", "json_ok", "{:.3f}"),
               ("schema ok", "schema_ok", "{:.3f}"), ("ROUGE-L", "all_rougeL", "{:.4f}"),
               ("ROUGE-1", "all_rouge1", "{:.4f}"), ("BERTScore", "bertscore_mean", "{:.4f}"),
               ("faithful", "faithfulness", "{:.4f}"), ("keyword F1", "keywords_f1", "{:.4f}"),
               ("EOS stop", "stopped_on_eos", "{:.3f}"), ("out tok", "out_tokens", "{:.0f}"),
               ("judge", "judge_mean", "{:.3f}"), ("judge Δ vs teacher", "judge_delta", "{:+.3f}"),
               ("win/tie vs teacher", "judge_win", "{:.2f}")]


def _table(rows: list[list[str]], header: list[str], align_left: int = 1) -> str:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join([":--" if i < align_left else "--:" for i in range(len(header))]) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def _val_loss_at(metrics: list[dict], examples: int) -> float | None:
    cand = [r for r in metrics if r["kind"] == "val_loss" and r.get("examples", -1) <= examples]
    return cand[-1]["loss"] if cand else None


def collect(s: Settings, run: str) -> dict:
    metrics = read_metrics(s, run)
    state_path = s.run_dir(run) / "state.json"
    run_path = s.run_dir(run) / "run.json"
    train = [r for r in metrics if r["kind"] == "train"]
    chunks = [r for r in metrics if r["kind"] in ("chunk_start", "chunk_end")]
    gpus = sorted({r.get("gpu", "") for r in metrics if r["kind"] == "chunk_start" and r.get("gpu")})
    tps = [r["tokens_per_s"] for r in train if r.get("tokens_per_s")]
    return {
        "run": run, "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "state": json.loads(state_path.read_text()) if state_path.exists() else {},
        "config": scrub(json.loads(run_path.read_text())) if run_path.exists() else {},
        "dataset": json.loads((s.state_dir / "build.json").read_text())
        if (s.state_dir / "build.json").exists() else {},
        "evals": {"val": read_evals(s, run, "val"), "test": read_evals(s, run, "test")},
        "health": {"gpus": gpus, "chunks": len([c for c in chunks if c["kind"] == "chunk_start"]),
                   "tokens_per_s_median": round(sorted(tps)[len(tps) // 2], 1) if tps else None,
                   "vram_peak_gb": max((r.get("vram_gb") or 0) for r in train) if train else None},
        **s.provenance(),
    }


def report_paths(s: Settings, run: str) -> tuple[Path, Path]:
    """(REPORT.md, results.json) for a run. ``main`` keeps the original names; every other run gets
    its own, because one file shared by three runs was overwritten by whichever wrote last - Student
    C's invalid evaluation replaced Run A's report on 2026-10-18."""
    if run == "main":
        return s.reports_dir / "REPORT.md", s.reports_dir / "results.json"
    return s.reports_dir / f"REPORT-{run}.md", s.reports_dir / f"results-{run}.json"


def write_report(s: Settings, run: str = "main") -> list[Path]:
    data = collect(s, run)
    s.reports_dir.mkdir(parents=True, exist_ok=True)
    out_md, out_json = report_paths(s, run)
    out_json.write_text(json.dumps(data, indent=2, default=str))

    st, cfg, ds = data["state"], data["config"], data["dataset"]
    metrics = read_metrics(s, run)
    lines = [f"# Scientific-paper summarisation: {s.base_model.split('/')[-1]} fine-tuned on "
             f"{st.get('examples', 0):,} papers", "",
             f"Generated {data['generated']} from `runs/{run}/` · commit `{data['git_commit']}` · "
             f"base model `{s.base_model}`", "",
             "## Setup", "",
             _table([["base model", s.base_model],
                     ["adapter", f"LoRA r={s.lora_r}, alpha={s.lora_alpha}, dropout={s.lora_dropout}, "
                                 f"on {', '.join(s.lora_targets)}"],
                     ["trainable parameters", f"{cfg.get('params_trainable', 0):,} of "
                                              f"{cfg.get('params_total', 0):,} "
                                              f"({cfg.get('trainable_pct', 0)}%)"],
                     ["sequence length", f"{s.max_seq_len:,} tokens "
                                         f"(prompt ≤ {s.max_prompt_tokens:,}, reserve {s.gen_reserve_tokens:,})"],
                     ["batch", f"{s.grad_accum} papers per optimiser step, one paper per forward"],
                     ["learning rate", f"{s.lr:g}, warmup {s.warmup_frac:.0%}, "
                                       f"stable, cosine decay over the last {s.decay_frac:.0%} "
                                       f"to {s.decay_floor:.0%}"],
                     ["loss", f"cross-entropy on target tokens only ({s.loss_impl}), token-weighted"],
                     ["precision", "bf16 weights, fp32 adapter and loss, SDPA attention"],
                     ["milestones", ", ".join(f"{m:,}" for m in s.milestones)],
                     ["seed", str(s.seed)]], ["setting", "value"], align_left=2), ""]

    if ds:
        lines += ["## Data", "",
                  f"Teacher summaries: {ds['index']['summary_records']:,} papers "
                  f"summarised by {len(ds['index']['teachers'])} models, "
                  f"{ds['index']['eligible']:,} eligible after filtering "
                  f"({ds['index']['dropped_total']:,} dropped: "
                  + ", ".join(f"{k} {v}" for k, v in ds["index"]["drops"].items()) + ").", "",
                  "```", format_table(ds), "```", "",
                  _table([[k, f"{ds['splits'][k]['used']:,}",
                           f"{ds['splits'][k]['total_tokens'].get('sum', 0) / 1e6:.0f}M",
                           f"{ds['splits'][k]['fits_whole_body_pct']:.1f}%",
                           ", ".join(f"{t.split('/')[-1]} {n:,}" for t, n in ds["splits"][k]["teachers"].items())]
                          for k in SPLITS],
                         ["split", "papers", "tokens", "whole body fits", "teachers"]), ""]

    evals = data["evals"]["val"]
    if evals:
        rows = []
        for e in evals:
            m = e["metrics"]
            vl = (e.get("checkpoint") or {}).get("val_loss", {}).get("loss") or _val_loss_at(metrics, e["examples"])
            cells = [f"{e['examples']:,}" if e["examples"] else "0 (base)"]
            for _, key, fmt in _CURVE_COLS:
                v = vl if key == "val_loss" else m.get(key)
                cells.append(fmt.format(v) if isinstance(v, (int, float)) else "–")
            rows.append(cells)
        lines += ["## Learning curve (validation split)", "",
                  f"Each point is one checkpoint of the single run, evaluated by greedy generation on "
                  f"{evals[-1]['papers']} held-out papers "
                  f"(faithfulness on the first {s.faithfulness_n}).", "",
                  _table(rows, ["papers seen"] + [c[0] for c in _CURVE_COLS]), "",
                  f"![learning curve](curves/{run}/loss.png)", "",
                  f"![quality](curves/{run}/quality.png)", "",
                  f"![schema](curves/{run}/schema.png)", ""]

    for e in data["evals"]["test"]:
        m = e["metrics"]
        lines += [f"## Test split — checkpoint `{e['ckpt']}` ({e['papers']:,} papers)", "",
                  _table([[name, fmt.format(m[key]) if key in m else "–"]
                          for name, key, fmt in _CURVE_COLS if key != "val_loss"],
                         ["metric", "value"], align_left=1), ""]
        if e.get("by_teacher"):
            lines += ["By teacher model:", "",
                      _table([[t, f"{v['papers']}", f"{v.get('all_rougeL', 0):.4f}",
                               f"{v.get('bertscore_mean', 0):.4f}", f"{v.get('faithfulness', 0):.4f}"]
                              for t, v in e["by_teacher"].items()],
                             ["teacher", "papers", "ROUGE-L", "BERTScore", "faithfulness"]), ""]
        if e.get("by_truncation"):
            lines += ["By how the paper was fitted into the prompt:", "",
                      _table([[t, f"{v['papers']}", f"{v.get('all_rougeL', 0):.4f}",
                               f"{v.get('faithfulness', 0):.4f}"]
                              for t, v in e["by_truncation"].items()],
                             ["body fitted", "papers", "ROUGE-L", "faithfulness"]), ""]

    h = data["health"]
    lines += ["## Run", "",
              _table([["papers seen", f"{st.get('examples', 0):,} of {s.n_train:,}"],
                      ["optimiser steps", f"{st.get('step', 0):,}"],
                      ["tokens forwarded", f"{st.get('tokens', 0) / 1e6:.0f}M "
                                           f"({st.get('target_tokens', 0) / 1e6:.0f}M scored)"],
                      ["GPU hours", f"{st.get('seconds', 0) / 3600:.2f}"],
                      ["chunks (resumes)", str(h["chunks"])],
                      ["median throughput", f"{h['tokens_per_s_median']} tokens/s"],
                      ["peak VRAM", f"{h['vram_peak_gb']} GiB"],
                      ["GPUs used", ", ".join(h["gpus"]) or "–"]],
                     ["", "value"], align_left=1), "",
              "## Reproducing", "",
              "```bash", "python scripts/pipeline.py --site <lumi|cuda> status   # every stage of every student",
              "python scripts/pipeline.py --site <lumi|cuda> next     # submit what is ready (see docs/PIPELINE.md)",
              "scisum report && scisum curves", "```", "",
              "---", "",
              "This report covers one run. `scripts/analysis/collect_results.py` compares the students,",
              "and `paper_docs/` holds the original project's findings and threats to validity.", ""]
    out_md.write_text("\n".join(lines))
    log.info("report: %s", out_md)
    return [out_md, out_json]
