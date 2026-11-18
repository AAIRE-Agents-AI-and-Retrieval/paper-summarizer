"""``scisum status``: dataset, training, evaluations and Slurm jobs on one screen. Read-only."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from scisum.config import Settings
from scisum.curves import read_evals, read_metrics
from scisum.dataset import SPLITS, collect_stats


def _fmt_age(ts: float) -> str:
    d = max(0.0, time.time() - ts)
    return f"{d / 60:.0f}m ago" if d < 7200 else f"{d / 3600:.1f}h ago"


def print_status(s: Settings, run: str = "main") -> None:
    print(f"SciSum status  {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}  run={run}")

    print("\ndataset")
    idx = s.corpus_dir / "index.json"
    if idx.exists():
        d = json.loads(idx.read_text())
        print(f"  eligible papers: {d['eligible']} of {d['summary_records']} teacher summaries "
              f"(dropped {d['dropped_total']}: {', '.join(f'{k}={v}' for k, v in list(d['drops'].items())[:4])})")
    for split in SPLITS:
        st = collect_stats(s, split)
        cap = {"train": s.n_train, "val": s.n_val, "test": s.n_test}[split]
        if st["parts"]:
            print(f"  {split:5} {st['papers']:6d} built (cap {cap}), {st['parts']} parts, "
                  f"{st['prompt_tokens'] / 1e6:.1f}M prompt + {st['target_tokens'] / 1e6:.1f}M target tokens"
                  + (f", drops {st['drops']}" if st["drops"] else ""))

    print("\ntraining")
    state_path = s.run_dir(run) / "state.json"
    if not state_path.exists():
        print("  not started")
    else:
        st = json.loads(state_path.read_text())
        rows = [r for r in read_metrics(s, run) if r["kind"] == "train"]
        last = rows[-1] if rows else {}
        cap = s.n_train
        pct = 100 * st["examples"] / max(cap, 1)
        print(f"  {st['examples']}/{cap} papers ({pct:.1f}%), step {st['step']}, "
              f"{st['tokens'] / 1e6:.1f}M tokens, {st['seconds'] / 3600:.2f} GPU-hours, "
              f"{st['chunks']} chunks, finished={st['finished']}")
        if last:
            print(f"  last point: loss {last.get('loss')}, lr {last.get('lr'):.2e}, "
                  f"{last.get('tokens_per_s')} tok/s, {last.get('vram_gb')} GiB peak, "
                  f"eta {last.get('eta_s', 0) / 3600:.1f}h  ({_fmt_age(last['t'])})")
        print(f"  milestones done: {st['milestones_done'] or '-'}")
        for d in sorted((s.run_dir(run) / "ckpt").glob("ex*")):
            flags = [f.name for f in d.iterdir() if f.name in ("READY", "EVALUATED")]
            print(f"    {d.name:10} {'+'.join(flags) or 'incomplete'}")

    print("\nevaluations")
    evals = read_evals(s, run) + read_evals(s, run, "test")
    if not evals:
        print("  none yet")
    else:
        hdr = (f"  {'ckpt':10} {'split':5} {'papers':>6} {'json':>6} {'schema':>7} {'rougeL':>7} "
               f"{'bertsc':>7} {'faith':>7} {'outtok':>7}")
        print(hdr)
        for e in evals:
            m = e["metrics"]
            print(f"  {e['ckpt']:10} {e['split']:5} {e['papers']:6d} {m.get('json_ok', 0):6.3f} "
                  f"{m.get('schema_ok', 0):7.3f} {m.get('all_rougeL', 0):7.4f} "
                  f"{m.get('bertscore_mean', 0):7.4f} {m.get('faithfulness', 0):7.4f} "
                  f"{m.get('out_tokens', 0):7.0f}")

    print("\nslurm")
    try:
        q = subprocess.run(["squeue", "--me", "-h", "-o", "%.10i %.16j %.8T %.10M %.10l %R"],
                           capture_output=True, text=True, timeout=20).stdout
        jobs = [ln for ln in q.splitlines() if "sum-" in ln]
        print("\n".join("  " + j for j in jobs) if jobs else "  no SciSum jobs")
    except (OSError, subprocess.SubprocessError):
        pass
    curves = Path(s.reports_dir / "curves" / run)
    if curves.exists():
        png = sorted(curves.glob("*.png"))
        if png:
            print(f"\ncurves: {len(png)} figures in {curves} (newest {_fmt_age(max(p.stat().st_mtime for p in png))})")
