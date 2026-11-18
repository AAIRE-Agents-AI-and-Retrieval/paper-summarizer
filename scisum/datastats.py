"""The dataset card, as numbers: ``state/build.json`` plus a printable table.

Everything the training and evaluation plans depend on comes from here — how many tokens one pass
over the training split costs, how often the body had to be cut to fit the prompt budget, how the
three teacher models are spread over the splits — so it is computed from the shards themselves
rather than from the plan.
"""

from __future__ import annotations

import gzip
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np

from scisum.config import Settings
from scisum.dataset import SPLITS, collect_stats, load_splits
from scisum.shards import ShardSet

log = logging.getLogger(__name__)


def _dist(a: np.ndarray) -> dict:
    if a.size == 0:
        return {}
    return {"mean": round(float(a.mean()), 1), "median": int(np.median(a)),
            "p90": int(np.percentile(a, 90)), "p99": int(np.percentile(a, 99)),
            "max": int(a.max()), "min": int(a.min()), "sum": int(a.sum())}


def _meta_rows(s: Settings, split: str) -> list[dict]:
    rows: list[dict] = []
    d = s.model_data_dir / "prompts" / split
    for p in sorted(d.glob("part-*.jsonl.gz")):
        with gzip.open(p, "rt") as f:
            rows.extend(json.loads(line)["meta"] for line in f)
    if rows:
        return rows
    shards = ShardSet(s.tokens_dir(split))
    return shards.meta()


def split_stats(s: Settings, split: str) -> dict:
    rows = _meta_rows(s, split)
    cap = {"train": s.n_train, "val": s.n_val, "test": s.n_test}[split]
    used = rows[:cap]
    pt = np.array([r["prompt_tokens"] for r in used], dtype=np.int64)
    tt = np.array([r["target_tokens"] for r in used], dtype=np.int64)
    pages = np.array([r["pages"] or 0 for r in used], dtype=np.int64)
    return {
        "built": len(rows), "used": len(used), "cap": cap,
        "prompt_tokens": _dist(pt), "target_tokens": _dist(tt), "total_tokens": _dist(pt + tt),
        "pages": _dist(pages),
        "truncation": dict(Counter(r["trunc_kind"] for r in used)),
        "fits_whole_body_pct": round(100 * sum(1 for r in used if r["fits"]) / max(len(used), 1), 2),
        "teachers": dict(Counter(r["teacher"] for r in used)),
        "cohorts": dict(Counter(r["cohort"] for r in used)),
        "teacher_prompt_tokens_mean": round(float(np.mean([r["teacher_prompt_tokens"] or 0 for r in used])), 1),
    }


def write_dataset_stats(s: Settings, out: Path | None = None) -> dict:
    sp = load_splits(s)
    stats = {"splits": {k: split_stats(s, k) for k in SPLITS},
             "build": {k: collect_stats(s, k) for k in SPLITS},
             "index": json.loads((s.corpus_dir / "index.json").read_text()),
             "split_plan": json.loads((s.corpus_dir / "splits_summary.json").read_text()),
             "curve_eval": len(sp["curve_eval"]), "loss_eval": len(sp["loss_eval"]),
             "settings": {k: getattr(s, k) for k in ("max_seq_len", "gen_reserve_tokens", "max_prompt_tokens",
                                                     "max_target_tokens", "grad_accum", "seed")},
             **s.provenance()}
    tr = stats["splits"]["train"]
    steps = -(-tr["used"] // s.grad_accum)
    stats["training_cost"] = {
        "examples": tr["used"], "optimiser_steps": steps,
        "tokens_per_epoch": tr["total_tokens"].get("sum", 0),
        "target_tokens_per_epoch": tr["target_tokens"].get("sum", 0),
        "milestones": {str(m): m for m in s.milestones},
    }
    # Per model as well as the current one: each student's dataset card has to survive the next
    # student's build, or the written record loses the numbers it was based on.
    (s.state_dir / f"build-{s.model_tag}.json").write_text(json.dumps(stats, indent=2, default=str))
    out = out or (s.state_dir / "build.json")
    Path(out).write_text(json.dumps(stats, indent=2, default=str))
    log.info("dataset: train %d (%d tokens), val %d, test %d",
             tr["used"], stats["training_cost"]["tokens_per_epoch"],
             stats["splits"]["val"]["used"], stats["splits"]["test"]["used"])
    return stats


def format_table(stats: dict) -> str:
    head = f"{'split':6} {'papers':>7} {'prompt tok (median/p90/max)':>32} {'target tok':>22} {'whole body':>11}"
    lines = [head, "-" * len(head)]
    for k in SPLITS:
        d = stats["splits"][k]
        p, t = d["prompt_tokens"], d["target_tokens"]
        lines.append(f"{k:6} {d['used']:7d} {p.get('median', 0):10d} /{p.get('p90', 0):7d} /{p.get('max', 0):7d}"
                     f"   {t.get('median', 0):7d} /{t.get('p90', 0):6d} /{t.get('max', 0):6d}"
                     f"   {d['fits_whole_body_pct']:8.1f}%")
    return "\n".join(lines)
