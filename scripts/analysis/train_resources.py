"""What one training run cost: wall time, GPU-hours, throughput, memory, and its held-out loss curve.

    python scripts/analysis/train_resources.py RUN [RUN ...] [--runs-dir runs] [--out-json F] [--out-md F]
        [--gpu-type "AMD MI250X GCD"] [--gpu-hour-price 0.0]

Reads only what the trainer writes: ``runs/<run>/metrics.jsonl`` (``chunk_start`` / ``train`` / ``val_loss``
/ ``chunk_end`` records) and ``runs/<run>/run.json``. Works for SFT runs (``scisum.train``) and preference
runs (``scisum.pref_train``, whose ``train`` records carry ``dpo``/``margin``/``pref_acc``).

How the numbers are made:

* **Wall time** is the sum over chunks (one chunk per job; a resumed run has several) of the time between
  the chunk's ``chunk_start`` and its ``chunk_end`` record, or its last record if the job was killed. Model
  loading before ``chunk_start`` and queueing are not included.
* **GPU-hours** = sum over chunks of wall time x the chunk's ``world`` (ranks = GPUs; on LUMI one rank is
  one MI250X GCD).
* **tokens/s per GPU** is the median of the ``train`` records' ``tokens_per_s`` (each rank's own rate).
  For preference runs, which do not log it, it is total tokens / (wall time x GPUs).
* **Peak memory** is the largest ``vram_peak_gb`` / ``vram_gb``: PyTorch's peak allocated memory on rank 0,
  which is below what the device showed (allocator cache, other processes).
* **Cost per 1,000 papers** = GPU-hours / examples x 1,000 (for a preference run, examples are pairs).
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analysis_lib import fmt, md_table, read_json, read_jsonl  # noqa: E402

SCHEMA = "scisum.train_resources/1"


def chunks(records: list[dict]) -> list[dict]:
    """Split the metrics stream into chunks (jobs). Records before the first ``chunk_start`` form an
    implicit chunk (old logs). Each chunk: start/end time, world, gpu, host, job, finished flag."""
    out: list[dict] = []
    cur: dict | None = None
    for r in records:
        if r.get("kind") == "chunk_start":
            if cur is not None:
                out.append(cur)
            cur = {"t0": r.get("t"), "t1": r.get("t"), "world": int(r.get("world") or 1), "gpu": r.get("gpu"),
                   "host": r.get("host"), "job": r.get("job"), "finished": False, "exit_code": None,
                   "reason": None, "vram_peak_gb": None, "start_examples": r.get("examples", 0)}
            continue
        if cur is None:
            cur = {"t0": r.get("t"), "t1": r.get("t"), "world": 1, "gpu": None, "host": None, "job": None,
                   "finished": False, "exit_code": None, "reason": None, "vram_peak_gb": None,
                   "start_examples": 0}
        if r.get("t") is not None:
            cur["t1"] = r["t"] if cur["t1"] is None else max(cur["t1"], r["t"])
            if cur["t0"] is None:
                cur["t0"] = r["t"]
        if r.get("kind") == "chunk_end":
            cur.update(finished=r.get("exit_code") == 0, exit_code=r.get("exit_code"), reason=r.get("reason"),
                       vram_peak_gb=r.get("vram_peak_gb"))
            out.append(cur)
            cur = None
    if cur is not None:
        out.append(cur)
    for c in out:
        c["seconds"] = max(0.0, (c["t1"] or 0) - (c["t0"] or 0))
    return out


def val_curve(records: list[dict]) -> list[dict]:
    """Held-out loss points: per target token (as logged) and in nats per paper (comparable across
    tokenizers, as in the project's learning curves)."""
    pts = []
    for r in records:
        if r.get("kind") != "val_loss" or r.get("loss") is None:
            continue
        papers = r.get("papers") or 0
        tt = r.get("target_tokens") or 0
        pts.append({"examples": r.get("examples", 0), "step": r.get("step"), "loss_per_token": r["loss"],
                    "ppl": r.get("ppl"), "papers": papers,
                    "nats_per_paper": round(r["loss"] * tt / papers, 2) if papers and tt else None})
    return pts


def summarise_run(run_dir: Path, gpu_type: str | None = None, gpu_hour_price: float = 0.0) -> dict:
    """Everything about one run's cost. Missing files give ``{"run", "missing": [...]}``."""
    run = run_dir.name
    records = read_jsonl(run_dir / "metrics.jsonl")
    meta = read_json(run_dir / "run.json") or {}
    missing = [n for n, ok in (("metrics.jsonl", bool(records)), ("run.json", bool(meta))) if not ok]
    if not records:
        return {"schema": SCHEMA, "run": run, "missing": missing}
    ch = chunks(records)
    train = [r for r in records if r.get("kind") == "train"]
    kind = "preference" if any("dpo" in r for r in train) else "sft"
    wall = sum(c["seconds"] for c in ch)
    gpu_s = sum(c["seconds"] * c["world"] for c in ch)
    worlds = sorted({c["world"] for c in ch})
    last = train[-1] if train else {}
    examples = int(last.get("examples") or 0)
    tokens = int(last.get("tokens") or 0)
    tps = [r["tokens_per_s"] for r in train if r.get("tokens_per_s")]
    tps_gpu = st.median(tps) if tps else (tokens / gpu_s if gpu_s else None)
    peaks = [c["vram_peak_gb"] for c in ch if c.get("vram_peak_gb")] + [r["vram_gb"] for r in train if r.get("vram_gb")]
    gpus = sorted({c["gpu"] for c in ch if c.get("gpu")})
    out = {
        "schema": SCHEMA, "run": run, "kind": kind, "missing": missing,
        "base_model": meta.get("base_model"), "params_total": meta.get("params_total"),
        "params_trainable": meta.get("params_trainable"), "lr": meta.get("lr"),
        "chunks": len(ch), "finished": bool(ch and ch[-1]["finished"]),
        "interrupted_chunks": sum(1 for c in ch if not c["finished"]),
        "world": worlds[0] if len(worlds) == 1 else worlds,
        "gpu_type": gpu_type or (gpus[0] if len(gpus) == 1 else (gpus or None)),
        "wall_hours": round(wall / 3600, 3), "gpu_hours": round(gpu_s / 3600, 3),
        "steps": last.get("step"), "examples": examples, "tokens": tokens,
        "tokens_per_s_per_gpu": round(tps_gpu, 1) if tps_gpu else None,
        "peak_vram_gb": round(max(peaks), 2) if peaks else None,
        "gpu_hours_per_1000_examples": round(gpu_s / 3600 / examples * 1000, 4) if examples else None,
        "final_train_loss": last.get("loss"),
        "val_loss": val_curve(records),
    }
    if gpu_hour_price:
        out["cost"] = round(out["gpu_hours"] * gpu_hour_price, 2)
        out["gpu_hour_price"] = gpu_hour_price
    if kind == "preference" and train:
        out["final_pref_acc"] = last.get("pref_acc")
        out["final_margin"] = last.get("margin")
    return out


def to_markdown(rows: list[dict]) -> str:
    header = ["run", "kind", "GPUs", "GPU type", "wall h", "GPU-h", "examples", "tokens (M)", "tok/s/GPU",
              "peak GB", "GPU-h / 1k ex", "chunks", "finished"]
    body = []
    for r in rows:
        if "kind" not in r:
            body.append([r["run"], "missing: " + ", ".join(r.get("missing", []))] + [""] * (len(header) - 2))
            continue
        body.append([r["run"], r["kind"], r["world"], r["gpu_type"] or "–", fmt(r["wall_hours"], 2),
                     fmt(r["gpu_hours"], 2), f"{r['examples']:,}", fmt(r["tokens"] / 1e6, 1),
                     fmt(r["tokens_per_s_per_gpu"], 0), fmt(r["peak_vram_gb"], 1),
                     fmt(r["gpu_hours_per_1000_examples"], 3), r["chunks"], "yes" if r["finished"] else "no"])
    return md_table(header, body)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+", help="run names (directories under --runs-dir)")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--gpu-type", default=None, help="label for the GPU when the logs do not name it")
    ap.add_argument("--gpu-hour-price", type=float, default=0.0, help="optional price per GPU-hour")
    ap.add_argument("--out-json")
    ap.add_argument("--out-md")
    a = ap.parse_args(argv)
    rows = [summarise_run(Path(a.runs_dir) / r, a.gpu_type, a.gpu_hour_price) for r in a.runs]
    md = to_markdown(rows)
    print(md)
    if a.out_json:
        Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out_json).write_text(json.dumps(rows, indent=1))
    if a.out_md:
        Path(a.out_md).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out_md).write_text(md + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
