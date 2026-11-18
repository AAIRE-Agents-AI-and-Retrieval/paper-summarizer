"""The SciSum job watchdog: what is running, what looks wrong, and what to do about it (login node, seconds).

    .venv/bin/python scripts/watchdog.py            # prints a table and the alerts; writes state/watchdog/status.json
    .venv/bin/python scripts/watchdog.py --quiet    # alerts only (exit 1 if any)

Written 2026-11-09 (I8 in ``paper_docs/12_reactive_training.md``). Every check comes from a failure
we had to find by hand:

* **stalled training**: a running ``sum-train-*`` or ``sum-rx-*`` job whose ``metrics.jsonl`` has not
  changed for 25 minutes (Run A lost 8.7 h to a filesystem stall; the trainer touches the file every step).
* **crash in a log**: ``Traceback``, ``CUDA out of memory``, ``NaN`` or ``Killed`` in the last 200 lines
  of a running job's log.
* **finished without its adapter**: ``state.json`` says finished but no ``ckpt/ex<N>`` exists (round 3:
  submitted without ``SUM_IMP_N``, the milestone was never reached, ``scripts/promote_final.py`` fixed it).
* **dependency that can never start**: a pending job whose reason is ``DependencyNeverSatisfied``.
* **redundant evaluation workers**: a ``sum-evalw-*`` or ``sum-final-*`` job for a run that is evaluated by
  its own ``sum-imp-eval-*`` job (round 3: self-requeuing workers held cards for nothing).

It only reads; it never cancels or submits. Acting on an alert is the operator's (or the session's) job.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STALL_S = 25 * 60
BAD = re.compile(r"Traceback|CUDA out of memory|OutOfMemoryError|\bNaN\b|Killed|DUE TO TIME LIMIT")


def squeue() -> list[dict]:
    out = subprocess.run(["squeue", "--me", "-h", "-o", "%i|%j|%T|%r|%N|%M|%S"], capture_output=True, text=True).stdout
    rows = []
    for line in out.splitlines():
        i, name, state, reason, node, elapsed, start = (line.split("|") + [""] * 7)[:7]
        if name.startswith("sum-"):
            rows.append({"id": i, "name": name, "state": state, "reason": reason, "node": node, "elapsed": elapsed, "start": start})
    return rows


def run_of(name: str) -> str | None:
    if name == "sum-rx-smoke":
        return "smoke-rx"
    m = re.match(r"sum-(?:train|rx|evalw|final|imp-eval|imp-peval)-(.+)$", name)
    return m.group(1) if m else None


def log_of(job: dict) -> Path:
    return ROOT / "runs" / "logs" / f"{job['name']}_{job['id']}.log"


def checks(jobs: list[dict]) -> list[str]:
    alerts = []
    now = time.time()
    imp_eval_runs = {run_of(j["name"]) for j in jobs if j["name"].startswith("sum-imp-")}
    for j in jobs:
        run = run_of(j["name"])
        if j["state"] == "PENDING" and "DependencyNeverSatisfied" in j["reason"]:
            alerts.append(f"{j['name']} ({j['id']}): dependency can never be satisfied; cancel or resubmit")
        if j["state"] == "RUNNING":
            lp = log_of(j)
            if lp.exists():
                tail = lp.read_text(errors="replace").splitlines()[-200:]
                hits = [ln for ln in tail if BAD.search(ln)]
                if hits:
                    alerts.append(f"{j['name']} ({j['id']}): log shows {hits[-1].strip()[:140]!r}")
            if run and (j["name"].startswith("sum-train-") or j["name"].startswith("sum-rx")):
                m = ROOT / "runs" / run / "metrics.jsonl"
                if m.exists() and now - m.stat().st_mtime > STALL_S:
                    alerts.append(f"{j['name']} ({j['id']}): metrics.jsonl unchanged for {(now - m.stat().st_mtime) / 60:.0f} min (stalled?)")
        if j["name"].startswith(("sum-evalw-", "sum-final-")) and run in imp_eval_runs:
            alerts.append(f"{j['name']} ({j['id']}): redundant, {run} is evaluated by its sum-imp-eval job")
    for st in (ROOT / "runs").glob("*/state.json"):
        try:
            s = json.loads(st.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        ck = st.parent / "ckpt"
        if s.get("finished") and ck.is_dir() and not any(p.name.startswith("ex") for p in ck.iterdir()) \
                and now - st.stat().st_mtime < 7 * 86400:
            alerts.append(f"{st.parent.name}: finished but no ckpt/ex<N> (run scripts/promote_final.py)")
    return alerts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    jobs = squeue()
    alerts = checks(jobs)
    out = ROOT / "state" / "watchdog"
    out.mkdir(parents=True, exist_ok=True)
    (out / "status.json").write_text(json.dumps({"t": time.strftime("%Y-%m-%d %H:%M"), "jobs": jobs, "alerts": alerts}, indent=1))
    if not a.quiet:
        print(f"{'job':>9}  {'name':38} {'state':11} {'node':10} {'elapsed':>10}  reason")
        for j in jobs:
            print(f"{j['id']:>9}  {j['name']:38} {j['state']:11} {j['node']:10} {j['elapsed']:>10}  {j['reason']}")
    print("\n".join(f"ALERT: {x}" for x in alerts) if alerts else "no alerts")
    sys.exit(1 if alerts and a.quiet else 0)


if __name__ == "__main__":
    main()
