"""Publish a finished preference run's final adapter as its milestone ``ckpt/ex<N>`` (GPU for the loss).

    python scripts/promote_final.py --run q35preflcrd3              # with the run's model profile sourced
    python scripts/promote_final.py --run q35preflcrd3 --no-val-loss  # CPU only; READY carries no loss

Written 2026-11-08. The round-3 runs were submitted without ``SUM_IMP_N``, so their milestone list was
the profile default ``[1500]`` while they had 1,163 or 1,073 pairs. Training is unaffected: it covers
``cap = min(n_train, pairs)`` pairs with the schedule computed from ``cap`` (146 steps, as round 2's
``preflc``), and the last rolling checkpoint is written after the final step. But the milestone was never
reached, so no ``ex<N>`` (the directory the evaluation reads) and no final held-out loss were written.

This copies ``ckpt/<LATEST>/adapter`` to ``ckpt/ex<pairs>`` with a READY marker like a milestone's
(``scisum.train._save_milestone``), after checking that the run finished, that the rolling checkpoint is
complete (DONE) and that it was written at the final pair. The held-out loss is computed as at a milestone
(``scisum.train.val_loss`` on ``loss_eval_n`` papers of the ``val_loss`` split) and appended to the run's
``metrics.jsonl`` with ``"promoted": true``. Idempotent: an existing ``ex<N>`` is left alone.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scisum.config import Settings  # noqa: E402


def final_checkpoint(run_dir: Path) -> tuple[Path, dict]:
    """The last rolling checkpoint of a finished run, checked to be complete and at the final pair."""
    state = json.loads((run_dir / "state.json").read_text())
    if not state.get("finished"):
        raise SystemExit(f"{run_dir.name}: not finished (state.json); nothing to promote")
    step_dir = run_dir / "ckpt" / (run_dir / "ckpt" / "LATEST").read_text().strip()
    if not (step_dir / "DONE").exists() or not (step_dir / "adapter").is_dir():
        raise SystemExit(f"{step_dir}: incomplete rolling checkpoint (no DONE or no adapter)")
    st = json.loads((step_dir / "state.json").read_text())
    if st.get("examples") != state["examples"] or st.get("step") != state["step"]:
        raise SystemExit(f"{step_dir}: written at pair {st.get('examples')}, the run ended at {state['examples']}")
    return step_dir, state


def promote(s: Settings, run: str, val: bool) -> Path:
    run_dir = s.run_dir(run)
    step_dir, state = final_checkpoint(run_dir)
    n = int(state["examples"])
    dest = run_dir / "ckpt" / f"ex{n}"
    if (dest / "READY").exists():
        print(f"{dest} already exists; nothing to do")
        return dest
    loss = None
    if val:
        from scisum.model import load_model
        from scisum.shards import ShardSet
        from scisum.train import val_loss

        model = load_model(s, train=True, adapter=str(step_dir / "adapter"), device="cuda:0")
        loss = val_loss(model, ShardSet(s.tokens_dir("val_loss")), "cuda:0", s.loss_eval_n)
        print(f"held-out loss at {n} pairs: {loss['loss']:.4f} ({loss['papers']} papers)")
        with (run_dir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps({"t": round(time.time(), 3), "kind": "val_loss", "examples": n, "step": state["step"],
                                "milestone": n, "promoted": True, **loss}) + "\n")
    tmp = run_dir / "ckpt" / f".writing-ex{n}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    shutil.copytree(step_dir / "adapter", tmp / "adapter")
    (tmp / "READY").write_text(json.dumps({
        "milestone": n, "examples": n, "step": state["step"], "tokens": state.get("tokens"),
        "target_tokens": state.get("target_tokens"), "val_loss": loss, "base_model": s.base_model,
        "promoted_from": f"ckpt/{step_dir.name}",
        "written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=2))
    if dest.exists():
        shutil.rmtree(dest)
    tmp.replace(dest)
    print(f"promoted {step_dir} -> {dest}")
    return dest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--no-val-loss", action="store_true")
    a = ap.parse_args()
    promote(Settings(), a.run, not a.no_val_loss)
