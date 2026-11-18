"""Preference training: push the student *away* from its own worse summaries, while still imitating.

``scisum.train`` maximises the likelihood of a target. Selection (``scisum.improve``) can only add
good summaries to that set; it cannot say "not this one". A preference pair can: for the same
paper, a *chosen* summary (the highest reward among the teacher's and the student's own samples)
and a *rejected* one (the lowest-reward valid sample, with invented numbers or missing facts).

The objective per pair is DPO plus a likelihood term on the chosen summary:

    L = -log sigmoid( beta * [ (log p(chosen) - log p_ref(chosen)) - (log p(rejected) - log p_ref(rejected)) ] )
        + w * mean_token_NLL(chosen)

* The reference ``p_ref`` is the adapter the run starts from, not the base model: the point is to
  improve the student, not to stay near an untrained model. Its log-probabilities are computed once
  before the first update, stored in ``runs/<run>/ref_logps.npy`` and never recomputed (after an
  update they would be wrong), so a resumed run reads them back.
* The likelihood term is what keeps the imitation: when the chosen summary is the teacher's it is
  plain distillation of that paper, and without it DPO is known to lower the likelihood of both
  members of a pair and drift. It also bounds how far the margin can be bought with length.
* A pair costs two forward and backward passes of a 10-30k-token sequence, so a run is about twice
  the cost of the same number of single targets.

This is a second training loop rather than a branch inside ``scisum.train``'s, on purpose: that loop
has trained three models and its resume, stop and accounting paths are not to be disturbed for an
experiment. What is shared (state, checkpoints, schedule, metrics, the validation loss) is imported.

The shards hold the pairs as consecutive examples, chosen first: ``2k`` and ``2k + 1``.
Exit codes as ``scisum.train``: 0 finished, 2 stopped early, 3 fatal.
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from scisum.config import Settings
from scisum.model import load_model, preference_coefficients, seq_logp, trainable_summary
from scisum.shards import ShardSet
from scisum.train import (
    STOP,
    MetricLog,
    RunState,
    _dist_setup,
    _eta,
    _install_signal_handlers,
    _new_window,
    _nullcontext,
    _save_milestone,
    _save_rolling,
    checkpoints,
    rolling_every,
    val_loss,
    wsd_factor,
)

log = logging.getLogger(__name__)


@torch.no_grad()
def reference_logps(model, shards: ShardSet, pairs: list[int], n_pairs: int, device: str, world: int, rank: int,
                    partial: Path | None = None, heartbeat: Path | None = None, chunk: int = 16) -> np.ndarray:
    """``[n_pairs, 2]`` summed log-probabilities of (chosen, rejected) under the model as it is now.

    Only ``pairs`` are computed (a self-check needs four, not the whole set); the others stay NaN.
    For the whole set this is hours of forward passes, so progress is saved to ``partial`` after every
    ``chunk`` pairs and read back by a restart, and ``heartbeat`` (the run's metrics file) is touched so
    the job's watchdog, which kills a trainer that writes no metrics for 20 minutes, sees it working.
    On 2026-11-04 the first version computed every pair even in the self-check, saved nothing until
    the end, and would have been killed by the watchdog and restarted from zero indefinitely.
    """
    from scisum.model import seq_logp

    was_training = model.training
    model.eval()
    out = np.full((n_pairs, 2), np.nan)
    if partial is not None and partial.exists():
        prev = np.load(partial)
        if prev.shape == out.shape:
            out = prev
    todo = [k for k in pairs if np.isnan(out[k]).any()]
    if rank == 0:
        log.info("  reference log-probabilities: %d of %d pairs to compute (%d saved before)", len(todo), len(pairs), len(pairs) - len(todo))
    t0 = time.monotonic()
    for c in range(0, len(todo), chunk):
        part = todo[c:c + chunk]
        buf = torch.zeros(len(part), 2, dtype=torch.float64, device=device)
        for i in range(rank, len(part), world):
            for j in (0, 1):
                ex = shards.get(2 * part[i] + j)
                ids = torch.from_numpy(ex.tokens.astype(np.int64)).unsqueeze(0).to(device)
                lp, _ = seq_logp(model, ids, ex.prompt_len)
                buf[i, j] = float(lp)
        if world > 1:
            torch.distributed.all_reduce(buf)
        out[part] = buf.cpu().numpy()
        if rank == 0:
            if partial is not None:
                tmp = partial.with_name(partial.stem + ".tmp.npy")
                np.save(tmp, out)
                tmp.replace(partial)
            if heartbeat is not None:
                heartbeat.touch()
            done = c + len(part)
            el = time.monotonic() - t0
            log.info("  reference log-probabilities: %d/%d pairs (%.0f s, %.1f s a pair)", done, len(todo), el, el / done)
    model.train(was_training)
    return out


def check_reference(model, shards: ShardSet, ref: np.ndarray, pair: int, device: str, rel: float = 0.01,
                    floor: float = 2.0) -> tuple[float, float]:
    """Recompute one pair's reference log-probabilities and compare them with a stored file's.

    A run may start from a reference file computed by another run on the same pairs and the same
    starting adapter (round 2's loss-only arm reuses round 1's: 1.7 GPU-hours). If the pairs or the
    adapter differ, the file is wrong and every update with it is wrong, so the first pair is
    recomputed and must agree within ``rel`` of its size or ``floor`` nats, whichever is larger (bf16
    on another card type moves a summed log-probability of ~-450 by well under 1). Returns the two
    absolute differences.
    """
    from scisum.model import seq_logp

    was_training = model.training
    model.eval()
    diffs = []
    with torch.no_grad():
        for j in (0, 1):
            ex = shards.get(2 * pair + j)
            ids = torch.from_numpy(ex.tokens.astype(np.int64)).unsqueeze(0).to(device)
            lp, _ = seq_logp(model, ids, ex.prompt_len)
            stored = float(ref[pair, j])
            diffs.append(abs(float(lp) - stored))
            if diffs[-1] > max(floor, rel * abs(stored)):
                raise RuntimeError(f"stored reference log-probability of pair {pair} ({'chosen' if j == 0 else 'rejected'}) is "
                                   f"{stored:.2f}, the starting adapter gives {float(lp):.2f}: the file belongs to other pairs "
                                   f"or another adapter")
    model.train(was_training)
    return diffs[0], diffs[1]


def run_pref_train(s: Settings, run: str, *, max_examples: int | None = None, max_seconds: float | None = None,
                   resume: bool = True, smoke: int = 0, shortest: bool = False, longest: bool = False) -> int:
    t_start = time.monotonic()
    _install_signal_handlers()
    world, rank, local = _dist_setup()
    device = f"cuda:{local}"
    run_dir = s.run_dir(run)
    (run_dir / "ckpt").mkdir(parents=True, exist_ok=True)
    ckpt, state = None, RunState()
    for candidate in (checkpoints(run_dir) if resume else []):
        try:
            state = RunState.load(candidate / "state.json")
        except (OSError, ValueError, TypeError) as e:  # noqa: PERF203 - one bad checkpoint, try the next
            log.warning("checkpoint %s is unreadable (%s); falling back", candidate.name, e)
            continue
        ckpt = candidate
        log.info("resuming from %s (%d pairs, step %d)", ckpt.name, state.examples, state.step)
        break

    shards = ShardSet(s.tokens_dir(s.train_split))
    loss_shards = ShardSet(s.tokens_dir("val_loss"))
    n_pairs = len(shards) // 2
    cap = min(max_examples or s.n_train, n_pairs)
    if smoke:
        cap = min(cap, smoke)
    order = list(range(0, 2 * cap, 2))                       # the chosen example of each pair
    if shortest or longest:   # for the preflight and for the memory self-check on the longest pairs
        sign = 1 if shortest else -1
        order = sorted(range(0, 2 * n_pairs, 2), key=lambda i: sign * max(int(shards.lengths[i]), int(shards.lengths[i + 1])))[:cap]
    elif s.pref_order_seed:   # a second seed: same pairs, another order (deterministic, so a resumed run continues it)
        random.Random(s.pref_order_seed).shuffle(order)
    total_steps = max(1, math.ceil(cap / (s.grad_accum * world)))
    warmup = max(1, int(total_steps * s.warmup_frac))
    decay_steps = int(total_steps * s.decay_frac)
    if state.examples >= cap and state.finished:
        log.info("run %s already finished at %d pairs", run, state.examples)
        return 0

    adapter = str(ckpt / "adapter") if ckpt is not None else (s.init_adapter or None)
    if adapter is None:
        raise RuntimeError("preference training needs a starting adapter (SCISUM_INIT_ADAPTER): it is the reference")
    model = load_model(s, train=True, adapter=adapter, device=device)

    ref_path = run_dir / "ref_logps.npy"
    needed = sorted({i // 2 for i in order})
    ref = np.load(ref_path) if ref_path.exists() else None
    if ref is not None and ref.shape == (n_pairs, 2) and not np.isnan(ref[needed]).any() and state.examples == 0:
        dc, dr = check_reference(model, shards, ref, needed[0], device)
        if rank == 0:
            log.info("stored reference log-probabilities agree with the starting adapter (pair %d: %.3f / %.3f nats)",
                     needed[0], dc, dr)
    if ref is None or ref.shape != (n_pairs, 2) or np.isnan(ref[needed]).any():
        if state.examples > 0:
            raise RuntimeError(f"{ref_path} is missing or incomplete but training has begun: the reference cannot be recomputed")
        log.info("computing the reference log-probabilities of %d pairs with the starting adapter", len(needed))
        ref = reference_logps(model, shards, needed, n_pairs, device, world, rank,
                              partial=run_dir / "ref_logps.partial.npy", heartbeat=run_dir / "metrics.jsonl")
        if rank == 0:
            tmp = ref_path.with_name("ref_logps.tmp.npy")
            np.save(tmp, ref)
            tmp.replace(ref_path)
        if world > 1:
            torch.distributed.barrier()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=s.lr, betas=s.adam_betas, weight_decay=s.weight_decay, eps=1e-8, fused=True)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda st: wsd_factor(st, total_steps, warmup, decay_steps, s.decay_floor))
    if ckpt is not None and (ckpt / "optim.pt").exists():
        try:
            blob = torch.load(ckpt / "optim.pt", map_location="cpu", weights_only=False)
            opt.load_state_dict(blob["optimizer"])
            sched.load_state_dict(blob["scheduler"])
            torch.set_rng_state(blob["cpu_rng"])
            torch.cuda.set_rng_state(blob["cuda_rng"], device=device)
        except (OSError, RuntimeError, KeyError, EOFError) as e:
            log.warning("optimiser state in %s will not load (%s); continuing with a fresh optimiser", ckpt.name, e)
            for _ in range(state.step):
                sched.step()
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local], find_unused_parameters=False)
    inner = model.module if world > 1 else model
    metrics = MetricLog(run_dir / "metrics.jsonl", rank)
    if rank == 0:
        info = {"run": run, "objective": "preference", "pairs": cap, "world": world, "total_steps": total_steps,
                "lr": s.lr, "beta": s.pref_beta, "sft_weight": s.pref_sft_weight, "len_alpha": s.pref_len_alpha, "order_seed": s.pref_order_seed,
                "grad_accum": s.grad_accum,
                "init_adapter": s.init_adapter, "settings": json.loads(s.model_dump_json()),
                **trainable_summary(inner), **s.provenance()}
        (run_dir / "run.json").write_text(json.dumps(info, indent=2, default=str))
        log.info("run %s: %d pairs, %d steps, beta %.3f, sft weight %.2f, length alpha %.4f, resuming at %d",
                 run, cap, total_steps, s.pref_beta, s.pref_sft_weight, s.pref_len_alpha, state.examples)
        metrics.add("chunk_start", cap=cap, world=world, total_steps=total_steps, examples=state.examples,
                    step=state.step, host=os.uname().nodename, job=os.environ.get("SLURM_JOB_ID", ""),
                    gpu=torch.cuda.get_device_name(device) if torch.cuda.is_available() else "")
    if state.examples == 0:
        r = val_loss(inner, loss_shards, device, s.loss_eval_n, world, rank)
        if rank == 0:
            metrics.add("val_loss", examples=0, step=0, **r)
            log.info("val loss at 0 pairs (the starting adapter): %.4f", r["loss"])

    torch.cuda.reset_peak_memory_stats(device)
    state.chunks += 1
    window = {**_new_window(), "dpo": 0.0, "nll": 0.0, "margin": 0.0, "acc": 0.0, "len_diff": 0.0, "pairs": 0}
    exit_code = 2
    next_loss_eval = ((state.examples // max(s.loss_eval_every, 1)) + 1) * s.loss_eval_every
    try:
        while state.examples < cap:
            batch_start = state.examples
            batch_end = min(batch_start + s.grad_accum * world, cap)
            batch = order[batch_start:batch_end]
            mine = [i for n, i in enumerate(batch) if (batch_start + n) % world == rank]
            idle_rank = world > 1 and not mine
            if idle_rank:
                mine = [batch[0]]
            opt.zero_grad(set_to_none=True)
            step_tokens = 0
            kn = {"beta": s.pref_beta, "sft_weight": s.pref_sft_weight, "len_alpha": s.pref_len_alpha,
                  "clip": s.max_grad_norm}
            for k, idx in enumerate(mine):
                ex, ex_r = shards.get(idx), shards.get(idx + 1)
                ids = torch.from_numpy(ex.tokens.astype(np.int64)).unsqueeze(0).to(device, non_blocking=True)
                ids_r = torch.from_numpy(ex_r.tokens.astype(np.int64)).unsqueeze(0).to(device, non_blocking=True)
                sync = (k == len(mine) - 1) or world == 1
                ctx = model.no_sync() if (world > 1 and not sync) else _nullcontext()
                # The two summaries are backpropagated one at a time (see preference_coefficients): the
                # memory of one sequence, so a 48 GB card is enough. The first backward never syncs.
                aux = preference_coefficients(inner, ids, ids_r, ex.prompt_len, float(ref[idx // 2, 0]),
                                              float(ref[idx // 2, 1]), kn["beta"], kn["sft_weight"], kn["len_alpha"])
                # every pair counts the same: each rank's share is scaled so that DDP's average over
                # ranks is the mean over the pairs of the step
                scale = 0.0 if idle_rank else world / max(len(batch), 1)
                lp_c, _ = seq_logp(inner, ids, ex.prompt_len)
                with (model.no_sync() if world > 1 else _nullcontext()):
                    (aux["coef_chosen"] * scale * lp_c).backward()
                del lp_c
                lp_r, _ = seq_logp(inner, ids_r, ex.prompt_len)
                with ctx:
                    (aux["coef_rejected"] * scale * lp_r).backward()
                del lp_r
                if idle_rank:
                    continue
                step_tokens += int(len(ex.tokens) + len(ex_r.tokens))
                window["pairs"] += 1
                window["tokens"] += int(len(ex.tokens) + len(ex_r.tokens))
                for key in ("dpo", "nll", "margin", "acc", "len_diff"):
                    window[key] += aux[key]
            grad_norm = float(torch.nn.utils.clip_grad_norm_(params, kn["clip"]))
            if math.isfinite(grad_norm):
                opt.step()
                updated = True
            else:
                updated = False
            if not updated:
                state.skipped_steps += 1
                log.warning("step %d: gradient norm is %s; skipping the update", state.step + 1, grad_norm)
            sched.step()
            state.step += 1
            state.examples = batch_end
            state.tokens += step_tokens * world
            window["steps"] += 1
            if rank == 0:
                # A heartbeat for the job's watchdog, which kills a trainer that writes no metrics for 20
                # minutes: a preference step is two sequences and two extra forwards, about 3.4 minutes,
                # and five steps of long pairs took 1,234 s, so the watchdog killed a healthy run (2026-11-04).
                (run_dir / "metrics.jsonl").touch()
            if rank == 0 and (state.step % s.log_every_steps == 0 or state.examples >= cap):
                n = max(window["pairs"], 1)
                dt = max(time.monotonic() - window["t0"], 1e-9)
                eta_s = (cap - state.examples) * dt / max(window["pairs"] * world, 1)
                metrics.add("train", step=state.step, examples=state.examples, tokens=state.tokens,
                            loss=round((window["dpo"] + s.pref_sft_weight * window["nll"]) / n, 5),
                            dpo=round(window["dpo"] / n, 5), nll=round(window["nll"] / n, 5),
                            margin=round(window["margin"] / n, 4), pref_acc=round(window["acc"] / n, 4),
                            len_diff=round(window["len_diff"] / n, 1),
                            lr=sched.get_last_lr()[0], grad_norm=round(grad_norm, 4), eta_s=round(eta_s),
                            seconds=round(time.monotonic() - t_start + state.seconds, 1),
                            vram_gb=round(torch.cuda.max_memory_allocated(device) / 2**30, 2))
                log.info("step %5d  pair %5d/%d  dpo %.4f  nll %.4f  margin %+.3f  acc %.2f  gnorm %.2f  eta %s",
                         state.step, state.examples, cap, window["dpo"] / n, window["nll"] / n,
                         window["margin"] / n, window["acc"] / n, grad_norm, _eta(eta_s))
                window = {**_new_window(), "dpo": 0.0, "nll": 0.0, "margin": 0.0, "acc": 0.0, "len_diff": 0.0, "pairs": 0}

            due = [m for m in s.milestones if m not in state.milestones_done and state.examples >= m]
            if state.examples >= next_loss_eval or due:
                r = val_loss(inner, loss_shards, device, s.loss_eval_n, world, rank)
                if rank == 0:
                    metrics.add("val_loss", examples=state.examples, step=state.step, milestone=due[-1] if due else None, **r)
                    log.info("val loss at %d pairs: %.4f", state.examples, r["loss"])
                if state.examples >= next_loss_eval:
                    next_loss_eval = ((state.examples // max(s.loss_eval_every, 1)) + 1) * s.loss_eval_every
                for m in due:
                    if rank == 0:
                        _save_milestone(s, run_dir, inner, state, m, r)
                    state.milestones_done.append(m)
                if world > 1:
                    torch.distributed.barrier()
            stop = STOP["flag"] or (max_seconds is not None and (time.monotonic() - t_start) > max_seconds)
            if world > 1:
                flag = torch.tensor([1.0 if stop else 0.0], device=device)
                torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
                stop = bool(flag.item())
            if stop:
                log.warning("stopping: %s", STOP["reason"] or "time limit for this chunk")
                break
            if rank == 0 and state.examples % rolling_every(s.save_every_examples, cap, s.grad_accum * world) < s.grad_accum * world:
                _save_rolling(run_dir, inner, opt, sched, state, t_start)
    except torch.cuda.OutOfMemoryError:
        log.exception("CUDA OOM at pair %d", state.examples)
        exit_code = 3
    else:
        if state.examples >= cap:
            state.finished = True
            exit_code = 0
    finally:
        if rank == 0:
            if exit_code in (0, 2):
                try:
                    _save_rolling(run_dir, inner, opt, sched, state, t_start)
                except Exception:  # noqa: BLE001 - never let the final save hide the real failure
                    log.exception("the final checkpoint could not be written")
            metrics.add("chunk_end", examples=state.examples, step=state.step, exit_code=exit_code,
                        skipped_steps=state.skipped_steps, seconds=round(time.monotonic() - t_start, 1),
                        vram_peak_gb=round(torch.cuda.max_memory_allocated(device) / 2**30, 2))
            metrics.close()
        if world > 1:
            torch.distributed.destroy_process_group()
    log.info("chunk done: pairs=%d step=%d exit=%d", state.examples, state.step, exit_code)
    return exit_code


def pair_summary(path: Path) -> dict:
    """How the run went, from its metrics: first and last window of the preference statistics."""
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    tr = [r for r in rows if r.get("kind") == "train"]
    if not tr:
        return {}
    pick = lambda r: {k: r[k] for k in ("step", "dpo", "nll", "margin", "pref_acc")}  # noqa: E731
    return {"first": pick(tr[0]), "last": pick(tr[-1])}
