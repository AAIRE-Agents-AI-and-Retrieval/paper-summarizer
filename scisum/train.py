"""The training loop: one pass over the 70 000 training papers, resumable and instrumented.

Design, and why
---------------
**One example per micro-batch, gradient accumulation for the batch.** Examples are 2k-16k tokens
long. Padding them into a rectangular batch would waste 25-35% of the compute, and packing them
into fixed bins needs block-diagonal attention and makes "after N examples" ambiguous. A single
variable-length sequence per forward wastes nothing, keeps the GPU busy (12k tokens is a large
batch in token terms), and makes the milestones exact.

**Token-weighted accumulation.** Each micro-loss is the mean over that example's target tokens, so
averaging the micro-losses would weight a 400-token summary like a 4 000-token one. Every micro-loss
is scaled by ``n_tokens_i / n_tokens_in_the_whole_batch`` instead (times the world size, because
DDP averages gradients across ranks). The batch's token total is known in advance from the index,
so no communication is needed to get it right.

**Warmup-stable-decay schedule**, not cosine-to-70k. The milestones (1k, 5k, 10k, 20k, 50k, 70k)
are intermediate checkpoints of one run; with a cosine schedule each would sit at a different
learning rate and the curve would measure the schedule as much as the data. WSD holds the learning
rate flat so that every checkpoint is "the same recipe, more data", and decays only over the last
10% of the run.

**Resumability.** The training order is fixed on disk, so the only state needed to resume is the
number of examples consumed: the loop seeks there. A rolling checkpoint (adapter, optimiser,
scheduler, RNG, counters) is written every ``save_every_examples``, and on ``SIGUSR1`` (Slurm sends
it 10 minutes before the walltime) the current chunk saves and exits 2, which tells the job script
to resubmit. Nothing is ever written in place: a temporary directory is renamed.

Exit codes: 0 finished, 2 stopped early (resubmit), 3 fatal.
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import signal
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from scisum.config import Settings
from scisum.model import load_model, loss_fn, trainable_summary
from scisum.shards import ShardSet

log = logging.getLogger(__name__)

STOP = {"flag": False, "reason": ""}


def _install_signal_handlers() -> None:
    def handler(signum, _frame):
        STOP["flag"] = True
        STOP["reason"] = signal.Signals(signum).name
        log.warning("%s received: stopping at the next micro-step boundary", STOP["reason"])

    for sig in (signal.SIGUSR1, signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, handler)


# ---- run state ----------------------------------------------------------------------------------
@dataclass
class RunState:
    examples: int = 0
    step: int = 0
    tokens: int = 0
    target_tokens: int = 0
    seconds: float = 0.0
    chunks: int = 0
    skipped_steps: int = 0
    milestones_done: list[int] = field(default_factory=list)
    finished: bool = False

    @classmethod
    def load(cls, path: Path) -> RunState:
        if path.exists():
            return cls(**json.loads(path.read_text()))
        return cls()

    def save(self, path: Path) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2))
        tmp.replace(path)


class MetricLog:
    """Append-only JSONL: every number the curves and the report are drawn from."""

    def __init__(self, path: Path, rank: int = 0):
        self.path = path
        self.rank = rank
        self.f = path.open("a") if rank == 0 else None

    def add(self, kind: str, **fields) -> None:
        if self.f is None:
            return
        rec = {"t": round(time.time(), 3), "kind": kind, **fields}
        self.f.write(json.dumps(rec) + "\n")
        self.f.flush()

    def close(self) -> None:
        if self.f is not None:
            self.f.close()


# ---- schedule -----------------------------------------------------------------------------------
def wsd_factor(step: int, total: int, warmup: int, decay_steps: int, floor: float) -> float:
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    decay_start = max(total - decay_steps, warmup)
    if step < decay_start or total <= decay_start:
        return 1.0
    frac = min(1.0, (step - decay_start) / max(total - decay_start, 1))
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * frac))


# ---- distributed --------------------------------------------------------------------------------
def _dist_setup() -> tuple[int, int, int]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        torch.distributed.init_process_group("nccl", rank=rank, world_size=world)
    torch.cuda.set_device(local)
    return world, rank, local


# ---- evaluation during training -----------------------------------------------------------------
@torch.no_grad()
def val_loss(model, shards: ShardSet, device: str, limit: int | None = None,
             world: int = 1, rank: int = 0) -> dict:
    """Teacher-forced loss on held-out papers: sum of target-token losses / number of tokens.

    Collective: every rank scores its share and the totals are reduced, so the other GPUs are not
    idle while rank 0 evaluates (14 evaluations over the run, two minutes each).
    """
    was_training = model.training
    model.eval()
    n = min(limit or len(shards), len(shards))
    total_loss = 0.0
    total_tok = 0
    t0 = time.monotonic()
    fn = loss_fn("head_slice")
    for i in range(rank, n, world):
        ex = shards.get(i)
        ids = torch.from_numpy(ex.tokens.astype(np.int64)).unsqueeze(0).to(device)
        loss, ntok = fn(model, ids, ex.prompt_len)
        total_loss += float(loss) * ntok
        total_tok += ntok
    if world > 1:
        t = torch.tensor([total_loss, float(total_tok)], device=device, dtype=torch.float64)
        torch.distributed.all_reduce(t)
        total_loss, total_tok = float(t[0]), int(t[1])
    model.train(was_training)
    mean = total_loss / max(total_tok, 1)
    return {"papers": n, "loss": round(mean, 5), "ppl": round(math.exp(min(mean, 20)), 3),
            "target_tokens": total_tok, "seconds": round(time.monotonic() - t0, 1)}


# ---- training -----------------------------------------------------------------------------------
def run_train(s: Settings, run: str, *, max_examples: int | None = None, max_seconds: float | None = None,
              resume: bool = True, smoke: int = 0, longest: bool = False,
              shortest: bool = False) -> int:
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
        log.info("resuming from %s (%d examples, step %d, %.2f GPU-hours so far)",
                 ckpt.name, state.examples, state.step, state.seconds / 3600)
        break

    train_shards = ShardSet(s.tokens_dir(s.train_split))
    loss_shards = ShardSet(s.tokens_dir("val_loss"))
    cap = min(max_examples or s.n_train, len(train_shards))
    if smoke:
        cap = min(cap, smoke)
    target_tokens = train_shards.target_tokens()
    # The training order is the order the shards were written in. ``--longest`` replaces it with the
    # longest examples in the split, which is how a self-check proves the memory ceiling fits.
    # ``--shortest`` is its opposite and proves nothing about memory: it is for the preflight, which
    # runs on whatever small GPU is free within minutes and only has to exercise the code path.
    order = list(range(cap))
    if longest or shortest:
        sign = 1 if shortest else -1
        order = sorted(range(len(train_shards)), key=lambda i: sign * int(train_shards.lengths[i]))[:cap]
        lens = [int(train_shards.lengths[i]) for i in order]
        log.info("%s-first self-check: %d examples of %d-%d tokens",
                 "shortest" if shortest else "longest", len(order), min(lens), max(lens))
    total_steps = max(1, math.ceil(cap / (s.grad_accum * world)))
    warmup = max(1, int(total_steps * s.warmup_frac))
    decay_steps = int(total_steps * s.decay_frac)

    if state.examples >= cap and state.finished:
        log.info("run %s already finished at %d examples", run, state.examples)
        return 0

    # A resumed run continues its own checkpoint; a new run may start from another run's adapter
    # (`init_adapter`): the optimiser, the schedule and the example counter are then fresh.
    adapter = str(ckpt / "adapter") if ckpt is not None else (s.init_adapter or None)
    model = load_model(s, train=True, adapter=adapter, device=device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=s.lr, betas=s.adam_betas, weight_decay=s.weight_decay,
                            eps=1e-8, fused=True)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda st: wsd_factor(st, total_steps, warmup, decay_steps, s.decay_floor))
    if ckpt is not None and (ckpt / "optim.pt").exists():
        # map_location="cpu": the RNG states are CPU byte tensors and must stay that way, and
        # `load_state_dict` moves each optimiser state to the device of its parameter anyway.
        try:
            blob = torch.load(ckpt / "optim.pt", map_location="cpu", weights_only=False)
            opt.load_state_dict(blob["optimizer"])
            sched.load_state_dict(blob["scheduler"])
            torch.set_rng_state(blob["cpu_rng"])
            torch.cuda.set_rng_state(blob["cuda_rng"], device=device)
            log.info("resumed optimiser and scheduler at step %d", state.step)
        except (OSError, RuntimeError, KeyError, EOFError) as e:
            # The adapter is the trained state; Adam's moments are not worth failing a run over.
            log.warning("optimiser state in %s will not load (%s); continuing with a fresh "
                        "optimiser from step %d", ckpt.name, e, state.step)
            for _ in range(state.step):
                sched.step()
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local],
                                                          find_unused_parameters=False)
    inner = model.module if world > 1 else model
    compute_loss = loss_fn(s.loss_impl)

    metrics = MetricLog(run_dir / "metrics.jsonl", rank)
    if rank == 0:
        info = {"run": run, "cap": cap, "world": world, "total_steps": total_steps, "warmup": warmup,
                "decay_steps": decay_steps, "train_examples": len(train_shards),
                "train_tokens": int(train_shards.lengths.sum()), "lr": s.lr, "grad_accum": s.grad_accum,
                "settings": json.loads(s.model_dump_json()), **trainable_summary(inner), **s.provenance(),
                # The class that was trained, which is not always the one config.json declares
                # (Qwen3.5-9B declares its vision-language wrapper); evaluation must serve this one.
                "model_class": type(inner.get_base_model() if hasattr(inner, "get_base_model") else inner).__name__}
        (run_dir / "run.json").write_text(json.dumps(info, indent=2, default=str))
        log.info("run %s: %d examples (cap), %d steps, %d trainable params, world=%d, resuming at %d",
                 run, cap, total_steps, info["params_trainable"], world, state.examples)
        # What CUDA can actually address, which is not what nvidia-smi prints on a MIG slice: the
        # parent card reports 80 GB while the process is given 39.25 GiB. CLUSTER.md documents the
        # trap; an OOM report that quotes the parent card sends you looking in the wrong place.
        if torch.cuda.is_available():
            log.info("visible GPU memory: %.2f GiB on %s",
                     torch.cuda.get_device_properties(0).total_memory / 2**30,
                     torch.cuda.get_device_name(0))
        metrics.add("chunk_start", **{k: info[k] for k in ("cap", "world", "total_steps")},
                    examples=state.examples, step=state.step, host=os.uname().nodename,
                    job=os.environ.get("SLURM_JOB_ID", ""), gpu=torch.cuda.get_device_name(local))


    if state.examples == 0:  # the zero point of every curve: the base model, adapter at identity
        r = val_loss(inner, loss_shards, device, s.loss_eval_n, world, rank)
        if rank == 0:
            metrics.add("val_loss", examples=0, step=0, **r)
            log.info("val loss at 0 examples (%s): %.4f (ppl %.2f, %d papers, %.0fs)",
                     "the starting adapter" if adapter else "base model",
                     r["loss"], r["ppl"], r["papers"], r["seconds"])

    torch.cuda.reset_peak_memory_stats(device)
    state.chunks += 1
    window = _new_window()
    exit_code = 2
    next_loss_eval = ((state.examples // max(s.loss_eval_every, 1)) + 1) * s.loss_eval_every

    try:
        while state.examples < cap:
            batch_start = state.examples
            batch_end = min(batch_start + s.grad_accum * world, cap)
            batch = order[batch_start:batch_end]
            n_batch_tokens = int(sum(int(target_tokens[i]) for i in batch))
            mine = [i for n, i in enumerate(batch) if (batch_start + n) % world == rank]
            # A rank with nothing to do would never all-reduce and the others would wait for ever;
            # it repeats one example with a zero weight instead. Only possible in a final partial
            # batch (or a tiny smoke run), but a hang eleven hours in is expensive.
            idle_rank = world > 1 and not mine
            if idle_rank:
                mine = [batch[0]]
            opt.zero_grad(set_to_none=True)
            step_tokens = step_target = 0
            for k, idx in enumerate(mine):
                ex = train_shards.get(idx)
                ids = torch.from_numpy(ex.tokens.astype(np.int64)).unsqueeze(0).to(device, non_blocking=True)
                sync = (k == len(mine) - 1) or world == 1
                ctx = model.no_sync() if (world > 1 and not sync) else _nullcontext()
                with ctx:
                    loss, ntok = compute_loss(inner, ids, ex.prompt_len)
                    scale = 0.0 if idle_rank else world * ntok / max(n_batch_tokens, 1)
                    (loss * scale).backward()
                if idle_rank:
                    continue
                step_tokens += int(len(ex.tokens))
                step_target += ntok
                window["loss"] += float(loss.detach()) * ntok
                window["tokens"] += int(len(ex.tokens))
                window["target_tokens"] += ntok
                window["examples"] += 1
            grad_norm = float(torch.nn.utils.clip_grad_norm_(params, s.max_grad_norm))
            if math.isfinite(grad_norm):
                opt.step()
            else:
                # One bad batch must not poison 132 M parameters for the rest of the run; the
                # schedule still advances so the run stays comparable, and the count is logged.
                state.skipped_steps += 1
                log.warning("step %d: gradient norm is %s; skipping the update (%d skipped so far)",
                            state.step + 1, grad_norm, state.skipped_steps)
            sched.step()
            state.step += 1
            state.examples = batch_end
            state.tokens += step_tokens * world  # this rank's share scaled to the whole batch
            state.target_tokens += step_target * world
            window["steps"] += 1

            if rank == 0 and (state.step % s.log_every_steps == 0 or state.examples >= cap):
                dt = max(time.monotonic() - window["t0"], 1e-9)
                mean = window["loss"] / max(window["target_tokens"], 1)
                tps = window["tokens"] / dt
                eta_s = (cap - state.examples) * dt / max(window["examples"] * world, 1)
                vram = torch.cuda.max_memory_allocated(device) / 2**30
                metrics.add("train", step=state.step, examples=state.examples, tokens=state.tokens,
                            loss=round(mean, 5), lr=sched.get_last_lr()[0], grad_norm=round(grad_norm, 4),
                            tokens_per_s=round(tps, 1), eta_s=round(eta_s),
                            seconds=round(time.monotonic() - t_start + state.seconds, 1),
                            vram_gb=round(vram, 2))
                log.info("step %6d  ex %6d/%d  loss %.4f  lr %.2e  gnorm %.2f  %6.0f tok/s  %4.1f GB  eta %s",
                         state.step, state.examples, cap, mean, sched.get_last_lr()[0], grad_norm, tps,
                         vram, _eta(eta_s))
                window = _new_window()

            due_milestones = [m for m in s.milestones
                              if m not in state.milestones_done and state.examples >= m]
            if state.examples >= next_loss_eval or due_milestones:
                r = val_loss(inner, loss_shards, device, s.loss_eval_n, world, rank)
                if rank == 0:
                    metrics.add("val_loss", examples=state.examples, step=state.step,
                                milestone=due_milestones[-1] if due_milestones else None, **r)
                    log.info("val loss at %d examples: %.4f (ppl %.2f, %.0fs)%s", state.examples,
                             r["loss"], r["ppl"], r["seconds"],
                             f" [milestone {due_milestones}]" if due_milestones else "")
                if state.examples >= next_loss_eval:
                    next_loss_eval = ((state.examples // max(s.loss_eval_every, 1)) + 1) * s.loss_eval_every
                for m in due_milestones:
                    if rank == 0:
                        _save_milestone(s, run_dir, inner, state, m, r)
                    state.milestones_done.append(m)
                if world > 1:
                    torch.distributed.barrier()

            over_time = max_seconds is not None and (time.monotonic() - t_start) > max_seconds
            stop = STOP["flag"] or over_time
            if world > 1:  # one rank stopping while the others continue would hang on the next sync
                flag = torch.tensor([1.0 if stop else 0.0], device=device)
                torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
                stop = bool(flag.item())
            if stop:
                log.warning("stopping: %s", STOP["reason"] or "time limit for this chunk")
                break
            if rank == 0 and state.examples % rolling_every(s.save_every_examples, cap, s.grad_accum * world) < s.grad_accum * world:
                _save_rolling(run_dir, inner, opt, sched, state, t_start)
    except torch.cuda.OutOfMemoryError:
        log.exception("CUDA OOM at example %d", state.examples)
        exit_code = 3
    else:
        if state.examples >= cap:
            state.finished = True
            exit_code = 0
    finally:
        if rank == 0:
            # Only a clean end writes a checkpoint. After a crash the process state is suspect
            # (this is where a NaN would be), and the last rolling checkpoint is at most
            # `save_every_examples` papers behind and known good.
            if exit_code in (0, 2):
                try:
                    _save_rolling(run_dir, inner, opt, sched, state, t_start)
                except Exception:  # noqa: BLE001 - never let the final save hide the real failure
                    log.exception("the final checkpoint could not be written")
            else:
                log.warning("not writing a checkpoint after an unclean exit; the run resumes from %s",
                            (latest_checkpoint(run_dir) or Path("nothing")).name)
            metrics.add("chunk_end", examples=state.examples, step=state.step, exit_code=exit_code,
                        reason=STOP["reason"] or ("finished" if exit_code == 0 else "limit"),
                        skipped_steps=state.skipped_steps,
                        seconds=round(time.monotonic() - t_start, 1),
                        vram_peak_gb=round(torch.cuda.max_memory_allocated(device) / 2**30, 2))
            metrics.close()
        if world > 1:
            torch.distributed.destroy_process_group()
    log.info("chunk done: examples=%d step=%d exit=%d", state.examples, state.step, exit_code)
    return exit_code


def _new_window() -> dict:
    return {"loss": 0.0, "tokens": 0, "target_tokens": 0, "examples": 0, "t0": time.monotonic(), "steps": 0}


def _eta(seconds: float) -> str:
    return f"{seconds / 3600:.1f}h" if seconds > 3600 else f"{seconds / 60:.0f}m"


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


CKPT_KEEP = 2  # complete rolling checkpoints kept on disk


def _publish(tmp: Path, dest: Path) -> None:
    """Rename a fully written directory into place: one atomic step, no half-visible state."""
    if dest.exists():
        shutil.rmtree(dest)
    tmp.replace(dest)


def _atomic_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def checkpoints(run_dir: Path) -> list[Path]:
    """Complete rolling checkpoints, newest first.

    A checkpoint is a directory written under a temporary name and renamed when everything in it is
    on disk, with a ``DONE`` marker written last; ``LATEST`` names the current one. Resuming
    therefore cannot land on a half-written checkpoint, and if the newest one will not load
    (a truncated ``optim.pt``, a bad filesystem moment) the previous one is still there.
    """
    ck = run_dir / "ckpt"
    names: list[str] = []
    pointer = ck / "LATEST"
    if pointer.exists():
        names.append(pointer.read_text().strip())
    names += sorted((d.name for d in ck.glob("step-*") if d.is_dir()), reverse=True)
    out: list[Path] = []
    for name in names:
        d = ck / name
        if d not in out and (d / "DONE").exists() and (d / "adapter" / "adapter_config.json").exists():
            out.append(d)
    return out


def latest_checkpoint(run_dir: Path) -> Path | None:
    """The newest rolling checkpoint that is complete, or ``None`` before the first one."""
    found = checkpoints(run_dir)
    return found[0] if found else None



def rolling_every(save_every_examples: int, cap: int, per_step: int) -> int:
    """Papers between rolling checkpoints: the setting, but at least eight per run.

    The setting (2,500) was chosen for 70,000-paper runs. A 1,520-paper continuation never reached it,
    so when it ran out of memory at paper 1,104 on 2026-11-04 there was no checkpoint and the retry
    started from zero, 2.6 hours lost. Rounded to whole steps.
    """
    if cap // 8 >= save_every_examples:
        return save_every_examples                        # long runs: exactly as before
    every = max(cap // 8, per_step)
    return max(per_step, every - every % per_step)


def _save_rolling(run_dir: Path, model, opt, sched, state: RunState, t_start: float) -> Path:
    """One complete, self-contained checkpoint: adapter, optimiser, scheduler, RNG and counters."""
    ck = run_dir / "ckpt"
    ck.mkdir(parents=True, exist_ok=True)
    dest = ck / f"step-{state.step:08d}"
    tmp = ck / f".writing-step-{state.step:08d}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    model.save_pretrained(str(tmp / "adapter"))
    torch.save({"optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state()},
               tmp / "optim.pt")
    st = RunState(**{**asdict(state), "seconds": round(state.seconds + time.monotonic() - t_start, 1)})
    (tmp / "state.json").write_text(json.dumps(asdict(st), indent=2))
    (tmp / "DONE").write_text(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    _publish(tmp, dest)
    _atomic_text(ck / "LATEST", dest.name)
    st.save(run_dir / "state.json")  # the monitoring view: status, the evaluation worker, reports
    for old_ckpt in sorted((d for d in ck.glob("step-*") if d.is_dir()), reverse=True)[CKPT_KEEP:]:
        shutil.rmtree(old_ckpt, ignore_errors=True)
    log.info("checkpoint written: %d examples, step %d -> %s", state.examples, state.step, dest.name)
    return dest


def _save_milestone(s: Settings, run_dir: Path, model, state: RunState, milestone: int, loss: dict) -> None:
    """A milestone adapter plus its READY marker, published in one rename.

    The evaluation worker only looks at directories that carry READY, and because the directory
    appears complete or not at all, it can never pick up an adapter that is still being written.
    """
    dest = run_dir / "ckpt" / f"ex{milestone}"
    tmp = run_dir / "ckpt" / f".writing-ex{milestone}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    model.save_pretrained(str(tmp / "adapter"))
    (tmp / "READY").write_text(json.dumps({
        "milestone": milestone, "examples": state.examples, "step": state.step, "tokens": state.tokens,
        "target_tokens": state.target_tokens, "val_loss": loss, "base_model": s.base_model,
        "written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=2))
    _publish(tmp, dest)
    log.info("milestone %d: adapter saved to %s", milestone, dest)
