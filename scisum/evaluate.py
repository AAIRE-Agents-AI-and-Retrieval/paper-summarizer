"""Generation with vLLM, scoring, and the worker that evaluates milestones as they appear.

Generation and scoring are two separate processes on purpose: vLLM reserves most of the GPU for its
KV cache and does not give it back inside the process, and the scorers (BERTScore, NLI) need a GPU
of their own. ``scisum eval worker`` runs them as subprocesses, one checkpoint at a time, so a
training job and an evaluation job can share the cluster without fighting over memory.

Everything is content-addressed by ``<split>-<ckpt>-n<papers>``: predictions, per-paper scores and
the aggregate land next to each other under ``runs/<run>/eval/`` and are never recomputed if they
are already there (``--force`` overrides). That is what makes the whole evaluation resumable.
"""

from __future__ import annotations

import gzip
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

from scisum.config import Settings
from scisum.dataset import load_splits, read_prompts
from scisum.metrics import aggregate, group_by, parse_prediction, score_one

log = logging.getLogger(__name__)

_BODY_MARK = "\nFull text:\n"
_KEYS_MARK = "\nReturn a JSON object with exactly these keys:"
_ABSTRACT_MARK = "\nAbstract:\n"


def source_text(prompt: str) -> str:
    """The paper text inside a prompt: the abstract and the body, without the instructions."""
    body = ""
    if _BODY_MARK in prompt:
        body = prompt.split(_BODY_MARK, 1)[1]
        body = body.split(_KEYS_MARK, 1)[0]
    abstract = ""
    if _ABSTRACT_MARK in prompt:
        abstract = prompt.split(_ABSTRACT_MARK, 1)[1]
        abstract = abstract.split(_BODY_MARK, 1)[0].split(_KEYS_MARK, 1)[0]
    return (abstract.strip() + "\n\n" + body.strip()).strip() or prompt


def eval_name(split: str, ckpt: str, n: int) -> str:
    return f"{split}-{ckpt}-n{n}"


def eval_dir(s: Settings, run: str) -> Path:
    d = s.run_dir(run) / "eval"
    d.mkdir(parents=True, exist_ok=True)
    return d


def adapter_path(s: Settings, run: str, ckpt: str) -> Path | None:
    if ckpt in ("base", "none"):
        return None
    p = s.run_dir(run) / "ckpt" / ckpt / "adapter"
    if not (p / "adapter_config.json").exists():
        raise FileNotFoundError(f"no adapter at {p}")
    return p


def load_records(s: Settings, split: str, subset: str, limit: int | None) -> list[dict]:
    """Prompt records for an evaluation, in a fixed order, capped at the split's asked-for size."""
    sp = load_splits(s)
    cap = {"val": s.n_val, "test": s.n_test, "train": s.n_train}[split]
    wanted = set(sp[split][:cap])
    if subset == "curve":
        wanted &= set(sp["curve_eval"])
    elif subset.startswith("ids:"):   # an explicit list of papers (a JSON array of ids), in split order
        wanted &= set(json.loads(Path(subset[4:]).read_text()))
    recs = [r for r in read_prompts(s, split) if r["id"] in wanted]
    order = {rid: i for i, rid in enumerate(sp[split])}
    recs.sort(key=lambda r: order.get(r["id"], 10**9))
    return recs[:limit] if limit else recs


# ---- generation ---------------------------------------------------------------------------------
def vllm_architecture_override(config: dict) -> dict:
    """``hf_overrides`` that make vLLM build the class training built, or ``{}`` if it already does.

    Training loads with ``AutoModelForCausalLM``; vLLM loads whatever ``config.architectures``
    declares. For most models those agree. ``Qwen/Qwen3.5-9B`` declares the vision-language
    ``Qwen3_5ForConditionalGeneration``, under which the text layers sit at a different prefix, so a
    PEFT adapter trained on the text-only ``Qwen3_5ForCausalLM`` matched none of vLLM's modules:
    LoRA was enabled, nothing was applied, and three different checkpoints produced byte-identical
    base-model output (2026-10-18). Forcing the training class fixes the module names; vLLM's
    text-only class maps the checkpoint's ``model.language_model.`` weights itself.
    """
    from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

    trained = MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.get(config.get("model_type", ""))
    declared = (config.get("architectures") or [None])[0]
    if trained and declared and trained != declared:
        return {"architectures": [trained]}
    return {}


def serve_model_args(s: Settings) -> dict:
    """The keyword arguments that choose *which* model vLLM builds: path, revision, architecture.

    One function for generation and for scripts/probe_vllm_adapter.py, so the probe tests exactly
    what an evaluation serves. ``serve_model`` points at a local text-only copy when the hub
    checkpoint is a vision-language model whose vision tower vLLM's text-only class cannot load.
    """
    from transformers import AutoConfig

    if s.serve_model:
        path = str(Path(s.serve_model).resolve())
        args: dict = {"model": path, "tokenizer": path}
        cfg = AutoConfig.from_pretrained(path, trust_remote_code=s.trust_remote_code).to_dict()
    else:
        args = {"model": s.base_model,
                **({"revision": s.base_model_revision} if s.base_model_revision else {})}
        cfg = AutoConfig.from_pretrained(s.base_model, **s.hf_kwargs()).to_dict()
    overrides = vllm_architecture_override(cfg)
    if overrides:
        args["hf_overrides"] = overrides
    return args


def check_adapter_applied(llm, lora, prompts: list[str], max_tokens: int = 24) -> None:
    """Fail loudly if the adapter changes nothing: the same engine, with and without it.

    A trained adapter that is applied changes a greedy continuation within a few tokens - the
    student opens with JSON where the base model opens with prose. Identical output on every probe
    means vLLM accepted the adapter and matched none of its weights, which it does without a
    warning. Comparing against a separately generated base run would not catch this: enabling LoRA
    changes the kernels, so even an unapplied adapter does not reproduce the base output exactly.
    """
    from vllm import SamplingParams

    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    plain = [o.outputs[0].text for o in llm.generate(prompts, sp, use_tqdm=False)]
    tuned = [o.outputs[0].text for o in llm.generate(prompts, sp, use_tqdm=False, lora_request=lora)]
    if plain == tuned:
        raise RuntimeError(
            f"the LoRA adapter {lora.lora_path} has no effect on {len(prompts)} probe prompts: vLLM "
            "loaded it but matched none of its modules (check the architecture vLLM resolved "
            "against the one training used)")
    log.info("adapter check: %d of %d probe prompts change with the adapter",
             sum(a != b for a, b in zip(plain, tuned, strict=True)), len(prompts))


def run_generate(s: Settings, *, split: str, run: str, ckpt: str, limit: int | None = None,
                 subset: str = "curve", max_model_len: int | None = None, force: bool = False,
                 gpu_frac: float = 0.90, batch_papers: int = 100,
                 max_seconds: float | None = None) -> dict:
    """Generate the summaries for one checkpoint, in batches, appending as it goes.

    Batching keeps the job resumable (15 000 test papers take hours and a job can be killed at its
    walltime), bounds memory, and gives a progress line per batch. Already-generated papers are
    skipped, so re-running after an interruption costs nothing.

    The batch is small (100 papers) because the thing it protects against is losing *all* of an
    interrupted run: a 9 B model evaluating the untrained base checkpoint on an L40S managed fewer
    than 500 papers in a 55-minute slot and so wrote nothing at all.

    ``max_seconds`` stops between batches and reports ``complete=False``. Without it the caller is
    killed at its walltime mid-batch, which on 2026-10-18 lost the evaluation worker its chance to
    queue a successor and stalled a run's milestones at 4 of 7.
    """
    t_start = time.monotonic()
    # The engine window is the training window: a prompt is at most max_seq_len - gen_reserve and
    # the answer at most gen_max_new_tokens <= gen_reserve, so the two always fit inside it.
    max_model_len = max_model_len or s.max_seq_len
    recs = load_records(s, split, subset, limit)
    name = eval_name(split, ckpt, len(recs))
    out = eval_dir(s, run) / f"{name}.pred.jsonl.gz"
    meta_path = eval_dir(s, run) / f"{name}.gen.json"
    if force:
        out.unlink(missing_ok=True)
        meta_path.unlink(missing_ok=True)
    done: set[str] = set()
    if out.exists():
        with gzip.open(out, "rt") as f:
            done = {json.loads(line)["id"] for line in f}
        if len(done) >= len(recs) and meta_path.exists():
            log.info("predictions already complete: %s", out)
            return json.loads(meta_path.read_text())
    todo = [r for r in recs if r["id"] not in done]
    log.info("%s: %d papers, %d already generated, %d to go", name, len(recs), len(done), len(todo))

    adapter = adapter_path(s, run, ckpt)
    import torch
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    t0 = time.monotonic()
    # vLLM defaults to one GPU. A job that asked for two and used one spent 4 h of a 5.5 h slot at
    # 0.33 papers/s with the second card idle at 0% (2026-10-18), so the engine takes what it was
    # given unless `eval_tensor_parallel` says otherwise.
    tp = s.eval_tensor_parallel or max(1, torch.cuda.device_count())
    served = serve_model_args(s)
    overrides = served.get("hf_overrides", {})
    log.info("vLLM serves %s%s", served["model"],
             f" as {overrides['architectures'][0]}" if overrides else "")
    llm = LLM(dtype="bfloat16", max_model_len=max_model_len, tensor_parallel_size=tp,
              gpu_memory_utilization=gpu_frac, enable_lora=adapter is not None,
              max_lora_rank=s.lora_r, max_loras=1, seed=0, disable_log_stats=True,
              trust_remote_code=s.trust_remote_code, **served)
    load_s = time.monotonic() - t0
    lora = LoRARequest(ckpt, 1, str(adapter)) if adapter else None
    if lora and todo:
        check_adapter_applied(llm, lora, [r["prompt"] for r in todo[:4]])

    gen_s = 0.0
    n_out = n_in = 0
    for start in range(0, len(todo), batch_papers):
        batch = todo[start:start + batch_papers]
        params = [SamplingParams(temperature=s.gen_temperature, top_p=1.0, max_tokens=min(
            s.gen_max_new_tokens, max_model_len - int(r["meta"]["prompt_tokens"]) - 8)) for r in batch]
        t1 = time.monotonic()
        # use_tqdm=False: a progress bar in a Slurm log is noise, and vLLM 0.31 divides by its
        # elapsed time when it prints one, which is zero if tqdm is disabled by environment.
        outs = llm.generate([r["prompt"] for r in batch], params, use_tqdm=False,
                            **({"lora_request": lora} if lora else {}))
        dt = time.monotonic() - t1
        gen_s += dt
        with gzip.open(out, "at") as f:  # gzip members concatenate; readers see one stream
            for r, o in zip(batch, outs, strict=True):
                c = o.outputs[0]
                n_out += len(c.token_ids)
                n_in += len(o.prompt_token_ids)
                f.write(json.dumps({"id": r["id"], "text": c.text, "finish_reason": c.finish_reason,
                                    "out_tokens": len(c.token_ids),
                                    "prompt_tokens": len(o.prompt_token_ids)}, ensure_ascii=False) + "\n")
        log.info("  %d/%d papers, %.0f output tok/s, %.2f papers/s",
                 start + len(batch), len(todo), n_out / max(gen_s, 1e-9), (start + len(batch)) / max(gen_s, 1e-9))
        if max_seconds is not None and time.monotonic() - t_start > max_seconds:
            log.warning("generation time limit reached: %d of %d papers; the successor continues",
                        start + len(batch), len(todo))
            break

    with gzip.open(out, "rt") as f:
        total = sum(1 for _ in f)
    meta = {"name": name, "run": run, "ckpt": ckpt, "split": split, "subset": subset, "papers": total,
            "requested": len(recs), "generated_now": len(todo), "complete": total >= len(recs),
            "tensor_parallel": tp, "served_model": served["model"],
            "vllm_architecture": (overrides.get("architectures") or [None])[0],
            "adapter": str(adapter) if adapter else None, "max_model_len": max_model_len,
            "temperature": s.gen_temperature, "load_s": round(load_s, 1), "generate_s": round(gen_s, 1),
            "prompt_tokens": n_in, "output_tokens": n_out,
            "output_tokens_per_s": round(n_out / max(gen_s, 1e-9), 1),
            "papers_per_s": round(len(todo) / max(gen_s, 1e-9), 3),
            "gpu": os.environ.get("SLURM_JOB_GPUS", ""), **s.provenance()}
    meta_path.write_text(json.dumps(meta, indent=2))
    log.info("generated %d summaries in %.0fs (%.0f output tok/s)", len(todo), gen_s, meta["output_tokens_per_s"])
    return meta


# ---- scoring ------------------------------------------------------------------------------------
def _bertscore(s: Settings, preds: list[dict | None], refs: list[dict], device: str) -> list[dict]:
    """BERTScore F1 per field — per field because roberta-large truncates at 512 tokens and a whole
    summary does not fit in one encoder pass."""
    from scisum.bertscore import BertScorer
    from scisum.metrics import field_text

    fields = ("tldr", "problem", "method", "results", "contributions")
    scorer = BertScorer(model_name=s.bertscore_model, layer=s.bertscore_layer, device=device)
    cands: list[str] = []
    gold: list[str] = []
    for p, r in zip(preds, refs, strict=True):
        for k in fields:
            cands.append(field_text(p, k))
            gold.append(field_text(r, k))
    scores = scorer.score(cands, gold)
    out = []
    for i in range(len(preds)):
        row = {f"bertscore_{k}": scores[i * len(fields) + j]["f1"] for j, k in enumerate(fields)}
        row["bertscore_mean"] = round(sum(row.values()) / len(fields), 5)
        out.append(row)
    return out


def run_score(s: Settings, *, split: str, run: str, ckpt: str, limit: int | None = None,
              subset: str = "curve", faithfulness: bool = True, bertscore: bool = True,
              force: bool = False, device: str = "cuda") -> dict:
    recs = load_records(s, split, subset, limit)
    name = eval_name(split, ckpt, len(recs))
    pred_path = eval_dir(s, run) / f"{name}.pred.jsonl.gz"
    res_path = eval_dir(s, run) / f"{name}.json"
    if res_path.exists() and not force:
        log.info("scores already exist: %s", res_path)
        return json.loads(res_path.read_text())
    if not pred_path.exists():
        raise FileNotFoundError(f"no predictions at {pred_path}; run `scisum eval generate` first")

    with gzip.open(pred_path, "rt") as f:
        preds = {r["id"]: r for r in (json.loads(line) for line in f)}
    from scisum.dataset import tokenizer
    from scisum.prompt import assistant_wrapper

    prefix, _ = assistant_wrapper(tokenizer(s))   # what this model's template makes it say first
    rows: list[dict] = []
    objs: list[dict | None] = []
    t0 = time.monotonic()
    for r in recs:
        p = preds.get(r["id"])
        if p is None:
            continue
        row = score_one(p["text"], r["summary"], finish_reason=p["finish_reason"],
                        out_tokens=p["out_tokens"], target_prefix=prefix)
        obj, _ = parse_prediction(p["text"].lstrip()[len(prefix):]
                                  if p["text"].lstrip().startswith(prefix) else p["text"])
        objs.append(obj)
        row.update({"id": r["id"], "teacher": r["meta"]["teacher"], "cohort": r["meta"]["cohort"],
                    "trunc_kind": r["meta"]["trunc_kind"], "fits": r["meta"]["fits"],
                    "prompt_tokens": r["meta"]["prompt_tokens"], "pages": r["meta"]["pages"]})
        rows.append(row)
    text_s = time.monotonic() - t0

    if bertscore and rows:
        t1 = time.monotonic()
        for row, extra in zip(rows, _bertscore(s, objs, [r["summary"] for r in recs if r["id"] in preds],
                                               device), strict=True):
            row.update(extra)
        log.info("BERTScore: %.0fs", time.monotonic() - t1)
    if faithfulness and rows:
        from scisum.faithfulness import NliScorer, score_papers

        n = min(s.faithfulness_n, len(rows))
        t1 = time.monotonic()
        scorer = NliScorer(s.nli_model, device=device)
        by_id = {r["id"]: r for r in recs}
        items = [(row["id"], objs[i], source_text(by_id[row["id"]]["prompt"])) for i, row in enumerate(rows[:n])]
        scores = score_papers(items, scorer)
        for row in rows[:n]:
            row.update(scores.get(row["id"], {}))
        log.info("faithfulness on %d papers: %.0fs", n, time.monotonic() - t1)

    scores_path = eval_dir(s, run) / f"{name}.scores.jsonl.gz"
    with gzip.open(Path(f"{scores_path}.tmp"), "wt") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    Path(f"{scores_path}.tmp").replace(scores_path)

    ready = s.run_dir(run) / "ckpt" / ckpt / "READY"
    result = {"name": name, "run": run, "ckpt": ckpt, "split": split, "subset": subset,
              "papers": len(rows), "scored_seconds": round(time.monotonic() - t0, 1),
              "text_metrics_seconds": round(text_s, 1),
              "checkpoint": json.loads(ready.read_text()) if ready.exists() else {},
              "generation": json.loads((eval_dir(s, run) / f"{name}.gen.json").read_text())
              if (eval_dir(s, run) / f"{name}.gen.json").exists() else {},
              "metrics": aggregate(rows), "by_teacher": group_by(rows, "teacher"),
              "by_truncation": group_by(rows, "trunc_kind"), **s.provenance()}
    tmp = Path(f"{res_path}.tmp")
    tmp.write_text(json.dumps(result, indent=2))
    tmp.replace(res_path)
    m = result["metrics"]
    log.info("%s: json_ok=%.3f schema_ok=%.3f rougeL=%.4f bertscore=%.4f faithfulness=%.4f",
             name, m.get("json_ok", 0), m.get("schema_ok", 0), m.get("all_rougeL", 0),
             m.get("bertscore_mean", 0), m.get("faithfulness", 0))
    return result


def rescore_text(s: Settings, *, run: str, split: str = "val") -> list[dict]:
    """Recompute the text-only metrics of every stored evaluation, keeping the GPU-derived ones.

    For when a metric definition changes but the generations have not: nothing is regenerated, the
    BERTScore and faithfulness columns are carried over from the stored per-paper rows, and the
    aggregate is rebuilt. Runs on CPU in seconds.
    """
    from scisum.dataset import tokenizer
    from scisum.prompt import assistant_wrapper

    prefix, _ = assistant_wrapper(tokenizer(s))
    keep = ("bertscore_", "faithfulness", "unsupported_rate", "claims")
    out = []
    for res_path in sorted(eval_dir(s, run).glob(f"{split}-*.json")):
        if res_path.name.endswith(".gen.json"):
            continue
        result = json.loads(res_path.read_text())
        name = result["name"]
        scores_path = eval_dir(s, run) / f"{name}.scores.jsonl.gz"
        pred_path = eval_dir(s, run) / f"{name}.pred.jsonl.gz"
        if not (scores_path.exists() and pred_path.exists()):
            continue
        with gzip.open(pred_path, "rt") as f:
            preds = {r["id"]: r for r in (json.loads(line) for line in f)}
        with gzip.open(scores_path, "rt") as f:
            old_rows = [json.loads(line) for line in f]
        recs = {r["id"]: r for r in load_records(s, split, result.get("subset", "curve"), None)}
        rows = []
        for old in old_rows:
            p, rec = preds.get(old["id"]), recs.get(old["id"])
            if p is None or rec is None:
                rows.append(old)
                continue
            row = score_one(p["text"], rec["summary"], finish_reason=p["finish_reason"],
                            out_tokens=p["out_tokens"], target_prefix=prefix)
            row.update({k: v for k, v in old.items() if k.startswith(keep) or k in
                        ("id", "teacher", "cohort", "trunc_kind", "fits", "prompt_tokens", "pages")})
            rows.append(row)
        tmp = Path(f"{scores_path}.tmp")
        with gzip.open(tmp, "wt") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        tmp.replace(scores_path)
        result["metrics"] = aggregate(rows)
        result["by_teacher"] = group_by(rows, "teacher")
        result["by_truncation"] = group_by(rows, "trunc_kind")
        result["rescored_text_metrics"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        res_path.write_text(json.dumps(result, indent=2))
        out.append({"name": name, "papers": len(rows), "json_ok": result["metrics"].get("json_ok"),
                    "schema_ok": result["metrics"].get("schema_ok")})
        log.info("rescored %s: json_ok=%.3f schema_ok=%.3f", name, result["metrics"].get("json_ok", 0),
                 result["metrics"].get("schema_ok", 0))
    return out


# ---- the worker ---------------------------------------------------------------------------------
def _cli(*args: str) -> int:
    cmd = [sys.executable, "-m", "scisum.cli", *args]
    log.info("+ %s", " ".join(cmd))
    return subprocess.run(cmd, check=False).returncode


def pending_checkpoints(s: Settings, run: str) -> list[str]:
    """Milestone checkpoints that are ready and not yet evaluated, in milestone order."""
    base = s.run_dir(run) / "ckpt"
    out = []
    for d in sorted(base.glob("ex*")):
        if (d / "READY").exists() and not (d / "EVALUATED").exists():
            out.append(d.name)
    return sorted(out, key=lambda n: int(n[2:]))


def run_worker(s: Settings, *, run: str, poll_s: float = 60.0, max_seconds: float | None = None,
               split: str = "val", once: bool = False, limit: int | None = None) -> dict:
    """Evaluate every milestone checkpoint as training produces it (and the base model first)."""
    t0 = time.monotonic()
    done: list[str] = []
    while True:
        todo = pending_checkpoints(s, run)
        for ckpt in todo:
            # `is not None`: a budget of 0 means no time left, not no limit.
            if max_seconds is not None and time.monotonic() - t0 > max_seconds:
                break
            lim = ["--limit", str(limit)] if limit else []
            # The time budget goes *into* generation: checked only between checkpoints, a slow model
            # ran one checkpoint straight through the walltime and the job died before it could
            # queue its successor (Student C, 2026-10-18, milestones stalled at 4 of 7).
            ms = [] if max_seconds is None else [
                "--max-seconds", str(int(max(60.0, max_seconds - (time.monotonic() - t0))))]
            rc = _cli("eval", "generate", "--run", run, "--ckpt", ckpt, "--split", split, *lim, *ms)
            if rc == 3:
                log.info("%s: generation reached the time limit; the successor continues", ckpt)
                break
            if rc == 0:
                rc = _cli("eval", "score", "--run", run, "--ckpt", ckpt, "--split", split, *lim)
            # Network-bound and free; a judge that fails must not cost the checkpoint its score.
            if rc == 0 and s.judge_enabled and _cli(
                    "eval", "judge", "--run", run, "--ckpt", ckpt, "--split", split, *lim) != 0:
                log.warning("the judge failed for %s; its scores stand without it", ckpt)
            if rc == 0:
                (s.run_dir(run) / "ckpt" / ckpt / "EVALUATED").write_text(time.strftime("%FT%TZ", time.gmtime()))
                done.append(ckpt)
                _cli("curves", "--run", run)
            else:
                log.error("evaluation of %s failed with %d; leaving it pending", ckpt, rc)
        over = max_seconds is not None and time.monotonic() - t0 > max_seconds
        if once or over:
            break
        state = s.run_dir(run) / "state.json"
        if state.exists() and json.loads(state.read_text()).get("finished") and not pending_checkpoints(s, run):
            log.info("training is finished and every checkpoint is evaluated")
            break
        time.sleep(poll_s)
    return {"run": run, "evaluated": done, "pending": pending_checkpoints(s, run),
            "seconds": round(time.monotonic() - t0, 1)}
