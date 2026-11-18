"""Reward-selected distillation: train the student to imitate the teacher *and* to be right.

Plain distillation maximises the likelihood of the teacher's text. That teaches the teacher's
wording, and also its omissions and its occasional errors; it cannot tell a summary that says the
right thing differently from one that says the wrong thing similarly. Measured on 299 test papers
(paper_docs/05_results.md) the students trail the teacher by *omission* (26.8% of questions
unanswered against 23.1%), and the fabrications that do occur are invented numbers.

This module adds the missing signal as a round of expert iteration, with the teacher kept as an
anchor in the objective rather than as the only target:

1. **Pool.** A set of papers the policy has not trained on (validation papers outside the
   monitoring subset), with the teacher's summary for each.
2. **Sample.** The current policy writes ``imp_k`` summaries per paper at temperature > 0.
3. **Questions.** A model writes ``imp_n_questions`` specific questions with gold answers from the
   paper's full text. Per paper, two reference readers separate the questions worth keeping: the
   whole paper (must answer correctly: the question is valid and answerable) and nothing at all
   (must *not* answer correctly: the question does not leak its own answer).
4. **Reward.** Every candidate (the teacher's summary and each sample) is read by a model that sees
   only that summary, and graded against the gold:

       R = fact + lambda * imitation - beta * unsupported_numbers - gamma * excess_length

   ``fact`` is the mean over kept questions of +1 correct, +0.25 partial, 0 not stated, -1 wrong:
   a wrong answer costs more than a missing one, so hedging is not free and fabricating is worse.
   ``imitation`` is ROUGE-L against the teacher, which is 1 for the teacher itself, so the teacher
   is replaced only by a sample that is more right by more than it is less like the teacher.
   ``unsupported_numbers`` is the share of the summary's numbers that appear nowhere in the paper.
5. **Select.** Each paper's training target is its highest-reward candidate, and a sample replaces
   the teacher only if it beats it by ``imp_margin`` (grader noise would otherwise make the choice
   random). The control arm trains on the same papers with the teacher's targets alone, so any
   difference is the selection and not the extra training.

The roles are crossed with the evaluation's: here a different model writes the questions than the
one that writes the test's, and a different one answers and grades, so the student is not tuned to
the instrument that scores it. It is still a model's reading of the paper; see T16.

Everything is resumable and append-only, so a killed job is restarted, not repeated.
"""

from __future__ import annotations

import ast
import asyncio
import collections
import gzip
import hashlib
import json
import logging
import math
import re
import statistics as st
import time
from pathlib import Path

from scisum.config import Settings
from scisum.factcheck import (
    ANS_PROMPT,
    ANS_SYSTEM,
    CLOSED_PROMPT,
    GEN_PROMPT,
    GEN_SYSTEM,
    GRADE_SYSTEM,
    Chat,
    answer_and_grade,
    append_jsonl,
    parse_obj,
    read_jsonl,
    render_for_reader,
)
from scisum.metrics import parse_prediction, render_summary, rouge
from scisum.prompt import SUMMARY_KEYS, canonical_target, validate_summary

log = logging.getLogger(__name__)

QUOTA = {"problem": 1, "motivation": 1, "approach": 3, "setup": 3, "finding": 3, "limitation": 1, "scope": 2}
VALUE = {"correct": 1.0, "partial": 0.25, "not_stated": 0.0, "ungraded": 0.0, "incorrect": -1.0}


# ---- paths and small I/O -------------------------------------------------------------------------
# What a pool paper's questions and reference readers are does not depend on the student: they come from the
# paper. They are shipped once (``<data root>/improve_pool/``) and copied into each student's own
# ``state/<imp_dir>`` the first time it is used, so every student is rewarded on the same questions and no
# grader time is spent writing them again. The pool itself (``pool.jsonl.gz``) is NOT shared: its prompts
# are rendered with the student's own chat template and token budget, so each student builds its own from
# the same fixed paper order. Delete a copied file to have it regenerated with the current graders.
SHARED_POOL_FILES = ("questions.jsonl.gz", "questions-failed.jsonl.gz", "refs.jsonl.gz")


def shared_pool_dir(s: Settings) -> Path:
    """``SCISUM_IMP_SHARED_DIR``, default ``<data root>/improve_pool``."""
    import os

    from scisum.factcheck import data_root

    return Path(os.environ.get("SCISUM_IMP_SHARED_DIR") or (data_root(s) / "improve_pool"))


def seed_shared(s: Settings, d: Path) -> list[str]:
    """Copy the student-independent files into ``d`` where they are missing; returns what was copied."""
    import shutil

    src = shared_pool_dir(s)
    if not src.is_dir() or src.resolve() == d.resolve():
        return []
    copied = []
    for name in SHARED_POOL_FILES:
        if (src / name).exists() and not (d / name).exists():
            tmp = d / (name + ".tmp")
            shutil.copyfile(src / name, tmp)
            tmp.rename(d / name)
            copied.append(name)
    if copied:
        log.info("seeded %s with the shared %s from %s", d.name, ", ".join(copied), src)
    return copied


def idir(s: Settings) -> Path:
    d = s.state_dir / s.imp_dir
    d.mkdir(parents=True, exist_ok=True)
    seed_shared(s, d)
    return d


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with gzip.open(tmp, "wt") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


# ---- the reward, as pure functions ---------------------------------------------------------------
_NUM = re.compile(r"(?<![\w.])[+-]?\d[\d,]*(?:\.\d+)?(?:[eE][+-]?\d+)?")


def numbers(text: str) -> set[str]:
    """The numbers worth checking in a text, normalised.

    Single digits are skipped (counts and enumerations: "three contributions"), and so are years,
    which a summary may state from general knowledge. Commas and signs are dropped, so "1,000" and
    "1000" match.
    """
    out = set()
    for m in _NUM.finditer(text):
        tok = m.group(0).replace(",", "").lstrip("+-")
        digits = re.sub(r"\D", "", tok.split("e")[0].split("E")[0])
        if len(digits) < 2:
            continue
        if re.fullmatch(r"(19|20)\d\d", tok):
            continue
        out.add(tok.rstrip("."))
    return out


def numeric_support(obj: dict, paper: str) -> tuple[int, int]:
    """(numbers in the summary, how many of them occur nowhere in the paper)."""
    mine = numbers(render_summary(obj))
    theirs = numbers(paper)
    return len(mine), len(mine - theirs)


def question_weights(full: list[str], closed: list[str]) -> list[int]:
    """Keep a question only if the whole paper answers it correctly and a guesser does not.

    The first condition drops questions whose gold answer is wrong or ambiguous (even a perfect
    reader misses them: about one in nine); the second drops questions that answer themselves.
    """
    return [int(f == "correct" and c != "correct") for f, c in zip(full, closed, strict=True)]


def fact_reward(verdicts: list[str], weights: list[int], min_questions: int) -> float | None:
    n = sum(weights)
    if n < min_questions:
        return None
    return sum(w * VALUE.get(v, 0.0) for v, w in zip(verdicts, weights, strict=True)) / n


def features(obj: dict | None, finish: str, teacher: dict, paper: str) -> dict:
    """Everything about a candidate that needs no model: validity, imitation, numbers, length."""
    if obj is None or finish != "stop" or not validate_summary(obj).ok:
        return {"valid": False}
    mine, theirs = render_summary(obj), render_summary(teacher)
    total, bad = numeric_support(obj, paper)
    return {"valid": True, "rougeL": rouge(mine, theirs)["rougeL"], "numbers": total, "unsupported": bad,
            "unsupported_frac": bad / total if total else 0.0, "length_ratio": len(mine) / max(len(theirs), 1)}


def composite(fact: float, feat: dict, s: Settings) -> float:
    """R = fact + lambda * imitation - beta * unsupported_numbers - gamma * excess_length."""
    excess = max(0.0, feat["length_ratio"] - s.imp_len_free)
    return fact + s.imp_lambda * feat["rougeL"] - s.imp_beta_num * feat["unsupported_frac"] - s.imp_gamma_len * excess


def choose(rewards: list[float | None], margin: float) -> int:
    """Index of the candidate to train on. ``rewards[0]`` is the teacher's.

    A sample replaces the teacher only if it is better by ``margin``; ``None`` marks a candidate
    that is invalid or could not be scored and is never chosen.
    """
    best = 0
    base = rewards[0] if rewards[0] is not None else float("-inf")
    top = base + margin
    for i, r in enumerate(rewards[1:], start=1):
        if r is not None and r > top:
            best, top = i, r
    return best


def parse_candidate(text: str, prefix: str) -> dict | None:
    """The summary object in a generated reply, with the template's own prefix removed."""
    t = text.lstrip()
    p = prefix.lstrip()
    if p and t.startswith(p):
        t = t[len(p):]
    obj, _ = parse_prediction(t)
    return obj if isinstance(obj, dict) else None


# ---- 1. the pool ---------------------------------------------------------------------------------
def select_pool(s: Settings) -> list[dict]:
    """Validation papers outside the monitoring subset, with their prompts and the teacher's summary."""
    path = idir(s) / "pool.jsonl.gz"
    if path.exists():
        return read_jsonl(path)[: s.imp_pool]
    from scisum.dataset import load_splits, read_prompts
    from scisum.evaluate import source_text

    sp = load_splits(s)
    keep_out = set(sp["curve_eval"])
    ids = [i for i in sp["val"][:s.n_val] if i not in keep_out][: s.imp_pool * (40 if s.imp_max_prompt_tokens else 1)]
    cap = s.imp_max_prompt_tokens        # only the smoke test sets it: scan further and keep the first papers that fit
    wanted = set(ids)
    rows: dict[str, dict] = {}
    for rec in read_prompts(s, "val"):
        if rec["id"] in wanted and (not cap or (rec["meta"].get("prompt_tokens") or 0) <= cap):
            rows[rec["id"]] = {
                "id": rec["id"], "prompt": rec["prompt"], "target": rec["target"],
                "teacher": {k: rec["summary"][k] for k in SUMMARY_KEYS},
                "paper": source_text(rec["prompt"])[: s.fact_paper_chars],
                "cohort": rec["meta"].get("cohort"), "teacher_model": rec["meta"].get("teacher"),
                "prompt_tokens": rec["meta"].get("prompt_tokens")}
    out = [rows[i] for i in ids if i in rows][: s.imp_pool]
    # The shared questions and reference verdicts were made from Student C's copy of each paper (``paper``).
    # Keep that copy for every student, so the reward's questions, refs and claim check see the same text;
    # only the prompt (chat template, tokenizer, budget) is the student's own.
    shared = shared_pool_dir(s) / "pool.jsonl.gz"
    if shared.exists():
        theirs = {r["id"]: r for r in read_jsonl(shared)}
        for r in out:
            if r["id"] in theirs:
                r["paper"] = theirs[r["id"]]["paper"]
        log.info("pool: 'paper' taken from the shared pool for %d of %d papers",
                 sum(r["id"] in theirs for r in out), len(out))
    write_jsonl_atomic(path, out)
    log.info("pool: %d papers (%d requested)", len(out), len(ids))
    return out


# ---- 2. samples from the current policy ----------------------------------------------------------
def run_sample(s: Settings, max_seconds: float | None = None, batch_papers: int = 40) -> dict:
    """``imp_k`` samples per pool paper from ``init_adapter``, appended every ``batch_papers``."""
    t0 = time.monotonic()
    pool = select_pool(s)
    path = idir(s) / "samples.jsonl.gz"
    done = {r["id"] for r in read_jsonl(path)}
    todo = [p for p in pool if p["id"] not in done]
    log.info("sample: %d papers, %d done, %d to go", len(pool), len(done), len(todo))
    if not todo:
        return {"sampled": 0, "remaining": 0}
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    from scisum.evaluate import check_adapter_applied, serve_model_args

    adapter = Path(s.init_adapter).resolve()
    if not (adapter / "adapter_config.json").exists():
        raise FileNotFoundError(f"no adapter at {adapter}: set SCISUM_INIT_ADAPTER to the policy to improve")
    llm = LLM(dtype="bfloat16", max_model_len=s.max_seq_len, tensor_parallel_size=s.eval_tensor_parallel or 1,
              gpu_memory_utilization=0.90, enable_lora=True, max_lora_rank=s.lora_r, max_loras=1, seed=0,
              disable_log_stats=True, trust_remote_code=s.trust_remote_code, **serve_model_args(s))
    lora = LoRARequest("policy", 1, str(adapter))
    check_adapter_applied(llm, lora, [p["prompt"] for p in todo[:4]])
    made = 0
    for i in range(0, len(todo), batch_papers):
        if max_seconds is not None and time.monotonic() - t0 > max_seconds:
            log.warning("sampling time limit reached after %d papers; the successor continues", made)
            break
        batch = todo[i:i + batch_papers]
        room = [min(s.gen_max_new_tokens, s.max_seq_len - int(p["prompt_tokens"] or 0) - 8) for p in batch]
        prompts = [p["prompt"] for p in batch]
        outs = llm.generate(prompts, [SamplingParams(n=s.imp_k, temperature=s.imp_temperature, top_p=s.imp_top_p,
                                                     max_tokens=m) for m in room],
                            use_tqdm=False, lora_request=lora)
        greedy = (llm.generate(prompts, [SamplingParams(temperature=0.0, max_tokens=m) for m in room],
                               use_tqdm=False, lora_request=lora) if s.imp_greedy else [None] * len(batch))
        rows = []
        for p, o, g in zip(batch, outs, greedy, strict=True):
            cands = ([{"text": g.outputs[0].text, "finish": g.outputs[0].finish_reason, "tokens": len(g.outputs[0].token_ids),
                       "kind": "greedy"}] if g is not None else [])
            cands += [{"text": c.text, "finish": c.finish_reason, "tokens": len(c.token_ids), "kind": "sampled"} for c in o.outputs]
            rows.append({"id": p["id"], "samples": cands})
        append_jsonl(path, rows)
        made += len(batch)
        log.info("  sampled %d/%d papers (%.0f s)", i + len(batch), len(todo), time.monotonic() - t0)
    return {"sampled": made, "remaining": len(todo) - made}


# ---- 3-4. questions, reference readers, candidate readers ----------------------------------------
def _chats(s: Settings, concurrency: int):
    writer = Chat(s, concurrency, s.imp_writer, GEN_SYSTEM, 3_500)
    ans = Chat(s, concurrency, s.imp_grader, ANS_SYSTEM, 1_800)
    grd = Chat(s, concurrency, s.imp_grader, GRADE_SYSTEM, 1_500)
    return writer, ans, grd


def _qtext(questions: list[dict]) -> str:
    return "\n".join(f"{i + 1}. {q['question']}" for i, q in enumerate(questions))


async def _chunks(items: list, fn, size: int = 64):
    for i in range(0, len(items), size):
        yield [r for r in await asyncio.gather(*(fn(x) for x in items[i:i + size])) if r is not None]


async def stage_questions(s: Settings, pool: list[dict], writer: Chat, deadline: float) -> int:
    path, bad = idir(s) / "questions.jsonl.gz", idir(s) / "questions-failed.jsonl.gz"
    done = {r["id"] for r in read_jsonl(path)} | {r["id"] for r in read_jsonl(bad)}
    todo = [p for p in pool if p["id"] not in done]
    n = s.imp_n_questions
    quota = ", ".join(f"{v} {k}" for k, v in QUOTA.items())
    log.info("questions: %d to write", len(todo))

    async def one(p: dict):
        for attempt in range(3):
            if time.monotonic() > deadline:
                return None
            text, meta = await writer.one(GEN_PROMPT.format(n=n, quota=quota, paper=p["paper"]))
            qs = [q for q in parse_obj(text, "questions") if isinstance(q, dict) and q.get("question") and q.get("answer")]
            if len(qs) >= n // 2:
                return {"id": p["id"], "questions": [{"type": str(q.get("type", "")), "question": str(q["question"]),
                                                      "answer": str(q["answer"])} for q in qs[:n]]}
            log.warning("%s: %d usable questions (attempt %d, %s)", p["id"], len(qs), attempt + 1,
                        meta.get("error") or meta.get("finish_reason"))
        append_jsonl(bad, [{"id": p["id"]}])
        return None

    made = 0
    async for rows in _chunks(todo, one):
        append_jsonl(path, rows)
        made += len(rows)
        log.info("  questions %d/%d", made, len(todo))
    return made


async def stage_refs(s: Settings, pool: list[dict], ans: Chat, grd: Chat, deadline: float) -> int:
    """Closed-book and whole-paper readers, per paper: which questions are worth keeping."""
    qs = {r["id"]: r["questions"] for r in read_jsonl(idir(s) / "questions.jsonl.gz")}
    path = idir(s) / "refs.jsonl.gz"
    done = {r["id"] for r in read_jsonl(path)}
    todo = [p for p in pool if p["id"] in qs and p["id"] not in done]
    log.info("reference readers: %d papers", len(todo))

    async def one(p: dict):
        if time.monotonic() > deadline:
            return None
        q = qs[p["id"]]
        closed_prompt = CLOSED_PROMPT.format(n=len(q), questions=_qtext(q))
        full_prompt = ANS_PROMPT.format(n=len(q), summary=render_for_reader({"paper_text": p["paper"]}), questions=_qtext(q))
        closed, full = await asyncio.gather(answer_and_grade(ans, grd, closed_prompt, q),
                                            answer_and_grade(ans, grd, full_prompt, q))
        return {"id": p["id"], "closed": closed, "full": full, "weights": question_weights(full, closed)}

    made = 0
    async for rows in _chunks(todo, one, size=24):
        append_jsonl(path, rows)
        made += len(rows)
        log.info("  references %d/%d", made, len(todo))
    return made


def candidate_objs(s: Settings, pool_row: dict, samples: list[dict], prefix: str) -> list[tuple[dict | None, str]]:
    """[(summary object or None, finish reason)] for the teacher and then each sample."""
    out = [(pool_row["teacher"], "stop")]
    for c in samples:
        out.append((parse_candidate(c["text"], prefix), c["finish"]))
    return out


def _fingerprint(obj: dict) -> str:
    return hashlib.sha1(render_for_reader(obj).encode()).hexdigest()


async def stage_candidates(s: Settings, pool: list[dict], ans: Chat, grd: Chat, deadline: float) -> int:
    """Read every valid candidate and grade it. Identical candidates are read once."""
    from scisum.dataset import tokenizer
    from scisum.prompt import assistant_wrapper

    prefix, _ = assistant_wrapper(tokenizer(s))
    qs = {r["id"]: r["questions"] for r in read_jsonl(idir(s) / "questions.jsonl.gz")}
    refs = {r["id"]: r for r in read_jsonl(idir(s) / "refs.jsonl.gz")}
    samples = {r["id"]: r["samples"] for r in read_jsonl(idir(s) / "samples.jsonl.gz")}
    path = idir(s) / "cand-grades.jsonl.gz"
    done = {r["id"] for r in read_jsonl(path)}
    todo = [p for p in pool if p["id"] in qs and p["id"] in refs and p["id"] in samples and p["id"] not in done
            and sum(refs[p["id"]]["weights"]) >= s.imp_min_questions]
    log.info("candidates: %d papers to read", len(todo))

    async def one(p: dict):
        if time.monotonic() > deadline:
            return None
        q = qs[p["id"]]
        cands = candidate_objs(s, p, samples[p["id"]], prefix)
        cache: dict[str, asyncio.Future] = {}
        verdicts: list[list[str] | None] = []
        tasks = []
        for obj, finish in cands:
            if obj is None or finish != "stop" or not validate_summary(obj).ok:
                tasks.append(None)
                continue
            key = _fingerprint(obj)
            if key not in cache:
                prompt = ANS_PROMPT.format(n=len(q), summary=render_for_reader(obj), questions=_qtext(q))
                cache[key] = asyncio.ensure_future(answer_and_grade(ans, grd, prompt, q))
            tasks.append(cache[key])
        for t in tasks:
            verdicts.append(None if t is None else await t)
        return {"id": p["id"], "verdicts": verdicts}

    made = 0
    async for rows in _chunks(todo, one, size=20):
        append_jsonl(path, rows)
        made += len(rows)
        log.info("  candidates %d/%d papers", made, len(todo))
    return made


async def run_reward(s: Settings, max_seconds: float = 3000, concurrency: int = 24) -> dict:
    """Score the pool in a window that stays just ahead of the sampler.

    Sampling is the slow stage, and it goes through the pool in order, so grading every paper's
    questions first would spend hours of endpoint time on papers the round never reaches and delay
    the grading of the ones it has. Each pass works on the first ``sampled + imp_lookahead`` papers:
    candidates of papers that are sampled, then reference readers, then questions. When a pass finds
    nothing to do it waits a minute for the sampler.
    """
    deadline = time.monotonic() + max_seconds
    pool = select_pool(s)
    writer, ans, grd = _chats(s, concurrency)
    totals = {"questions": 0, "references": 0, "candidates": 0}
    last_log = 0.0
    while time.monotonic() < deadline:
        sampled = len({r["id"] for r in read_jsonl(idir(s) / "samples.jsonl.gz")})
        window = pool[: min(len(pool), sampled + s.imp_lookahead)]
        # The three stages run at once: the lookahead (questions, reference readers) is needed an hour from
        # now and must not delay the grading of papers that are ready now. A paper whose references finish
        # during this pass is picked up by the next one.
        cand, refs_made, questions_made = await asyncio.gather(
            stage_candidates(s, window, ans, grd, deadline), stage_refs(s, window, ans, grd, deadline),
            stage_questions(s, window, writer, deadline))
        did = {"candidates": cand, "references": refs_made, "questions": questions_made}
        for k, v in did.items():
            totals[k] += v
        if reward_remaining(s, pool) == 0:
            break
        if not any(did.values()):
            if time.monotonic() - last_log > 600:
                log.info("waiting for the sampler: %d papers sampled, window %d", sampled, len(window))
                last_log = time.monotonic()
            await asyncio.sleep(min(60.0, max(deadline - time.monotonic(), 0)))
    for c in (writer, ans, grd):
        await c.close()
    totals["remaining"] = reward_remaining(s, pool)
    return totals


def reward_remaining(s: Settings, pool: list[dict]) -> int:
    """Papers still missing questions, reference readers, or (when samples exist) candidate grades."""
    d = idir(s)
    have_q = {r["id"] for r in read_jsonl(d / "questions.jsonl.gz")} | {r["id"] for r in read_jsonl(d / "questions-failed.jsonl.gz")}
    refs = {r["id"]: r for r in read_jsonl(d / "refs.jsonl.gz")}
    graded = {r["id"] for r in read_jsonl(d / "cand-grades.jsonl.gz")}
    failed = {r["id"] for r in read_jsonl(d / "questions-failed.jsonl.gz")}
    left = sum(p["id"] not in have_q for p in pool)
    left += sum(p["id"] not in refs for p in pool if p["id"] in have_q and p["id"] not in failed)
    for p in pool:
        r = refs.get(p["id"])
        # a paper with enough usable questions is finished only once its samples exist AND are read
        if r is not None and sum(r["weights"]) >= s.imp_min_questions and p["id"] not in graded:
            left += 1
    return left


# ---- 5. selection --------------------------------------------------------------------------------
def finished_ids(s: Settings, pool: list[dict]) -> set[str]:
    """Papers that are completely done: sampled, and either scored or known to have too few usable questions.

    Selection and the training sets use only these, so the round can stop at whatever size the time
    allows (the pool is processed in order) and every arm still sees the same papers.
    """
    d = idir(s)
    sampled = {r["id"] for r in read_jsonl(d / "samples.jsonl.gz")}
    failed = {r["id"] for r in read_jsonl(d / "questions-failed.jsonl.gz")}
    refs = {r["id"]: r for r in read_jsonl(d / "refs.jsonl.gz")}
    graded = {r["id"] for r in read_jsonl(d / "cand-grades.jsonl.gz")}
    out = set()
    for p in pool:
        i = p["id"]
        if i not in sampled:
            continue
        if i in failed or (i in refs and sum(refs[i]["weights"]) < s.imp_min_questions) or i in graded:
            out.add(i)
    return out


def progress(s: Settings) -> dict:
    """Where the round stands, per stage, over the pool."""
    d = idir(s)
    pool = select_pool(s)
    ids = {p["id"] for p in pool}
    count = lambda f: len({r["id"] for r in read_jsonl(d / f)} & ids)  # noqa: E731
    return {"pool": len(pool), "sampled": count("samples.jsonl.gz"), "questions": count("questions.jsonl.gz"),
            "question_failures": count("questions-failed.jsonl.gz"), "references": count("refs.jsonl.gz"),
            "candidates_graded": count("cand-grades.jsonl.gz"), "finished": len(finished_ids(s, pool))}


def score_paper(s: Settings, p: dict, questions: list[dict], ref: dict | None, samples: list[dict] | None,
                verdicts: list[list[str] | None] | None, prefix: str) -> dict:
    """Every candidate's reward for one paper, and the index chosen."""
    cands = candidate_objs(s, p, samples or [], prefix)
    row: dict = {"id": p["id"], "n_candidates": len(cands), "chosen": 0, "reason": "no_reward"}
    if ref is None or verdicts is None or sum(ref["weights"]) < s.imp_min_questions:
        return row
    rewards: list[float | None] = []
    detail = []
    for (obj, finish), v in zip(cands, verdicts, strict=True):
        feat = features(obj, finish, p["teacher"], p["paper"])
        fact = fact_reward(v, ref["weights"], s.imp_min_questions) if v is not None and feat["valid"] else None
        rewards.append(composite(fact, feat, s) if fact is not None else None)
        detail.append({**feat, "fact": fact, "reward": rewards[-1]})
    chosen = choose(rewards, s.imp_margin)
    row.update({"chosen": chosen, "reason": "teacher" if chosen == 0 else "sample", "kept_questions": sum(ref["weights"]),
                "candidates": detail})
    return row


def run_select(s: Settings) -> dict:
    """Score every candidate and write the choice per paper, plus the statistics of the round."""
    from scisum.dataset import tokenizer
    from scisum.prompt import assistant_wrapper

    prefix, _ = assistant_wrapper(tokenizer(s))
    d = idir(s)
    pool = select_pool(s)
    qs = {r["id"]: r["questions"] for r in read_jsonl(d / "questions.jsonl.gz")}
    refs = {r["id"]: r for r in read_jsonl(d / "refs.jsonl.gz")}
    samples = {r["id"]: r["samples"] for r in read_jsonl(d / "samples.jsonl.gz")}
    graded = {r["id"]: r["verdicts"] for r in read_jsonl(d / "cand-grades.jsonl.gz")}
    done = finished_ids(s, pool)
    rows = [score_paper(s, p, qs.get(p["id"], []), refs.get(p["id"]), samples.get(p["id"]), graded.get(p["id"]), prefix)
            for p in pool if p["id"] in done]
    write_jsonl_atomic(d / "selection.jsonl.gz", rows)
    stats = selection_stats(rows)
    stats["settings"] = {k: getattr(s, k) for k in ("imp_k", "imp_temperature", "imp_lambda", "imp_beta_num", "imp_gamma_len",
                                                    "imp_len_free", "imp_margin", "imp_min_questions", "imp_writer", "imp_grader")}
    (d / "selection-stats.json").write_text(json.dumps(stats, indent=1))
    return stats


def selection_stats(rows: list[dict]) -> dict:
    scored = [r for r in rows if "candidates" in r]
    c = collections.Counter(r["reason"] for r in rows)
    mean = lambda xs: round(st.fmean(xs), 4) if xs else None  # noqa: E731
    teacher = [r["candidates"][0] for r in scored]
    samples = [x for r in scored for x in r["candidates"][1:]]
    valid = [x for x in samples if x["valid"]]
    picked = [r["candidates"][r["chosen"]] for r in scored]
    swapped = [r for r in scored if r["chosen"] > 0]
    return {
        "papers": len(rows), "scored": len(scored), "teacher_kept": c["teacher"], "sample_chosen": c["sample"],
        "no_reward": c["no_reward"], "sample_share_of_scored": round(c["sample"] / max(len(scored), 1), 4),
        "samples": len(samples), "samples_valid": round(len(valid) / max(len(samples), 1), 4),
        "mean_fact": {"teacher": mean([x["fact"] for x in teacher if x["fact"] is not None]),
                      "samples": mean([x["fact"] for x in valid if x["fact"] is not None]),
                      "chosen": mean([x["fact"] for x in picked if x["fact"] is not None])},
        "mean_reward": {"teacher": mean([x["reward"] for x in teacher if x["reward"] is not None]),
                        "samples": mean([x["reward"] for x in valid if x["reward"] is not None]),
                        "chosen": mean([x["reward"] for x in picked if x["reward"] is not None])},
        "mean_rougeL_to_teacher": {"samples": mean([x["rougeL"] for x in valid]),
                                   "chosen_when_sample": mean([r["candidates"][r["chosen"]]["rougeL"] for r in swapped])},
        "unsupported_numbers": {"teacher": mean([x["unsupported_frac"] for x in teacher if x["valid"]]),
                                "samples": mean([x["unsupported_frac"] for x in valid]),
                                "chosen": mean([x["unsupported_frac"] for x in picked if x["valid"]])},
        "length_ratio": {"samples": mean([x["length_ratio"] for x in valid]),
                         "chosen_when_sample": mean([r["candidates"][r["chosen"]]["length_ratio"] for r in swapped])},
        "kept_questions_mean": mean([r["kept_questions"] for r in scored]),
    }


# ---- 6. training sets ----------------------------------------------------------------------------
def pick_pair(rewards: list[float | None], chosen: int, gap: float, lengths: list[float] | None = None,
              rule: str = "lowest") -> int | None:
    """The sample to reject for a paper, or ``None`` if there is no clearly worse one.

    Only student samples are rejected (never the teacher's summary): the aim is to move the student
    away from its own worse outputs. Every candidate must trail the chosen one by ``gap``. Among those,
    ``rule`` picks:

    * ``"lowest"`` (round 1): the lowest-reward sample.
    * ``"closest_length"`` (round 2): the sample whose length is closest to the chosen summary's, on a
      log scale (ties: the lower reward). ``lengths`` are any per-candidate lengths on one scale.

    Why round 2 changed it: in round 1's 1,073 pairs the rejected summary was the shorter one in 57%
    (median length 0.967 of the chosen, in characters), and the preference arm learned to write 1.26x
    the teacher's length. A small length bias in the pairs is known to be amplified by DPO (Park et al.
    2024). Matching lengths within each pair leaves the reward gap as the only thing that separates
    them. Measured on round 1's candidates before round 2 was trained (with a length allowance of 1.0x
    the teacher's): 1,163 pairs, rejected/chosen median length 1.013, mean reward gap 0.372.
    """
    base = rewards[chosen]
    if base is None:
        return None
    behind = [i for i, r in enumerate(rewards) if i not in (0, chosen) and r is not None and base - r >= gap]
    if not behind:
        return None
    if rule == "lowest":
        return min(behind, key=lambda i: rewards[i])
    if rule == "closest_length":
        if lengths is None:
            raise ValueError("closest_length needs the candidates' lengths")
        return min(behind, key=lambda i: (abs(math.log(max(lengths[i], 1e-9) / max(lengths[chosen], 1e-9))), rewards[i]))
    raise ValueError(f"unknown pair rule {rule!r}")


# ---- round 4: a declared filter on pairs, and several pairs per paper ------------------------------
FILTER_PARTS = ("fact", "claim", "len", "rouge", "num", "reward")
FILTER_NAMES = tuple(f"{side}_{k}" for side in ("c", "r") for k in FILTER_PARTS) + ("c_teacher", "gap")
_FILTER_FUNCS = {"abs": abs, "min": min, "max": max}
_FILTER_NODES = (ast.Expression, ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not, ast.USub, ast.BinOp, ast.Add,
                 ast.Sub, ast.Mult, ast.Div, ast.Compare, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq, ast.Name,
                 ast.Load, ast.Constant, ast.Call)


def compile_filter(expr: str):
    """A pair filter: a boolean expression over the stored parts of the chosen (``c_``) and rejected (``r_``) summary.

    Names: ``c_fact``/``r_fact`` (fact reward), ``c_claim``/``r_claim`` (share of statements the claim check finds
    unsupported or contradicted; 0 when unlabelled, as the reward charges nothing then), ``c_len``/``r_len``
    (length relative to the teacher's summary), ``c_rouge``/``r_rouge`` (ROUGE-L against the teacher),
    ``c_num``/``r_num`` (share of numbers the paper does not contain), ``c_reward``/``r_reward``, ``gap``
    (their difference) and ``c_teacher`` (the chosen summary is the teacher's). Only comparisons, ``and``/``or``/
    ``not``, ``+ - * /``, numbers and ``abs``/``min``/``max`` are accepted: the expression comes from a model
    (the round-4 designer), so it is parsed and checked node by node, never passed to ``eval`` unchecked.
    Returns ``None`` for an empty expression, else a function of the name -> value mapping.
    """
    if not expr or not expr.strip():
        return None
    tree = ast.parse(expr.strip(), mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _FILTER_NODES):
            raise ValueError(f"pair filter: {type(node).__name__} is not allowed")
        if isinstance(node, ast.Name) and node.id not in FILTER_NAMES and node.id not in _FILTER_FUNCS:
            raise ValueError(f"pair filter: unknown name {node.id!r}")
        if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name) and node.func.id in _FILTER_FUNCS
                                               and not node.keywords):
            raise ValueError("pair filter: only abs(), min() and max() may be called")
        if isinstance(node, ast.Constant) and (isinstance(node.value, bool) or not isinstance(node.value, (int, float))):
            raise ValueError(f"pair filter: constant {node.value!r} is not a number")
    code = compile(tree, "<pair filter>", "eval")
    return lambda env: bool(eval(code, {"__builtins__": {}, **_FILTER_FUNCS}, dict(env)))  # noqa: S307 (checked above)


def pair_env(cands: list[dict], shares: list[float | None] | None, rewards: list[float | None], chosen: int,
             rej: int) -> dict:
    """The names of ``compile_filter`` for one (chosen, rejected) pair of a selection row."""
    env: dict = {"c_teacher": chosen == 0}
    for side, i in (("c", chosen), ("r", rej)):
        c = cands[i]
        share = shares[i] if shares and i < len(shares) and shares[i] is not None else 0.0
        env.update({f"{side}_fact": c.get("fact") or 0.0, f"{side}_claim": share, f"{side}_len": c.get("length_ratio") or 0.0,
                    f"{side}_rouge": c.get("rougeL") or 0.0, f"{side}_num": c.get("unsupported_frac") or 0.0,
                    f"{side}_reward": rewards[i]})
    env["gap"] = rewards[chosen] - rewards[rej]
    return env


def pick_pairs(rewards: list[float | None], chosen: int, gap: float, lengths: list[float] | None = None,
               rule: str = "lowest", n: int = 1, keep=None) -> list[int]:
    """Up to ``n`` samples to reject for a paper, in the order ``rule`` ranks them (``pick_pair`` is its first).

    ``keep(i)`` (the pair filter) drops a candidate before the ranking. With ``n`` 1 and no filter this is
    ``pick_pair``; a unit test compares the two.
    """
    base = rewards[chosen]
    if base is None:
        return []
    behind = [i for i, r in enumerate(rewards) if i not in (0, chosen) and r is not None and base - r >= gap]
    if keep is not None:
        behind = [i for i in behind if keep(i)]
    if rule == "lowest":
        key = lambda i: (rewards[i],)  # noqa: E731
    elif rule == "closest_length":
        if lengths is None:
            raise ValueError("closest_length needs the candidates' lengths")
        key = lambda i: (abs(math.log(max(lengths[i], 1e-9) / max(lengths[chosen], 1e-9))), rewards[i])  # noqa: E731
    else:
        raise ValueError(f"unknown pair rule {rule!r}")
    return sorted(behind, key=key)[: max(n, 0)]


def is_pair_arm(arm: str) -> bool:
    """The fixed pair arms of rounds 1-3, and the designer's round-4 arms (``r4a``, ``r4b``, ...)."""
    return arm in PAIR_ARMS or re.fullmatch(r"r\d+[a-z][a-z0-9]*", arm) is not None


def rescore(pick: dict, s: Settings, claim_share: list[float | None] | None = None) -> tuple[list[float | None], int]:
    """A selection row's rewards and choice under the current reward settings, from its stored parts.

    ``run_select`` stores every candidate's fact reward, ROUGE-L, unsupported-number share and length
    ratio, so a different weighting (round 2 charges length beyond 1.0x the teacher's, not 1.2x) needs
    no new model call. Under the settings ``run_select`` used it reproduces the stored rewards exactly.

    Round 3 (``imp_claim_weight`` > 0): each candidate also loses ``imp_claim_weight`` times the share of
    its statements that the claim check (``run_claims``) found unsupported or contradicted. A candidate
    the verifier could not label (``None``) takes no penalty.
    """
    rewards: list[float | None] = []
    for i, c in enumerate(pick.get("candidates", [])):
        ok = c.get("valid") and c.get("fact") is not None
        r = composite(c["fact"], c, s) if ok else None
        if r is not None and s.imp_claim_weight and claim_share and i < len(claim_share) and claim_share[i] is not None:
            r -= s.imp_claim_weight * claim_share[i]
        rewards.append(r)
    return rewards, (choose(rewards, s.imp_margin) if rewards else 0)


# ---- round 3: the claim check inside the reward ----------------------------------------------------
def claims_path(s: Settings, model: str | None = None) -> Path:
    return idir(s) / f"claims.{model or s.imp_claim_model}.jsonl.gz"


def claim_shares(s: Settings) -> dict[str, list[float | None]]:
    """paper id -> per candidate (teacher first): share of labelled statements not supported, or None."""
    return {r["id"]: r["unsupported_share"] for r in read_jsonl(claims_path(s))}


async def run_claims(s: Settings, max_seconds: float = 3000, concurrency: int = 24, model: str | None = None) -> dict:
    """Claim-check every scored candidate of the selection (teacher and samples) against the paper.

    The statements and the prompt are the evaluation's (``scisum.claimcheck``); the verifier is
    ``imp_claim_model`` (Gemma-4-31B), not the evaluation's Qwen3.8-27B, as the fact reward's grader is
    the other model of the two. Resumable: one row per paper, appended.
    """
    from scisum.claimcheck import MAX_CLAIMS, SYSTEM, check_one
    from scisum.dataset import tokenizer
    from scisum.faithfulness import claims_of
    from scisum.prompt import assistant_wrapper

    deadline = time.monotonic() + max_seconds
    model = model or s.imp_claim_model
    prefix, _ = assistant_wrapper(tokenizer(s))
    d = idir(s)
    picks = {r["id"]: r for r in read_jsonl(d / "selection.jsonl.gz")}
    samples = {r["id"]: r["samples"] for r in read_jsonl(d / "samples.jsonl.gz")}
    path = claims_path(s, model)
    done = {r["id"] for r in read_jsonl(path)}
    todo = [p for p in select_pool(s) if p["id"] in picks and p["id"] not in done]
    chat = Chat(s, concurrency, model, SYSTEM, 3_000)

    async def check(p: dict, obj: dict | None, c: dict) -> tuple[float | None, list[str]]:
        claims = claims_of(obj, MAX_CLAIMS) if obj is not None and c.get("valid") and c.get("fact") is not None else []
        if not claims:
            return None, []
        verdicts, _ = await check_one(chat, p["paper"], claims)
        lab = [v["verdict"] for v in verdicts]
        k = [x for x in lab if x != "ungraded"]
        return (sum(x != "supported" for x in k) / len(k) if k else None), lab

    async def one(p: dict) -> dict | None:
        if time.monotonic() > deadline:
            return None
        objs = candidate_objs(s, p, samples.get(p["id"], []), prefix)
        cands = picks[p["id"]].get("candidates", [])
        res = await asyncio.gather(*(check(p, obj, cands[i] if i < len(cands) else {}) for i, (obj, _) in enumerate(objs)))
        return {"id": p["id"], "unsupported_share": [r[0] for r in res], "verdicts": [r[1] for r in res]}

    made = 0
    for i in range(0, len(todo), 32):
        rows = [r for r in await asyncio.gather(*(one(p) for p in todo[i:i + 32])) if r is not None]
        append_jsonl(path, rows)
        made += len(rows)
        log.info("claims: %d/%d papers", made, len(todo))
        if time.monotonic() > deadline:
            break
    await chat.close()
    left = len([p for p in select_pool(s) if p["id"] in picks]) - len({r["id"] for r in read_jsonl(path)})
    return {"model": model, "checked": made, "remaining": left}


PAIR_ARMS = ("pref", "preflc", "prefcc")


def build_arm(s: Settings, arm: str, part_size: int = 500) -> dict:
    """Tokenise one arm's targets into tokens/<imp_dir>-<arm>.

    ``ctl``: the teacher's targets. ``rft``: the selected targets. ``pref`` and ``preflc``: preference
    pairs, with rewards recomputed under the current settings (``rescore``) and the rejected sample
    picked by ``imp_pair_rule``. ``preflc`` is built with the round-2 settings in
    ``slurm/models/q35-preflc.env``; the settings used are written to ``build.json`` in the directory.
    """
    from scisum.dataset import render_target, tokenizer
    from scisum.shards import ShardWriter

    if arm not in ("ctl", "rft") and not is_pair_arm(arm):
        raise ValueError(f"arm must be ctl, rft, one of {PAIR_ARMS} or a round arm such as r4a")
    pairs_arm = is_pair_arm(arm)
    flt = compile_filter(s.imp_pair_filter) if pairs_arm else None
    tok = tokenizer(s)
    from scisum.prompt import assistant_wrapper

    prefix, _ = assistant_wrapper(tok)
    d = idir(s)
    pool = select_pool(s)
    picks = {r["id"]: r for r in read_jsonl(d / "selection.jsonl.gz")}
    samples = {r["id"]: r["samples"] for r in read_jsonl(d / "samples.jsonl.gz")}
    final = s.tokens_dir(f"{s.imp_dir}-{arm}")        # tokens/improve-rft; the smoke test has its own
    tmp = final.with_name(final.name + ".tmp")
    if tmp.exists():
        import shutil
        shutil.rmtree(tmp)
    writer, part, n_in_part, n, swapped, lens, gaps = None, 0, 0, 0, 0, [], []

    def render(p: dict, idx: int) -> tuple[str, str]:
        if idx == 0:
            return p["target"], "teacher"
        obj = candidate_objs(s, p, samples[p["id"]], prefix)[idx][0]
        return render_target(s, tok, obj), samples[p["id"]][idx - 1].get("kind", "sampled")

    def emit(prompt_ids: list[int], items: list[tuple[list[int], dict]]) -> None:
        nonlocal writer, part, n_in_part, n
        if writer is None:
            writer = ShardWriter(tmp, part)
        for target_ids, meta in items:
            writer.add(prompt_ids, target_ids, meta)
        n += 1
        n_in_part += 1
        if n_in_part >= part_size:
            writer.close()
            writer, part, n_in_part = None, part + 1, 0

    shares = claim_shares(s) if (s.imp_claim_weight or flt is not None) else {}
    if s.imp_claim_weight:
        missing = [i for i in picks if i in {p["id"] for p in pool} and i not in shares]
        if missing:
            raise RuntimeError(f"imp_claim_weight is set but {len(missing)} papers have no claim check ({claims_path(s)})")
    for p in pool:
        if p["id"] not in picks:      # not finished yet: left out of every arm, so the arms stay on the same papers
            continue
        pick = picks[p["id"]]
        rewards, rechosen = rescore(pick, s, shares.get(p["id"]))
        idx = rechosen if pairs_arm else pick["chosen"] if arm == "rft" else 0
        target, source = render(p, idx)
        prompt_ids = tok(p["prompt"], add_special_tokens=False).input_ids
        target_ids = tok(target, add_special_tokens=False).input_ids
        if len(prompt_ids) + len(target_ids) > s.max_seq_len or len(target_ids) > s.max_target_tokens:
            continue
        if pairs_arm:    # pairs: chosen first, then rejected, as consecutive examples
            cands = pick.get("candidates", [])
            lengths = [c.get("length_ratio") or 0.0 for c in cands]
            keep = None
            if flt is not None:
                share = shares.get(p["id"])
                keep = lambda i: flt(pair_env(cands, share, rewards, idx, i))  # noqa: E731, B023 (used in this iteration only)
            rejs = pick_pairs(rewards, idx, s.imp_pair_gap, lengths, s.imp_pair_rule, s.imp_pairs_per_paper, keep) if rewards else []
            for rej in rejs:
                rtarget, rsource = render(p, rej)
                rtarget_ids = tok(rtarget, add_special_tokens=False).input_ids
                if len(prompt_ids) + len(rtarget_ids) > s.max_seq_len or len(rtarget_ids) > s.max_target_tokens:
                    continue
                lens.append(len(rtarget_ids) / max(len(target_ids), 1))
                gaps.append(rewards[idx] - rewards[rej])
                emit(prompt_ids, [(target_ids, {"id": p["id"], "role": "chosen", "source": source, "arm": arm,
                                                "reward": rewards[idx]}),
                                  (rtarget_ids, {"id": p["id"], "role": "rejected", "source": rsource, "arm": arm,
                                                 "reward": rewards[rej]})])
                swapped += idx > 0
            continue
        swapped += idx > 0
        emit(prompt_ids, [(target_ids, {"id": p["id"], "source": source, "arm": arm})])
    if writer is not None:
        writer.close()
    out = {"arm": arm, "pairs" if pairs_arm else "examples": n, "from_samples": swapped, "directory": str(final)}
    if lens:
        out["rejected_over_chosen_length_median"] = round(st.median(lens), 3)
        out["rejected_shorter_share"] = round(sum(x < 1 for x in lens) / len(lens), 4)
        out["reward_gap_mean"] = round(st.fmean(gaps), 4)
    out["settings"] = {k: getattr(s, k) for k in ("imp_lambda", "imp_beta_num", "imp_gamma_len", "imp_len_free",
                                                  "imp_margin", "imp_pair_gap", "imp_pair_rule", "imp_min_questions",
                                                  "imp_claim_weight", "imp_claim_model", "imp_pair_filter",
                                                  "imp_pairs_per_paper")}
    if tmp.exists():
        (tmp / "build.json").write_text(json.dumps(out, indent=1))
    if final.exists():
        import shutil
        shutil.rmtree(final)
    tmp.replace(final)
    return out


# ---- smoke test: does the reward tell good summaries from bad ones? -------------------------------
SYNTH = ("copy", "perturbed_numbers", "truncated", "other_paper")


def _perturb_numbers(text: str) -> str:
    """Change the last digit of every checkable number: the same sentence with invented figures."""
    def bump(m: re.Match) -> str:
        tok = m.group(0)
        if tok.replace(",", "").lstrip("+-") not in numbers(tok):
            return tok
        digits = [i for i, ch in enumerate(tok) if ch.isdigit()]
        i = digits[-1]
        return tok[:i] + str((int(tok[i]) + 3) % 10) + tok[i + 1:]
    return _NUM.sub(bump, text)


def synthetic_candidates(teacher: dict, other: dict) -> list[dict]:
    """Four summaries with a known defect, in the order of ``SYNTH``."""
    def first_sentence(t: str) -> str:
        return re.split(r"(?<=[.!?])\s", t.strip(), maxsplit=1)[0]

    perturbed = {k: (_perturb_numbers(v) if isinstance(v, str) else [_perturb_numbers(x) for x in v] if k == "contributions" else v)
                 for k, v in teacher.items()}
    truncated = {**teacher, "method": first_sentence(teacher["method"]), "results": first_sentence(teacher["results"]),
                 "limitations": first_sentence(teacher["limitations"]), "contributions": teacher["contributions"][:2]}
    return [dict(teacher), perturbed, truncated, {k: other[k] for k in SUMMARY_KEYS}]


def write_synthetic_samples(s: Settings) -> int:
    from scisum.dataset import tokenizer
    from scisum.prompt import assistant_wrapper

    prefix, suffix = assistant_wrapper(tokenizer(s))
    pool = select_pool(s)
    rows = []
    for i, p in enumerate(pool):
        other = pool[(i + 1) % len(pool)]["teacher"]
        rows.append({"id": p["id"], "samples": [
            {"text": prefix + canonical_target(c), "finish": "stop", "tokens": 0} for c in synthetic_candidates(p["teacher"], other)]})
    write_jsonl_atomic(idir(s) / "samples.jsonl.gz", rows)
    return len(rows)


def synth_report(s: Settings) -> dict:
    """Mean fact reward and composite reward per synthetic defect; the teacher is index 0."""
    rows = [r for r in read_jsonl(idir(s) / "selection.jsonl.gz") if "candidates" in r]
    out = {"papers_scored": len(rows)}
    for j, name in enumerate(("teacher", *SYNTH)):
        facts = [r["candidates"][j]["fact"] for r in rows if r["candidates"][j].get("fact") is not None]
        rews = [r["candidates"][j]["reward"] for r in rows if r["candidates"][j].get("reward") is not None]
        unsup = [r["candidates"][j]["unsupported_frac"] for r in rows if r["candidates"][j].get("valid")]
        out[name] = {"fact": round(st.fmean(facts), 3) if facts else None, "reward": round(st.fmean(rews), 3) if rews else None,
                     "unsupported_numbers": round(st.fmean(unsup), 3) if unsup else None, "n": len(facts)}
    out["chosen_teacher_or_copy"] = sum(r["chosen"] in (0, 1) for r in rows)
    return out


# ---- the evaluation set for the continuation runs ---------------------------------------------------
def write_eval_ids(s: Settings, first: int = 500) -> dict:
    """The test papers the continuations are evaluated on: every fact-test paper, plus the first ``first``.

    Generating the whole first 3,000 test papers for each arm needs a full GPU for hours; the fact
    test needs only the papers whose questions exist, and the reference metrics need a few hundred.
    The starting model has predictions for all of them already.
    """
    from scisum.dataset import load_splits

    ids: set[str] = set()
    for d in ("factcheck", "factcheck-b"):
        p = s.state_dir / d / "papers.json"
        if p.exists():
            ids |= {r["id"] for r in json.loads(p.read_text())}
    n_fact = len(ids)
    ids |= set(load_splits(s)["test"][: s.n_test][:first])
    ordered = [i for i in load_splits(s)["test"][: s.n_test] if i in ids]
    path = idir(s) / "eval_ids.json"
    path.write_text(json.dumps(ordered))
    return {"papers": len(ordered), "fact_test_papers": n_fact, "file": str(path)}


# ---- is the selection signal or luck? ---------------------------------------------------------------
def typical_sample(row: dict) -> int | None:
    """The index of the unchosen valid sample whose reward is the median: what a random sample looks like."""
    cands = [(c["reward"], i) for i, c in enumerate(row["candidates"]) if i > 0 and i != row["chosen"] and c.get("reward") is not None]
    if not cands:
        return None
    cands.sort()
    return cands[len(cands) // 2][1]


async def run_validate(s: Settings, max_papers: int = 250, max_seconds: float = 3000, concurrency: int = 24) -> dict:
    """Re-score the teacher, the chosen sample and a typical unchosen sample with the evaluation's instrument.

    The selection used one writer, reader and grader; the evaluation uses others
    (``fact_gen_model`` writes ten questions, ``judge_model`` reads and grades). If the chosen sample
    is really better than a typical one, it stays better on questions the selection never saw.
    """
    from scisum.dataset import tokenizer
    from scisum.factcheck import QTYPES
    from scisum.prompt import assistant_wrapper

    deadline = time.monotonic() + max_seconds
    prefix, _ = assistant_wrapper(tokenizer(s))
    d = idir(s)
    pool = select_pool(s)
    sel = {r["id"]: r for r in read_jsonl(d / "selection.jsonl.gz")}
    samples = {r["id"]: r["samples"] for r in read_jsonl(d / "samples.jsonl.gz")}
    path = d / "validation.jsonl.gz"
    done = {r["id"] for r in read_jsonl(path)}
    todo = [p for p in pool if sel.get(p["id"], {}).get("reason") == "sample" and p["id"] not in done][: max(max_papers - len(done), 0)]
    writer = Chat(s, concurrency, s.fact_gen_model, GEN_SYSTEM, 3_000)
    ans = Chat(s, concurrency, s.judge_model, ANS_SYSTEM, 1_500)
    grd = Chat(s, concurrency, s.judge_model, GRADE_SYSTEM, 1_200)
    quota, n = ", ".join(f"{v} {k}" for k, v in QTYPES.items()), sum(QTYPES.values())
    log.info("validation: %d papers to score (%d done)", len(todo), len(done))

    async def one(p: dict):
        if time.monotonic() > deadline:
            return None
        row = sel[p["id"]]
        j = typical_sample(row)
        if j is None:
            return None
        questions = []
        for _ in range(3):
            text, _meta = await writer.one(GEN_PROMPT.format(n=n, quota=quota, paper=p["paper"]))
            questions = [q for q in parse_obj(text, "questions") if isinstance(q, dict) and q.get("question") and q.get("answer")][:n]
            if len(questions) >= n // 2:
                break
        if len(questions) < n // 2:
            return None
        objs = candidate_objs(s, p, samples[p["id"]], prefix)
        qtext = _qtext(questions)

        async def read(obj: dict) -> list[str]:
            return await answer_and_grade(ans, grd, ANS_PROMPT.format(n=len(questions), summary=render_for_reader(obj), questions=qtext), questions)
        teacher, chosen, typical = await asyncio.gather(read(objs[0][0]), read(objs[row["chosen"]][0]), read(objs[j][0]))
        return {"id": p["id"], "chosen": row["chosen"], "typical": j, "teacher": teacher, "chosen_verdicts": chosen,
                "typical_verdicts": typical, "selection_reward_gap": row["candidates"][row["chosen"]]["reward"] - row["candidates"][j]["reward"]}

    made = 0
    async for rows in _chunks(todo, one, size=16):
        append_jsonl(path, rows)
        made += len(rows)
        log.info("  validated %d/%d", made, len(todo))
    for c in (writer, ans, grd):
        await c.close()
    return {"validated_now": made, "validated_total": len(done) + made}


def validation_report(s: Settings, boots: int = 2000, seed: int = 0) -> dict:
    """Paired differences in fact recall on the independent questions."""
    import random

    from scisum.factcheck import paper_recall

    rows = read_jsonl(idir(s) / "validation.jsonl.gz")
    rnd = random.Random(seed)

    def boot(d: list[float]) -> dict:
        ms = sorted(st.fmean(rnd.choices(d, k=len(d))) for _ in range(boots))
        return {"mean": round(st.fmean(d), 4), "ci95": [round(ms[int(0.025 * boots)], 4), round(ms[int(0.975 * boots)], 4)]}

    rec = {k: [paper_recall({"verdicts": r[k]}) for r in rows] for k in ("teacher", "chosen_verdicts", "typical_verdicts")}
    diff = lambda a, b: [x - y for x, y in zip(rec[a], rec[b], strict=True)]  # noqa: E731
    out = {"papers": len(rows),
           "recall": {k.replace("_verdicts", ""): round(st.fmean(v), 4) for k, v in rec.items()},
           "chosen_minus_teacher": boot(diff("chosen_verdicts", "teacher")),
           "chosen_minus_typical_sample": boot(diff("chosen_verdicts", "typical_verdicts")),
           "typical_sample_minus_teacher": boot(diff("typical_verdicts", "teacher")),
           "selection_reward_gap_mean": round(st.fmean(r["selection_reward_gap"] for r in rows), 4)}
    (idir(s) / "validation-report.json").write_text(json.dumps(out, indent=1))
    return out


async def run_validate_pairs(s: Settings, max_papers: int = 200, max_seconds: float = 3000, concurrency: int = 24) -> dict:
    """Are the preference pairs ordered correctly on questions the selection never saw?

    For papers that have a pair (a chosen summary, usually the teacher's, and a clearly worse student
    sample), both summaries are read against ten new questions written and graded by the evaluation's
    models. A pair is useful training data only if the chosen one answers more of them.
    """
    from scisum.dataset import tokenizer
    from scisum.factcheck import QTYPES
    from scisum.prompt import assistant_wrapper

    deadline = time.monotonic() + max_seconds
    prefix, _ = assistant_wrapper(tokenizer(s))
    d = idir(s)
    pool = select_pool(s)
    sel = {r["id"]: r for r in read_jsonl(d / "selection.jsonl.gz")}
    samples = {r["id"]: r["samples"] for r in read_jsonl(d / "samples.jsonl.gz")}
    path = d / "validation-pairs.jsonl.gz"
    done = {r["id"] for r in read_jsonl(path)}
    todo = []
    for p in pool:
        row = sel.get(p["id"])
        if row is None or "candidates" not in row or p["id"] in done:
            continue
        j = pick_pair([c.get("reward") for c in row["candidates"]], row["chosen"], s.imp_pair_gap)
        if j is not None:
            todo.append((p, row, j))
    todo = todo[: max(max_papers - len(done), 0)]
    writer = Chat(s, concurrency, s.fact_gen_model, GEN_SYSTEM, 3_000)
    ans = Chat(s, concurrency, s.judge_model, ANS_SYSTEM, 1_500)
    grd = Chat(s, concurrency, s.judge_model, GRADE_SYSTEM, 1_200)
    quota, n = ", ".join(f"{v} {k}" for k, v in QTYPES.items()), sum(QTYPES.values())

    async def one(item):
        p, row, j = item
        if time.monotonic() > deadline:
            return None
        questions = []
        for _ in range(3):
            text, _meta = await writer.one(GEN_PROMPT.format(n=n, quota=quota, paper=p["paper"]))
            questions = [q for q in parse_obj(text, "questions") if isinstance(q, dict) and q.get("question") and q.get("answer")][:n]
            if len(questions) >= n // 2:
                break
        if len(questions) < n // 2:
            return None
        objs = candidate_objs(s, p, samples[p["id"]], prefix)
        qtext = _qtext(questions)

        async def read(obj: dict) -> list[str]:
            return await answer_and_grade(ans, grd, ANS_PROMPT.format(n=len(questions), summary=render_for_reader(obj), questions=qtext), questions)
        chosen, rejected = await asyncio.gather(read(objs[row["chosen"]][0]), read(objs[j][0]))
        rw = [c.get("reward") for c in row["candidates"]]
        return {"id": p["id"], "chosen": row["chosen"], "rejected": j, "chosen_verdicts": chosen, "rejected_verdicts": rejected,
                "selection_reward_gap": rw[row["chosen"]] - rw[j]}

    made = 0
    async for rows in _chunks(todo, one, size=16):
        append_jsonl(path, rows)
        made += len(rows)
        log.info("  pairs validated %d/%d", made, len(todo))
    for c in (writer, ans, grd):
        await c.close()
    return {"validated_now": made, "validated_total": len(done) + made}


def pair_validation_report(s: Settings, boots: int = 2000, seed: int = 0) -> dict:
    import random

    from scisum.factcheck import paper_recall

    rows = read_jsonl(idir(s) / "validation-pairs.jsonl.gz")
    rnd = random.Random(seed)

    def boot(d: list[float]) -> dict:
        if len(d) < 2:
            return {"mean": None, "ci95": None}
        ms = sorted(st.fmean(rnd.choices(d, k=len(d))) for _ in range(boots))
        return {"mean": round(st.fmean(d), 4), "ci95": [round(ms[int(0.025 * boots)], 4), round(ms[int(0.975 * boots)], 4)]}

    def diff(sub: list[dict]) -> list[float]:
        return [paper_recall({"verdicts": r["chosen_verdicts"]}) - paper_recall({"verdicts": r["rejected_verdicts"]}) for r in sub]
    out = {"pairs": len(rows), "chosen_minus_rejected": boot(diff(rows)),
           "chosen_is_teacher": {"n": sum(r["chosen"] == 0 for r in rows), **boot(diff([r for r in rows if r["chosen"] == 0]))},
           "chosen_is_sample": {"n": sum(r["chosen"] > 0 for r in rows), **boot(diff([r for r in rows if r["chosen"] > 0]))},
           "share_chosen_ahead": round(sum(x > 0 for x in diff(rows)) / max(len(rows), 1), 4),
           "share_rejected_ahead": round(sum(x < 0 for x in diff(rows)) / max(len(rows), 1), 4),
           "selection_reward_gap_mean": round(st.fmean(r["selection_reward_gap"] for r in rows), 4) if rows else None}
    (idir(s) / "validation-pairs-report.json").write_text(json.dumps(out, indent=1))
    return out
