"""LLM-as-judge: how the student's summaries compare with the teacher's, scored by a third model.

ROUGE and BERTScore say how closely a summary matches one reference; the NLI metric says whether
its claims are supported. Neither answers the question this project exists to answer — *is the
student as good as the teacher it was distilled from?* — because the reference is the teacher, and
matching it exactly is not the same as being as useful.

So each paper is judged twice over, by `qwen38-27b` on an OpenAI-compatible endpoint
(a third family, not one of the teachers, and free for us):

* **Absolute**: the candidate summary is scored 1-5 on faithfulness, coverage, specificity and
  fluency against an excerpt of the paper.
* **Preference**: the candidate and the teacher's summary are shown as A and B — in an order fixed
  by a hash of the paper id, so the judge's position bias cannot align with ours — and the judge
  says which is more useful, or that they are equal. The win rate against the teacher is the
  headline number.

Caveats worth keeping in mind when reading the result: the judge sees an excerpt of the paper, not
all of it (`source_chars`), so its faithfulness score is weaker evidence than the NLI metric; and a
judge of this size prefers longer, more specific prose, which is a known bias of the method. It is
one view among the four, not the arbiter.
"""

from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import logging
import random
import time
from pathlib import Path

import httpx

from scisum.config import Settings
from scisum.metrics import parse_prediction, render_summary

log = logging.getLogger(__name__)

SYSTEM = """You are an exacting reviewer of scientific-paper summaries. You judge only what the text
supports, you never reward length for its own sake, and you answer with JSON and nothing else."""

RUBRIC = """Here is an excerpt of a scientific paper, and two structured summaries of it.

PAPER EXCERPT
-------------
{source}

SUMMARY A
---------
{a}

SUMMARY B
---------
{b}

Score **both** summaries 1-5 on each axis (5 is best), judging only against the excerpt above:
- "faithfulness": every claim is supported; inventions or wrong numbers are heavily penalised.
- "coverage": the problem, the method and the main results are all present.
- "specificity": concrete methods, datasets, metrics and numbers rather than vague description.
- "fluency": clear, dense, well-formed English appropriate for a search engine's index.

Then "preference": "A" if summary A is more useful to a researcher, "B" if summary B is, "tie" if
they are equally useful.

Return exactly this JSON object and nothing else:
{{"A": {{"faithfulness": n, "coverage": n, "specificity": n, "fluency": n}},
  "B": {{"faithfulness": n, "coverage": n, "specificity": n, "fluency": n}},
  "preference": "A"|"B"|"tie", "reason": "one sentence"}}"""

AXES = ("faithfulness", "coverage", "specificity", "fluency")


def _order(paper_id: str) -> bool:
    """True when the candidate is shown as A. Fixed per paper, so the order is reproducible."""
    return hashlib.sha256(paper_id.encode()).digest()[0] % 2 == 0


def build_prompt(source: str, candidate: str, reference: str, paper_id: str, source_chars: int) -> tuple[str, str]:
    cand_is_a = _order(paper_id)
    a, b = (candidate, reference) if cand_is_a else (reference, candidate)
    return RUBRIC.format(source=source[:source_chars], a=a, b=b), ("A" if cand_is_a else "B")


def parse_verdict(text: str, candidate_label: str) -> dict | None:
    """Both summaries' scores, mapped from A/B back to candidate and reference.

    Scoring the teacher's summary in the same call is what makes the numbers readable: a
    faithfulness of 2.3 means nothing until the reference, judged against the same excerpt by the
    same model in the same breath, scores 3.9.
    """
    obj, _ = parse_prediction(text)
    if not isinstance(obj, dict) or not isinstance(obj.get("A"), dict) or not isinstance(obj.get("B"), dict):
        return None
    ref_label = "B" if candidate_label == "A" else "A"
    try:
        cand = {k: float(obj[candidate_label][k]) for k in AXES}
        ref = {k: float(obj[ref_label][k]) for k in AXES}
    except (KeyError, TypeError, ValueError):
        return None
    if not all(1 <= v <= 5 for v in (*cand.values(), *ref.values())):
        return None
    out: dict = {f"judge_{k}": cand[k] for k in AXES}
    out.update({f"judge_ref_{k}": ref[k] for k in AXES})
    out["judge_mean"] = round(sum(cand.values()) / len(AXES), 4)
    out["judge_ref_mean"] = round(sum(ref.values()) / len(AXES), 4)
    out["judge_delta"] = round(out["judge_mean"] - out["judge_ref_mean"], 4)
    pref = str(obj.get("preference", "")).strip().upper()
    out["judge_win"] = float(pref == candidate_label)
    out["judge_tie"] = float(pref == "TIE")
    out["judge_loss"] = float(pref in ("A", "B") and pref != candidate_label)
    out["reason"] = str(obj.get("reason", ""))[:300]
    return out


class Judge:
    def __init__(self, s: Settings, concurrency: int = 8, model: str | None = None):
        from scisum.endpoints import resolve

        # one base URL per served model (scisum.endpoints): self-hosted graders run as separate servers
        ep = resolve(s, model or s.judge_model)
        self.s = s
        self.endpoint_source = ep.source
        self.sem = asyncio.Semaphore(concurrency)
        self.client = httpx.AsyncClient(
            base_url=ep.base_url, timeout=httpx.Timeout(180.0),
            headers={"Authorization": f"Bearer {ep.api_key}"})

    async def close(self) -> None:
        await self.client.aclose()

    def _body(self, prompt: str, max_tokens: int) -> dict:
        body = {"model": self.s.judge_model, "temperature": 0.0, "max_tokens": max_tokens,
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]}
        if not self.s.judge_thinking:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        return body

    async def one(self, prompt: str) -> tuple[str, dict]:
        """One judgement, with retries that give the model more room rather than changing its mode.

        `qwen38-27b` reasons before it answers, and on these prompts it spends the whole budget
        thinking and returns empty content with ``finish_reason: length`` — seven papers in eight at
        4 096 tokens. Raising the budget only moves the cliff, and letting some papers be judged
        with reasoning and others without would mix two scales in one number, so thinking is off for
        every call (``judge_thinking``) and a reply that still runs out of room is simply retried
        with more room.
        """
        last = ""
        for attempt in range(4):
            max_tokens = self.s.judge_max_tokens * (1 + attempt)
            async with self.sem:
                t0 = time.monotonic()
                try:
                    r = await self.client.post("/chat/completions", json=self._body(prompt, max_tokens))
                    r.raise_for_status()
                    data = r.json()
                    choice = data["choices"][0]
                    text = choice["message"].get("content") or ""
                    meta = {"seconds": round(time.monotonic() - t0, 2), "thinking": self.s.judge_thinking,
                            "max_tokens": max_tokens, "finish_reason": choice.get("finish_reason"),
                            "tokens": (data.get("usage") or {}).get("completion_tokens")}
                    if text.strip():
                        return text, meta
                    last = f"empty content (finish_reason={meta['finish_reason']})"
                except (httpx.HTTPError, KeyError, IndexError, ValueError) as e:
                    last = f"{type(e).__name__}: {e}"
                    log.warning("judge call failed (%s), attempt %d", last, attempt + 1)
            await asyncio.sleep(2 ** attempt + random.random())
        return "", {"error": last[:200]}


async def run_judge(s: Settings, *, run: str, ckpt: str, split: str = "val", limit: int | None = None,
                    subset: str = "curve", n: int | None = None, force: bool = False,
                    concurrency: int = 8, batch: int = 50, max_seconds: float | None = None) -> dict:
    """Judge one checkpoint's generated summaries against the teacher's. Resumable and idempotent."""
    from scisum.evaluate import eval_dir, eval_name, load_records, source_text

    recs = load_records(s, split, subset, limit)
    name = eval_name(split, ckpt, len(recs))
    pred_path = eval_dir(s, run) / f"{name}.pred.jsonl.gz"
    if not pred_path.exists():
        raise FileNotFoundError(f"no predictions at {pred_path}")
    out_path = eval_dir(s, run) / f"{name}.judge.jsonl.gz"
    if force:
        out_path.unlink(missing_ok=True)
    done: set[str] = set()
    if out_path.exists():
        with gzip.open(out_path, "rt") as f:
            done = {json.loads(line)["id"] for line in f}
    with gzip.open(pred_path, "rt") as f:
        preds = {r["id"]: r for r in (json.loads(line) for line in f)}

    wanted = [r for r in recs if r["id"] in preds][: (n or s.judge_n)]
    todo = [r for r in wanted if r["id"] not in done]
    log.info("judge %s: %d papers, %d already judged, %d to go", name, len(wanted), len(done), len(todo))
    if not todo:
        return summarise(out_path, name, run, ckpt, split, len(wanted))

    judge = Judge(s, concurrency)
    t0 = time.monotonic()

    async def work(rec: dict) -> dict:
        obj, _ = parse_prediction(preds[rec["id"]]["text"])
        candidate = render_summary(obj)
        prompt, label = build_prompt(source_text(rec["prompt"]), candidate or "(no valid summary)",
                                     render_summary(rec["summary"]), rec["id"], s.judge_source_chars)
        text, meta = await judge.one(prompt)
        verdict = parse_verdict(text, label) or {"error": "unparseable", "raw": text[:200]}
        return {"id": rec["id"], "candidate_label": label, **verdict, **meta}

    # In batches, appending as they land: the endpoint turned out to serve about one call a second
    # however many are in flight, so a thousand papers is twenty minutes, and a job killed at its
    # walltime must keep what it has. Re-running continues from the ids already in the file.
    done_now = 0
    try:
        for start in range(0, len(todo), batch):
            rows = await asyncio.gather(*(work(r) for r in todo[start:start + batch]))
            with gzip.open(out_path, "at") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")
            done_now += len(rows)
            rate = done_now / max(time.monotonic() - t0, 1e-9)
            log.info("  judged %d/%d (%.2f papers/s, %.0f min left)", done_now, len(todo), rate,
                     (len(todo) - done_now) / max(rate, 1e-9) / 60)
            if max_seconds and time.monotonic() - t0 > max_seconds:
                log.warning("judge time limit reached: %d of %d done, the rest on the next run",
                            done_now, len(todo))
                break
    finally:
        await judge.close()
    log.info("judged %d papers in %.0f min", done_now, (time.monotonic() - t0) / 60)
    return summarise(out_path, name, run, ckpt, split, len(wanted))


def summarise(out_path: Path, name: str, run: str, ckpt: str, split: str, wanted: int) -> dict:
    with gzip.open(out_path, "rt") as f:
        rows = [json.loads(line) for line in f]
    ok = [r for r in rows if "judge_mean" in r]
    agg: dict = {"name": name, "run": run, "ckpt": ckpt, "split": split, "judged": len(ok),
                 "requested": wanted, "unparseable": len(rows) - len(ok)}
    if ok:
        keys = [f"judge_{k}" for k in AXES] + [f"judge_ref_{k}" for k in AXES] + [
            "judge_mean", "judge_ref_mean", "judge_delta", "judge_win", "judge_tie", "judge_loss"]
        for k in keys:
            agg[k] = round(sum(r[k] for r in ok) / len(ok), 4)
        agg["judge_seconds_mean"] = round(sum(r.get("seconds", 0) for r in ok) / len(ok), 2)
    return agg
