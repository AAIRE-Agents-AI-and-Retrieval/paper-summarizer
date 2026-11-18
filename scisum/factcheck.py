"""Fact-level evaluation of a summary that does not use the teacher as the reference.

Every other metric here compares a summary with the teacher's. That measures imitation, and it
penalises a student that is right where the teacher is wrong. This one asks what a reader could
learn from the summary alone.

1. **Questions** are written from the *paper's full text*, never from any summary, by one model
   (``fact_gen_model``): ten specific questions per paper with short gold answers, spread over what
   the problem is, why it matters, what is proposed, how it was tested, what was found (with
   numbers), what limits it and under what conditions the results hold.
2. **Answers** are written by a second model from the summary *only*, with the instruction to say
   NOT STATED rather than guess. The same questions go to the teacher's summary, every student's and
   the untrained base models', so the teacher is a system under test, not the reference.
3. **Grades**: a third pass compares each answer with the gold answer: correct, partial, incorrect
   or not stated. ``recall`` is (correct + 0.5 partial) over all questions; ``accuracy`` is the
   share of *attempted* answers that were correct, which is the hallucination side.

Two models from different families write the questions and grade the answers, so one model's habits
cannot both set and mark the test. What remains: the gold answers are a model's reading of the
paper, so ``export_for_checking`` writes a sample for people to verify, and the numbers are best read
as a ranking of systems, not as absolute accuracy.
"""

from __future__ import annotations

import asyncio
import collections
import gzip
import hashlib
import json
import logging
import random
import statistics as st
import time
from pathlib import Path

from scisum.config import Settings
from scisum.judge import Judge
from scisum.metrics import parse_prediction

log = logging.getLogger(__name__)

QTYPES = {"problem": 1, "motivation": 1, "approach": 2, "setup": 2, "finding": 2, "limitation": 1, "scope": 1}
VERDICTS = ("correct", "partial", "incorrect", "not_stated")
# Systems under test. A label is a run name (contract of the student pipeline: ``<tag>-sft50k``,
# ``<tag>-prefcc``, ``<tag>-prefcc-s2``), optionally with ``@<checkpoint>`` (``k2h-3.7b-sft50k@ex20000``,
# ``k2h-3.7b-sft50k@base`` for the untrained model). Its summaries are the newest
# ``runs/<run>/eval/test-<ckpt>-n<papers>.pred.jsonl.gz``. Nothing has to be registered by hand.
#
# ``teacher`` is read from the index. The anchors of the earlier work keep their old labels so their
# scores can be re-graded and compared: Student C, its untrained base, and prefcc's two seeds. Their
# summaries ship under ``<data root>/anchors/<run>/eval/`` (or ``runs/<run>/eval/`` if regenerated).
TEACHER = "teacher"
ANCHORS = {  # label -> (run, evaluation name pattern)
    "C": ("q35", "test-ex20000-n*"), "C_base": ("q35", "test-base-n*"),
    "C_prefcc": ("q35prefcc", "test-ex*-n*"), "C_prefccs2": ("q35prefccs2", "test-ex*-n*"),
}
# backwards-compatible name: the fixed part of the registry (the teacher and the anchors)
SYSTEMS = {TEACHER: None, **ANCHORS}


def data_root(s: Settings) -> Path:
    """``SCISUM_DATA_ROOT`` (the release data: corpus, evaluation sets, anchors); default ``<repo>/data``."""
    import os

    root = getattr(s, "data_root", None) or os.environ.get("SCISUM_DATA_ROOT") or (s.repo_root / "data")
    return Path(root)


def _path_overrides() -> dict[str, str]:
    """``SCISUM_SYSTEM_PATHS``: inline JSON or a JSON file ``{label: path to a .pred.jsonl.gz}``."""
    import os

    raw = os.environ.get("SCISUM_SYSTEM_PATHS", "").strip()
    if not raw:
        return {}
    return json.loads(raw if raw.startswith("{") else Path(raw).read_text())


def _ckpt_key(path: Path) -> tuple:
    """Sort key of ``test-<ckpt>-n<papers>.pred.jsonl.gz``: base first, then by examples, then by papers."""
    import re

    m = re.match(r"test-(?:ex(\d+)|([a-z]+\d*))-n(\d+)\.pred\.jsonl\.gz$", path.name)
    if not m:
        return (-1, 0, path.name)
    return (int(m.group(1)) if m.group(1) else -1, int(m.group(3)), path.name)


def system_pred_path(s: Settings, label: str) -> Path | None:
    """Where the summaries of ``label`` are, or None if the label names no run with test summaries."""
    over = _path_overrides()
    if label in over:
        return Path(over[label])
    if label in ANCHORS:
        run, pattern = ANCHORS[label]
    else:
        run, _, ckpt = label.partition("@")
        pattern = f"test-{ckpt}-n*" if ckpt else "test-*-n*"
    for d in (s.run_dir(run) / "eval", data_root(s) / "anchors" / run / "eval"):
        found = sorted(d.glob(f"{pattern}.pred.jsonl.gz"), key=_ckpt_key)
        if label not in ANCHORS and "@" not in label:
            found = [f for f in found if not f.name.startswith("test-base-")] or found
        if found:
            return found[-1]
    return None


def read_system(s: Settings, label: str, ids: set[str]) -> dict[str, dict | None]:
    """paper id -> parsed summary of ``label`` (None when the system produced nothing usable)."""
    path = system_pred_path(s, label)
    if path is None:
        raise ValueError(f"unknown system {label!r}: no runs/{label.partition('@')[0]}/eval/test-*.pred.jsonl.gz "
                         f"(set SCISUM_SYSTEM_PATHS to point at its summaries)")
    got: dict[str, dict | None] = {}
    for r in read_jsonl(path):
        if r["id"] in ids:
            obj, _ = parse_prediction(r["text"])
            got[r["id"]] = obj if isinstance(obj, dict) else None
    return got


FIELDS = ("tldr", "problem", "contributions", "method", "results", "limitations", "keywords", "domain",
          "abstract", "paper_text")
# Controls that say what the test measures. closed_book: the answerer gets no summary and may guess, so
# its score is what the questions leak or the answerer already knows. abstract: the paper's own abstract
# as the "summary", the floor a summary should beat. full_paper: the whole text, the ceiling set by gold
# and grader noise (a perfect reader would score near 1).
CONTROLS = ("closed_book", "abstract", "full_paper")

GEN_SYSTEM = "You write exam questions about scientific papers. You answer with JSON and nothing else."
GEN_PROMPT = """Below is the text of a scientific paper. Write exactly {n} questions that test whether a
reader of a *summary* of this paper would have learned its most important facts.

Rules:
- Every question must be answerable from the paper text, with a short answer (at most 30 words) that
  you give, including exact numbers, names and conditions where the paper has them.
- Every question must be specific to this paper: a reader who has not read it, however expert,
  could not answer from general knowledge. Never ask "what is X?" about a textbook concept.
- Every question must stand alone: no "the authors", "this paper" or "the method" without saying
  which. A reader sees only the question.
- Cover these types, in this number: {quota}.
  problem = the research problem or gap; motivation = why it matters; approach = what is proposed
  or built; setup = data, systems or experiments used to test it; finding = a main result, with its
  number; limitation = something the authors say does not work or is left open; scope = a condition
  under which the results hold or fail.
- Choose the facts a researcher would most want to keep, not details of secondary importance.

Answer with JSON: {{"questions": [{{"type": "...", "question": "...", "answer": "..."}}, ...]}}

PAPER
-----
{paper}
"""
CLOSED_PROMPT = """Here are {n} questions about a scientific paper. You have NOT been given the paper.

Answer each from your own knowledge. If you do not know, give your best guess; never write NOT STATED.
Keep each answer under 30 words.

QUESTIONS
---------
{questions}

Answer with JSON: {{"answers": [{{"i": 1, "answer": "..."}}, ...]}}
"""
ANS_SYSTEM = "You answer questions using only the summary you are given. You answer with JSON and nothing else."
ANS_PROMPT = """Here is a summary of a scientific paper, and {n} questions about the paper.

Answer each question using ONLY the summary. If the summary does not state the answer, write exactly
NOT STATED. Do not guess and do not use outside knowledge. Keep each answer under 30 words.

SUMMARY
-------
{summary}

QUESTIONS
---------
{questions}

Answer with JSON: {{"answers": [{{"i": 1, "answer": "..."}}, ...]}}
"""
GRADE_SYSTEM = "You are a strict grader. You answer with JSON and nothing else."
GRADE_PROMPT = """For each item, compare the CANDIDATE answer with the GOLD answer to the QUESTION.

- correct: the candidate states the same fact as the gold answer (wording may differ; numbers and
  names that matter must agree).
- partial: it states part of the gold answer, or the right fact with a significant omission or a
  minor error.
- incorrect: it contradicts the gold answer or answers something else.
Do not reward detail the gold answer does not contain, and do not penalise a different but
equivalent wording.

{items}

Answer with JSON: {{"grades": [{{"i": 1, "verdict": "correct|partial|incorrect"}}, ...]}}
"""


class Chat(Judge):
    """The judge's retrying client, pointed at any model with any system prompt."""

    def __init__(self, s: Settings, concurrency: int, model: str, system: str, max_tokens: int):
        super().__init__(s, concurrency, model=model)
        self.model, self.system, self.max_tokens = model, system, max_tokens

    def _body(self, prompt: str, max_tokens: int) -> dict:
        body = {"model": self.model, "temperature": 0.0, "max_tokens": max(max_tokens, self.max_tokens),
                "messages": [{"role": "system", "content": self.system}, {"role": "user", "content": prompt}]}
        body["chat_template_kwargs"] = {"enable_thinking": False}
        return body


def failed_ids(s: Settings) -> set[str]:
    """Papers the question writer could not produce questions for, after three tries."""
    return {r["id"] for r in read_jsonl(fdir(s) / "questions-failed.jsonl.gz")}


def fdir(s: Settings) -> Path:
    d = s.state_dir / s.fact_dir
    d.mkdir(parents=True, exist_ok=True)
    return d


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: Path, rows: list[dict]) -> None:
    with gzip.open(path, "at") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def render_for_reader(obj: dict) -> str:
    """The same labelled text for every system, so no system is helped or hurt by its formatting."""
    out = []
    for k in FIELDS:
        v = obj.get(k)
        if isinstance(v, list):
            v = "\n".join(f"- {x}" for x in v)
        if isinstance(v, str) and v.strip():
            out.append(f"{k.upper()}: {v.strip()}")
    return "\n".join(out)


def stratified_ids(rows: list[dict], n: int, seed: int) -> list[str]:
    """``n`` papers, proportional to each archive's size (largest remainder), every random-cohort
    paper first: the random cohort is the only sample of arXiv at large and is small."""
    rnd = random.Random(seed)
    rand = [r for r in rows if r["cohort"] == "random"]
    top = [r for r in rows if r["cohort"] != "random"]
    chosen = [r["id"] for r in rand][: max(1, n // 10)]
    need = n - len(chosen)
    by = collections.defaultdict(list)
    for r in top:
        by[r["archive"]].append(r["id"])
    total = sum(len(v) for v in by.values())
    quota = {a: need * len(v) / total for a, v in by.items()}
    take = {a: int(q) for a, q in quota.items()}
    for a in sorted(quota, key=lambda a: quota[a] - take[a], reverse=True)[: need - sum(take.values())]:
        take[a] += 1
    for a, v in sorted(by.items()):
        chosen += rnd.sample(sorted(v), min(take[a], len(v)))
    return chosen


def select_papers(s: Settings) -> list[dict]:
    """The evaluation set, written once to state/factcheck/papers.json and then only read."""
    path = fdir(s) / "papers.json"
    if path.exists():
        return json.loads(path.read_text())
    from scisum.dataset import load_splits, read_prompts
    from scisum.evaluate import source_text

    pool = load_splits(s)["test"][: s.fact_pool]
    wanted = set(pool)
    from scisum.dataset import read_index

    meta: dict[str, dict] = {}
    for d in read_index(s):
        if d["id"] in wanted:
            cats = d["doc"].get("categories") or ["?"]
            meta[d["id"]] = {"id": d["id"], "cohort": d["cohort"], "archive": cats[0].split(".")[0],
                             "title": d["doc"].get("title", ""), "teacher_model": d["teacher"],
                             "teacher": d["summary"] if isinstance(d["summary"], dict) else json.loads(d["summary"])}
    taken: set[str] = set()
    if s.fact_exclude_dir:   # a second, disjoint question set: the first one's papers are not drawn again
        taken = {p["id"] for p in json.loads((s.state_dir / s.fact_exclude_dir / "papers.json").read_text())}
    ids = stratified_ids([meta[i] for i in pool if i in meta and i not in taken], s.fact_n_papers, s.seed)
    keep = set(ids)
    # The paper text comes from the current student's test prompts, so build them with the long budget
    # (28,672 prompt tokens, as every student profile here): the question writer must see the whole paper.
    # The shipped question sets already contain papers.json, so this only runs for a new set.
    if s.max_prompt_tokens < 28_672:
        log.warning("prompt budget %d < 28,672: the question writer will see less of each paper", s.max_prompt_tokens)
    for rec in read_prompts(s, "test"):
        if rec["id"] in keep:
            meta[rec["id"]]["paper"] = source_text(rec["prompt"])[: s.fact_paper_chars]
    out = [meta[i] for i in ids if "paper" in meta[i]]
    path.write_text(json.dumps(out))
    log.info("selected %d papers (%d random-cohort)", len(out), sum(p["cohort"] == "random" for p in out))
    return out


def system_summaries(s: Settings, papers: list[dict], labels: list[str] | tuple[str, ...] = ()) -> dict[str, dict[str, dict | None]]:
    """label -> paper id -> parsed summary (None when the system produced nothing usable).

    Always has the teacher and the controls; any other system only if it is in ``labels``.
    """
    ids = {p["id"] for p in papers}
    out: dict[str, dict[str, dict | None]] = {TEACHER: {p["id"]: p["teacher"] for p in papers}}
    out["closed_book"] = {p["id"]: {"__closed_book__": "1"} for p in papers}
    out["abstract"] = {p["id"]: {"abstract": p["paper"].split("\n\n", 1)[0]} for p in papers}
    out["full_paper"] = {p["id"]: {"paper_text": p["paper"]} for p in papers}
    for label in labels:
        if label not in out:
            out[label] = read_system(s, label, ids)
    return out


def parse_obj(text: str, key: str) -> list:
    obj, _ = parse_prediction(text)
    v = obj.get(key) if isinstance(obj, dict) else None
    return v if isinstance(v, list) else []


async def run_questions(s: Settings, papers: list[dict], concurrency: int, deadline: float) -> int:
    path = fdir(s) / "questions.jsonl.gz"
    done = {r["id"] for r in read_jsonl(path)} | failed_ids(s)
    todo = [p for p in papers if p["id"] not in done]
    log.info("questions: %d papers, %d done or failed", len(papers), len(done))
    if not todo:
        return 0
    chat = Chat(s, concurrency, s.fact_gen_model, GEN_SYSTEM, 3_000)
    quota = ", ".join(f"{v} {k}" for k, v in QTYPES.items())
    n = sum(QTYPES.values())

    async def one(p: dict) -> dict | None:
        """Up to three tries; a paper whose reply never parses is recorded as failed, not retried forever."""
        reason = ""
        for attempt in range(3):
            if time.monotonic() > deadline:
                return None
            text, meta = await chat.one(GEN_PROMPT.format(n=n, quota=quota, paper=p["paper"]))
            qs = [q for q in parse_obj(text, "questions") if isinstance(q, dict) and q.get("question") and q.get("answer")]
            if len(qs) >= n // 2:
                return {"id": p["id"], "questions": [{"type": str(q.get("type", "")), "question": str(q["question"]),
                                                      "answer": str(q["answer"])} for q in qs[:n]], "meta": meta}
            reason = f"{len(qs)} usable questions ({meta.get('error') or meta.get('finish_reason')})"
            log.warning("%s: %s, attempt %d", p["id"], reason, attempt + 1)
        append_jsonl(fdir(s) / "questions-failed.jsonl.gz", [{"id": p["id"], "reason": reason, "chars": len(text)}])
        return None

    made = 0
    for i in range(0, len(todo), 40):
        rows = [r for r in await asyncio.gather(*(one(p) for p in todo[i:i + 40])) if r]
        append_jsonl(path, rows)
        made += len(rows)
        log.info("  questions %d/%d", i + 40, len(todo))
    await chat.close()
    return made


async def run_answers(s: Settings, label: str, papers: list[dict], concurrency: int, deadline: float) -> int:
    summaries = system_summaries(s, papers, [label])[label]
    qs = {r["id"]: r["questions"] for r in read_jsonl(fdir(s) / "questions.jsonl.gz")}
    apath, gpath = fdir(s) / f"answers-{label}.jsonl.gz", fdir(s) / f"grades-{label}.jsonl.gz"
    done_a = {r["id"] for r in read_jsonl(apath)}
    todo = [p["id"] for p in papers if p["id"] in qs and p["id"] in summaries and p["id"] not in done_a]
    log.info("answers[%s]: %d to do", label, len(todo))
    if not todo:
        return 0
    ans = Chat(s, concurrency, s.judge_model, ANS_SYSTEM, 1_500)
    grd = Chat(s, concurrency, s.judge_model, GRADE_SYSTEM, 1_200)

    async def one(pid: str) -> dict | None:
        if time.monotonic() > deadline:
            return None
        obj, questions = summaries.get(pid), qs[pid]
        got: list[str] = ["NOT STATED"] * len(questions)
        if obj:
            qtext = "\n".join(f"{i + 1}. {q['question']}" for i, q in enumerate(questions))
            prompt = (CLOSED_PROMPT.format(n=len(questions), questions=qtext) if "__closed_book__" in obj else
                      ANS_PROMPT.format(n=len(questions), summary=render_for_reader(obj), questions=qtext))
            text, _ = await ans.one(prompt)
            for a in parse_obj(text, "answers"):
                try:
                    got[int(a["i"]) - 1] = str(a["answer"]).strip() or "NOT STATED"
                except (KeyError, ValueError, IndexError, TypeError):
                    continue
        verdicts = ["not_stated"] * len(questions)
        attempt = [i for i, a in enumerate(got) if not a.upper().startswith("NOT STATED")]
        if attempt:
            items = "\n\n".join(f"[{k + 1}] QUESTION: {questions[i]['question']}\nGOLD: {questions[i]['answer']}\n"
                                f"CANDIDATE: {got[i]}" for k, i in enumerate(attempt))
            text, _ = await grd.one(GRADE_PROMPT.format(items=items))
            for g in parse_obj(text, "grades"):
                try:
                    v = str(g["verdict"]).strip().lower()
                    verdicts[attempt[int(g["i"]) - 1]] = v if v in ("correct", "partial", "incorrect") else "incorrect"
                except (KeyError, ValueError, IndexError, TypeError):
                    continue
            for i in attempt:  # a grade the model failed to return is a missing grade, not a free pass
                if verdicts[i] == "not_stated":
                    verdicts[i] = "ungraded"
        return {"id": pid, "usable": obj is not None, "answers": got, "verdicts": verdicts,
                "types": [q["type"] for q in questions]}

    made = 0
    for i in range(0, len(todo), 40):
        rows = [r for r in await asyncio.gather(*(one(pid) for pid in todo[i:i + 40])) if r]
        append_jsonl(apath, [{k: r[k] for k in ("id", "usable", "answers")} for r in rows])
        append_jsonl(gpath, [{k: r[k] for k in ("id", "usable", "verdicts", "types")} for r in rows])
        made += len(rows)
        log.info("  %s %d/%d", label, min(i + 40, len(todo)), len(todo))
    await ans.close()
    await grd.close()
    return made


def score_rows(rows: list[dict]) -> dict:
    """Per-system rates over every question; unusable summaries count as all NOT STATED."""
    c = collections.Counter()
    for r in rows:
        c.update(r["verdicts"])
    n = sum(c.values())
    graded = n - c["ungraded"]
    attempted = c["correct"] + c["partial"] + c["incorrect"]
    return {"papers": len(rows), "questions": n, "ungraded": c["ungraded"],
            "usable_summary": round(sum(r["usable"] for r in rows) / max(len(rows), 1), 4),
            **{v: round(c[v] / max(graded, 1), 4) for v in VERDICTS},
            "recall": round((c["correct"] + 0.5 * c["partial"]) / max(graded, 1), 4),
            "accuracy_when_answering": round(c["correct"] / attempted, 4) if attempted else None,
            "attempted": round(attempted / max(graded, 1), 4)}


def paper_recall(r: dict) -> float:
    g = [v for v in r["verdicts"] if v != "ungraded"]
    return (sum(v == "correct" for v in g) + 0.5 * sum(v == "partial" for v in g)) / max(len(g), 1)


def report(s: Settings, seed: int = 0, boots: int = 2000) -> dict:
    papers = {p["id"]: p for p in select_papers(s)}
    # every system that has grades in this question set: the teacher, controls, anchors and any run
    labs = sorted({p.name[len("grades-"):-len(".jsonl.gz")] for p in fdir(s).glob("grades-*.jsonl.gz")})
    labs = [TEACHER, *CONTROLS, *(lab for lab in labs if lab not in (TEACHER, *CONTROLS))]
    grades = {lab: {r["id"]: r for r in read_jsonl(fdir(s) / f"grades-{lab}.jsonl.gz")} for lab in labs}
    res: dict = {"n_papers": len(papers), "question_failures": sorted(failed_ids(s)), "systems": {}}
    rnd = random.Random(seed)
    for lab, g in grades.items():
        if not g:
            continue
        rows = list(g.values())
        entry = score_rows(rows)
        by_type = collections.defaultdict(list)
        for r in rows:
            for t, v in zip(r["types"], r["verdicts"], strict=True):
                by_type[t].append(v)
        entry["recall_by_type"] = {t: round((sum(v == "correct" for v in vs if v != "ungraded") + 0.5 * sum(v == "partial" for v in vs))
                                            / max(sum(v != "ungraded" for v in vs), 1), 4) for t, vs in sorted(by_type.items())}
        for coh in ("top", "random"):
            sub = [r for r in rows if (papers[r["id"]]["cohort"] == "random") == (coh == "random")]
            entry[f"recall_{coh}"] = {"n": len(sub), "recall": round(st.fmean(paper_recall(r) for r in sub), 4) if sub else None}
        res["systems"][lab] = entry
    common = sorted(set.intersection(*(set(g) for g in grades.values() if g))) if any(grades.values()) else []
    res["n_common"] = len(common)
    # paired bootstrap over papers, on the papers every system has (the base models cover only some)
    if "teacher" in grades and common:
        base = {i: paper_recall(grades["teacher"][i]) for i in common}
        res["recall_minus_teacher"] = {}
        for lab, g in grades.items():
            if lab == "teacher" or not g:
                continue
            d = [paper_recall(g[i]) - base[i] for i in common]
            bs = sorted(st.fmean(rnd.choices(d, k=len(d))) for _ in range(boots))
            res["recall_minus_teacher"][lab] = {"mean": round(st.fmean(d), 4), "ci95": [round(bs[int(.025 * boots)], 4), round(bs[int(.975 * boots)], 4)]}
    res["paired"] = {}
    for a, b in (("A_base", "A"), ("A", "B"), ("A", "C"), ("B", "C"), ("C_base", "C")):
        if grades.get(a) and grades.get(b):
            ids = sorted(set(grades[a]) & set(grades[b]))
            d = [paper_recall(grades[b][i]) - paper_recall(grades[a][i]) for i in ids]
            bs = sorted(st.fmean(rnd.choices(d, k=len(d))) for _ in range(boots))
            res["paired"][f"{b} - {a}"] = {"n": len(ids), "mean": round(st.fmean(d), 4),
                                           "ci95": [round(bs[int(.025 * boots)], 4), round(bs[int(.975 * boots)], 4)]}
    (fdir(s) / "report.json").write_text(json.dumps(res, indent=1))
    return res


def export_for_checking(s: Settings, n: int = 40, out: Path | None = None) -> Path:
    """Questions with their gold answers and the supporting paper text, for a person to verify."""
    papers = {p["id"]: p for p in select_papers(s)}
    qs = read_jsonl(fdir(s) / "questions.jsonl.gz")
    rnd = random.Random(1)
    pick = rnd.sample(qs, min(n, len(qs)))
    lines = ["# Fact-check questions to verify", "",
             "For each question, mark the gold answer **right / wrong / ambiguous** against the paper. "
             "Paper text is the same the question writer saw (truncated to its first 3,000 characters here; "
             "`state/factcheck/papers.json` has all of it).", ""]
    for r in pick:
        p = papers[r["id"]]
        lines += [f"## {p['title']}  (`{r['id']}`, {p['archive']}, {p['cohort']})", "", f"> {p['paper'][:3000].replace(chr(10), ' ')}", ""]
        for i, q in enumerate(r["questions"], 1):
            lines += [f"{i}. **[{q['type']}]** {q['question']}", f"   - gold: {q['answer']}", "   - [ ] right  [ ] wrong  [ ] ambiguous", ""]
    path = out or Path("paper_docs/samples/factcheck_to_verify.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return path


def fingerprint(s: Settings) -> str:
    """What the question set depends on, so a changed prompt or model is a new set, not a silent mix."""
    return hashlib.sha256(json.dumps([GEN_PROMPT, s.fact_gen_model, s.fact_n_papers, s.fact_pool, s.seed]).encode()).hexdigest()[:12]


async def run_all(s: Settings, labels: list[str], *, concurrency: int = 8, max_seconds: float = 3000) -> dict:
    deadline = time.monotonic() + max_seconds
    papers = select_papers(s)
    fp = fdir(s) / "fingerprint"
    if fp.exists() and fp.read_text() != fingerprint(s):
        raise RuntimeError("the question set was made with a different prompt/model; move state/factcheck aside")
    fp.write_text(fingerprint(s))
    out: dict = {"questions": await run_questions(s, papers, concurrency, deadline)}
    for lab in labels:
        out[lab] = await run_answers(s, lab, papers, concurrency, deadline)
    out["remaining"] = remaining(s, labels)
    return out


def remaining(s: Settings, labels: list[str]) -> int:
    """Papers still missing a question set or a graded answer set, over the systems asked for."""
    papers = select_papers(s)
    have_q = {r["id"] for r in read_jsonl(fdir(s) / "questions.jsonl.gz")}
    left = sum(p["id"] not in have_q and p["id"] not in failed_ids(s) for p in papers)
    summ = system_summaries(s, papers, labels)
    for lab in labels:
        done = {r["id"] for r in read_jsonl(fdir(s) / f"grades-{lab}.jsonl.gz")}
        left += sum(p["id"] not in done and p["id"] in summ[lab] and p["id"] in have_q for p in papers)
    return left


async def cross_grade(s: Settings, label: str, model: str, concurrency: int = 8) -> dict:
    """Re-grade a system's attempted answers with a different model, and say how often the two agree.

    Agreement is on the three-way verdict (correct / partial / incorrect). Cohen's kappa corrects it
    for chance. A grader that disagrees with a second grader this often cannot be trusted to rank two
    systems that differ by less than that.
    """
    qs = {r["id"]: r["questions"] for r in read_jsonl(fdir(s) / "questions.jsonl.gz")}
    ans = {r["id"]: r["answers"] for r in read_jsonl(fdir(s) / f"answers-{label}.jsonl.gz")}
    orig = {r["id"]: r["verdicts"] for r in read_jsonl(fdir(s) / f"grades-{label}.jsonl.gz")}
    grd = Chat(s, concurrency, model, GRADE_SYSTEM, 1_200)

    async def one(pid: str) -> list[tuple[str, str]]:
        att = [i for i, a in enumerate(ans[pid]) if not a.upper().startswith("NOT STATED")]
        if not att:
            return []
        items = "\n\n".join(f"[{k + 1}] QUESTION: {qs[pid][i]['question']}\nGOLD: {qs[pid][i]['answer']}\n"
                             f"CANDIDATE: {ans[pid][i]}" for k, i in enumerate(att))
        text, _ = await grd.one(GRADE_PROMPT.format(items=items))
        got = {}
        for g in parse_obj(text, "grades"):
            try:
                got[int(g["i"]) - 1] = str(g["verdict"]).strip().lower()
            except (KeyError, ValueError, TypeError):
                continue
        return [(orig[pid][i], got[k]) for k, i in enumerate(att) if k in got and orig[pid][i] in ("correct", "partial", "incorrect")]

    pairs = [p for rows in await asyncio.gather(*(one(pid) for pid in ans if pid in qs and pid in orig)) for p in rows]
    await grd.close()
    n = len(pairs)
    labels = ("correct", "partial", "incorrect")
    agree = sum(a == b for a, b in pairs) / max(n, 1)
    pa = {k: sum(a == k for a, _ in pairs) / max(n, 1) for k in labels}
    pb = {k: sum(b == k for _, b in pairs) / max(n, 1) for k in labels}
    chance = sum(pa[k] * pb[k] for k in labels)
    return {"system": label, "second_grader": model, "items": n, "agreement": round(agree, 4),
            "kappa": round((agree - chance) / (1 - chance), 4) if chance < 1 else None,
            "original_mix": {k: round(pa[k], 3) for k in labels}, "second_mix": {k: round(pb[k], 3) for k in labels}}


async def answer_and_grade(ans: Chat, grd: Chat, reader_prompt: str, questions: list[dict]) -> list[str]:
    """One reader, one question set: the reader answers from ``reader_prompt``, the grader marks it.

    Returns a verdict per question: correct / partial / incorrect, ``not_stated`` when the reader
    declined, ``ungraded`` when it answered but the grader returned nothing for it (counted as
    neither right nor wrong by the callers). The same logic as ``run_answers``, reusable for any
    reader (a summary, the whole paper, nothing) and any pair of models.
    """
    n = len(questions)
    got = ["NOT STATED"] * n
    for _attempt in range(3):   # a reply that parses to no answers at all is a failed call, not a reader that knew nothing
        text, _ = await ans.one(reader_prompt)
        answers = parse_obj(text, "answers")
        if answers:
            break
    for a in answers:
        try:
            got[int(a["i"]) - 1] = str(a["answer"]).strip() or "NOT STATED"
        except (KeyError, ValueError, IndexError, TypeError):
            continue
    verdicts = ["not_stated"] * n
    attempted = [i for i, a in enumerate(got) if not a.upper().startswith("NOT STATED")]
    if attempted:
        items = "\n\n".join(f"[{k + 1}] QUESTION: {questions[i]['question']}\nGOLD: {questions[i]['answer']}\n"
                             f"CANDIDATE: {got[i]}" for k, i in enumerate(attempted))
        text, _ = await grd.one(GRADE_PROMPT.format(items=items))
        for g in parse_obj(text, "grades"):
            try:
                v = str(g["verdict"]).strip().lower()
                verdicts[attempted[int(g["i"]) - 1]] = v if v in ("correct", "partial", "incorrect") else "incorrect"
            except (KeyError, ValueError, IndexError, TypeError):
                continue
        for i in attempted:
            if verdicts[i] == "not_stated":
                verdicts[i] = "ungraded"
    return verdicts
