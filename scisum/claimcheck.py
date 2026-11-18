"""Claim-level faithfulness against the whole paper: every statement of a summary, checked by an LLM
that reads the full text.

Why (round 1 of ``scisum.improve``, 2026-11-05): the preference arm answered more fact-test questions
at an unchanged error rate per answer, while the two faithfulness checks that see only part of the
paper went down: NLI (DeBERTa against the three BM25-retrieved 3-sentence chunks closest to each
sentence) and the judge (a 9,000-character excerpt; it gives even the teacher about 2.1 of 5 on
faithfulness). A longer, more specific summary makes more statements that a partial view cannot
check, so a drop in those two does not tell invented content from content the instrument did not
see. This check removes the partial view.

Design:

* **Papers.** A seeded random sample of the fact-test papers (``state/factcheck*/papers.json``),
  whose stored ``paper`` is the text the summarisers read (Student C's prompt body, which is also the
  longest budget in use), cut at ``fact_paper_chars`` (90,000 characters) as for the question writer.
* **Statements.** ``scisum.faithfulness.claims_of``: the prose sentences of tldr, problem, method,
  results and limitations, and the contribution bullets, the same units NLI scores, but up to
  ``MAX_CLAIMS`` and not 40, so a longer summary is checked in full.
* **Verifier.** ``judge_model`` (Qwen3.8-27B, the evaluation's grader; thinking off, temperature 0) reads
  the paper and one summary's numbered statements and labels each ``supported``, ``not_in_paper`` or
  ``contradicted``, quoting the offending detail. One call per summary; the verifier is never told
  which system wrote it. A second model can be run on the same statements (``--model``) to see whether
  two verifiers agree.
* **Controls**, on the first ``n_controls`` papers, show what the instrument can detect: the teacher's
  summary with every checkable number changed (``scisum.improve._perturb_numbers``), and another
  paper's teacher summary.

A statement's verdict is that of its worst detail, so longer statements that combine more details are
more often flagged; the report gives counts per summary and shares per statement, and the claim shape
(statements per summary, words per statement) next to them.

Everything appends and is keyed by (system, paper), so a killed job resumes where it stopped.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import statistics as st
import time
from pathlib import Path

from scisum.config import Settings
from scisum.factcheck import Chat, append_jsonl, parse_obj, read_jsonl, read_system
from scisum.faithfulness import claims_of

log = logging.getLogger(__name__)

MAX_CLAIMS = 80
LABELS = ("supported", "not_in_paper", "contradicted")
CONTROLS = ("teacher_perturbed", "other_paper")
FACT_DIRS = ("factcheck", "factcheck-b")

SYSTEM = "You check summaries of scientific papers against the paper. You answer with JSON and nothing else."
PROMPT = """Here is the text of a scientific paper, followed by {n} numbered statements taken from a summary of it.

For each statement, decide whether the paper supports it:
- "supported": everything the statement asserts is stated in the paper or follows directly from it. Paraphrase, abbreviation and reasonable generalisation are fine.
- "not_in_paper": some specific detail of the statement (a number, a name, a dataset, a method, a comparison, a result, a limitation) does not appear in the paper and does not follow from it.
- "contradicted": some detail of the statement conflicts with the paper (a different number, the opposite direction, a wrong attribution).
A statement with several details gets the verdict of its worst detail; "contradicted" is worse than "not_in_paper".
For "not_in_paper" and "contradicted", quote the offending detail in "detail" (a few words), and for "contradicted" say what the paper says in "paper".
Judge only against this paper: do not use outside knowledge, and do not penalise a statement for leaving things out.

PAPER:
{paper}

STATEMENTS:
{statements}

Answer with JSON: {{"verdicts": [{{"i": 1, "verdict": "supported|not_in_paper|contradicted", "detail": "", "paper": ""}}, ...]}}
"""


def cdir(s: Settings) -> Path:
    d = s.state_dir / s.claim_dir
    d.mkdir(parents=True, exist_ok=True)
    return d


def out_path(s: Settings, label: str, model: str) -> Path:
    return cdir(s) / f"{label}.{model}.jsonl.gz"


def sample_papers(s: Settings, n: int, seed: int = 0) -> list[dict]:
    """``n`` fact-test papers, a seeded random sample of both question sets, in a fixed order."""
    papers: dict[str, dict] = {}
    for d in FACT_DIRS:
        p = s.state_dir / d / "papers.json"
        if p.exists():
            for r in json.loads(p.read_text()):
                papers.setdefault(r["id"], r)
    ids = sorted(papers)
    random.Random(seed).shuffle(ids)
    return [papers[i] for i in ids[:n]]


def summaries(s: Settings, label: str, papers: list[dict], n_controls: int) -> dict[str, dict | None]:
    """paper id -> parsed summary of ``label`` (a fact-test system, or a control)."""
    if label == "teacher":
        return {p["id"]: p["teacher"] for p in papers}
    if label == "teacher_perturbed":
        from scisum.improve import synthetic_candidates
        return {p["id"]: synthetic_candidates(p["teacher"], p["teacher"])[1] for p in papers[:n_controls]}
    if label == "other_paper":
        ctl = papers[:n_controls]
        return {p["id"]: ctl[(i + 1) % len(ctl)]["teacher"] for i, p in enumerate(ctl)} if len(ctl) > 1 else {}
    return read_system(s, label, {p["id"] for p in papers})


# One verdict record, read on its own: the fallback for a reply that is not valid JSON as a whole.
_RECORD = re.compile(r'"i"\s*:\s*"?(\d+)"?\s*,\s*"verdict"\s*:\s*"([^"]*)"(?:\s*,\s*"detail"\s*:\s*"((?:[^"\\]|\\.)*)")?')


def parse_verdicts(text: str, n: int) -> list[dict]:
    """``n`` verdict records in statement order; a statement the reply skipped is ``ungraded``.

    If the reply is not valid JSON as a whole, each record is read on its own (``i`` and ``verdict``,
    and ``detail`` when it is a well-formed string). Added 2026-11-07: 15 of 2,475 Qwen3.8-27B and 3 of
    529 Gemma-4-31B replies were complete except for one malformed field (text after a closing quote in
    ``paper``); at temperature 0 the three retries returned the same text, so the summary went unlabelled.
    """
    records = parse_obj(text, "verdicts")
    if not records:
        records = [{"i": m.group(1), "verdict": m.group(2), "detail": m.group(3) or ""} for m in _RECORD.finditer(text)]
    got: dict[int, dict] = {}
    for v in records:
        if not isinstance(v, dict):
            continue
        try:
            i = int(v.get("i"))
        except (TypeError, ValueError):
            continue
        lab = str(v.get("verdict", "")).strip().lower().replace(" ", "_").replace("-", "_")
        if 1 <= i <= n and lab in LABELS and i not in got:
            got[i] = {"verdict": lab, "detail": str(v.get("detail") or "")[:300], "paper": str(v.get("paper") or "")[:300]}
    return [got.get(i, {"verdict": "ungraded", "detail": "", "paper": ""}) for i in range(1, n + 1)]


async def check_one(chat: Chat, paper: str, claims: list[str], tries: int = 3) -> tuple[list[dict], dict]:
    statements = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(claims))
    prompt = PROMPT.format(n=len(claims), paper=paper, statements=statements)
    best = parse_verdicts("", len(claims))   # all ungraded until a reply labels something
    meta: dict = {}
    for attempt in range(tries):
        text, meta = await chat.one(prompt)
        v = parse_verdicts(text, len(claims))
        if sum(x["verdict"] != "ungraded" for x in v) > sum(x["verdict"] != "ungraded" for x in best):
            best = v
        if sum(x["verdict"] == "ungraded" for x in best) <= len(claims) // 10:
            break
        log.warning("%d of %d statements unlabelled (attempt %d, %s)", sum(x["verdict"] == "ungraded" for x in v),
                    len(claims), attempt + 1, meta.get("error") or meta.get("finish_reason"))
    return best, meta


def checked_ids(path: Path) -> set[str]:
    """Papers already checked. A row with statements but no verdicts at all was written before 2026-11-07
    (the reply could not be parsed); it is checked again, and the new row, appended later, wins."""
    return {r["id"] for r in read_jsonl(path) if r["verdicts"] or not r["claims"]}


async def run(s: Settings, labels: list[str], *, n: int = 200, n_controls: int = 40, model: str | None = None,
              concurrency: int = 16, max_seconds: float = 3000) -> dict:
    deadline = time.monotonic() + max_seconds
    model = model or s.judge_model
    papers = sample_papers(s, n)
    chat = Chat(s, concurrency, model, SYSTEM, 3_000)
    out: dict = {"papers": len(papers), "model": model}
    for lab in labels:
        summ = summaries(s, lab, papers, n_controls)
        path = out_path(s, lab, model)
        done = checked_ids(path)
        todo = [p for p in papers if p["id"] in summ and p["id"] not in done]

        async def one(p: dict, summ=summ):
            if time.monotonic() > deadline:
                return None
            obj = summ.get(p["id"])
            claims = claims_of(obj, MAX_CLAIMS)
            if not claims:
                return {"id": p["id"], "claims": [], "verdicts": [], "valid": obj is not None}
            verdicts, meta = await check_one(chat, p["paper"], claims)
            return {"id": p["id"], "claims": claims, "verdicts": verdicts, "valid": True,
                    "finish": meta.get("finish_reason"), "seconds": meta.get("seconds")}

        made = 0
        for i in range(0, len(todo), 32):
            rows = [r for r in await asyncio.gather(*(one(p) for p in todo[i:i + 32])) if r is not None]
            append_jsonl(path, rows)
            made += len(rows)
            log.info("  %s: %d/%d summaries checked", lab, made, len(todo))
            if time.monotonic() > deadline:
                break
        out[lab] = made
    await chat.close()
    out["remaining"] = remaining(s, labels, n, n_controls, model)
    return out


def remaining(s: Settings, labels: list[str], n: int, n_controls: int, model: str) -> int:
    papers = sample_papers(s, n)
    left = 0
    for lab in labels:
        summ = summaries(s, lab, papers, n_controls)
        done = checked_ids(out_path(s, lab, model))
        left += sum(p["id"] in summ and p["id"] not in done for p in papers)
    return left


# ---- report ------------------------------------------------------------------------------------------
def paper_stats(r: dict) -> dict:
    """Per summary: statements, how many of each verdict, and the shares among labelled statements."""
    v = [x["verdict"] for x in r["verdicts"]]
    lab = [x for x in v if x != "ungraded"]
    k = max(len(lab), 1)
    words = [len(c.split()) for c in r["claims"]]
    return {"claims": len(v), "labelled": len(lab), "supported": lab.count("supported"),
            "not_in_paper": lab.count("not_in_paper"), "contradicted": lab.count("contradicted"),
            "unsupported": lab.count("not_in_paper") + lab.count("contradicted"),
            "supported_share": lab.count("supported") / k, "contradicted_share": lab.count("contradicted") / k,
            "not_in_paper_share": lab.count("not_in_paper") / k,
            "words": sum(words), "words_per_claim": st.fmean(words) if words else 0.0}


def _boot(d: list[float], boots: int = 2000, seed: int = 0) -> list[float]:
    if len(d) < 2:
        return [float("nan")] * 3
    rnd = random.Random(seed)
    ms = sorted(st.fmean(rnd.choices(d, k=len(d))) for _ in range(boots))
    return [round(st.fmean(d), 4), round(ms[int(0.025 * boots)], 4), round(ms[int(0.975 * boots)], 4)]


METRICS = ("claims", "supported_share", "not_in_paper_share", "contradicted_share", "unsupported", "contradicted",
           "words_per_claim")


def report(s: Settings, model: str | None = None, base: tuple[str, ...] = ("C", "teacher")) -> dict:
    """Per system: means over papers; per pair (system - base): paired bootstrap over common papers."""
    model = model or s.judge_model
    rows: dict[str, dict[str, dict]] = {}
    unlabelled: dict[str, int] = {}
    for f in sorted(cdir(s).glob(f"*.{model}.jsonl.gz")):
        lab = f.name[: -len(f".{model}.jsonl.gz")]
        # the last row per paper wins (a re-check is appended); a summary with no labelled statement has no
        # verdict, so it is left out rather than counted as 0% supported with 0 unsupported statements
        by = {r["id"]: r for r in read_jsonl(f) if r.get("valid") and r["claims"]}
        stats = {i: paper_stats(r) for i, r in by.items()}
        rows[lab] = {i: p for i, p in stats.items() if p["labelled"]}
        unlabelled[lab] = len(stats) - len(rows[lab])
    out: dict = {"model": model, "systems": {}, "paired": {}}
    for lab, by in rows.items():
        if not by:
            continue
        ps = list(by.values())
        pooled = {k: sum(p[k] for p in ps) for k in ("labelled", "supported", "not_in_paper", "contradicted")}
        out["systems"][lab] = {"papers": len(ps), "unlabelled_summaries": unlabelled[lab], **{m: round(st.fmean(p[m] for p in ps), 4) for m in METRICS},
                               "pooled_supported": round(pooled["supported"] / max(pooled["labelled"], 1), 4),
                               "pooled_contradicted": round(pooled["contradicted"] / max(pooled["labelled"], 1), 4),
                               "pooled_not_in_paper": round(pooled["not_in_paper"] / max(pooled["labelled"], 1), 4),
                               "unlabelled_share": round(1 - pooled["labelled"] / max(sum(p["claims"] for p in ps), 1), 4)}
    for b in base:
        if b not in rows:
            continue
        for lab in rows:
            if lab == b or lab in CONTROLS:
                continue
            ids = sorted(set(rows[lab]) & set(rows[b]))
            if len(ids) < 2:
                continue
            out["paired"][f"{lab} - {b}"] = {"papers": len(ids), **{
                m: _boot([rows[lab][i][m] - rows[b][i][m] for i in ids]) for m in METRICS}}
    out["controls"] = control_sensitivity(s, model)
    (cdir(s) / f"report.{model}.json").write_text(json.dumps(out, indent=1))
    return out


def control_sensitivity(s: Settings, model: str) -> dict:
    """What the verifier catches: of the perturbed teacher's statements, those whose numbers were changed
    should be flagged and the unchanged ones should not; every statement of another paper's summary
    should be flagged. The teacher's own statements, on the same papers, are the baseline."""
    raw = {lab: {r["id"]: r for r in read_jsonl(out_path(s, lab, model))} for lab in ("teacher", *CONTROLS)}
    changed, unchanged = [], []
    for pid, r in raw["teacher_perturbed"].items():
        t = raw["teacher"].get(pid)
        if t is None or len(t["claims"]) != len(r["claims"]):
            continue
        for a, b, v in zip(t["claims"], r["claims"], r["verdicts"], strict=True):
            if v["verdict"] != "ungraded":
                (changed if a != b else unchanged).append(v["verdict"])
    other = [v["verdict"] for r in raw["other_paper"].values() for v in r["verdicts"] if v["verdict"] != "ungraded"]
    flag = lambda xs: round(sum(x != "supported" for x in xs) / len(xs), 4) if xs else None  # noqa: E731
    contra = lambda xs: round(sum(x == "contradicted" for x in xs) / len(xs), 4) if xs else None  # noqa: E731
    return {"perturbed_changed_statements": len(changed), "flagged_when_numbers_changed": flag(changed),
            "contradicted_when_numbers_changed": contra(changed),
            "perturbed_unchanged_statements": len(unchanged), "flagged_when_unchanged": flag(unchanged),
            "other_paper_statements": len(other), "flagged_other_paper": flag(other)}



# ---- a blinded sample for a person to check -----------------------------------------------------------
def export_for_people(s: Settings, labels: list[str], model: str | None = None, per_system: int = 15,
                      seed: int = 0) -> tuple[Path, Path]:
    """Statements for a person to verify against the paper: for each system, ``per_system`` flagged and
    ``per_system`` supported ones, shuffled together, each with the verifier's verdict and note and the
    paper passage most similar to the statement (BM25 over 3-sentence chunks; the passage is a pointer,
    the whole paper is the reference). Systems are replaced by codes; the key is a separate file.

    Written under ``paper_docs/samples/``, which is not tracked: it quotes paper text and summaries.
    """
    from scisum.faithfulness import Bm25, chunk_sentences, split_sentences

    model = model or s.judge_model
    rnd = random.Random(seed)
    papers = {p["id"]: p for p in sample_papers(s, 10_000)}
    codes = {lab: f"S{i + 1}" for i, lab in enumerate(rnd.sample(labels, len(labels)))}
    items = []
    for lab in labels:
        rows = [r for r in read_jsonl(out_path(s, lab, model)) if r.get("claims")]
        flagged = [(r, j) for r in rows for j, v in enumerate(r["verdicts"]) if v["verdict"] in ("not_in_paper", "contradicted")]
        supported = [(r, j) for r in rows for j, v in enumerate(r["verdicts"]) if v["verdict"] == "supported"]
        for r, j in rnd.sample(flagged, min(per_system, len(flagged))) + rnd.sample(supported, min(per_system, len(supported))):
            items.append((lab, r, j))
    rnd.shuffle(items)
    lines = ["# Claim check: statements for a person to verify", "",
             f"Verifier: {model}. For each statement, read the paper (the passage is only the most similar one) and mark",
             "whether the verifier's verdict is right. Systems are coded; the key is in the separate key file.", ""]
    chunks_cache: dict[str, tuple[list[str], Bm25]] = {}
    for n, (lab, r, j) in enumerate(items, start=1):
        pid = r["id"]
        if pid not in chunks_cache:
            ch = chunk_sentences(split_sentences(papers[pid]["paper"])) or [papers[pid]["paper"][:1500]]
            chunks_cache[pid] = (ch, Bm25(ch))
        ch, bm = chunks_cache[pid]
        v = r["verdicts"][j]
        passage = ch[bm.top_k(r["claims"][j], 1)[0]][:1200]
        lines += [f"## {n}. paper {pid}, system {codes[lab]}", "", f"**Statement:** {r['claims'][j]}", "",
                  f"**Verifier:** {v['verdict']}" + (f"; detail: {v['detail']}" if v["detail"] else "")
                  + (f"; paper says: {v['paper']}" if v["paper"] else ""), "", f"**Most similar passage:** {passage}", "",
                  "**Your verdict:** [ ] verifier right  [ ] verifier wrong: ______", ""]
    out = s.repo_root / "paper_docs" / "samples" / f"claimcheck_{model}.md"
    key = out.with_name(out.stem + "_key.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    key.write_text(json.dumps({v: k for k, v in codes.items()}, indent=1))
    return out, key
