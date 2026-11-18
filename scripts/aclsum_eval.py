"""Students against human-written reference summaries: ACLSum (Takeshita et al., 2024).

Every other reference in this project is a teacher model's. ACLSum (`sobamchan/aclsum`, MIT) has,
for 100 held-out ACL Anthology papers, summaries of three aspects - challenge, approach, outcome -
that are not the teacher's. We map them onto the fields the students already produce:

    challenge -> problem      approach -> method      outcome -> results

Caveats that belong next to the numbers: the dataset's `document` is a short extract (median 5,756
characters: abstract, introduction and conclusion, tokenised with spaces before punctuation), not the
46,000-character full text the students were trained on; the papers are NLP only, and the students
were trained mostly on physics; and an aspect summary is a sentence, a student's field a paragraph,
so ROUGE here is precision-limited. Read it as a ranking, not an absolute.

    prepare                 download, clean, write state/aclsum/data.jsonl
    generate RUN CKPT LABEL vLLM, one system, resumable; run under that system's model profile
    score                   ROUGE-L, BERTScore and an LLM coverage grade per aspect, per system
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import statistics as st
from pathlib import Path

from scisum.config import get_settings

OUT = Path("state/aclsum")
MAP = {"challenge": "problem", "approach": "method", "outcome": "results"}
SYSTEMS = {"A": ("main", "ex70000"), "B": ("k2", "ex70000"), "C": ("q35", "ex20000"), "A_base": ("main", "base")}


def detok(t: str) -> str:
    t = re.sub(r"\s+([,.;:!?%)\]])", r"\1", t)
    t = re.sub(r"([(\[$])\s+", r"\1", t)
    return re.sub(r"\s+('s|n't|'re|'ve)\b", r"\1", t).strip()


def read_jsonl(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def prepare() -> None:
    from huggingface_hub import hf_hub_download
    OUT.mkdir(parents=True, exist_ok=True)
    p = hf_hub_download("sobamchan/aclsum", "abstractive/test.jsonl", repo_type="dataset")
    rows = read_jsonl(p)
    with open(OUT / "data.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps({"id": r["id"], "document": detok(r["document"]),
                                **{k: r[k] for k in MAP}}) + "\n")
    print(f"{len(rows)} papers written")


def generate(run: str, ckpt: str, label: str) -> None:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    from scisum.dataset import render_prompt, tokenizer
    from scisum.evaluate import adapter_path, check_adapter_applied, serve_model_args
    s = get_settings()
    out = OUT / f"pred-{label}.jsonl"
    if out.exists() and len(read_jsonl(out)) >= 100:
        print(f"{label}: already done")
        return
    rows = read_jsonl(OUT / "data.jsonl")
    tok = tokenizer(s)
    prompts = [render_prompt(tok, {"title": "", "authors": [], "categories": [], "abstract": ""}, r["document"])
               for r in rows]
    adapter = adapter_path(s, run, ckpt)
    llm = LLM(dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.85, enable_lora=adapter is not None,
              max_lora_rank=s.lora_r, max_loras=1, seed=0, disable_log_stats=True,
              trust_remote_code=s.trust_remote_code, **serve_model_args(s))
    lora = LoRARequest(ckpt, 1, str(adapter)) if adapter else None
    if lora:
        check_adapter_applied(llm, lora, prompts[:4])
    sp = SamplingParams(temperature=0.0, max_tokens=3000)
    outs = llm.generate(prompts, sp, use_tqdm=False, **({"lora_request": lora} if lora else {}))
    tmp = out.with_suffix(".tmp")
    with open(tmp, "w") as f:
        for r, o in zip(rows, outs, strict=True):
            f.write(json.dumps({"id": r["id"], "text": o.outputs[0].text, "finish": o.outputs[0].finish_reason,
                                "tokens": len(o.outputs[0].token_ids)}) + "\n")
    tmp.rename(out)
    print(f"{label}: wrote {len(rows)}")


GRADE_SYSTEM = "You are a strict grader. You answer with JSON and nothing else."
GRADE = """An expert wrote a one-sentence summary of one aspect of a paper (HUMAN). A system produced text
for the same aspect (SYSTEM). For each item, does SYSTEM convey the main point of HUMAN?

- covers: it states the same main point (wording and extra detail may differ).
- partly: it states part of it, or the right topic but not the point.
- misses: it does not, or says something different.

{items}

Answer with JSON: {{"grades": [{{"i": 1, "verdict": "covers|partly|misses"}}, ...]}}"""


async def score() -> None:
    from scisum.bertscore import BertScorer
    from scisum.factcheck import Chat, parse_obj
    from scisum.metrics import field_text, parse_prediction, rouge
    s = get_settings()
    rows = {r["id"]: r for r in read_jsonl(OUT / "data.jsonl")}
    chat = Chat(s, 8, s.judge_model, GRADE_SYSTEM, 800)
    bert = BertScorer(device="cpu", batch_size=16)
    res: dict = {"n": len(rows), "systems": {}}
    for label in SYSTEMS:
        path = OUT / f"pred-{label}.jsonl"
        if not path.exists():
            continue
        preds = {r["id"]: r for r in read_jsonl(path)}
        objs = {i: parse_prediction(p["text"])[0] for i, p in preds.items()}
        entry: dict = {"parsed": round(sum(isinstance(o, dict) for o in objs.values()) / len(rows), 4), "aspects": {}}

        async def grade(i: str, objs=objs) -> dict[str, str]:
            o = objs.get(i)
            items = [(a, field_text(o, f)) for a, f in MAP.items()]
            text = "\n\n".join(f"[{k + 1}] ({a})\nHUMAN: {rows[i][a]}\nSYSTEM: {t or '(nothing)'}" for k, (a, t) in enumerate(items))
            out, _ = await chat.one(GRADE.format(items=text))
            got = {}
            for g in parse_obj(out, "grades"):
                try:
                    got[items[int(g["i"]) - 1][0]] = str(g["verdict"]).lower()
                except (KeyError, ValueError, IndexError, TypeError):
                    continue
            return got
        verdicts = dict(zip(rows, await asyncio.gather(*(grade(i) for i in rows)), strict=True))
        for a, f in MAP.items():
            pred = [field_text(objs.get(i), f) for i in rows]
            ref = [rows[i][a] for i in rows]
            rl = [rouge(p, r)["rougeL"] for p, r in zip(pred, ref, strict=True)]
            bs = [x["f1"] for x in bert.score(pred, ref)]
            v = [verdicts[i].get(a, "ungraded") for i in rows]
            entry["aspects"][a] = {"rougeL": round(st.fmean(rl), 4), "bertscore_f1": round(st.fmean(bs), 4),
                                   **{k: round(v.count(k) / len(v), 4) for k in ("covers", "partly", "misses", "ungraded")},
                                   "coverage": round((v.count("covers") + 0.5 * v.count("partly")) / max(len(v) - v.count("ungraded"), 1), 4),
                                   "mean_words": round(st.fmean(len(p.split()) for p in pred), 1)}
        entry["coverage_mean"] = round(st.fmean(x["coverage"] for x in entry["aspects"].values()), 4)
        # per-paper results, so differences between systems can be given a paired interval
        score = {"covers": 1.0, "partly": 0.5, "misses": 0.0}
        entry["per_paper_coverage"] = {i: round(st.fmean(score.get(verdicts[i].get(a), 0.0) for a in MAP), 4) for i in rows}
        res["systems"][label] = entry
        print(label, json.dumps(entry), flush=True)
    rnd = random.Random(0)
    res["paired_coverage_difference"] = {}
    for a in res["systems"]:
        for b in res["systems"]:
            if a < b:
                x, y = res["systems"][a]["per_paper_coverage"], res["systems"][b]["per_paper_coverage"]
                d = [y[i] - x[i] for i in rows]
                bs = sorted(st.fmean(rnd.choices(d, k=len(d))) for _ in range(2000))
                res["paired_coverage_difference"][f"{b} - {a}"] = {"mean": round(st.fmean(d), 4), "ci95": [round(bs[50], 4), round(bs[1950], 4)]}
    Path("paper_docs/data/aclsum_results.json").write_text(json.dumps(res, indent=1))
    await chat.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    g = sub.add_parser("generate")
    g.add_argument("run"), g.add_argument("ckpt"), g.add_argument("label")
    sub.add_parser("score")
    a = ap.parse_args()
    if a.cmd == "prepare":
        prepare()
    elif a.cmd == "generate":
        generate(a.run, a.ckpt, a.label)
    else:
        asyncio.run(score())


if __name__ == "__main__":
    main()
