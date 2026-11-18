"""Compare Student C with its continuations, on the same test papers, with paired intervals.

    python scripts/improve_compare.py            # CPU, a couple of minutes; run as a batch job
    python scripts/improve_compare.py --round 2  # adds the round-2 arms; writes improve_results_r2.json
    python scripts/improve_compare.py --round 3  # round 3 (claim reward, alpha 0.003, second seeds); improve_results_r3.json

Arms (a run, and the evaluation name inside it):
    C     the starting model, scored once on all 15,000 test papers (restricted here to the papers
          the continuations were evaluated on)
    ctl   C continued on the teacher's own targets for the same papers (the control)
    rft   C continued on reward-selected targets
    pref  C continued with the preference objective on reward-ranked pairs
  round 2 (length control; ``--round 2``, which leaves round 1's file as it is):
    preflc    pairs of similar length, length charged from 1.0x the teacher's (data)
    prefrd    round 1's pairs with the R-DPO length term (loss)
    preflcrd  both

Three families of measures, never mixed:
  * reference metrics against the teacher (ROUGE, BERTScore, keyword F1, format), per paper from the
    stored scores;
  * the fact test (scisum.factcheck): recall, strict correct, wrong and not-stated rates over the
    questions of both question sets, per paper, from the stored grades;
  * the judge (scisum.judge) on the first 1,000 test papers.
Differences are paired over papers and bootstrapped (2,000 resamples). An arm that is missing is
skipped. Writes paper_docs/data/improve_results.json.
"""

from __future__ import annotations

import gzip
import json
import random
import statistics as st
from pathlib import Path

ARMS = {"C": ("q35", "test-ex20000-n15000"), "ctl": ("q35ctl", "test-ex*-n*"),
        "rft": ("q35rft", "test-ex*-n*"), "pref": ("q35pref", "test-ex*-n*")}
FACT_LABEL = {"teacher": "teacher", "C": "C", "ctl": "C_ctl", "rft": "C_rft", "pref": "C_pref"}
FACT_DIRS = ("factcheck", "factcheck-b")
METRICS = ["all_rougeL", "all_rouge1", "all_rouge2", "bertscore_mean", "keywords_f1", "domain_exact", "json_ok",
           "schema_ok", "length_ratio", "stopped_on_eos", "distinct4", "out_tokens"]
PAIRS = (("ctl", "C"), ("rft", "C"), ("rft", "ctl"), ("pref", "C"), ("pref", "ctl"), ("pref", "rft"))
TEACHER_PAIRS = (("teacher", "rft"), ("teacher", "C"))
ARMS_R2 = {"preflc": ("q35preflc", "test-ex*-n*"), "prefrd": ("q35prefrd", "test-ex*-n*"),
           "preflcrd": ("q35preflcrd", "test-ex*-n*")}
FACT_R2 = {"preflc": "C_preflc", "prefrd": "C_prefrd", "preflcrd": "C_preflcrd"}
PAIRS_R2 = (("preflc", "C"), ("prefrd", "C"), ("preflcrd", "C"), ("preflc", "pref"), ("prefrd", "pref"),
            ("preflcrd", "pref"), ("preflcrd", "preflc"), ("preflcrd", "prefrd"), ("prefrd", "preflc"),
            ("pref", "C"))
TEACHER_PAIRS_R2 = (("preflc", "teacher"), ("prefrd", "teacher"), ("preflcrd", "teacher"), ("pref", "teacher"))
ARMS_R3 = {"prefcc": ("q35prefcc", "test-ex*-n*"), "preflcrd3": ("q35preflcrd3", "test-ex*-n*"),
           "preflcs2": ("q35preflcs2", "test-ex*-n*"), "prefrds2": ("q35prefrds2", "test-ex*-n*")}
FACT_R3 = {"prefcc": "C_prefcc", "preflcrd3": "C_preflcrd3", "preflcs2": "C_preflcs2", "prefrds2": "C_prefrds2"}
PAIRS_R3 = (("prefcc", "C"), ("preflcrd3", "C"), ("preflcs2", "C"), ("prefrds2", "C"), ("prefcc", "preflc"),
            ("preflcrd3", "preflc"), ("preflcs2", "preflc"), ("prefrds2", "prefrd"), ("prefcc", "prefrd"))
TEACHER_PAIRS_R3 = (("prefcc", "teacher"), ("preflcrd3", "teacher"), ("preflcs2", "teacher"), ("prefrds2", "teacher"))
ARMS_R4 = {"prefcc": ("q35prefcc", "test-ex*-n*"), "prefccs2": ("q35prefccs2", "test-ex*-n*"),
           "ccmix50": ("q35ccmix50", "test-mix50-n*"), "ccmix75": ("q35ccmix75", "test-mix75-n*"),
           "prefccrx": ("q35prefccrx", "test-ex*-n*")}
FACT_R4 = {a: f"C_{a}" for a in ARMS_R4}
PAIRS_R4 = (("prefcc", "C"), ("prefccs2", "C"), ("ccmix50", "C"), ("ccmix75", "C"), ("prefccrx", "C"),
            ("prefccs2", "prefcc"), ("ccmix50", "prefcc"), ("ccmix75", "prefcc"), ("prefccrx", "prefccs2"),
            ("prefccrx", "ccmix50"), ("prefccrx", "ccmix75"))
TEACHER_PAIRS_R4 = tuple((a, "teacher") for a in ARMS_R4)
OUT = "paper_docs/data/improve_results.json"
rnd = random.Random(0)


def use_round(r: int) -> None:
    """Rounds 2-4 add their arms and comparisons and write their own file; round 1's output file is not touched."""
    global PAIRS, TEACHER_PAIRS, OUT
    if r == 2:
        ARMS.update(ARMS_R2)
        FACT_LABEL.update(FACT_R2)
        PAIRS, TEACHER_PAIRS, OUT = PAIRS_R2, TEACHER_PAIRS_R2, "paper_docs/data/improve_results_r2.json"
    if r == 3:   # round 3 is compared with C and with the round-2 arms it varies; writes its own file
        ARMS.update(ARMS_R2)
        ARMS.update(ARMS_R3)
        FACT_LABEL.update(FACT_R2)
        FACT_LABEL.update(FACT_R3)
        PAIRS, TEACHER_PAIRS, OUT = PAIRS_R3, TEACHER_PAIRS_R3, "paper_docs/data/improve_results_r3.json"
    if r == 4:   # 12 §8: prefcc's two seeds, the C<->prefcc blends and the reactive run; C and the five only
        for a in ("ctl", "rft", "pref"):
            ARMS.pop(a)
            FACT_LABEL.pop(a)
        ARMS.update(ARMS_R4)
        FACT_LABEL.update(FACT_R4)
        PAIRS, TEACHER_PAIRS, OUT = PAIRS_R4, TEACHER_PAIRS_R4, "paper_docs/data/improve_results_r4.json"


def read_jsonl(path: Path) -> list[dict]:
    if "*" in path.name:   # a continuation run's checkpoint is named by the size of its training set
        found = sorted(path.parent.glob(path.name))
        path = found[-1] if found else path
    if not path.exists():
        return []
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


def boot(d: list[float], n: int = 2000) -> tuple[float, float, float]:
    if len(d) < 2:
        return (float("nan"),) * 3
    ms = sorted(st.fmean(rnd.choices(d, k=len(d))) for _ in range(n))
    return st.fmean(d), ms[int(0.025 * n)], ms[int(0.975 * n)]


def fmt(m: tuple[float, float, float], digits: int = 4) -> str:
    return f"{m[0]:+.{digits}f} [{m[1]:+.{digits}f}, {m[2]:+.{digits}f}]"


def robustness(grades: dict[str, dict[str, dict]], fids: list[str], n: int = 10000, seeds: int = 5) -> dict:
    """The primary measure (fact recall), checked three ways that the 2,000-resample interval is not.

    Added on 2026-11-04 after the selection arm's interval against the control came out at
    [+0.0004, +0.0134]: a lower bound that close to zero moves with the bootstrap's own randomness.
    (1) the 95% lower bound with 10,000 resamples under five seeds, (2) each question set alone
    (A: 299 papers, B: 600, written and graded the same way, disjoint), (3) a paired sign-flip
    permutation test (two-sided, 10,000 flips). The estimand is unchanged.
    """
    import numpy as np

    sets = {d: {r["id"] for r in read_jsonl(Path(f"state/{d}/grades-teacher.jsonl.gz"))} for d in FACT_DIRS}
    res = {}
    print("\nROBUSTNESS of the fact-recall differences")
    for a, b in PAIRS:
        if a not in grades or b not in grades:
            continue
        d = np.array([recall(grades[a][i]["verdicts"]) - recall(grades[b][i]["verdicts"]) for i in fids])
        lows, highs = [], []
        for seed in range(seeds):
            idx = np.random.default_rng(seed).integers(0, len(d), (n, len(d)))
            m = d[idx].mean(axis=1)
            lows.append(float(np.quantile(m, 0.025)))
            highs.append(float(np.quantile(m, 0.975)))
        flips = np.random.default_rng(0).choice([-1.0, 1.0], (n, len(d)))
        null = np.abs((flips * d).mean(axis=1))
        p = float((1 + (null >= abs(d.mean())).sum()) / (n + 1))
        per_set = {}
        for name, members in sets.items():
            sub = [x for i, x in zip(fids, d, strict=True) if i in members]
            per_set[name] = [len(sub), *boot(sub)]
        better, worse = float((d > 0).mean()), float((d < 0).mean())
        res[f"{a} - {b}"] = {"mean": float(d.mean()), "low_range": [min(lows), max(lows)],
                             "high_range": [min(highs), max(highs)], "perm_p": p, "per_set": per_set,
                             "papers_better": better, "papers_worse": worse}
        sets_txt = "; ".join(f"{k} (n={v[0]}) {fmt(tuple(v[1:]))}" for k, v in per_set.items())
        print(f"  {a} - {b}: {d.mean():+.4f}, 95% low {min(lows):+.4f}..{max(lows):+.4f} over {seeds} seeds, "
              f"permutation p={p:.4f}, better/worse in {better:.1%}/{worse:.1%} of papers")
        print(f"      by question set: {sets_txt}")
    return res


def recall(verdicts: list[str]) -> float:
    g = [v for v in verdicts if v != "ungraded"]
    return (sum(v == "correct" for v in g) + 0.5 * sum(v == "partial" for v in g)) / max(len(g), 1)


def rate(verdicts: list[str], which: str) -> float:
    g = [v for v in verdicts if v != "ungraded"]
    return sum(v == which for v in g) / max(len(g), 1)


def main() -> None:
    out: dict = {"arms": {}}
    scores = {}
    for arm, (run, name) in ARMS.items():
        rows = read_jsonl(Path(f"runs/{run}/eval/{name}.scores.jsonl.gz"))
        if rows:
            scores[arm] = {r["id"]: r for r in rows}
    arms = list(scores)
    ids = sorted(set.intersection(*(set(scores[a]) for a in arms))) if arms else []
    out["arms"] = arms
    out["papers"] = len(ids)
    print(f"arms: {arms}; {len(ids)} test papers common to all of them\n")

    # ---- reference metrics -------------------------------------------------------------------
    table = {}
    for m in METRICS:
        table[m] = {a: st.fmean(float(scores[a][i].get(m) or 0) for i in ids) for a in arms}
    print("REFERENCE METRICS (against the teacher)")
    print(f"{'metric':16s}" + "".join(f"{a:>10s}" for a in arms))
    for m in METRICS:
        print(f"{m:16s}" + "".join(f"{table[m][a]:10.4f}" for a in arms))
    paired = {}
    for a, b in PAIRS:
        if a in scores and b in scores:
            for m in ("all_rougeL", "bertscore_mean", "keywords_f1", "length_ratio"):
                d = [float(scores[a][i].get(m) or 0) - float(scores[b][i].get(m) or 0) for i in ids]
                paired[f"{a} - {b} :: {m}"] = boot(d)
    print("\nPAIRED DIFFERENCES")
    for k, v in paired.items():
        print(f"  {k:42s} {fmt(v)}")
    out["reference"] = {"means": table, "paired": {k: list(v) for k, v in paired.items()}}

    # faithfulness (NLI) is scored on a subset of papers
    nli = {a: [float(scores[a][i]["faithfulness"]) for i in ids if scores[a][i].get("faithfulness") is not None] for a in arms}
    print("\nNLI FAITHFULNESS (a canary; the papers it was computed on):",
          {a: (round(st.fmean(v), 4), len(v)) for a, v in nli.items() if v})
    out["nli_faithfulness"] = {a: [round(st.fmean(v), 4), len(v)] for a, v in nli.items() if v}
    nli_pair = {}
    for a, b in PAIRS:
        if a in scores and b in scores:
            both = [i for i in ids if scores[a][i].get("faithfulness") is not None
                    and scores[b][i].get("faithfulness") is not None]
            if both:
                nli_pair[f"{a} - {b}"] = boot([float(scores[a][i]["faithfulness"]) - float(scores[b][i]["faithfulness"])
                                               for i in both])
    for k, v in nli_pair.items():
        print(f"  {k:42s} {fmt(v)}")
    out["nli_paired"] = {k: list(v) for k, v in nli_pair.items()}

    # ---- the fact test, over both question sets ----------------------------------------------
    grades: dict[str, dict[str, dict]] = {}
    for arm, label in FACT_LABEL.items():
        merged: dict[str, dict] = {}
        for d in FACT_DIRS:
            for r in read_jsonl(Path(f"state/{d}/grades-{label}.jsonl.gz")):
                merged[r["id"]] = r
        if merged:
            grades[arm] = merged
    if "C" in grades and len(grades) > 2:
        fids = sorted(set.intersection(*(set(g) for g in grades.values())))
        print(f"\nFACT TEST: {len(fids)} papers with questions graded for every arm "
              f"({', '.join(grades)})")
        print(f"{'arm':8s}{'recall':>9s}{'correct':>9s}{'wrong':>8s}{'notstated':>11s}")
        facts = {}
        for a, g in grades.items():
            rows = [g[i]["verdicts"] for i in fids]
            facts[a] = {"recall": st.fmean(recall(v) for v in rows), "correct": st.fmean(rate(v, "correct") for v in rows),
                        "wrong": st.fmean(rate(v, "incorrect") for v in rows),
                        "not_stated": st.fmean(rate(v, "not_stated") for v in rows)}
            print(f"{a:8s}{facts[a]['recall']:9.4f}{facts[a]['correct']:9.4f}{facts[a]['wrong']:8.4f}{facts[a]['not_stated']:11.4f}")
        fpair = {}
        for a, b in (*PAIRS, *TEACHER_PAIRS):
            if a in grades and b in grades:
                for name, fn in (("recall", recall), ("correct", lambda v: rate(v, "correct")), ("wrong", lambda v: rate(v, "incorrect"))):
                    d = [fn(grades[a][i]["verdicts"]) - fn(grades[b][i]["verdicts"]) for i in fids]
                    fpair[f"{a} - {b} :: {name}"] = boot(d)
        print("\nPAIRED DIFFERENCES (fact test)")
        for k, v in fpair.items():
            print(f"  {k:42s} {fmt(v)}")
        out["fact"] = {"papers": len(fids), "means": facts, "paired": {k: list(v) for k, v in fpair.items()}}
        out["fact"]["robustness"] = robustness(grades, fids)

    # ---- the judge ----------------------------------------------------------------------------
    judged = {}
    for arm, (run, name) in ARMS.items():
        rows = [r for r in read_jsonl(Path(f"runs/{run}/eval/{name}.judge.jsonl.gz")) if "judge_mean" in r]
        if rows:
            judged[arm] = {r["id"]: r for r in rows}
    if len(judged) > 1:
        jids = sorted(set.intersection(*(set(j) for j in judged.values())))
        print(f"\nJUDGE (qwen38-27b vs the teacher's summary, {len(jids)} papers)")
        out["judge"] = {"papers": len(jids), "delta": {a: st.fmean(j[i]["judge_delta"] for i in jids) for a, j in judged.items()}}
        for a, j in judged.items():
            print(f"  {a:6s} delta {out['judge']['delta'][a]:+.4f}  win {st.fmean(j[i]['judge_win'] for i in jids):.3f}"
                  f"  tie {st.fmean(j[i]['judge_tie'] for i in jids):.3f}")
        # each dimension of the candidate's own score (1-5), and paired differences between arms: the
        # judge's faithfulness is the third instrument when the fact test and NLI disagree
        dims = ["judge_faithfulness", "judge_coverage", "judge_specificity", "judge_fluency", "judge_mean"]
        out["judge"]["dims"] = {a: {d: st.fmean(float(j[i][d]) for i in jids) for d in dims} for a, j in judged.items()}
        print(f"  {'arm':6s}" + "".join(f"{d.removeprefix('judge_'):>14s}" for d in dims))
        for a in judged:
            print(f"  {a:6s}" + "".join(f"{out['judge']['dims'][a][d]:14.3f}" for d in dims))
        jpair = {}
        for a, b in PAIRS:
            if a in judged and b in judged:
                for d in dims:
                    jpair[f"{a} - {b} :: {d}"] = boot([float(judged[a][i][d]) - float(judged[b][i][d]) for i in jids])
        for k, v in jpair.items():
            print(f"  {k:42s} {fmt(v, 3)}")
        out["judge"]["paired"] = {k: list(v) for k, v in jpair.items()}
    Path(OUT).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=1, choices=(1, 2, 3, 4))
    use_round(ap.parse_args().round)
    main()
