"""How much of an arm's fact-recall gain is length? (CPU, seconds; run as a batch job)

    python scripts/improve_length.py
    python scripts/improve_length.py --round 2   # with the round-2 arms; writes improve_length_r2.json
    python scripts/improve_length.py --round 3   # round 3's arms next to preflc/prefrd; improve_length_r3.json

Written on 2026-11-05 after the preference arm came out +0.071 recall over Student C with outputs 27%
longer, lower NLI faithfulness and lower ROUGE: the signature the protocol named in advance. A
longer summary can answer more questions without being a better one, so this separates the two ways
the fact test allows, with no new model calls:

  1. Precision among answered questions. Recall rises either because more questions are answered
     or because answers are more often right. The wrong and correct rates per *answered* question
     (not "not stated") say which, pooled over all questions of the 899 fact-test papers.
  2. Recall against length, paired within a paper. Per paper, the arm's recall minus C's is
     regressed on the log ratio of their output tokens. The intercept is the gain at equal length,
     under a linear model; the slope is what length buys. Bootstrapped over papers.
  3. The same difference by how much longer the arm wrote, and the papers where it was at most
     10% longer than C. These subsets are chosen on an outcome of training, so they describe, they
     do not identify.

Writes paper_docs/data/improve_length.json.
"""

from __future__ import annotations

import gzip
import json
import math
from pathlib import Path

import numpy as np

ARMS = {"C": ("q35", "test-ex20000-n15000"), "ctl": ("q35ctl", "test-ex1520-n1246"),
        "rft": ("q35rft", "test-ex1520-n1246"), "pref": ("q35pref", "test-ex1073-n1246")}
FACT_LABEL = {"teacher": "teacher", "C": "C", "ctl": "C_ctl", "rft": "C_rft", "pref": "C_pref"}
FACT_DIRS = ("factcheck", "factcheck-b")
BINS = [(0.0, 1.1), (1.1, 1.3), (1.3, 1.6), (1.6, float("inf"))]
REG_ARMS = ("ctl", "rft", "pref")
BIN_ARMS = ("pref",)
OUT = "paper_docs/data/improve_length.json"


def use_round(r: int) -> None:
    global REG_ARMS, BIN_ARMS, OUT
    if r == 2:
        ARMS.update({"preflc": ("q35preflc", "test-ex1163-n1246"), "prefrd": ("q35prefrd", "test-ex1073-n1246"),
                     "preflcrd": ("q35preflcrd", "test-ex1163-n1246")})
        FACT_LABEL.update({"preflc": "C_preflc", "prefrd": "C_prefrd", "preflcrd": "C_preflcrd"})
        REG_ARMS = ("pref", "preflc", "prefrd", "preflcrd")
        BIN_ARMS = ("pref", "preflc", "prefrd", "preflcrd")
        OUT = "paper_docs/data/improve_length_r2.json"
    if r == 3:   # round 3's arms next to the round-2 arms they vary; prefcc's checkpoint is named by its pairs
        ARMS.update({"preflc": ("q35preflc", "test-ex1163-n1246"), "prefrd": ("q35prefrd", "test-ex1073-n1246"),
                     "prefcc": ("q35prefcc", "test-ex*-n1246"), "preflcrd3": ("q35preflcrd3", "test-ex1163-n1246"),
                     "preflcs2": ("q35preflcs2", "test-ex1163-n1246"), "prefrds2": ("q35prefrds2", "test-ex1073-n1246")})
        FACT_LABEL.update({a: f"C_{a}" for a in ("preflc", "prefrd", "prefcc", "preflcrd3", "preflcs2", "prefrds2")})
        REG_ARMS = BIN_ARMS = ("preflc", "prefrd", "prefcc", "preflcrd3", "preflcs2", "prefrds2")
        OUT = "paper_docs/data/improve_length_r3.json"


def read_jsonl(path: Path) -> list[dict]:
    if "*" in path.name:   # a checkpoint named by the size of its training set, not known in advance
        found = sorted(path.parent.glob(path.name))
        path = found[-1] if found else path
    if not path.exists():
        return []
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


def recall(v: list[str]) -> float:
    g = [x for x in v if x != "ungraded"]
    return (sum(x == "correct" for x in g) + 0.5 * sum(x == "partial" for x in g)) / max(len(g), 1)


def ci(x: np.ndarray, n: int = 10000, seed: int = 0) -> list[float]:
    if len(x) < 2:
        return [float("nan")] * 3
    m = x[np.random.default_rng(seed).integers(0, len(x), (n, len(x)))].mean(axis=1)
    return [float(x.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]


def claim_shape() -> dict:
    """4. What NLI faithfulness is averaged over, per arm, on the 300 papers it was scored on.

    NLI scores each summary sentence (up to 40) against the 3-sentence source chunks it is closest
    to, and averages over sentences. More sentences, or longer ones that combine several facts, are
    harder to entail from one chunk, so the shape of the claims is reported next to the score.
    """
    import statistics as st
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from scisum.evaluate import parse_prediction
    from scisum.faithfulness import claims_of

    scores = {a: {r["id"]: r for r in read_jsonl(Path(f"runs/{run}/eval/{name}.scores.jsonl.gz"))}
              for a, (run, name) in ARMS.items()}
    ids = set.intersection(*({i for i, r in sc.items() if r.get("faithfulness") is not None} for sc in scores.values()))
    print(f"\n4. CLAIMS SCORED BY NLI ({len(ids)} papers)")
    res = {}
    for a, (run, name) in ARMS.items():
        preds = {r["id"]: r for r in read_jsonl(Path(f"runs/{run}/eval/{name}.pred.jsonl.gz")) if r["id"] in ids}
        n, words = [], []
        for i in sorted(ids):
            cl = claims_of(parse_prediction(preds[i]["text"])[0])
            n.append(len(cl))
            words += [len(c.split()) for c in cl]
        res[a] = {"papers": len(ids), "claims_per_paper": st.fmean(n), "capped_at_40": sum(x >= 40 for x in n),
                  "words_per_claim_mean": st.fmean(words), "words_per_claim_median": st.median(words),
                  "nli": st.fmean(float(scores[a][i]["faithfulness"]) for i in ids)}
        r = res[a]
        print(f"  {a:5s} claims/paper {r['claims_per_paper']:5.1f} (capped at 40: {r['capped_at_40']:3d})  words/claim "
              f"{r['words_per_claim_mean']:5.1f} (median {r['words_per_claim_median']:.0f})  NLI {r['nli']:.4f}")
    return res


def main() -> None:
    out: dict = {}
    grades = {}
    for arm, label in FACT_LABEL.items():
        merged = {}
        for d in FACT_DIRS:
            for r in read_jsonl(Path(f"state/{d}/grades-{label}.jsonl.gz")):
                merged[r["id"]] = r["verdicts"]
        grades[arm] = merged
    # An arm whose evaluation or fact grades are not written yet is skipped, as in improve_compare.py.
    for arm in [a for a, g in grades.items() if not g or (a in ARMS and not read_jsonl(
            Path("runs/{}/eval/{}.scores.jsonl.gz".format(*ARMS[a]))))]:
        print(f"skipping {arm}: no evaluation or no fact grades yet")
        del grades[arm]
        ARMS.pop(arm, None)
    global REG_ARMS, BIN_ARMS
    REG_ARMS = tuple(a for a in REG_ARMS if a in grades)
    BIN_ARMS = tuple(a for a in BIN_ARMS if a in grades)
    fids = sorted(set.intersection(*(set(g) for g in grades.values())))
    tokens = {}
    for arm, (run, name) in ARMS.items():
        tokens[arm] = {r["id"]: float(r.get("out_tokens") or 0) for r in
                       read_jsonl(Path(f"runs/{run}/eval/{name}.scores.jsonl.gz"))}
    print(f"{len(fids)} fact-test papers graded for every arm\n")

    # 1. precision among answered questions, pooled over questions
    print("1. PER ANSWERED QUESTION (pooled over all graded questions)")
    print(f"{'arm':8s}{'answered':>10s}{'correct|ans':>13s}{'partial|ans':>13s}{'wrong|ans':>11s}")
    prec = {}
    for arm, g in grades.items():
        v = [x for i in fids for x in g[i] if x != "ungraded"]
        ans = [x for x in v if x != "not_stated"]
        prec[arm] = {"questions": len(v), "answered": len(ans) / len(v),
                     "correct_given_answered": sum(x == "correct" for x in ans) / len(ans),
                     "partial_given_answered": sum(x == "partial" for x in ans) / len(ans),
                     "wrong_given_answered": sum(x == "incorrect" for x in ans) / len(ans)}
        p = prec[arm]
        print(f"{arm:8s}{p['answered']:10.4f}{p['correct_given_answered']:13.4f}"
              f"{p['partial_given_answered']:13.4f}{p['wrong_given_answered']:11.4f}")
    out["per_answered"] = prec

    # 2. paired regression of the recall difference on the log length ratio
    print("\n2. RECALL DIFFERENCE vs LOG LENGTH RATIO, paired per paper (arm - C)")
    reg = {}
    for arm in REG_ARMS:
        ids = [i for i in fids if tokens[arm].get(i) and tokens["C"].get(i)]
        dr = np.array([recall(grades[arm][i]) - recall(grades["C"][i]) for i in ids])
        dl = np.array([math.log(tokens[arm][i] / tokens["C"][i]) for i in ids])
        rng = np.random.default_rng(0)
        coefs = []
        for _ in range(10000):
            k = rng.integers(0, len(ids), len(ids))
            coefs.append(np.polyfit(dl[k], dr[k], 1))
        coefs = np.array(coefs)
        b, a = np.polyfit(dl, dr, 1)
        reg[arm] = {"papers": len(ids), "mean_log_ratio": float(dl.mean()),
                    "median_ratio": float(np.exp(np.median(dl))),
                    "slope": [float(b), *np.quantile(coefs[:, 0], [0.025, 0.975]).tolist()],
                    "intercept": [float(a), *np.quantile(coefs[:, 1], [0.025, 0.975]).tolist()],
                    "corr": float(np.corrcoef(dl, dr)[0, 1])}
        r = reg[arm]
        print(f"  {arm:5s} n={r['papers']}  median length ratio {r['median_ratio']:.3f}  "
              f"slope {r['slope'][0]:+.4f} [{r['slope'][1]:+.4f}, {r['slope'][2]:+.4f}] per unit log ratio  "
              f"intercept (equal length) {r['intercept'][0]:+.4f} [{r['intercept'][1]:+.4f}, {r['intercept'][2]:+.4f}]  "
              f"r={r['corr']:+.3f}")
    out["regression"] = reg

    # 3. by length bin (descriptive)
    bins = {}
    for arm in BIN_ARMS:
        print(f"\n3. {arm} - C RECALL BY HOW MUCH LONGER {arm} WROTE (descriptive: bins are chosen on an outcome)")
        ids = [i for i in fids if tokens[arm].get(i) and tokens["C"].get(i)]
        ratio = {i: tokens[arm][i] / tokens["C"][i] for i in ids}
        for lo, hi in BINS:
            sub = [i for i in ids if lo <= ratio[i] < hi]
            d = np.array([recall(grades[arm][i]) - recall(grades["C"][i]) for i in sub])
            dt = np.array([recall(grades[arm][i]) - recall(grades["teacher"][i]) for i in sub])
            key = f"{lo:.1f}-{hi:.1f}" if hi != float("inf") else f">={lo:.1f}"
            name = key if arm == "pref" and len(BIN_ARMS) == 1 else f"{arm} {key}"   # round 1's keys unchanged
            bins[name] = {"papers": len(sub), "pref_minus_C": ci(d), "pref_minus_teacher": ci(dt)}
            print(f"  ratio {key:9s} n={len(sub):4d}  {arm} - C {bins[name]['pref_minus_C'][0]:+.4f} "
                  f"[{bins[name]['pref_minus_C'][1]:+.4f}, {bins[name]['pref_minus_C'][2]:+.4f}]  "
                  f"{arm} - teacher {bins[name]['pref_minus_teacher'][0]:+.4f} "
                  f"[{bins[name]['pref_minus_teacher'][1]:+.4f}, {bins[name]['pref_minus_teacher'][2]:+.4f}]")
    out["bins"] = bins
    for arm in BIN_ARMS:
        d_all = np.array([recall(grades[arm][i]) - recall(grades["teacher"][i]) for i in fids])
        key = "pref_minus_teacher" if arm == "pref" else f"{arm}_minus_teacher"
        out[key] = ci(d_all)
        print(f"\n  {arm} - teacher, all {len(fids)} papers: {out[key][0]:+.4f} [{out[key][1]:+.4f}, {out[key][2]:+.4f}]")
    out["claims"] = claim_shape()
    Path(OUT).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, default=1, choices=(1, 2, 3))
    use_round(ap.parse_args().round)
    main()
