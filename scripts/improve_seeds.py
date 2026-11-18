"""Round 3's seed verdict: `preflc` and `prefrd` over two seeds each, against Student C.

    python scripts/improve_seeds.py   # CPU, about a minute; run as a batch job. Writes improve_seeds_r3.json

Protocol (`paper_docs/11_reward_selected_distillation.md`, "Round 3", written before training): for
`preflc` and `prefrd`, each paper's score is the mean of the two seeds' scores, and round 2's four
criteria are applied to that per-paper average:

  1. fact recall above C's, the 95% paired interval excludes zero (the permutation p is reported next
     to it against the Bonferroni threshold of 0.01 for five comparisons);
  2. length ratio to the teacher within 0.90-1.10 (mean over papers);
  3. ROUGE-L and BERTScore no more than 0.01 below C's (point estimates);
  4. the share of statements the claim check (Qwen3.8-27B) finds unsupported or contradicted, minus C's:
     the interval includes zero or lies below it.

Each seed is also reported alone against C, and seed 2 minus seed 1 as the measure of seed noise. The
measures are read as `scripts/improve_compare.py` and `scisum claimcheck report` read them: fact grades
merged over both question sets, the last claim-check row per paper, a summary with no labelled statement
left out. A paper enters a measure only if C and both seeds have it.

Also reported, descriptively and outside the protocol: ``prefcc`` (``preflc``'s pairing rule plus the
claim penalty) minus each ``preflc`` seed and minus their per-paper mean, next to the seed-2-minus-seed-1
difference, so the claim penalty's effect can be read against seed noise. Intervals: 2,000 paired bootstrap
resamples over papers (seed 0), as in `improve_compare.py`; permutation test: 10,000 sign flips.
"""

from __future__ import annotations

import gzip
import json
import random
import statistics as st
from pathlib import Path

C_RUN = ("q35", "test-ex20000-n15000")
SEEDS = {"preflc": (("q35preflc", "test-ex1163-n1246"), ("q35preflcs2", "test-ex1163-n1246"), "preflcs2"),
         "prefrd": (("q35prefrd", "test-ex1073-n1246"), ("q35prefrds2", "test-ex1073-n1246"), "prefrds2")}
PREFCC = ("q35prefcc", "test-ex1233-n1246")
FACT_DIRS = ("factcheck", "factcheck-b")
CLAIM_MODEL = "qwen38-27b"
BONFERRONI = 0.01
OUT = Path("paper_docs/data/improve_seeds_r3.json")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


def recall(verdicts: list[str]) -> float:
    g = [v for v in verdicts if v != "ungraded"]
    return (sum(v == "correct" for v in g) + 0.5 * sum(v == "partial" for v in g)) / max(len(g), 1)


def unsupported_share(r: dict) -> float | None:
    """Unsupported-or-contradicted share among labelled statements (1 - supported share); None if none labelled."""
    lab = [x["verdict"] for x in r["verdicts"] if x["verdict"] != "ungraded"]
    return None if not lab else 1 - lab.count("supported") / len(lab)


def boot(d: list[float], n: int = 2000, seed: int = 0) -> list[float]:
    if len(d) < 2:
        return [float("nan")] * 3
    rnd = random.Random(seed)
    ms = sorted(st.fmean(rnd.choices(d, k=len(d))) for _ in range(n))
    return [st.fmean(d), ms[int(0.025 * n)], ms[int(0.975 * n)]]


def perm_p(d: list[float], n: int = 10000, seed: int = 0) -> float:
    """Two-sided paired sign-flip test of mean(d) = 0 (as `improve_compare.robustness`)."""
    import numpy as np

    x = np.asarray(d, dtype=float)
    flips = np.random.default_rng(seed).choice([-1.0, 1.0], (n, len(x)))
    null = np.abs((flips * x).mean(axis=1))
    return float((1 + (null >= abs(x.mean())).sum()) / (n + 1))


def paired(base: dict[str, float], s1: dict[str, float], s2: dict[str, float]) -> dict:
    """Per paper: the two-seed mean minus C, each seed minus C, and seed 2 minus seed 1."""
    ids = sorted(set(base) & set(s1) & set(s2))
    avg = [(s1[i] + s2[i]) / 2 - base[i] for i in ids]
    return {"papers": len(ids), "ids": ids,
            "two_seed_minus_C": avg,
            "seed1_minus_C": [s1[i] - base[i] for i in ids],
            "seed2_minus_C": [s2[i] - base[i] for i in ids],
            "seed2_minus_seed1": [s2[i] - s1[i] for i in ids],
            "two_seed_mean": st.fmean((s1[i] + s2[i]) / 2 for i in ids) if ids else float("nan"),
            "seed_means": [st.fmean(s1[i] for i in ids), st.fmean(s2[i] for i in ids)] if ids else [],
            "C_mean": st.fmean(base[i] for i in ids) if ids else float("nan")}


def summarise(p: dict, with_p: bool = False) -> dict:
    out = {"papers": p["papers"], "C_mean": p["C_mean"], "two_seed_mean": p["two_seed_mean"],
           "seed_means": p["seed_means"]}
    for k in ("two_seed_minus_C", "seed1_minus_C", "seed2_minus_C", "seed2_minus_seed1"):
        out[k] = boot(p[k])
    if with_p:
        out["perm_p_two_seed"] = perm_p(p["two_seed_minus_C"])
    return out


def versus(x: dict[str, float], s1: dict[str, float], s2: dict[str, float]) -> dict:
    """Per paper: an arm minus each seed of another arm and minus their mean (descriptive, not a criterion)."""
    ids = sorted(set(x) & set(s1) & set(s2))
    return {"papers": len(ids),
            "minus_two_seed": boot([x[i] - (s1[i] + s2[i]) / 2 for i in ids]),
            "minus_seed1": boot([x[i] - s1[i] for i in ids]),
            "minus_seed2": boot([x[i] - s2[i] for i in ids]),
            "seed2_minus_seed1": boot([s2[i] - s1[i] for i in ids])}


def criteria(m: dict) -> dict:
    """Round 2's four criteria on the two-seed average (``m``: the summaries of each measure)."""
    r, ln, rl, bs, uc = (m.get(k) for k in ("recall", "length_ratio", "all_rougeL", "bertscore_mean", "unsupported_share"))
    c = {
        "1_recall": None if r is None else bool(r["two_seed_minus_C"][1] > 0),
        "2_length": None if ln is None else bool(0.90 <= ln["two_seed_mean"] <= 1.10),
        "3_reference": None if rl is None or bs is None else bool(rl["two_seed_minus_C"][0] >= -0.01
                                                                   and bs["two_seed_minus_C"][0] >= -0.01),
        "4_claims": None if uc is None else bool(uc["two_seed_minus_C"][1] <= 0),
    }
    c["all_four"] = None if None in c.values() else all(c.values())
    if r is not None and "perm_p_two_seed" in r:
        c["recall_p_below_bonferroni"] = bool(r["perm_p_two_seed"] < BONFERRONI)
    return c


def load_scores(run: str, name: str) -> dict[str, dict]:
    return {r["id"]: r for r in read_jsonl(Path(f"runs/{run}/eval/{name}.scores.jsonl.gz"))}


def load_recall(label: str) -> dict[str, float]:
    merged: dict[str, list[str]] = {}
    for d in FACT_DIRS:
        for r in read_jsonl(Path(f"state/{d}/grades-{label}.jsonl.gz")):
            merged[r["id"]] = r["verdicts"]
    return {i: recall(v) for i, v in merged.items()}


def load_claims(label: str) -> dict[str, float]:
    by = {r["id"]: r for r in read_jsonl(Path(f"state/claimcheck/{label}.{CLAIM_MODEL}.jsonl.gz"))
          if r.get("valid") and r["claims"]}
    out = {i: unsupported_share(r) for i, r in by.items()}
    return {i: v for i, v in out.items() if v is not None}


def prefcc_vs_preflc(c_scores: dict[str, dict]) -> dict:
    """``prefcc`` (``preflc``'s rule plus the claim penalty) against both ``preflc`` seeds; {} if a run is missing."""
    x = load_scores(*PREFCC)
    (run1, run2, label2) = SEEDS["preflc"]
    s1, s2 = load_scores(*run1), load_scores(*run2)
    if not (x and s1 and s2):
        return {}
    out = {}
    for k in ("length_ratio", "all_rougeL", "bertscore_mean"):
        col = [{i: float(r.get(k) or 0) for i, r in sc.items()} for sc in (x, s1, s2)]
        out[k] = versus(*col)
    for k, load in (("recall", load_recall), ("unsupported_share", load_claims)):
        col = [load("C_prefcc"), load("C_preflc"), load(f"C_{label2}")]
        if all(col):
            out[k] = versus(*col)
    return out


def main() -> None:
    c_scores = load_scores(*C_RUN)
    res: dict = {"protocol": "paper_docs/11 Round 3; per-paper mean of two seeds, round 2's four criteria",
                 "claim_verifier": CLAIM_MODEL, "arms": {}}
    for arm, (run1, run2, label2) in SEEDS.items():
        s1, s2 = load_scores(*run1), load_scores(*run2)
        if not s1 or not s2:
            print(f"{arm}: seed run missing ({run1[0] if not s1 else run2[0]}); skipped")
            continue
        m: dict = {}
        for k in ("length_ratio", "all_rougeL", "bertscore_mean", "out_tokens"):
            col = {name: {i: float(r.get(k) or 0) for i, r in sc.items()} for name, sc in (("C", c_scores), ("1", s1), ("2", s2))}
            col["C"] = {i: v for i, v in col["C"].items() if i in s1}   # C was scored on all 15,000 test papers
            m[k] = summarise(paired(col["C"], col["1"], col["2"]))
        rc, r1, r2 = load_recall("C"), load_recall(f"C_{arm}"), load_recall(f"C_{label2}")
        if r1 and r2:
            m["recall"] = summarise(paired(rc, r1, r2), with_p=True)
        cc, c1, c2 = load_claims("C"), load_claims(f"C_{arm}"), load_claims(f"C_{label2}")
        if c1 and c2:
            m["unsupported_share"] = summarise(paired(cc, c1, c2))
        res["arms"][arm] = {"seeds": [run1[0], run2[0]], "measures": m, "criteria": criteria(m)}

        print(f"\n{arm}: seeds {run1[0]} and {run2[0]}")
        print(f"  {'measure':18s}{'n':>6s}  {'two-seed mean - C':>32s}  {'seed 1 - C':>10s}  {'seed 2 - C':>10s}  "
              f"{'seed 2 - seed 1':>32s}")
        for k, v in m.items():
            t, d = v["two_seed_minus_C"], v["seed2_minus_seed1"]
            print(f"  {k:18s}{v['papers']:6d}  {t[0]:+.4f} [{t[1]:+.4f}, {t[2]:+.4f}]  {v['seed1_minus_C'][0]:+10.4f}  "
                  f"{v['seed2_minus_C'][0]:+10.4f}  {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]")
        if "recall" in m:
            print(f"  recall permutation p (two-seed) = {m['recall']['perm_p_two_seed']:.4f} (Bonferroni {BONFERRONI})")
        print(f"  length ratio (two-seed mean) {m['length_ratio']['two_seed_mean']:.3f}")
        print(f"  criteria: {res['arms'][arm]['criteria']}")
    res["prefcc_vs_preflc_seeds"] = vs = prefcc_vs_preflc(c_scores)
    if vs:
        print("\nprefcc against preflc's two seeds (descriptive: preflc's pairing rule plus the claim penalty; not a criterion)")
        print(f"  {'measure':18s}{'n':>6s}  {'prefcc - two-seed mean':>32s}  {'- seed 1':>9s}  {'- seed 2':>9s}  "
              f"{'seed 2 - seed 1':>32s}")
        for k, v in vs.items():
            t, d = v["minus_two_seed"], v["seed2_minus_seed1"]
            print(f"  {k:18s}{v['papers']:6d}  {t[0]:+.4f} [{t[1]:+.4f}, {t[2]:+.4f}]  {v['minus_seed1'][0]:+9.4f}  "
                  f"{v['minus_seed2'][0]:+9.4f}  {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]")
    OUT.write_text(json.dumps(res, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
