"""One results table for every system: quality, the four criteria against its own SFT model, training cost,
and serving cost. The input of `pareto.py` and of the decision report.

    python scripts/analysis/collect_results.py --out-dir reports/analysis              # the students in configs/
    python scripts/analysis/collect_results.py --systems my_systems.json --out-dir ...   # an explicit list

Writes ``results.json`` (schema ``scisum.results/1``), ``results.csv`` (one flat row per system) and
``results.md`` (tables to paste into a report). Everything is read from the artifacts the pipeline writes;
nothing is recomputed from summaries:

| measure | file | per paper |
|---|---|---|
| fact recall, strict recall | ``state/<fact dir>/grades-<label>.jsonl.gz`` (each question set's dir) | correct + 0.5 partial (strict: correct only) over graded questions |
| unsupported share | ``state/<claim dir>/<label>.<claim model>.jsonl.gz`` | 1 - supported share of labelled statements |
| length ratio, ROUGE-L, BERTScore, NLI, JSON validity, output tokens | ``runs/<run>/eval/<eval>.scores.jsonl.gz`` | as scored |
| judge faithfulness, judge delta to the teacher | ``runs/<run>/eval/<eval>.judge.jsonl.gz`` | as judged |
| training cost | ``runs/<run>/metrics.jsonl`` (via ``train_resources.py``) | |
| serving cost, latency, memory | ``runs/<run>/bench/*.json`` (``bench_inference.py``); else the evaluation run's ``.gen.json`` (flagged) | |

A system with two runs (the two pair orders of a preference arm) is judged on the per-paper mean of the
two, over the papers both have, as in the C -> prefcc verdicts. Comparisons are paired over the papers both
sides have; intervals are 95% percentile bootstraps (2,000 resamples, seed 0); recall's p-value is a paired
sign-flip permutation test. The four criteria (against the system's ``reference``, normally its own SFT
model): (1) recall interval above zero and p below alpha / k (Bonferroni over the k systems that have a
reference); (2) length ratio 0.90-1.10; (3) ROUGE-L and BERTScore no more than 0.010 below; (4) unsupported
share interval reaching zero or below. Anything missing is listed under ``missing`` and never guessed.

Systems file (JSON)::

    {"teacher": {"fact_label": "teacher", "claim_label": "teacher"},
     "systems": [{"name": "k2h-7b SFT", "student": "k2h-7b", "stage": "sft", "params": 9.0e9,
                  "runs": [{"run": "k2h-7b-sft50k", "eval": "test-ex*-n1246",
                            "fact_label": "k2h-7b-sft50k", "claim_label": "k2h-7b-sft50k"}],
                  "train_runs": ["k2h-7b-sft50k"]},
                 {"name": "k2h-7b prefcc", "student": "k2h-7b", "stage": "prefcc", "reference": "k2h-7b SFT",
                  "runs": [{"run": "k2h-7b-prefcc", ...}, {"run": "k2h-7b-prefcc-s2", ...}],
                  "train_runs": ["k2h-7b-sft50k", "k2h-7b-prefcc"], "verification_runs": ["k2h-7b-prefcc-s2"]}]}

Optional per system: ``"anchor": true`` (drawn in grey), ``"serving": {...}`` (a manual serving
measurement: ``papers_per_s``, ``gpus``, ``gpu_type``, ``source``), ``"bench": "path/glob.json"``.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_resources  # noqa: E402
from analysis_lib import (  # noqa: E402
    boot_ci,
    discover_students,
    fmt,
    fmt_ci,
    last_match,
    md_table,
    perm_p,
    read_json,
    read_jsonl,
    run_names,
)

SCHEMA = "scisum.results/1"
SCORE_KEYS = {"length_ratio": "length_ratio", "rougeL": "all_rougeL", "bertscore": "bertscore_mean",
              "nli": "faithfulness", "json_ok": "json_ok", "out_tokens": "out_tokens"}
JUDGE_KEYS = {"judge_faithfulness": "judge_faithfulness", "judge_delta": "judge_delta"}
MEASURES = ("recall", "strict_recall", "unsupported_share", "length_ratio", "rougeL", "bertscore", "nli",
            "judge_faithfulness", "judge_delta", "json_ok", "out_tokens")
VS_TEACHER = ("recall", "strict_recall", "unsupported_share")


# ---- per-paper loaders --------------------------------------------------------------------------


def recall_of(verdicts: list[str], strict: bool = False) -> float | None:
    g = [v for v in verdicts if v != "ungraded"]
    if not g:
        return None
    ok = sum(v == "correct" for v in g) + (0 if strict else 0.5 * sum(v == "partial" for v in g))
    return ok / len(g)


def load_fact(state: Path, fact_dirs: list[str], label: str) -> tuple[dict, dict]:
    """Recall and strict recall per paper, merged over the question sets (disjoint papers; a later set
    wins if a paper were in two)."""
    rec: dict[str, float] = {}
    strict: dict[str, float] = {}
    for d in fact_dirs:
        for r in read_jsonl(state / d / f"grades-{label}.jsonl.gz"):
            v = r.get("verdicts") or []
            a, b = recall_of(v), recall_of(v, strict=True)
            if a is not None:
                rec[r["id"]], strict[r["id"]] = a, b
    return rec, strict


def load_claims(state: Path, claim_dir: str, label: str, model: str) -> dict[str, float]:
    """Unsupported-or-contradicted share per paper; the last valid row per paper wins."""
    by = {r["id"]: r for r in read_jsonl(state / claim_dir / f"{label}.{model}.jsonl.gz")
          if r.get("valid") and r.get("claims")}
    out = {}
    for i, r in by.items():
        lab = [x.get("verdict") for x in r.get("verdicts") or [] if x.get("verdict") != "ungraded"]
        if lab:
            out[i] = 1 - lab.count("supported") / len(lab)
    return out


def load_run(runs: Path, state: Path, spec: dict, fact_dirs: list[str], claim_dir: str, claim_model: str) -> tuple[dict, list[str], dict]:
    """All per-paper measures of one run; returns (measures, missing, eval meta)."""
    m: dict[str, dict[str, float]] = {}
    missing: list[str] = []
    meta: dict = {}
    run, label = spec["run"], spec.get("fact_label") or spec["run"]
    ev = runs / run / "eval"
    pattern = spec.get("eval", "test-ex*-n*")
    scores_p = last_match(ev, f"{pattern}.scores.jsonl.gz")
    if scores_p is None:
        missing.append(f"{run}: no scores ({pattern}.scores.jsonl.gz)")
    else:
        name = scores_p.name[: -len(".scores.jsonl.gz")]
        meta = {"eval": name, "gen": read_json(ev / f"{name}.gen.json")}
        rows = read_jsonl(scores_p)
        for k, src in SCORE_KEYS.items():
            vals = {r["id"]: float(r[src]) for r in rows if r.get(src) is not None}
            if vals:
                m[k] = vals
        jrows = [r for r in read_jsonl(ev / f"{name}.judge.jsonl.gz") if "judge_faithfulness" in r]
        if jrows:
            for k, src in JUDGE_KEYS.items():
                m[k] = {r["id"]: float(r[src]) for r in jrows if r.get(src) is not None}
        else:
            missing.append(f"{run}: no judge file")
    rec, strict = load_fact(state, fact_dirs, label)
    if rec:
        m["recall"], m["strict_recall"] = rec, strict
    else:
        missing.append(f"{run}: no fact grades for label {label!r}")
    cl = load_claims(state, claim_dir, spec.get("claim_label") or label, claim_model)
    if cl:
        m["unsupported_share"] = cl
    else:
        missing.append(f"{run}: no claim check for label {spec.get('claim_label') or label!r} ({claim_model})")
    return m, missing, meta


def combine(runs_m: list[dict]) -> dict[str, dict[str, float]]:
    """Per-paper mean over the runs of a system, on the papers every run has."""
    if len(runs_m) == 1:
        return runs_m[0]
    out = {}
    for k in MEASURES:
        if all(k in r for r in runs_m):
            ids = set.intersection(*(set(r[k]) for r in runs_m))
            out[k] = {i: sum(r[k][i] for r in runs_m) / len(runs_m) for i in ids}
    return out


def paired(a: dict[str, float], b: dict[str, float]) -> list[float]:
    return [a[i] - b[i] for i in sorted(set(a) & set(b))]


# ---- systems ------------------------------------------------------------------------------------


def auto_systems(students_dir: Path, eval_glob: str) -> dict:
    """The default system list: every student profile, its SFT model and its prefcc arm (two seeds)."""
    systems = []
    for st_ in discover_students(students_dir):
        tag, rn = st_["tag"], run_names(st_["tag"])
        role = {"anchor": True} if st_["anchor"] else {}
        sft = f"{tag} SFT"
        systems.append({"name": sft, "student": tag, "stage": "sft", "params": st_["params"], **role,
                        "runs": [{"run": rn["sft"], "eval": eval_glob}], "train_runs": [rn["sft"]]})
        systems.append({"name": f"{tag} prefcc", "student": tag, "stage": "prefcc", "params": st_["params"],
                        "reference": sft, **role,
                        "runs": [{"run": rn["prefcc"], "eval": eval_glob}, {"run": rn["prefcc_s2"], "eval": eval_glob}],
                        "train_runs": [rn["sft"], rn["prefcc"]], "verification_runs": [rn["prefcc_s2"]]})
    return {"teacher": {"fact_label": "teacher", "claim_label": "teacher"}, "systems": systems}


def pick_bench(runs: Path, sys_spec: dict, prefer: tuple[str, ...] = ("lora", "merged", "base")) -> tuple[dict | None, str | None]:
    """The benchmark JSON of a system: an explicit ``bench`` glob, else ``runs/<first run>/bench/*.json``,
    preferring the evaluated serving mode (LoRA adapter) and then the newest."""
    if sys_spec.get("bench"):
        cands = [Path(x) for x in sorted(glob.glob(sys_spec["bench"]))]
    else:
        cands = sorted((runs / sys_spec["runs"][0]["run"] / "bench").glob("*.json"))
    docs = [(p, read_json(p)) for p in cands]
    docs = [(p, d) for p, d in docs if d and str(d.get("schema", "")).startswith("scisum.bench_inference/")]
    if not docs:
        return None, None
    rank = {m: i for i, m in enumerate(prefer)}
    docs.sort(key=lambda pd: (rank.get(pd[1].get("adapter_mode"), 9), -(pd[0].stat().st_mtime)))
    return docs[0][1], str(docs[0][0])


def serving_of(runs: Path, sys_spec: dict, eval_meta: dict) -> dict:
    """Serving cost fields, with their source: a controlled benchmark, a manual entry, or the eval run."""
    bench, path = pick_bench(runs, sys_spec)
    if bench:
        t, lat, mem = bench.get("throughput") or {}, bench.get("latency") or {}, bench.get("memory") or {}
        hw = (bench.get("hardware") or {}).get("gpus") or [{}]
        return {"cost_source": "benchmark", "bench_file": path, "gpu_type": hw[0].get("name"),
                "gpus": t.get("gpus"), "gpu_s_per_paper": t.get("gpu_seconds_per_paper"),
                "papers_per_hour": t.get("papers_per_hour"), "output_tokens_per_s": t.get("output_tokens_per_s"),
                "ttft_p50_s": (lat.get("ttft_s") or {}).get("p50"), "e2e_p50_s": (lat.get("e2e_s") or {}).get("p50"),
                "e2e_p95_s": (lat.get("e2e_s") or {}).get("p95"), "tpot_p50_s": (lat.get("tpot_s") or {}).get("p50"),
                "weights_gb": mem.get("weights_gb"), "kv_gb_per_paper_p95": mem.get("kv_gb_per_paper_p95"),
                "load_s": bench.get("load_s"), "adapter_mode": bench.get("adapter_mode")}
    manual = sys_spec.get("serving")
    if manual:
        pps = manual.get("papers_per_s")
        gpus = manual.get("gpus", 1)
        return {"cost_source": "manual: " + manual.get("source", ""), "gpu_type": manual.get("gpu_type"),
                "gpus": gpus, "gpu_s_per_paper": gpus / pps if pps else None,
                "papers_per_hour": 3600 * pps if pps else None,
                "output_tokens_per_s": manual.get("output_tokens_per_s")}
    gen = (eval_meta or {}).get("gen") or {}
    if gen.get("papers_per_s"):
        gpus = gen.get("tensor_parallel") or 1
        return {"cost_source": "evaluation run (not a controlled benchmark)", "gpu_type": None, "gpus": gpus,
                "gpu_s_per_paper": gpus / gen["papers_per_s"], "papers_per_hour": 3600 * gen["papers_per_s"],
                "output_tokens_per_s": gen.get("output_tokens_per_s"), "load_s": gen.get("load_s")}
    return {"cost_source": None}


def load_system(spec: dict, runs: Path, state: Path, fact_dirs: list[str], claim_dir: str,
                claim_model: str) -> tuple[dict, list[str], list[dict]]:
    per_run, missing, metas = [], [], []
    for r in spec["runs"]:
        m, miss, meta = load_run(runs, state, r, fact_dirs, claim_dir, claim_model)
        per_run.append(m)
        missing += miss
        metas.append(meta)
    return combine(per_run), missing, metas


def summarise_system(spec: dict, pp: dict, missing: list[str], metas: list[dict], runs: Path, teacher: dict,
                     gpu_type: str | None) -> dict:
    missing = list(missing)
    measures = {k: {"papers": len(pp[k]), "mean_ci": boot_ci([pp[k][i] for i in sorted(pp[k])])}
                for k in MEASURES if k in pp}
    vs_t = {}
    for k in VS_TEACHER:
        if k in pp and k in teacher:
            d = paired(pp[k], teacher[k])
            vs_t[k] = {"papers": len(d), "diff_ci": boot_ci(d)}
    trains = [train_resources.summarise_run(runs / r, gpu_type) for r in spec.get("train_runs", [])]
    verif = [train_resources.summarise_run(runs / r, gpu_type) for r in spec.get("verification_runs", [])]
    for t in trains + verif:
        if "kind" not in t:
            missing.append(f"{t['run']}: no training metrics")
    have = [t for t in trains if "kind" in t]
    train = {"runs": [t["run"] for t in trains],
             "gpu_hours": round(sum(t["gpu_hours"] for t in have), 3) if have else None,
             "wall_hours": round(sum(t["wall_hours"] for t in have), 3) if have else None,
             "peak_vram_gb": max((t["peak_vram_gb"] or 0) for t in have) if have else None,
             "gpu_types": sorted({str(t["gpu_type"]) for t in have if t.get("gpu_type")}),
             "verification_gpu_hours": round(sum(t.get("gpu_hours", 0) for t in verif if "kind" in t), 3) if verif else None,
             "detail": trains + verif}
    serving = serving_of(runs, spec, metas[0] if metas else {})
    if not serving.get("cost_source"):
        missing.append("no serving measurement (run bench_inference.py)")
    out = {"name": spec["name"], "student": spec.get("student", spec["name"]), "stage": spec.get("stage", ""),
           "anchor": bool(spec.get("anchor")), "params": spec.get("params"), "reference": spec.get("reference"),
           "runs": [{"run": r["run"], "eval": (metas[i] or {}).get("eval")} for i, r in enumerate(spec["runs"])],
           "measures": measures, "vs_teacher": vs_t, "train": train, "serving": serving, "missing": missing}
    if out["params"] is None:
        out["params"] = next((t.get("params_total") for t in trains if t.get("params_total")), None)
    return out


def restrict_to_common(pps: dict[str, dict], teacher: dict) -> dict[str, dict]:
    """Every measure restricted to the papers that all systems having it share (and the teacher, for the
    measures it has), so that absolute values are comparable across systems."""
    out = {name: dict(pp) for name, pp in pps.items()}
    for k in MEASURES:
        sets = [set(pp[k]) for pp in pps.values() if k in pp] + ([set(teacher[k])] if k in teacher else [])
        if not sets:
            continue
        common = set.intersection(*sets)
        for pp in out.values():
            if k in pp:
                pp[k] = {i: v for i, v in pp[k].items() if i in common}
        if k in teacher:
            teacher[k] = {i: v for i, v in teacher[k].items() if i in common}
    return out


def criteria(vs: dict, sys_pp: dict, alpha_k: float) -> dict:
    """The four criteria of the C -> prefcc protocol, from paired differences against the reference."""
    def get(k):
        return vs.get(k)
    r, rl, bs, uc = get("recall"), get("rougeL"), get("bertscore"), get("unsupported_share")
    ln = sys_pp.get("length_ratio_paired_mean")
    c = {"1_recall": None if r is None else bool(r["diff_ci"][1] > 0 and r["p"] < alpha_k),
         "2_length": None if ln is None else bool(0.90 <= ln <= 1.10),
         "3_reference": None if rl is None or bs is None else bool(rl["diff_ci"][0] >= -0.01 and bs["diff_ci"][0] >= -0.01),
         "4_claims": None if uc is None else bool(uc["diff_ci"][1] <= 0)}
    c["all_four"] = None if None in c.values() else all(c.values())
    c["alpha_per_comparison"] = alpha_k
    return c


def flat_row(s: dict) -> dict:
    m, sv, tr = s["measures"], s["serving"], s["train"]

    def mean(k):
        return m[k]["mean_ci"][0] if k in m else None

    def lo(k):
        return m[k]["mean_ci"][1] if k in m else None

    def hi(k):
        return m[k]["mean_ci"][2] if k in m else None

    vr = (s.get("vs_reference") or {}).get("measures", {})
    crit = (s.get("vs_reference") or {}).get("criteria", {})
    return {"name": s["name"], "student": s["student"], "stage": s["stage"], "anchor": s["anchor"],
            "params_b": round(s["params"] / 1e9, 3) if s.get("params") else None,
            "recall": mean("recall"), "recall_lo": lo("recall"), "recall_hi": hi("recall"),
            "recall_papers": m.get("recall", {}).get("papers"),
            "recall_vs_teacher": (s["vs_teacher"].get("recall") or {}).get("diff_ci", [None])[0],
            "strict_recall": mean("strict_recall"), "unsupported_share": mean("unsupported_share"),
            "unsupported_lo": lo("unsupported_share"), "unsupported_hi": hi("unsupported_share"),
            "length_ratio": mean("length_ratio"), "rougeL": mean("rougeL"), "bertscore": mean("bertscore"),
            "nli": mean("nli"), "judge_faithfulness": mean("judge_faithfulness"), "json_ok": mean("json_ok"),
            "recall_vs_reference": (vr.get("recall") or {}).get("diff_ci", [None])[0],
            "recall_p_vs_reference": (vr.get("recall") or {}).get("p"),
            "criteria_all_four": crit.get("all_four"),
            "train_gpu_hours": tr.get("gpu_hours"), "train_peak_vram_gb": tr.get("peak_vram_gb"),
            "gpu_s_per_paper": sv.get("gpu_s_per_paper"), "papers_per_hour": sv.get("papers_per_hour"),
            "ttft_p50_s": sv.get("ttft_p50_s"), "e2e_p50_s": sv.get("e2e_p50_s"), "e2e_p95_s": sv.get("e2e_p95_s"),
            "serve_weights_gb": sv.get("weights_gb"), "kv_gb_per_paper_p95": sv.get("kv_gb_per_paper_p95"),
            "serve_gpu_type": sv.get("gpu_type"), "cost_source": sv.get("cost_source"),
            "missing": len(s["missing"])}


def collect(spec: dict, runs: Path, state: Path, fact_dirs: list[str], claim_dir: str, claim_model: str,
            alpha: float = 0.05, gpu_type: str | None = None, common_papers: bool = False) -> dict:
    t = spec.get("teacher") or {"fact_label": "teacher", "claim_label": "teacher"}
    teacher: dict[str, dict[str, float]] = {}
    rec, strict = load_fact(state, fact_dirs, t["fact_label"])
    if rec:
        teacher["recall"], teacher["strict_recall"] = rec, strict
    cl = load_claims(state, claim_dir, t["claim_label"], claim_model)
    if cl:
        teacher["unsupported_share"] = cl

    loaded = {s_["name"]: load_system(s_, runs, state, fact_dirs, claim_dir, claim_model) for s_ in spec["systems"]}
    pps = {name: v[0] for name, v in loaded.items()}
    if common_papers:
        pps = restrict_to_common(pps, teacher)
    teacher_summary = {k: {"papers": len(v), "mean_ci": boot_ci([v[i] for i in sorted(v)])} for k, v in teacher.items()}
    systems = [summarise_system(s_, pps[s_["name"]], loaded[s_["name"]][1], loaded[s_["name"]][2], runs, teacher,
                                gpu_type) for s_ in spec["systems"]]
    k = max(1, sum(1 for s_ in systems if s_.get("reference")))
    for s_ in systems:
        ref = s_.get("reference")
        if not ref:
            continue
        if ref not in pps:
            s_["missing"].append(f"reference {ref!r} is not in the system list")
            continue
        a, b = pps[s_["name"]], pps[ref]
        vs = {}
        for key in MEASURES:
            if key in a and key in b:
                d = paired(a[key], b[key])
                vs[key] = {"papers": len(d), "diff_ci": boot_ci(d)}
                if key == "recall":
                    vs[key]["p"] = perm_p(d)
        lr = a.get("length_ratio", {})
        common = sorted(set(lr) & set(b.get("length_ratio", {})))
        extra = {"length_ratio_paired_mean": (sum(lr[i] for i in common) / len(common)) if common else None}
        s_["vs_reference"] = {"reference": ref, "measures": vs, "criteria": criteria(vs, extra, alpha / k),
                              "length_ratio_paired_mean": extra["length_ratio_paired_mean"]}
    for s_ in systems:
        s_["flat"] = flat_row(s_)
    return {"schema": SCHEMA, "created": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "inputs": {"fact_dirs": fact_dirs, "claim_dir": claim_dir, "claim_model": claim_model, "alpha": alpha,
                       "bonferroni_k": k, "common_papers": common_papers},
            "teacher": teacher_summary, "systems": systems}


# ---- outputs ------------------------------------------------------------------------------------


def to_markdown(res: dict) -> str:
    out = []
    t = res["teacher"]
    tr = t.get("recall", {}).get("mean_ci")
    out.append("## Quality on the test papers\n")
    out.append(f"Teacher: fact recall {fmt_ci(tr, signed=False)} ({t.get('recall', {}).get('papers', 0)} papers), "
               f"unsupported share {fmt_ci(t.get('unsupported_share', {}).get('mean_ci'), signed=False)}.\n")
    rows = []
    for s in res["systems"]:
        m = s["measures"]

        def g(k, d=4, m=m):
            return fmt(m[k]["mean_ci"][0], d) if k in m else "–"
        rows.append([s["name"], fmt_ci(m.get("recall", {}).get("mean_ci"), signed=False),
                     fmt_ci((s["vs_teacher"].get("recall") or {}).get("diff_ci")), g("strict_recall"),
                     g("unsupported_share"), g("length_ratio", 3), g("rougeL"), g("bertscore"), g("nli"),
                     g("judge_faithfulness", 3), g("json_ok", 4)])
    out.append(md_table(["system", "fact recall [95% CI]", "recall − teacher", "strict recall", "unsupported share",
                         "length ratio", "ROUGE-L", "BERTScore", "NLI", "judge faithfulness (1–5)", "JSON valid"], rows))
    ref_rows = []
    for s in res["systems"]:
        v = s.get("vs_reference")
        if not v:
            continue
        mm, c = v["measures"], v["criteria"]

        def yn(x):
            return "–" if x is None else ("yes" if x else "no")
        rp = mm.get("recall", {}).get("p")
        ref_rows.append([s["name"], v["reference"], fmt_ci(mm.get("recall", {}).get("diff_ci")),
                         "–" if rp is None else f"{rp:.4f}", fmt(v.get("length_ratio_paired_mean"), 3),
                         fmt_ci(mm.get("rougeL", {}).get("diff_ci")), fmt_ci(mm.get("bertscore", {}).get("diff_ci")),
                         fmt_ci(mm.get("unsupported_share", {}).get("diff_ci")), fmt_ci(mm.get("nli", {}).get("diff_ci")),
                         fmt_ci(mm.get("judge_faithfulness", {}).get("diff_ci"), 3),
                         yn(c.get("1_recall")), yn(c.get("2_length")), yn(c.get("3_reference")), yn(c.get("4_claims")),
                         yn(c.get("all_four"))])
    if ref_rows:
        out.append("\n## Against the reference (normally the student's own SFT model)\n")
        alpha, k = res["inputs"]["alpha"], res["inputs"]["bonferroni_k"]
        out.append(f"Criterion 1 needs the recall interval above zero and p below {alpha / k:.4f} "
                   f"(alpha {alpha} / {k} comparisons). Criteria: 1 recall, 2 length 0.90–1.10, "
                   "3 ROUGE-L and BERTScore ≥ −0.010, 4 unsupported share not higher.\n")
        out.append(md_table(["system", "reference", "recall − ref", "p", "length ratio", "ROUGE-L − ref", "BERTScore − ref",
                             "unsupported − ref", "NLI − ref", "judge − ref", "1", "2", "3", "4", "all four"], ref_rows))
    out.append("\n## Cost\n")
    rows = []
    for s in res["systems"]:
        f, sv = s["flat"], s["serving"]
        rows.append([s["name"], fmt(f["params_b"], 2), fmt(f["train_gpu_hours"], 1),
                     ", ".join(s["train"].get("gpu_types") or []) or "–", fmt(f["train_peak_vram_gb"], 1),
                     fmt(f["gpu_s_per_paper"], 2), fmt(f["papers_per_hour"], 0), fmt(f["ttft_p50_s"], 2),
                     fmt(f["e2e_p50_s"], 1), fmt(f["e2e_p95_s"], 1), fmt(f["serve_weights_gb"], 1),
                     fmt(f["kv_gb_per_paper_p95"], 2), sv.get("gpu_type") or "–", sv.get("cost_source") or "–"])
    out.append(md_table(["system", "params (B)", "train GPU-h", "train GPU", "train peak GB", "GPU-s / paper",
                         "papers / h", "TTFT p50 s", "e2e p50 s", "e2e p95 s", "weights GB", "KV GB / paper p95",
                         "serving GPU", "cost source"], rows))
    miss = [(s["name"], m) for s in res["systems"] for m in s["missing"]]
    if miss:
        out.append("\n## Missing inputs\n")
        out.append(md_table(["system", "missing"], miss, ["l", "l"]))
    return "\n".join(out) + "\n"


def write_outputs(res: dict, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pj, pc, pm = out_dir / "results.json", out_dir / "results.csv", out_dir / "results.md"
    pj.write_text(json.dumps(res, indent=1))
    rows = [s["flat"] for s in res["systems"]]
    with open(pc, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["name"])
        w.writeheader()
        w.writerows(rows)
    pm.write_text(to_markdown(res))
    return [pj, pc, pm]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--systems", help="systems JSON (default: every configs/students/*.env, SFT and prefcc)")
    ap.add_argument("--students-dir", default="configs/students")
    ap.add_argument("--eval-glob", default="test-ex*-n1246", help="evaluation name pattern of the test runs")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--state-dir", default="state")
    ap.add_argument("--fact-dirs", default="factcheck,factcheck-b", help="question-set directories under state/")
    ap.add_argument("--claim-dir", default="claimcheck")
    ap.add_argument("--claim-model", default="qwen38-27b", help="the evaluation-side claim verifier")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--gpu-type", default=None, help="training GPU label when the logs do not name it")
    ap.add_argument("--common-papers", action="store_true",
                    help="restrict every measure to the papers all systems share (comparable absolute values)")
    ap.add_argument("--out-dir", default="reports/analysis")
    a = ap.parse_args(argv)
    spec = json.loads(Path(a.systems).read_text()) if a.systems else auto_systems(Path(a.students_dir), a.eval_glob)
    res = collect(spec, Path(a.runs_dir), Path(a.state_dir), [d for d in a.fact_dirs.split(",") if d], a.claim_dir,
                  a.claim_model, a.alpha, a.gpu_type, a.common_papers)
    for p in write_outputs(res, Path(a.out_dir)):
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
