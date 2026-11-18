"""Tests of scripts/analysis: parsing, statistics, aggregation, the Pareto front, the decision rule, plots.

Small synthetic fixtures only; no GPU, no network, no model."""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import pytest

ANALYSIS = Path(__file__).resolve().parent.parent / "scripts" / "analysis"
sys.path.insert(0, str(ANALYSIS))

import analysis_lib as al  # noqa: E402
import bench_inference as bi  # noqa: E402
import collect_results as cr  # noqa: E402
import learning_curves as lc  # noqa: E402
import pareto as pa  # noqa: E402
import train_resources as tr  # noqa: E402


def write_jsonl(path: Path, rows: list[dict], gz: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if gz else open
    with opener(path, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# ---- analysis_lib -------------------------------------------------------------------------------


def test_read_jsonl_skips_a_truncated_last_line(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"a": 1}\n{"a": 2}\n{"a": ')
    assert al.read_jsonl(p) == [{"a": 1}, {"a": 2}]
    assert al.read_jsonl(tmp_path / "missing.jsonl.gz") == []


def test_parse_env_file_handles_export_quotes_comments_and_expansion(tmp_path):
    p = tmp_path / "s.env"
    p.write_text("# comment\nexport A=1\nB='x y'  # not part\nexport C=\"$A-${B}\"\nD=plain # trailing\n"
                 "E=${MISSING:-dflt}\nsource other.env\nexport L='[\"q_proj\",\"k_proj\"]'\n")
    env = al.parse_env_file(p)
    assert env == {"A": "1", "B": "x y", "C": "1-x y", "D": "plain", "E": "dflt", "L": '["q_proj","k_proj"]'}


def test_last_match_orders_checkpoints_numerically(tmp_path):
    for n in ("test-ex9000-n10", "test-ex50000-n10", "test-ex5000-n10"):
        (tmp_path / f"{n}.scores.jsonl.gz").write_text("")
    assert al.last_match(tmp_path, "test-ex*-n10.scores.jsonl.gz").name == "test-ex50000-n10.scores.jsonl.gz"
    assert al.last_match(tmp_path, "nothing*") is None


def test_quantile_and_describe():
    assert al.quantile([1, 2, 3, 4], 0.5) == 2.5
    assert al.quantile([5], 0.95) == 5
    d = al.describe([1.0, 2.0, 3.0, None])
    assert d["n"] == 3 and d["mean"] == 2.0 and d["p50"] == 2.0
    assert al.describe([])["mean"] is None


def test_boot_ci_is_deterministic_and_brackets_the_mean():
    d = [0.1 * (i % 7) - 0.2 for i in range(200)]
    a, b = al.boot_ci(d), al.boot_ci(d)
    assert a == b and a[1] <= a[0] <= a[2]
    assert al.boot_ci([1.0])[0] == 1.0


def test_perm_p_detects_a_shift_and_not_noise():
    assert al.perm_p([0.05 + 0.001 * i for i in range(100)], n=2000) < 0.01
    assert al.perm_p([(-1) ** i * 0.05 for i in range(100)], n=2000) > 0.5


def test_pareto_front_dominance_ties_and_missing_values():
    pts = [{"q": 0.60, "c": 1.0}, {"q": 0.55, "c": 0.5}, {"q": 0.50, "c": 0.8},   # 2 dominated by 1
           {"q": 0.60, "c": 1.2}, {"q": 0.62, "c": None}, {"q": 0.65, "c": 3.0}]
    assert al.pareto_front(pts, "q", "c") == [1, 0, 5]
    # lower is better on quality: unsupported share
    pts2 = [{"u": 0.03, "c": 1.0}, {"u": 0.02, "c": 2.0}, {"u": 0.04, "c": 1.5}]
    assert al.pareto_front(pts2, "u", "c", higher_is_better=False) == [0, 1]


def test_student_colour_follows_the_student_not_its_rank():
    a = al.student_style("qwen35-4b")
    b = al.student_style("qwen35-4b", known=["qwen35-4b"])
    assert a["color"] == al.PALETTE[2] and b["color"] == al.PALETTE[0]
    assert al.student_style("k2h-3.7b")["marker"] != al.student_style("minicpm5-2b")["marker"]


def test_discover_students_orders_by_the_fixed_list(tmp_path):
    for tag, params in (("granite42-3b", "3.66e9"), ("k2h-3.7b", "5.06e9"), ("zz-new", "")):
        (tmp_path / f"{tag}.env").write_text(f"export STUDENT_TAG={tag}\nexport STUDENT_PARAMS={params}\n"
                                             f"export SCISUM_BASE_MODEL=org/{tag}\n")
    rows = al.discover_students(tmp_path)
    assert [r["tag"] for r in rows] == ["k2h-3.7b", "granite42-3b", "zz-new"]
    assert rows[0]["params"] == 5.06e9 and rows[2]["params"] is None


# ---- train_resources ----------------------------------------------------------------------------


def sft_metrics(run_dir: Path) -> None:
    rows = [{"t": 0, "kind": "chunk_start", "world": 2, "gpu": "GPU-X", "examples": 0},
            {"t": 10, "kind": "val_loss", "examples": 0, "step": 0, "papers": 200, "loss": 0.7, "target_tokens": 200000},
            {"t": 100, "kind": "train", "step": 5, "examples": 40, "tokens": 400000, "loss": 0.65, "tokens_per_s": 2000.0,
             "vram_gb": 30.0},
            # killed: no chunk_end; resumed
            {"t": 1000, "kind": "chunk_start", "world": 2, "gpu": "GPU-X", "examples": 40},
            {"t": 1800, "kind": "train", "step": 10, "examples": 80, "tokens": 800000, "loss": 0.6, "tokens_per_s": 2200.0,
             "vram_gb": 31.0},
            {"t": 1900, "kind": "val_loss", "examples": 80, "step": 10, "papers": 200, "loss": 0.5, "target_tokens": 200000},
            {"t": 4600, "kind": "chunk_end", "examples": 80, "exit_code": 0, "vram_peak_gb": 32.5}]
    write_jsonl(run_dir / "metrics.jsonl", rows, gz=False)
    (run_dir / "run.json").write_text(json.dumps({"base_model": "org/m", "params_total": 4e9, "lr": 1e-4}))


def test_train_resources_sums_chunks_and_counts_gpus(tmp_path):
    sft_metrics(tmp_path / "r")
    s = tr.summarise_run(tmp_path / "r")
    assert s["kind"] == "sft" and s["chunks"] == 2 and s["interrupted_chunks"] == 1 and s["finished"]
    assert s["wall_hours"] == pytest.approx((100 + 3600) / 3600, abs=1e-3)
    assert s["gpu_hours"] == pytest.approx(2 * (100 + 3600) / 3600, abs=1e-3)
    assert s["tokens_per_s_per_gpu"] == 2100.0 and s["peak_vram_gb"] == 32.5 and s["examples"] == 80
    assert s["val_loss"][0]["nats_per_paper"] == pytest.approx(0.7 * 200000 / 200)
    assert "| r | sft | 2 |" in tr.to_markdown([s])
    assert tr.summarise_run(tmp_path / "nothing")["missing"] == ["metrics.jsonl", "run.json"]


def test_train_resources_preference_run_without_tokens_per_s(tmp_path):
    rows = [{"t": 0, "kind": "chunk_start", "world": 1, "examples": 0},
            {"t": 3600, "kind": "train", "step": 5, "examples": 40, "tokens": 3600000, "dpo": 0.5, "pref_acc": 0.8,
             "margin": 1.0, "vram_gb": 36.0},
            {"t": 3600, "kind": "chunk_end", "exit_code": 0, "vram_peak_gb": 37.0}]
    write_jsonl(tmp_path / "p" / "metrics.jsonl", rows, gz=False)
    s = tr.summarise_run(tmp_path / "p", gpu_type="MI250X GCD")
    assert s["kind"] == "preference" and s["tokens_per_s_per_gpu"] == 1000.0 and s["gpu_type"] == "MI250X GCD"
    assert s["final_pref_acc"] == 0.8


# ---- collect_results ----------------------------------------------------------------------------


def make_world(tmp_path: Path, n: int = 40) -> tuple[Path, Path, dict]:
    """runs/ and state/ for a teacher, one SFT system and a two-seed preference system."""
    runs, state = tmp_path / "runs", tmp_path / "state"
    ids = [f"p{i:03d}" for i in range(n)]

    def verdicts(k_correct: int) -> list[str]:
        return ["correct"] * k_correct + ["partial"] + ["not_stated"] * (9 - k_correct)

    write_jsonl(state / "factcheck" / "grades-teacher.jsonl.gz",
                [{"id": i, "verdicts": verdicts(6)} for i in ids])
    write_jsonl(state / "claimcheck" / "teacher.qwen38-27b.jsonl.gz",
                [{"id": i, "valid": True, "claims": ["a"] * 20,
                  "verdicts": [{"verdict": "supported"}] * 19 + [{"verdict": "not_in_paper"}]} for i in ids])
    for run, kc, unsup, lr in (("s-sft50k", 5, 1, 0.98), ("s-prefcc", 6, 1, 1.0), ("s-prefcc-s2", 6, 1, 1.02)):
        write_jsonl(state / "factcheck" / f"grades-{run}.jsonl.gz",
                    [{"id": i, "verdicts": verdicts(kc if j % 4 else kc - 1)} for j, i in enumerate(ids)])
        write_jsonl(state / "claimcheck" / f"{run}.qwen38-27b.jsonl.gz",
                    [{"id": i, "valid": True, "claims": ["a"] * 20,
                      "verdicts": [{"verdict": "supported"}] * (20 - unsup) + [{"verdict": "contradicted"}] * unsup}
                     for i in ids])
        ev = runs / run / "eval"
        write_jsonl(ev / "test-ex100-n40.scores.jsonl.gz",
                    [{"id": i, "length_ratio": lr, "all_rougeL": 0.47, "bertscore_mean": 0.45, "faithfulness": 0.45,
                      "json_ok": 1.0, "out_tokens": 1000} for i in ids])
        write_jsonl(ev / "test-ex100-n40.judge.jsonl.gz",
                    [{"id": i, "judge_faithfulness": 2.0, "judge_delta": 0.0} for i in ids])
        (ev / "test-ex100-n40.gen.json").write_text(json.dumps({"papers_per_s": 0.5, "tensor_parallel": 1}))
        sft_metrics(runs / run)
    spec = {"teacher": {"fact_label": "teacher", "claim_label": "teacher"},
            "systems": [{"name": "s SFT", "student": "s", "stage": "sft", "params": 3e9,
                         "runs": [{"run": "s-sft50k", "eval": "test-ex*-n40"}], "train_runs": ["s-sft50k"]},
                        {"name": "s prefcc", "student": "s", "stage": "prefcc", "reference": "s SFT", "params": 3e9,
                         "runs": [{"run": "s-prefcc", "eval": "test-ex*-n40"}, {"run": "s-prefcc-s2", "eval": "test-ex*-n40"}],
                         "train_runs": ["s-sft50k", "s-prefcc"], "verification_runs": ["s-prefcc-s2"]}]}
    return runs, state, spec


def test_recall_of_counts_partial_as_half_and_strict_as_zero():
    v = ["correct", "partial", "not_stated", "incorrect", "ungraded"]
    assert cr.recall_of(v) == pytest.approx(1.5 / 4)
    assert cr.recall_of(v, strict=True) == pytest.approx(1 / 4)
    assert cr.recall_of(["ungraded"]) is None


def test_collect_two_seed_mean_criteria_and_outputs(tmp_path):
    runs, state, spec = make_world(tmp_path)
    res = cr.collect(spec, runs, state, ["factcheck", "factcheck-b"], "claimcheck", "qwen38-27b")
    sft, pref = res["systems"]
    assert res["teacher"]["recall"]["mean_ci"][0] == pytest.approx(6.5 / 10)
    assert sft["measures"]["recall"]["papers"] == 40
    # the preference system is the per-paper mean of its two seeds: length (1.00 + 1.02) / 2
    assert pref["measures"]["length_ratio"]["mean_ci"][0] == pytest.approx(1.01)
    vr = pref["vs_reference"]
    assert vr["measures"]["recall"]["diff_ci"][0] == pytest.approx(0.1)
    assert vr["criteria"]["1_recall"] is True and vr["criteria"]["2_length"] is True
    assert vr["criteria"]["3_reference"] is True and vr["criteria"]["4_claims"] is True
    assert vr["criteria"]["all_four"] is True
    assert pref["train"]["gpu_hours"] == pytest.approx(2 * sft["train"]["gpu_hours"])
    assert pref["serving"]["cost_source"].startswith("evaluation run")
    assert pref["serving"]["gpu_s_per_paper"] == pytest.approx(2.0)
    out = cr.write_outputs(res, tmp_path / "out")
    md = (tmp_path / "out" / "results.md").read_text()
    assert "s prefcc" in md and "| yes | yes | yes | yes | yes |" in md
    assert (tmp_path / "out" / "results.csv").read_text().splitlines()[0].startswith("name,student,stage")
    assert all(p.exists() for p in out)


def test_collect_reports_missing_inputs_instead_of_guessing(tmp_path):
    runs, state, spec = make_world(tmp_path)
    spec["systems"][0]["runs"][0]["fact_label"] = "nobody"
    (state / "claimcheck" / "s-sft50k.qwen38-27b.jsonl.gz").unlink()
    res = cr.collect(spec, runs, state, ["factcheck"], "claimcheck", "qwen38-27b")
    miss = res["systems"][0]["missing"]
    assert any("no fact grades for label 'nobody'" in m for m in miss)
    assert any("no claim check" in m for m in miss)
    assert "recall" not in res["systems"][0]["measures"]
    assert res["systems"][1]["vs_reference"]["criteria"]["1_recall"] is None


def test_restrict_to_common_papers(tmp_path):
    pps = {"a": {"recall": {"x": 1.0, "y": 0.5}}, "b": {"recall": {"y": 0.4, "z": 0.3}}}
    teacher = {"recall": {"x": 0.9, "y": 0.8, "z": 0.7}}
    out = cr.restrict_to_common(pps, teacher)
    assert out["a"]["recall"] == {"y": 0.5} and out["b"]["recall"] == {"y": 0.4} and teacher["recall"] == {"y": 0.8}


def test_benchmark_file_wins_over_the_eval_run(tmp_path):
    runs, state, spec = make_world(tmp_path)
    bench = {"schema": "scisum.bench_inference/1", "adapter_mode": "lora", "load_s": 30.0,
             "throughput": {"gpus": 1, "gpu_seconds_per_paper": 0.8, "papers_per_hour": 4500, "output_tokens_per_s": 1200},
             "latency": {"ttft_s": {"p50": 0.4}, "e2e_s": {"p50": 20.0, "p95": 31.0}, "tpot_s": {"p50": 0.02}},
             "memory": {"weights_gb": 7.5, "kv_gb_per_paper_p95": 1.2}, "hardware": {"gpus": [{"name": "AMD MI250X"}]}}
    (runs / "s-prefcc" / "bench").mkdir(parents=True)
    (runs / "s-prefcc" / "bench" / "ex100.json").write_text(json.dumps(bench))
    res = cr.collect(spec, runs, state, ["factcheck"], "claimcheck", "qwen38-27b")
    sv = res["systems"][1]["serving"]
    assert sv["cost_source"] == "benchmark" and sv["gpu_s_per_paper"] == 0.8 and sv["e2e_p95_s"] == 31.0
    assert sv["gpu_type"] == "AMD MI250X"


def test_auto_systems_follow_the_run_naming_contract(tmp_path):
    (tmp_path / "k2h-7b.env").write_text("export STUDENT_TAG=k2h-7b\nexport STUDENT_PARAMS=9.0e9\n")
    spec = cr.auto_systems(tmp_path, "test-ex*-n1246")
    names = [s["name"] for s in spec["systems"]]
    assert names == ["k2h-7b SFT", "k2h-7b prefcc"]
    assert [r["run"] for r in spec["systems"][1]["runs"]] == ["k2h-7b-prefcc", "k2h-7b-prefcc-s2"]
    assert spec["systems"][1]["reference"] == "k2h-7b SFT"


# ---- pareto -------------------------------------------------------------------------------------


def results_fixture() -> dict:
    def sysrow(name, student, stage, recall, gpu_s, crit=None, anchor=False):
        return {"name": name, "flat": {"name": name, "student": student, "stage": stage, "anchor": anchor,
                                       "params_b": 3.0, "recall": recall, "recall_lo": recall - 0.01,
                                       "recall_hi": recall + 0.01, "recall_vs_teacher": recall - 0.61,
                                       "unsupported_share": 0.03, "unsupported_lo": 0.02, "unsupported_hi": 0.04,
                                       "criteria_all_four": crit, "gpu_s_per_paper": gpu_s, "e2e_p95_s": 30.0,
                                       "train_gpu_hours": 50.0, "serve_weights_gb": 6.0, "kv_gb_per_paper_p95": 1.0,
                                       "serve_gpu_type": "G"}}
    return {"teacher": {"recall": {"mean_ci": [0.61, 0.6, 0.62]}, "unsupported_share": {"mean_ci": [0.03, 0.02, 0.04]}},
            "systems": [sysrow("a SFT", "a", "sft", 0.57, 0.5), sysrow("a prefcc", "a", "prefcc", 0.60, 0.5, True),
                        sysrow("b SFT", "b", "sft", 0.58, 0.9), sysrow("b prefcc", "b", "prefcc", 0.605, 0.9, False),
                        sysrow("C", "C", "sft", 0.59, 2.0, anchor=True)]}


def test_decision_rule_floor_criteria_and_cost(tmp_path):
    pts = pa.points_from(results_fixture())
    assert pts[0]["serve_mem_gb"] == 7.0
    rows, pick = pa.decide(pts, recall_floor=-0.02, require_criteria=True, max_unsupported=None, max_e2e_p95=None,
                           load=86400, utilisation=1.0)
    by = {r["name"]: r for r in rows}
    assert by["a SFT"]["eligible"] is False                      # recall 0.57 is 0.04 below the teacher
    assert by["b prefcc"]["eligible"] is False                   # fails the four criteria
    assert by["C"]["eligible"] is False                          # anchors are never picked
    assert pick["name"] == "a prefcc" and pick["gpus_at_load"] == pytest.approx(0.5)
    rows2, pick2 = pa.decide(pts, -0.02, False, None, max_e2e_p95=10.0, load=86400, utilisation=1.0)
    assert pick2 is None and all(r["checks"]["latency cap"] is False for r in rows2)


def test_pareto_main_writes_figures_and_decision(tmp_path):
    p = tmp_path / "results.json"
    p.write_text(json.dumps(results_fixture()))
    assert pa.main([str(p), "--out-dir", str(tmp_path / "o"), "--costs", "gpu_s_per_paper,params_b"]) == 0
    files = {f.name for f in (tmp_path / "o").iterdir()}
    assert {"pareto_recall_vs_gpu_s_per_paper.png", "pareto_unsupported_share_vs_params_b.svg", "decision.md"} <= files
    dec = json.loads((tmp_path / "o" / "decision.json").read_text())
    assert dec["front_recall_vs_gpu_s"] == ["a prefcc", "b prefcc"]
    assert "Cheapest system that meets the rule: a prefcc" in (tmp_path / "o" / "decision.md").read_text()


# ---- bench_inference ----------------------------------------------------------------------------


def test_kv_bytes_count_only_attention_layers():
    hybrid = {"text_config": {"num_hidden_layers": 4, "num_attention_heads": 16, "num_key_value_heads": 4,
                              "hidden_size": 2048, "head_dim": 256,
                              "layer_types": ["linear_attention", "linear_attention", "linear_attention",
                                              "full_attention"]}}
    k = bi.kv_bytes_per_token(hybrid)
    assert k["kv_layers"] == 1 and k["fixed_state_layers"] == 3 and k["bytes_per_token"] == 2 * 1 * 4 * 256 * 2
    dense = {"num_hidden_layers": 40, "num_attention_heads": 40, "num_key_value_heads": 8, "hidden_size": 2560}
    assert bi.kv_bytes_per_token(dense)["bytes_per_token"] == 2 * 40 * 8 * 64 * 2


def test_max_new_tokens_respects_the_window():
    assert bi.max_new_tokens(1000, 3000, 32768) == 3000
    assert bi.max_new_tokens(31000, 3000, 32768) == 32768 - 31000 - 8
    assert bi.max_new_tokens(40000, 3000, 32768) == 1


class FakeEngine:
    """Generates n tokens per request; the fake clock advances 1 s per request + 0.01 s per output token."""

    gpus = 2

    def __init__(self, clock):
        self.clock = clock

    def generate(self, prompts, max_tokens):
        out = []
        for p, m in zip(prompts, max_tokens, strict=True):
            n = min(m, 100)
            self.clock.t += 1.0 + 0.01 * n
            out.append({"text": '<think></think>{"tldr": "x"}' if "ok" in p else "no json", "out_tokens": n,
                        "prompt_tokens": 50, "finish_reason": "stop", "metrics_ttft_s": None})
        return out


class Clock:
    t = 0.0

    def __call__(self):
        return self.t


def test_latency_and_throughput_with_a_fake_engine():
    clock = Clock()
    eng = FakeEngine(clock)
    recs = [{"id": f"p{i}", "prompt": "ok" if i % 2 else "bad", "prompt_tokens": 50} for i in range(6)]
    lat = bi.measure_latency(eng, recs, cap=3000, max_model_len=32768, warmup=2, clock=clock)
    assert lat["n"] == 4 and lat["ttft_source"] == "1-token generation"
    assert lat["ttft_s"]["p50"] == pytest.approx(1.01)            # one token
    assert lat["e2e_s"]["p50"] == pytest.approx(2.0)              # 100 tokens
    assert lat["tpot_s"]["p50"] == pytest.approx((2.0 - 1.01) / 99)
    thr = bi.measure_throughput(eng, recs, cap=3000, max_model_len=32768, repeats=3, clock=clock)
    assert thr["n"] == 6 and thr["repeats"] == 3 and thr["wall_s"] == pytest.approx(12.0)
    assert thr["gpu_seconds_per_paper"] == pytest.approx(2 * 12.0 / 6)
    assert thr["json_parse_share"] == pytest.approx(0.5) and thr["hit_token_limit_share"] == 0.0


def test_load_prompts_follows_the_id_list(tmp_path):
    d = tmp_path / "prompts" / "test"
    write_jsonl(d / "part-0000.jsonl.gz", [{"id": f"p{i}", "prompt": f"P{i}", "meta": {"prompt_tokens": 10 + i}}
                                          for i in range(5)])
    recs = bi.load_prompts(d, ["p3", "p1", "zz"], 5)
    assert [r["id"] for r in recs] == ["p3", "p1"] and recs[0]["prompt_tokens"] == 13
    assert [r["id"] for r in bi.load_prompts(d, None, 2)] == ["p0", "p1"]
    with pytest.raises(SystemExit):
        bi.load_prompts(tmp_path / "none", None, 1)


def test_apply_env_sets_the_student_profile(tmp_path, monkeypatch):
    p = tmp_path / "s.env"
    p.write_text("export SCISUM_BASE_MODEL=org/m\nexport STUDENT_TAG=m\n")
    monkeypatch.delenv("SCISUM_BASE_MODEL", raising=False)
    env = bi.apply_env([str(p), ""])
    import os

    assert env["STUDENT_TAG"] == "m" and os.environ["SCISUM_BASE_MODEL"] == "org/m"


# ---- learning_curves ----------------------------------------------------------------------------


def test_student_of_and_binning():
    assert lc.student_of("k2h-3.7b-sft50k") == "k2h-3.7b"
    assert lc.student_of("k2h-3.7b-prefcc-s2") == "k2h-3.7b"
    pts = [(float(x), 1.0 / x) for x in range(1, 1001)]
    b = lc.binned(pts, bins=10)
    assert len(b) <= 10 and b[0][0] < b[-1][0]


def test_learning_curves_main_writes_overview_and_pages(tmp_path):
    runs = tmp_path / "runs"
    sft_metrics(runs / "s-sft50k")
    ev = runs / "s-sft50k" / "eval"
    ev.mkdir(parents=True, exist_ok=True)
    for n, rl in ((40, 0.44), (80, 0.46)):
        (ev / f"val-ex{n}-n10.json").write_text(json.dumps({"metrics": {"all_rougeL": rl, "schema_ok": 0.99}}))
    write_jsonl(runs / "s-prefcc" / "metrics.jsonl",
                [{"t": 0, "kind": "chunk_start", "world": 1},
                 {"t": 5, "kind": "train", "step": 5, "examples": 40, "tokens": 1, "dpo": 0.5, "nll": 0.45,
                  "margin": 0.4, "pref_acc": 0.7, "len_diff": 10},
                 {"t": 9, "kind": "train", "step": 10, "examples": 80, "tokens": 2, "dpo": 0.3, "nll": 0.44,
                  "margin": 1.4, "pref_acc": 0.9, "len_diff": -3}], gz=False)
    out = tmp_path / "curves"
    assert lc.main(["s-sft50k", "s-prefcc", "--runs-dir", str(runs), "--out-dir", str(out)]) == 0
    names = {p.name for p in out.iterdir()}
    assert {"curves_sft_val_rougeL.png", "curves_pref_margin.svg", "run_s-sft50k.png", "run_s-prefcc.png"} <= names
    x, y = lc.series(lc.read_run(runs / "s-sft50k"), "val_rougeL")
    assert x == [40, 80] and y == [0.44, 0.46]
