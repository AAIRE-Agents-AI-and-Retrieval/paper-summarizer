"""Round 3's two-seed verdict (`scripts/improve_seeds.py`) and the round-3 length analysis set-up."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


seeds = load("improve_seeds")


def test_unsupported_share_ignores_ungraded_and_returns_none_without_labels():
    r = {"verdicts": [{"verdict": v} for v in ("supported", "not_in_paper", "contradicted", "ungraded")]}
    assert seeds.unsupported_share(r) == pytest.approx(2 / 3)
    assert seeds.unsupported_share({"verdicts": [{"verdict": "ungraded"}]}) is None


def test_paired_uses_per_paper_mean_on_papers_all_three_have():
    base = {"a": 0.5, "b": 0.5, "c": 0.5}
    s1 = {"a": 0.6, "b": 0.7, "x": 1.0}
    s2 = {"a": 0.8, "b": 0.5}
    p = seeds.paired(base, s1, s2)
    assert p["ids"] == ["a", "b"]
    assert p["two_seed_minus_C"] == pytest.approx([0.2, 0.1])
    assert p["seed2_minus_seed1"] == pytest.approx([0.2, -0.2])
    assert p["seed_means"] == pytest.approx([0.65, 0.65])
    assert p["C_mean"] == pytest.approx(0.5)


def test_versus_is_arm_minus_seeds_on_common_papers():
    x = {"a": 1.0, "b": 0.0, "c": 5.0}
    s1 = {"a": 0.5, "b": 0.5}
    s2 = {"a": 0.0, "b": 0.5}
    v = seeds.versus(x, s1, s2)
    assert v["papers"] == 2
    assert v["minus_two_seed"][0] == pytest.approx(((1.0 - 0.25) + (0.0 - 0.5)) / 2)
    assert v["minus_seed1"][0] == pytest.approx(0.0)
    assert v["minus_seed2"][0] == pytest.approx(0.25)
    assert v["seed2_minus_seed1"][0] == pytest.approx(-0.25)


def _m(two_seed_minus_c, mean=None):
    return {"two_seed_minus_C": list(two_seed_minus_c), "two_seed_mean": mean}


def test_criteria_as_written_in_the_protocol():
    m = {"recall": {**_m([0.02, 0.01, 0.03]), "perm_p_two_seed": 0.0001},
         "length_ratio": _m([0.0, 0, 0], mean=1.05),
         "all_rougeL": _m([-0.006, -0.01, -0.002]),
         "bertscore_mean": _m([-0.00999, -0.014, -0.006]),
         "unsupported_share": _m([0.005, -0.0001, 0.011])}
    c = seeds.criteria(m)
    assert c == {"1_recall": True, "2_length": True, "3_reference": True, "4_claims": True, "all_four": True,
                 "recall_p_below_bonferroni": True}
    m["bertscore_mean"] = _m([-0.01004, -0.014, -0.006])          # round 2's preflc miss stays a miss
    m["unsupported_share"] = _m([0.0085, 0.0031, 0.0144])          # interval above zero fails
    m["length_ratio"] = _m([0, 0, 0], mean=0.89)
    m["recall"] = {**_m([0.01, -0.001, 0.02]), "perm_p_two_seed": 0.02}
    c = seeds.criteria(m)
    assert c == {"1_recall": False, "2_length": False, "3_reference": False, "4_claims": False, "all_four": False,
                 "recall_p_below_bonferroni": False}


def test_criteria_missing_measure_gives_no_verdict():
    c = seeds.criteria({"length_ratio": _m([0, 0, 0], mean=1.0)})
    assert c["2_length"] is True and c["1_recall"] is None and c["all_four"] is None


def test_perm_p_detects_a_clear_shift_and_not_noise():
    assert seeds.perm_p([0.1] * 50, n=2000) < 0.01
    assert seeds.perm_p([0.1, -0.1] * 25, n=2000) > 0.5


def test_improve_length_round3_arms():
    length = load("improve_length")
    length.use_round(3)
    assert length.OUT.endswith("improve_length_r3.json")
    assert set(length.REG_ARMS) == {"preflc", "prefrd", "prefcc", "preflcrd3", "preflcs2", "prefrds2"}
    assert all(a in length.ARMS and a in length.FACT_LABEL for a in length.REG_ARMS)
    assert length.FACT_LABEL["prefcc"] == "C_prefcc"
