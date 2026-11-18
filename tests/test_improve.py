"""The reward and the selection, without a model or the network."""

from __future__ import annotations

import pytest

from scisum.config import Settings
from scisum.improve import (
    SYNTH,
    _perturb_numbers,
    choose,
    composite,
    fact_reward,
    features,
    numbers,
    numeric_support,
    parse_candidate,
    question_weights,
    rescore,
    score_paper,
    selection_stats,
    synthetic_candidates,
)

PAPER = "We train on 12,000 images and reach 93.4% accuracy, 2.5 points above the 90.9 baseline in 2019."


def good(extra: str = "") -> dict:
    return {
        "tldr": "A new image classifier reaches 93.4% accuracy on a benchmark of 12,000 images." + extra,
        "problem": "Existing classifiers lose accuracy when images are scarce and noisy.",
        "contributions": ["A data-efficient training recipe for classifiers", "A benchmark of 12,000 labelled images"],
        "method": "The model is pretrained on unlabelled images and fine-tuned with a contrastive objective. Ablations follow.",
        "results": "It reaches 93.4% accuracy, 2.5 points above the 90.9 baseline. Gains hold across seeds.",
        "limitations": "Only tested on natural images.",
        "keywords": ["image classification", "contrastive learning", "data efficiency"],
        "domain": "computer vision",
    }


def test_numbers_skip_single_digits_and_years_and_ignore_commas():
    assert numbers("12,000 images, 3 seeds, in 2019, 93.4%, 0.5, 10-20, 2.5e-3") == {"12000", "93.4", "0.5", "10", "20", "2.5e-3"}
    assert "3" not in numbers("3 seeds") and "2019" not in numbers("in 2019")
    assert numbers("1,000") == numbers("1000") == {"1000"}


def test_numeric_support_counts_numbers_the_paper_does_not_contain():
    total, bad = numeric_support(good(), PAPER)
    assert (total, bad) == (4, 0)                                  # 93.4, 12000, 2.5, 90.9 are all in the paper
    wrong = good()
    wrong["results"] = "It reaches 97.8% accuracy, 7.1 points above the 90.9 baseline."
    total, bad = numeric_support(wrong, PAPER)
    assert bad == 2 and total == 5


def test_perturbed_numbers_are_unsupported_and_single_digits_untouched():
    out = _perturb_numbers("reaches 93.4% on 12,000 images in 3 runs")
    assert "93.4" not in out and "12,000" not in out and "3 runs" in out
    assert numbers(out).isdisjoint(numbers("93.4 12,000"))


def test_question_weights_keep_only_valid_questions_that_do_not_leak():
    full = ["correct", "correct", "partial", "incorrect", "correct"]
    closed = ["not_stated", "correct", "not_stated", "not_stated", "partial"]
    assert question_weights(full, closed) == [1, 0, 0, 0, 1]


def test_a_wrong_answer_costs_more_than_a_missing_one_and_too_few_questions_give_no_reward():
    w = [1, 1, 1, 1, 1, 1]
    assert fact_reward(["correct"] * 6, w, 5) == 1.0
    assert fact_reward(["not_stated"] * 6, w, 5) == 0.0
    assert fact_reward(["incorrect"] * 6, w, 5) == -1.0
    assert fact_reward(["correct", "partial", "not_stated", "incorrect", "correct", "ungraded"], w, 5) == pytest.approx((2.25 - 1) / 6)
    assert fact_reward(["correct"] * 4, [1, 1, 1, 1], 5) is None
    assert fact_reward(["correct", "incorrect", "correct"], [1, 0, 1], 2) == 1.0     # dropped questions do not count


def test_composite_trades_facts_against_imitation_numbers_and_length():
    s = Settings(imp_lambda=0.2, imp_beta_num=0.5, imp_gamma_len=1.0, imp_len_free=1.2)
    base = {"rougeL": 0.5, "unsupported_frac": 0.0, "length_ratio": 1.0}
    assert composite(0.3, base, s) == pytest.approx(0.4)
    assert composite(0.3, {**base, "unsupported_frac": 0.4}, s) == pytest.approx(0.2)          # invented numbers cost
    assert composite(0.3, {**base, "length_ratio": 1.5}, s) == pytest.approx(0.4 - 0.3)         # only the excess over 1.2
    assert composite(0.3, {**base, "length_ratio": 1.1}, s) == pytest.approx(0.4)               # length is free below it


def test_choose_replaces_the_teacher_only_by_a_margin_and_never_picks_an_unscored_one():
    assert choose([0.40, 0.42, 0.41], margin=0.05) == 0           # better, but not by enough
    assert choose([0.40, 0.46, 0.41], margin=0.05) == 1
    assert choose([0.40, 0.50, 0.60], margin=0.05) == 2           # the best of several
    assert choose([0.40, None, 0.9, None], margin=0.05) == 2
    assert choose([None, 0.1, 0.2], margin=0.05) == 2             # an unscorable teacher can be replaced
    assert choose([0.4], margin=0.05) == 0


def test_a_candidate_that_did_not_finish_or_is_invalid_has_no_features():
    t = good()
    assert features(t, "stop", t, PAPER)["valid"]
    assert not features(t, "length", t, PAPER)["valid"]
    assert not features(None, "stop", t, PAPER)["valid"]
    bad = {**t, "keywords": ["one"]}
    assert not features(bad, "stop", t, PAPER)["valid"]
    f = features(t, "stop", t, PAPER)
    assert f["rougeL"] == pytest.approx(1.0) and f["length_ratio"] == pytest.approx(1.0)


def test_the_template_prefix_is_removed_even_when_it_starts_with_a_newline():
    import json
    text = "\n</think>\n\n" + json.dumps(good())
    assert parse_candidate(text, "\n</think>\n\n") == good()
    assert parse_candidate("not json at all", "\n</think>\n\n") is None


def _ref(weights):
    return {"weights": weights}


def test_score_paper_picks_the_better_sample_and_falls_back_to_the_teacher(monkeypatch):
    import json
    s = Settings(imp_margin=0.05, imp_min_questions=3)
    teacher = good()
    better = good(" It also runs in real time.")
    p = {"id": "x", "teacher": teacher, "paper": PAPER}
    samples = [{"text": json.dumps(better), "finish": "stop"}, {"text": "garbage", "finish": "stop"}]
    ref = _ref([1, 1, 1, 1, 1])
    # teacher answers 2 of 5, the good sample 4 of 5, the garbage one is not graded at all
    verdicts = [["correct", "correct", "not_stated", "not_stated", "not_stated"],
                ["correct", "correct", "correct", "correct", "not_stated"], None]
    row = score_paper(s, p, [], ref, samples, verdicts, prefix="")
    assert row["chosen"] == 1 and row["reason"] == "sample"
    assert row["candidates"][2]["valid"] is False and row["candidates"][2]["reward"] is None
    same = [verdicts[0], verdicts[0], None]
    assert score_paper(s, p, [], ref, samples, same, prefix="")["chosen"] == 0
    assert score_paper(s, p, [], _ref([1, 0, 0, 0, 0]), samples, verdicts, prefix="")["reason"] == "no_reward"   # too few questions
    assert score_paper(s, p, [], None, samples, None, prefix="")["chosen"] == 0


def test_selection_stats_describe_the_round():
    s = Settings(imp_margin=0.0, imp_min_questions=3)
    import json
    p = {"id": "x", "teacher": good(), "paper": PAPER}
    samples = [{"text": json.dumps(good(" Extra.")), "finish": "stop"}]
    rows = [score_paper(s, p, [], _ref([1] * 5), samples,
                        [["correct", "not_stated", "not_stated", "not_stated", "not_stated"],
                         ["correct", "correct", "correct", "not_stated", "not_stated"]], prefix=""),
            {"id": "y", "chosen": 0, "reason": "no_reward", "n_candidates": 2}]
    st = selection_stats(rows)
    assert st["papers"] == 2 and st["scored"] == 1 and st["sample_chosen"] == 1 and st["no_reward"] == 1
    assert st["mean_fact"]["chosen"] > st["mean_fact"]["teacher"]
    assert st["samples_valid"] == 1.0


def test_synthetic_defects_have_the_defect_they_claim():
    t, other = good(), {**good(), "tldr": "A study of galaxy clusters in deep surveys of the sky."}
    c = synthetic_candidates(t, other)
    assert len(c) == len(SYNTH)
    assert c[0] == t
    assert numbers(c[1]["results"]).isdisjoint(numbers(t["results"]))                       # invented figures
    assert len(c[2]["method"]) < len(t["method"]) and len(c[2]["contributions"]) == 2     # omissions
    assert c[3]["tldr"] == other["tldr"]                                                   # another paper's summary


def test_a_paper_is_finished_only_when_sampled_and_either_scored_or_known_to_have_too_few_questions(tmp_path, monkeypatch):
    import scisum.improve as imp

    monkeypatch.setattr(imp, "idir", lambda s: tmp_path)
    w = imp.write_jsonl_atomic
    pool = [{"id": c} for c in "abcdefg"]
    w(tmp_path / "samples.jsonl.gz", [{"id": c, "samples": []} for c in "abcdef"])          # g was never sampled
    w(tmp_path / "questions-failed.jsonl.gz", [{"id": "b"}])                                # the writer failed on b
    w(tmp_path / "refs.jsonl.gz", [{"id": "c", "weights": [1, 1, 0, 0, 0, 0]},              # too few usable questions
                                   {"id": "d", "weights": [1] * 8},                          # scored
                                   {"id": "e", "weights": [1] * 8}])                         # usable but not scored yet
    w(tmp_path / "cand-grades.jsonl.gz", [{"id": "d", "verdicts": []}])
    s = Settings(imp_min_questions=5)
    assert imp.finished_ids(s, pool) == {"b", "c", "d"}          # a (no questions yet), e (not scored), f (no refs), g (unsampled) wait


def test_the_typical_sample_is_the_median_unchosen_one():
    from scisum.improve import typical_sample
    cands = [{"reward": 0.9}, {"reward": 0.5}, {"reward": 0.8}, {"reward": None}, {"reward": 0.2}, {"reward": 0.6}]
    # teacher (0) and the chosen sample (2) are excluded; the unscored one is skipped; of 0.5, 0.2, 0.6 the median is 0.5
    assert typical_sample({"chosen": 2, "candidates": cands}) == 1
    assert typical_sample({"chosen": 1, "candidates": [{"reward": 0.9}, {"reward": 0.5}]}) is None


def test_rescore_reproduces_the_stored_rewards_and_charges_length_under_a_tighter_allowance():
    import json
    s = Settings(imp_margin=0.05, imp_min_questions=3)
    teacher = good()
    longer = {**good(), "results": good()["results"] + " It also runs in real time on a phone. Training takes four hours on"
              " one card. Errors concentrate on blurred images and rare classes. Removing the contrastive stage costs"
              " most of the gain, and halving the unlabelled pool costs a quarter of it."}   # longer, not repetitive
    p = {"id": "x", "teacher": teacher, "paper": PAPER}
    samples = [{"text": json.dumps(longer), "finish": "stop"}, {"text": "garbage", "finish": "stop"}]
    verdicts = [["correct", "correct", "not_stated", "not_stated", "not_stated"],
                ["correct", "correct", "correct", "correct", "not_stated"], None]
    row = score_paper(s, p, [], _ref([1, 1, 1, 1, 1]), samples, verdicts, prefix="")
    rewards, chosen = rescore(row, s)
    assert rewards == [c["reward"] for c in row["candidates"]] and chosen == row["chosen"] == 1    # exactly, not approximately
    ratio = row["candidates"][1]["length_ratio"]
    assert 1.2 < ratio < 1.7
    tight = Settings(imp_margin=0.05, imp_min_questions=3, imp_len_free=1.0)
    r2, c2 = rescore(row, tight)
    assert r2[1] == pytest.approx(rewards[1] - 0.2) and r2[0] == rewards[0]       # the 1.0-1.2x band now costs 0.2
    assert r2[2] is None and c2 in (0, 1)
    assert rescore({"id": "y", "chosen": 0, "reason": "no_reward"}, s) == ([], 0)


def test_the_claim_penalty_lowers_each_candidate_by_its_unsupported_share_and_only_when_switched_on():
    row = {"id": "z", "chosen": 0, "candidates": [
        {"valid": True, "fact": 0.6, "rougeL": 0.5, "unsupported_frac": 0.0, "length_ratio": 1.0},
        {"valid": True, "fact": 0.7, "rougeL": 0.4, "unsupported_frac": 0.0, "length_ratio": 1.0},
        {"valid": False}]}
    off = Settings(imp_margin=0.05)
    on = Settings(imp_margin=0.05, imp_claim_weight=1.0)
    plain, chosen = rescore(row, off, [0.0, 0.2, None])
    assert chosen == 1 and rescore(row, off) == (plain, chosen)          # weight 0: the shares change nothing
    r, c = rescore(row, on, [0.0, 0.2, None])
    assert r[0] == pytest.approx(plain[0]) and r[1] == pytest.approx(plain[1] - 0.2) and r[2] is None
    assert c == 0                                       # the sample's 20% unsupported statements cost it the lead
    r, _ = rescore(row, on, [None, None, None])
    assert r == plain                                   # an unlabelled candidate takes no penalty
