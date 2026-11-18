"""Scoring: parsing a reply, schema compliance, and the overlap metrics."""

from __future__ import annotations

import json

from scisum.metrics import (
    aggregate,
    parse_prediction,
    render_summary,
    repetition,
    schema_report,
    score_one,
    set_f1,
    token_f1,
)

GOOD = {"tldr": "A method for X that improves Y by 3 points.",
        "problem": "Y is hard because of Z.",
        "contributions": ["a new loss", "a benchmark", "an ablation"],
        "method": "A transformer trained with the new loss on D.",
        "results": "Improves Y from 71.0 to 74.0 on D.",
        "limitations": "Only English data.",
        "keywords": ["summarisation", "transformers", "benchmark", "ablation", "loss"],
        "domain": "natural language processing"}


def test_parse_clean_json():
    obj, diag = parse_prediction(json.dumps(GOOD))
    assert obj == GOOD and diag["json_ok"] and not diag["repaired"]


def test_parse_repairs_fences_and_trailing_text():
    for raw in (f"```json\n{json.dumps(GOOD)}\n```",
                f"Here is the summary:\n{json.dumps(GOOD)}\nLet me know if you need more.",
                f"{json.dumps(GOOD)}{{"):
        obj, diag = parse_prediction(raw)
        assert obj == GOOD, raw[:40]
        assert diag["repaired"] and not diag["json_ok"]


def test_parse_failure_is_reported_not_raised():
    obj, diag = parse_prediction("I cannot summarise this paper.")
    assert obj is None and not diag["json_ok"]
    assert schema_report(obj)["schema_ok"] is False


def test_schema_report_counts_fields_and_ranges():
    r = schema_report(GOOD)
    assert r["schema_ok"] and r["fields_present"] == 8 and r["extra_fields"] == 0
    assert r["keywords_in_range"] and r["contributions_in_range"]
    bad = {**GOOD, "keywords": ["one"], "contributions": ["a"] * 9, "extra": 1}
    r = schema_report(bad)
    assert r["extra_fields"] == 1 and not r["keywords_in_range"] and not r["contributions_in_range"]
    assert schema_report({**GOOD, "contributions": "not a list"})["type_errors"] == 1


def test_perfect_prediction_scores_one():
    r = score_one(json.dumps(GOOD), GOOD)
    assert r["json_ok"] and r["schema_ok"]
    assert r["all_rouge1"] == 1.0 and r["all_rougeL"] == 1.0
    assert r["keywords_f1"] == 1.0 and r["domain_exact"] == 1.0
    assert r["length_ratio"] == 1.0 and r["stopped_on_eos"]


def test_degraded_prediction_scores_lower():
    worse = {**GOOD, "results": "Improves Y on D.", "keywords": ["summarisation", "x", "y", "z", "w"]}
    r = score_one(json.dumps(worse), GOOD)
    assert 0 < r["results_rougeL"] < 1          # partial overlap with "Improves Y from 71.0 to 74.0 on D."
    assert r["all_rougeL"] < 1 and r["keywords_f1"] == 0.2
    assert r["json_ok"] and r["schema_ok"]


def test_unparseable_prediction_scores_zero_without_crashing():
    r = score_one("I'm sorry, I can't help with that.", GOOD)
    assert not r["json_ok"] and not r["schema_ok"]
    assert r["all_rougeL"] == 0.0 and r["keywords_f1"] == 0.0 and r["pred_chars"] == 0


def test_set_and_token_f1():
    assert set_f1(["Deep Learning"], ["deep learning"])["f1"] == 1.0
    assert set_f1(["a", "b"], ["b", "c"])["f1"] == 0.5
    assert set_f1([], [])["f1"] == 1.0
    assert token_f1("information retrieval", "information retrieval") == 1.0
    assert 0 < token_f1("information retrieval", "retrieval") < 1


def test_repetition_detects_loops():
    assert repetition("the model learns a useful representation of the paper text")["distinct4"] == 1.0
    loop = ("the model learns a useful representation " * 8)
    assert repetition(loop)["distinct4"] < 0.3
    assert repetition(loop)["max_repeat10"] >= 4


def test_render_is_stable_and_aggregate_handles_missing():
    assert render_summary(GOOD).splitlines()[0].startswith("tldr:")
    assert render_summary(None) == ""
    agg = aggregate([{"a": 1.0, "b": 2}, {"a": 2.0}])
    assert agg["papers"] == 2 and agg["a"] == 1.5 and agg["b"] == 2


def test_a_template_prefix_that_starts_with_whitespace_is_still_removed():
    """Qwen3.5's assistant turn opens with "\\n</think>\\n\\n"; Run C read 0.000 clean JSON until this."""
    from scisum.metrics import score_one
    ref = {"tldr": "t", "problem": "p", "contributions": ["c"], "method": "m", "results": "r",
           "limitations": "l", "keywords": ["k"], "domain": "d"}
    body = '{"tldr": "t", "problem": "p", "contributions": ["c"], "method": "m", "results": "r", ' \
           '"limitations": "l", "keywords": ["k"], "domain": "d"}'
    for prefix, text in (("\n</think>\n\n", "\n</think>\n\n" + body),       # Qwen3.5
                         ("</ifm|think>", "</ifm|think>" + body),            # K2-Horizon
                         ("", body)):                                         # Qwen3-4B
        r = score_one(text, ref, target_prefix=prefix)
        assert r["json_ok"] and not r["repaired"], (prefix, r["parse_error"])
