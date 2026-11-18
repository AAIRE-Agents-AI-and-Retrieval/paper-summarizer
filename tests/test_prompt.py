"""The target format and the quality filter that decides what becomes a training example."""

from __future__ import annotations

import json

from scisum.prompt import SUMMARY_KEYS, canonical_target, distinct_n, validate_summary
from tests.test_metrics import GOOD


def test_canonical_target_is_json_in_schema_order():
    t = canonical_target({**GOOD, "doc_id": "arxiv:1", "model": "x"})
    obj = json.loads(t)
    assert list(obj) == list(SUMMARY_KEYS)  # extra bookkeeping fields are dropped
    assert '"tldr": ' in t and "\\u" not in t


def test_validate_accepts_short_but_real_domains():
    assert validate_summary({**GOOD, "domain": "optics"}).ok
    assert validate_summary({**GOOD, "limitations": ""}).ok  # may be empty by schema


def test_validate_rejects_defects():
    cases = {"missing": {k: v for k, v in GOOD.items() if k != "method"},
             "type": {**GOOD, "contributions": "a, b"},
             "empty": {**GOOD, "tldr": "short"},
             "contributions_count": {**GOOD, "contributions": ["only one"]},
             "keywords_count": {**GOOD, "keywords": ["a", "b"]}}
    for expected, obj in cases.items():
        chk = validate_summary(obj)
        assert not chk.ok and chk.reason.split(":")[0] == expected, (expected, chk)
    assert not validate_summary("not an object").ok


def test_repetition_filter_keeps_dense_science_and_rejects_loops():
    formula = {**GOOD, "method": "We study H = p^2 / 2m + x^2 (i x)^N for N >= 2, with H = p^2 / 2m."}
    assert validate_summary(formula).ok
    looped = {**GOOD, "results": "the model improves the metric by a large margin " * 12}
    assert not validate_summary(looped).ok
    assert distinct_n("a b c d e f g h i j k l") == 1.0
    assert distinct_n("a b c d e f g h i j " * 6) < 0.5


def test_the_api_key_never_appears_in_a_settings_dump(monkeypatch):
    """run.json, the reports and the paper tables all serialise Settings; the live key once reached git that way."""
    from scisum.config import Settings
    fake = "sk-" + "t" * 24                                        # built at runtime: no key-shaped literal in the source
    monkeypatch.setenv("JUDGE_API_KEY", fake)
    monkeypatch.setenv("SCISUM_JUDGE_BASE", "https://endpoint.invalid/v1")
    s = Settings()
    assert s.judge_api_key == fake                   # it is still usable by the judge
    assert fake not in s.model_dump_json() and fake not in repr(s) and "judge_api_key" not in s.model_dump()
    assert s.judge_base == "https://endpoint.invalid/v1"   # the endpoint is private too: usable, never dumped
    assert "endpoint.invalid" not in s.model_dump_json() and "judge_base" not in s.model_dump()


def test_scrub_drops_the_private_settings_from_old_dumps():
    """run.json files written before 2026-11-08 carry the endpoint; reports copy them into git."""
    from scisum.config import scrub
    old = {"settings": {"judge_base": "https://endpoint.invalid/v1", "judge_model": "m", "nested": [{"judge_api_key": "x"}]},
           "seed": 0}
    assert scrub(old) == {"settings": {"judge_model": "m", "nested": [{}]}, "seed": 0}
