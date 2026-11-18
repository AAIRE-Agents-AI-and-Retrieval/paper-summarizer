"""The claim check's parsing, statistics and controls, without the network."""

from __future__ import annotations

import json

import pytest

from scisum.claimcheck import PROMPT, paper_stats, parse_verdicts, report, sample_papers, summaries
from scisum.config import Settings
from scisum.factcheck import append_jsonl


def test_parse_verdicts_keeps_order_and_marks_what_the_reply_skipped():
    text = json.dumps({"verdicts": [
        {"i": 2, "verdict": "contradicted", "detail": "93.4%", "paper": "92.1%"},
        {"i": 1, "verdict": "Supported"},
        {"i": 1, "verdict": "contradicted"},            # a second answer for the same statement is ignored
        {"i": 4, "verdict": "not in paper", "detail": "ImageNet"},
        {"i": 9, "verdict": "supported"},               # out of range
        {"i": "x", "verdict": "supported"},
        {"i": 3, "verdict": "probably"},                # not a label
    ]})
    v = parse_verdicts(text, 4)
    assert [x["verdict"] for x in v] == ["supported", "contradicted", "ungraded", "not_in_paper"]
    assert v[1]["paper"] == "92.1%" and v[3]["detail"] == "ImageNet"
    assert [x["verdict"] for x in parse_verdicts("no json here", 2)] == ["ungraded", "ungraded"]


def test_paper_stats_counts_and_shares_over_labelled_statements():
    r = {"claims": ["one two three", "four five", "six", "seven eight"],
         "verdicts": [{"verdict": "supported"}, {"verdict": "contradicted"}, {"verdict": "ungraded"},
                      {"verdict": "not_in_paper"}]}
    p = paper_stats(r)
    assert (p["claims"], p["labelled"], p["supported"], p["contradicted"], p["not_in_paper"], p["unsupported"]) == (4, 3, 1, 1, 1, 2)
    assert p["supported_share"] == pytest.approx(1 / 3) and p["contradicted_share"] == pytest.approx(1 / 3)
    assert p["words"] == 8 and p["words_per_claim"] == pytest.approx(2.0)


def _papers(tmp_path, n=6):
    for d, lo in (("factcheck", 0), ("factcheck-b", n // 2)):
        (tmp_path / "state" / d).mkdir(parents=True)
        rows = [{"id": f"p{i}", "paper": f"Paper {i} reports 12.5% and 300 samples.",
                 "teacher": {"tldr": f"Paper {i} reports 12.5% on 300 samples of data.", "problem": "x" * 30,
                             "contributions": ["a contribution that is long enough", "another one long enough to count"],
                             "method": "y" * 30, "results": "It reaches 12.5% on 300 samples in total.",
                             "limitations": "", "keywords": ["a", "b", "c"], "domain": "cs"}}
                for i in range(lo, lo + n // 2)]
        (tmp_path / "state" / d / "papers.json").write_text(json.dumps(rows))
    return Settings(repo_root=tmp_path)


def test_the_sample_is_seeded_and_spans_both_question_sets(tmp_path):
    s = _papers(tmp_path)
    a, b = sample_papers(s, 4), sample_papers(s, 4)
    assert [p["id"] for p in a] == [p["id"] for p in b] and len(a) == 4
    assert sorted(p["id"] for p in sample_papers(s, 10)) == [f"p{i}" for i in range(6)]


def test_controls_have_their_defect(tmp_path):
    s = _papers(tmp_path)
    papers = sample_papers(s, 6)
    pert = summaries(s, "teacher_perturbed", papers, 3)
    other = summaries(s, "other_paper", papers, 3)
    assert set(pert) == set(other) == {p["id"] for p in papers[:3]}
    first = papers[0]
    assert "12.5" not in pert[first["id"]]["results"] and "300" not in pert[first["id"]]["results"]
    assert other[first["id"]] == papers[1]["teacher"]
    with pytest.raises(ValueError):
        summaries(s, "nobody", papers, 3)


def test_the_prompt_formats_and_the_report_pairs_systems_on_common_papers(tmp_path):
    assert "{" not in PROMPT.format(n=2, paper="P", statements="1. a\n2. b").split("Answer with JSON")[0]
    s = Settings(repo_root=tmp_path)
    d = tmp_path / "state" / "claimcheck"
    d.mkdir(parents=True)
    def row(i, verdicts):
        return {"id": f"p{i}", "valid": True, "claims": ["a claim"] * len(verdicts),
                "verdicts": [{"verdict": v} for v in verdicts]}
    append_jsonl(d / "C.m.jsonl.gz", [row(i, ["supported", "supported", "contradicted"]) for i in range(5)])
    append_jsonl(d / "C_pref.m.jsonl.gz", [row(i, ["supported", "not_in_paper", "contradicted", "contradicted"]) for i in range(4)])
    out = report(s, model="m", base=("C",))
    assert out["systems"]["C"]["papers"] == 5 and out["systems"]["C_pref"]["claims"] == 4
    pair = out["paired"]["C_pref - C"]
    assert pair["papers"] == 4 and pair["contradicted"][0] == pytest.approx(1.0) and pair["unsupported"][0] == pytest.approx(2.0)


def test_the_export_is_blinded_and_shows_flagged_and_supported_statements(tmp_path):
    from scisum.claimcheck import export_for_people
    s = _papers(tmp_path)
    d = tmp_path / "state" / "claimcheck"
    d.mkdir(parents=True)
    rows = [{"id": "p0", "valid": True, "claims": ["Paper 0 reports 12.5% on 300 samples.", "It uses a quantum computer."],
             "verdicts": [{"verdict": "supported", "detail": "", "paper": ""},
                          {"verdict": "not_in_paper", "detail": "quantum computer", "paper": ""}]}]
    append_jsonl(d / "C.m.jsonl.gz", rows)
    append_jsonl(d / "teacher.m.jsonl.gz", rows)
    out, key = export_for_people(s, ["teacher", "C"], model="m", per_system=1)
    text = out.read_text()
    assert text.count("**Statement:**") == 4 and "quantum computer" in text and "system C" not in text
    assert set(json.loads(key.read_text()).values()) == {"teacher", "C"}


def test_a_reply_with_one_malformed_field_is_read_record_by_record():
    # the shape of the 2026-11-07 failures: text after the closing quote of one "paper" field
    text = ('{\n  "verdicts": [\n    {"i": 1, "verdict": "supported", "detail": "", "paper": ""},\n'
            '    {"i": 2, "verdict": "contradicted", "detail": "1.117 +/- 0.098 for LHCb",\n'
            '     "paper": "the value becomes 1.117" (not specifically LHCb).\n    },\n'
            '    {"i": 3, "verdict": "Not in paper", "detail": "a \\"quoted\\" word", "paper": ""}\n  ]\n}')
    v = parse_verdicts(text, 4)
    assert [x["verdict"] for x in v] == ["supported", "contradicted", "not_in_paper", "ungraded"]
    assert v[1]["detail"] == "1.117 +/- 0.098 for LHCb" and v[2]["detail"] == 'a \\"quoted\\" word'


def test_unlabelled_summaries_are_missing_and_a_recheck_replaces_the_old_row(tmp_path):
    from scisum.claimcheck import checked_ids

    s = Settings(repo_root=tmp_path)
    d = tmp_path / "state" / "claimcheck"
    d.mkdir(parents=True)
    ok = {"valid": True, "claims": ["a", "b"], "verdicts": [{"verdict": "supported"}, {"verdict": "supported"}]}
    append_jsonl(d / "C.m.jsonl.gz", [{"id": "p0", **ok}, {"id": "p1", **ok}, {"id": "p2", **ok}])
    append_jsonl(d / "C_x.m.jsonl.gz", [
        {"id": "p0", **ok},
        {"id": "p1", "valid": True, "claims": ["a", "b"], "verdicts": []},                    # written before the fix
        {"id": "p2", "valid": True, "claims": ["a", "b"], "verdicts": [{"verdict": "ungraded"}] * 2},
        {"id": "p3", "valid": False, "claims": [], "verdicts": []}])
    assert checked_ids(d / "C_x.m.jsonl.gz") == {"p0", "p2", "p3"}
    out = report(s, model="m", base=("C",))
    assert out["systems"]["C_x"]["papers"] == 1 and out["systems"]["C_x"]["unlabelled_summaries"] == 2
    assert out["systems"]["C_x"]["supported_share"] == pytest.approx(1.0)
    append_jsonl(d / "C_x.m.jsonl.gz", [{"id": "p1", "valid": True, "claims": ["a", "b"],
                                          "verdicts": [{"verdict": "supported"}, {"verdict": "contradicted"}]}])
    assert "p1" in checked_ids(d / "C_x.m.jsonl.gz")
    out = report(s, model="m", base=("C",))
    assert out["systems"]["C_x"]["papers"] == 2 and out["paired"]["C_x - C"]["papers"] == 2
    assert out["paired"]["C_x - C"]["contradicted"][0] == pytest.approx(0.5)
