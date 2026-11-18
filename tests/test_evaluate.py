"""Pulling the paper text back out of a prompt, and the naming that makes evaluations resumable."""

from scisum.evaluate import eval_name, source_text
from scisum.prompt import build_user_message

DOC = {"title": "A Paper", "authors": ["A B"], "categories": ["cs.CL"], "abstract": "The abstract."}
BODY = "Introduction\nThe body of the paper, with its results."


def test_source_text_recovers_abstract_and_body():
    prompt = build_user_message(DOC, BODY, abstract_only=False)
    src = source_text(prompt)
    assert "The abstract." in src and "results." in src
    assert "Return a JSON object" not in src and "Categories:" not in src


def test_source_text_falls_back_to_the_whole_prompt():
    assert source_text("no markers here") == "no markers here"


def test_abstract_only_prompt_has_no_body_section():
    prompt = build_user_message(DOC, "", abstract_only=True)
    assert "Full text:" not in prompt
    assert source_text(prompt).strip() == "The abstract."


def test_eval_name_is_content_addressed():
    assert eval_name("val", "ex5000", 1000) == "val-ex5000-n1000"


def test_the_worker_hands_its_time_budget_to_generation(monkeypatch, tmp_path):
    """A slow model must stop between batches, not run one checkpoint through the walltime."""
    import scisum.evaluate as ev
    from scisum.config import Settings

    calls: list[tuple] = []

    def fake_cli(*args):
        calls.append(args)
        return 3 if args[:2] == ("eval", "generate") else 0   # generation hit its limit

    monkeypatch.setattr(ev, "_cli", fake_cli)
    monkeypatch.setattr(ev, "pending_checkpoints", lambda s, run: ["ex5000", "ex10000"])
    out = ev.run_worker(Settings(), run="r", max_seconds=3600, once=True)

    gen = [c for c in calls if c[:2] == ("eval", "generate")]
    assert len(gen) == 1, "after a time-limited stop the worker must not start the next checkpoint"
    assert "--max-seconds" in gen[0]
    assert not any(c[:2] == ("eval", "score") for c in calls), "an unfinished checkpoint is never scored"
    assert out["evaluated"] == []


def test_generation_reports_whether_it_finished():
    import inspect

    from scisum.evaluate import run_generate
    assert "max_seconds" in inspect.signature(run_generate).parameters
    src = inspect.getsource(run_generate)
    assert '"complete": total >= len(recs)' in src
    assert "tensor_parallel_size=tp" in src, "an engine must use every GPU the job was given"


def test_vllm_is_told_to_build_the_class_training_built():
    """Qwen3.5-9B declares its VL wrapper; serving that left the adapter matching no module."""
    from scisum.evaluate import vllm_architecture_override as ov
    assert ov({"model_type": "qwen3_5", "architectures": ["Qwen3_5ForConditionalGeneration"]}) == \
        {"architectures": ["Qwen3_5ForCausalLM"]}
    assert ov({"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}) == {}       # run A
    assert ov({"model_type": "k2_horizon", "architectures": ["K2HorizonForCausalLM"]}) == {}  # B: remote code
    assert ov({}) == {}


class _FakeOut:
    def __init__(self, text):
        self.outputs = [type("C", (), {"text": text})()]


class _FakeLLM:
    def __init__(self, applied: bool):
        self.applied = applied

    def generate(self, prompts, sp, use_tqdm=False, lora_request=None):
        tag = "{" if (lora_request and self.applied) else "Thinking"
        return [_FakeOut(f"{tag} {p}") for p in prompts]


def test_an_adapter_that_changes_nothing_is_an_error():
    import pytest

    from scisum.evaluate import check_adapter_applied
    lora = type("L", (), {"lora_path": "/x/adapter"})()
    check_adapter_applied(_FakeLLM(applied=True), lora, ["a", "b"])          # applied: passes
    with pytest.raises(RuntimeError, match="has no effect"):
        check_adapter_applied(_FakeLLM(applied=False), lora, ["a", "b"])


def test_a_zero_budget_means_no_time_left(monkeypatch):
    import scisum.evaluate as ev
    from scisum.config import Settings

    calls: list = []
    monkeypatch.setattr(ev, "_cli", lambda *a: calls.append(a) or 0)
    monkeypatch.setattr(ev, "pending_checkpoints", lambda s, run: ["ex500"])
    ev.run_worker(Settings(), run="r", max_seconds=0, once=True)
    assert calls == [], "max_seconds=0 was read as 'no limit' and started a generation"


def test_each_run_has_its_own_report():
    from scisum.config import Settings
    from scisum.report import report_paths
    s = Settings()
    assert report_paths(s, "main")[0].name == "REPORT.md"               # unchanged for run A
    assert report_paths(s, "q35")[0].name == "REPORT-q35.md"
    assert report_paths(s, "k2") != report_paths(s, "q35")


def test_the_served_model_is_the_text_only_copy_when_one_is_configured(monkeypatch, tmp_path):
    import transformers

    from scisum.config import Settings
    from scisum.evaluate import serve_model_args

    declared = {}

    class _Cfg:
        def __init__(self, d):
            self.d = d

        def to_dict(self):
            return self.d

    hub_cfg = {"model_type": "qwen3_5", "architectures": ["Qwen3_5ForConditionalGeneration"]}

    def fake(path, **kw):
        return _Cfg(declared.get(str(path), hub_cfg))

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", staticmethod(fake))
    hub = serve_model_args(Settings(base_model="Qwen/Qwen3.5-9B", base_model_revision="c202236"))
    assert hub["model"] == "Qwen/Qwen3.5-9B" and hub["revision"] == "c202236"
    assert hub["hf_overrides"] == {"architectures": ["Qwen3_5ForCausalLM"]}

    copy = tmp_path / "text-only"
    declared[str(copy.resolve())] = {"model_type": "qwen3_5", "architectures": ["Qwen3_5ForCausalLM"]}
    local = serve_model_args(Settings(base_model="Qwen/Qwen3.5-9B", serve_model=str(copy)))
    assert local["model"] == local["tokenizer"] == str(copy.resolve())
    assert "revision" not in local and "hf_overrides" not in local     # the copy already declares it


# ---- the fact-level evaluation -------------------------------------------------------------
def test_stratified_sample_is_proportional_and_keeps_the_random_cohort():
    from scisum.factcheck import stratified_ids
    rows = ([{"id": f"a{i}", "archive": "astro-ph", "cohort": "top"} for i in range(600)]
            + [{"id": f"c{i}", "archive": "cs", "cohort": "top"} for i in range(300)]
            + [{"id": f"m{i}", "archive": "math", "cohort": "top"} for i in range(60)]
            + [{"id": f"r{i}", "archive": "cs", "cohort": "random"} for i in range(20)])
    ids = stratified_ids(rows, 100, seed=1)
    assert len(ids) == len(set(ids)) == 100
    assert sum(i.startswith("r") for i in ids) == 10                     # n // 10 random-cohort papers first
    counts = {p: sum(i.startswith(p) for i in ids) for p in "acm"}
    assert counts["a"] > counts["c"] > counts["m"] >= 4                   # proportional, none vanishes
    assert ids == stratified_ids(rows, 100, seed=1)                       # deterministic


def test_scoring_counts_unusable_summaries_as_missing_and_ungraded_as_neither():
    from scisum.factcheck import paper_recall, score_rows
    rows = [{"usable": True, "verdicts": ["correct", "partial", "incorrect", "not_stated"], "types": []},
            {"usable": False, "verdicts": ["not_stated"] * 4, "types": []},
            {"usable": True, "verdicts": ["correct", "ungraded", "ungraded", "correct"], "types": []}]
    s = score_rows(rows)
    assert s["ungraded"] == 2 and s["questions"] == 12
    assert s["not_stated"] == round(5 / 10, 4) and s["recall"] == round((3 + .5) / 10, 4)
    assert s["accuracy_when_answering"] == round(3 / 5, 4) and s["usable_summary"] == round(2 / 3, 4)
    assert paper_recall(rows[0]) == (1 + .5) / 4 and paper_recall(rows[2]) == 1.0


def test_every_system_is_shown_to_the_reader_in_the_same_form():
    from scisum.factcheck import render_for_reader
    a = render_for_reader({"tldr": "t", "contributions": ["x", "y"], "keywords": "k"})
    assert "TLDR: t" in a and "CONTRIBUTIONS: - x\n- y" in a and "KEYWORDS: k" in a
    assert render_for_reader({}) == ""
