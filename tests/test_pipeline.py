"""scripts/pipeline.py without Slurm: the stage graph, what "done" means, idempotent submission, the task runner,
and the site layer (flags per job kind, the LUMI python wrapper)."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def load():
    spec = importlib.util.spec_from_file_location("pipeline_under_test", REPO / "scripts" / "pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod      # dataclasses resolve string annotations through sys.modules
    spec.loader.exec_module(mod)
    return mod


class FakeSite:
    name = "fake"
    grader_mode = "serve"

    def __init__(self, cache: Path):
        self.flags = {k: f"--kind={k}" for k in ("train", "gen", "grader", "cpu", "cpu-big", "driver")}
        self.hf_hub_cache = cache


@pytest.fixture
def pl(tmp_path, monkeypatch):
    m = load()
    for name, rel in (("REPO", ""), ("STATE", "state/pipeline"), ("LEDGER", "state/pipeline/ledger.jsonl"),
                      ("GRADER_DIR", "state/graders"), ("LOGS", "runs/logs")):
        monkeypatch.setattr(m, name, tmp_path / rel if rel else tmp_path)
    monkeypatch.delenv("SCISUM_DATA_ROOT", raising=False)
    monkeypatch.delenv("SUM_EVAL_IDS", raising=False)
    (tmp_path / "configs" / "students").mkdir(parents=True)
    (tmp_path / "configs" / "recipes").mkdir(parents=True)
    (tmp_path / "configs" / "students" / "tiny.env").write_text(
        "export STUDENT_TAG=tiny\nexport SCISUM_BASE_MODEL=Org/Tiny-1B\nexport SCISUM_BASE_MODEL_REVISION=abc123\n")
    (tmp_path / "configs" / "recipes" / "sft50k.env").write_text("export SCISUM_N_TRAIN=50000\n")
    (tmp_path / "configs" / "recipes" / "prefcc.env").write_text(
        "export SCISUM_IMP_DIR=improve-${STUDENT_TAG}\nexport SCISUM_N_TRAIN=${SUM_IMP_N:-1500}\n")
    (tmp_path / "data" / "eval").mkdir(parents=True)
    (tmp_path / "data" / "eval" / "eval_ids.json").write_text(json.dumps([f"p{i}" for i in range(4)]))
    m.SITE = FakeSite(tmp_path / "hf")
    return m


def mark_done(m, tag, stage):
    m.write_json(m.marker(tag, stage, "done"), {"t": m.now()})


def fetched(m, root: Path):
    snap = root / "hf" / "models--Org--Tiny-1B" / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")


def finished_run(root: Path, run: str, ckpt: str):
    d = root / "runs" / run
    (d / "ckpt" / ckpt / "adapter").mkdir(parents=True)
    (d / "state.json").write_text(json.dumps({"finished": True, "examples": int(ckpt[2:])}))


def args(**kw):
    base = {"student": None, "dry_run": False, "no_deps": False, "retry_failed": False, "with_report": False,
            "no_anchors": True, "json": False, "force": False, "max_seconds": 0}
    base.update(kw)
    return type("A", (), base)()


# ---- the graph --------------------------------------------------------------------------------------
def test_every_need_exists_and_comes_first():
    m = load()
    seen = {s.name for s in m.GLOBAL_STAGES}
    for st in m.STUDENT_STAGES:
        for need in st.needs:
            assert need in seen, f"{st.name} needs {need}, which is not defined before it"
        seen.add(st.name)
    assert all(s.grader in m.GRADERS for s in m.STUDENT_STAGES if s.kind == "grader")
    assert {s.name for s in m.STUDENT_STAGES if s.chain} == {"sft", "dpo-s1", "dpo-s2"}
    # the evaluation side never grades with the training side's model, and the reverse
    assert {s.grader for s in m.STUDENT_STAGES if s.name.startswith(("sft-grade", "dpo-grade"))} == {m.EVAL_SIDE}
    assert {s.grader for s in m.STUDENT_STAGES if s.name.startswith("cand-")} == {m.TRAIN_SIDE}


def test_run_names_follow_the_contract(pl):
    s = pl.Student("tiny")
    assert (s.sft_run, s.pref_run(1), s.pref_run(2)) == ("tiny-sft50k", "tiny-prefcc", "tiny-prefcc-s2")
    assert s.sft_adapter == "runs/tiny-sft50k/ckpt/ex50000/adapter"
    assert (s.model_tag, s.imp_dir, s.pair_split) == ("tiny-1b", "improve-tiny", "improve-tiny-prefcc")


def test_the_dpo_stage_env_carries_the_pairs_and_the_seed(pl, tmp_path):
    b = tmp_path / "data" / "tiny-1b" / "tokens" / "improve-tiny-prefcc"
    b.mkdir(parents=True)
    (b / "build.json").write_text(json.dumps({"pairs": 1233}))
    env = pl.stage_env(pl.BY_NAME["dpo-s2"], "tiny")
    assert env["SUM_RUN"] == "tiny-prefcc-s2" and env["SCISUM_PREF_ORDER_SEED"] == "1"
    assert env["SCISUM_N_TRAIN"] == "1233" and env["SCISUM_MILESTONES"] == "[1233]" and env["SUM_IMP_N"] == "1233"
    assert env["SCISUM_INIT_ADAPTER"] == "runs/tiny-sft50k/ckpt/ex50000/adapter"
    assert env["SCISUM_TRAIN_SPLIT"] == "improve-tiny-prefcc" and env["SUM_NO_EVAL"] == "1"
    assert pl.stage_env(pl.BY_NAME["dpo-s1"], "tiny")["SCISUM_PREF_ORDER_SEED"] == "0"
    assert pl.stage_env(pl.BY_NAME["sft"], "tiny") == {"SUM_RUN": "tiny-sft50k", "SUM_MILESTONE_EVAL": "1"}


def test_read_env_resolves_references(pl, tmp_path):
    env = pl.read_env(tmp_path / "configs" / "recipes" / "prefcc.env", {"STUDENT_TAG": "tiny"})
    assert env["SCISUM_IMP_DIR"] == "improve-tiny" and env["SCISUM_N_TRAIN"] == "1500"


# ---- status ---------------------------------------------------------------------------------------------
def test_status_from_artifacts_and_the_queue(pl, tmp_path):
    site = pl.SITE
    tags = ["tiny"]
    mark_done(pl, "_global", "prepare")
    fetched(pl, tmp_path)
    (tmp_path / "state").mkdir(exist_ok=True)
    (tmp_path / "state" / "build-tiny-1b.json").write_text("{}")
    mark_done(pl, "tiny", "build")
    q = pl.Queue("101|sum-tiny-sft|RUNNING|1:00:00|node1\n")
    st = lambda name: pl.state_of(pl.BY_NAME[name], "tiny", q, site, tags)["state"]  # noqa: E731
    assert st("fetch") == "done" and st("build") == "done"
    assert st("sft") == "running" and st("sft-eval") == "waiting" and st("pool") == "ready"
    finished_run(tmp_path, "tiny-sft50k", "ex50000")
    q2 = pl.Queue("")
    assert pl.state_of(pl.BY_NAME["sft"], "tiny", q2, site, tags)["state"] == "done"
    assert pl.state_of(pl.BY_NAME["sft-eval"], "tiny", q2, site, tags)["state"] == "ready"


def test_a_failed_last_job_shows_as_failed(pl, tmp_path):
    mark_done(pl, "_global", "prepare")
    fetched(pl, tmp_path)
    (tmp_path / "state" / "build-tiny-1b.json").parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / "state" / "build-tiny-1b.json").write_text("{}")
    mark_done(pl, "tiny", "build")
    pl.ledger("submitted", "tiny", "sft", job_id="42")
    q = pl.Queue("", sacct=lambda jid: {"state": "OUT_OF_MEMORY", "exit": "0:125", "end": "now"} if jid == "42" else {})
    r = pl.state_of(pl.BY_NAME["sft"], "tiny", q, pl.SITE, ["tiny"])
    assert r["state"] == "failed" and "OUT_OF_MEMORY" in r["detail"]


def test_squeue_and_sacct_parsing(pl):
    jobs = pl.parse_squeue("1|a|RUNNING|0:10|n1\n2|a|PENDING|0:00|(Priority)\nbad line\n")
    assert [j["id"] for j in jobs["a"]] == ["1", "2"] and jobs["a"][1]["reason"] == "(Priority)"
    acct = pl.parse_sacct("7|FAILED|1:0|2026-11-17T10:00:00\n8|CANCELLED by 123|0:15|x\n")
    assert acct["7"]["state"] == "FAILED" and acct["8"]["state"] == "CANCELLED"


# ---- submission ---------------------------------------------------------------------------------------------
def test_next_submits_ready_stages_once(pl, tmp_path, monkeypatch):
    sent = []
    monkeypatch.setattr(pl, "submit", lambda site, st, tag, dry, after=None, extra_env=None:
                        sent.append((st.name, tag, tuple(after or ()))) or f"J{len(sent)}")
    monkeypatch.setattr(pl, "students", lambda only=None, with_anchor=False: ["tiny"])
    pl.cmd_next(args(), pl.SITE, pl.Queue(""))
    names = [s[0] for s in sent]
    assert names[:2] == ["prepare", "fetch"], names
    # build waits for prepare and fetch, which were just submitted: it goes in behind them
    build = [s for s in sent if s[0] == "build"]
    assert build and set(build[0][2]) == {"J1", "J2"}
    # the training chain follows build; stages behind a chain wait for the next pass
    assert "sft" in names and "sft-eval" not in names
    # second pass: everything submitted is in the queue, so nothing is submitted again
    sent.clear()
    queued = [(x, "tiny" if pl.BY_NAME[x].scope == "student" else "_global") for x in names]
    queue = "".join(f"{i}|{pl.job_name(pl.BY_NAME[n], t)}|PENDING|0:00|(Dependency)\n" for i, (n, t) in enumerate(queued))
    pl.cmd_next(args(), pl.SITE, pl.Queue(queue))
    assert sent == [], sent


def test_a_grader_job_is_submitted_once_for_ready_grading(pl, tmp_path, monkeypatch):
    sent = []
    monkeypatch.setattr(pl, "submit", lambda site, st, tag, dry, after=None, extra_env=None:
                        sent.append((st.name, st.grader)) or "G1")
    monkeypatch.setattr(pl, "students", lambda only=None, with_anchor=False: ["tiny"])
    mark_done(pl, "_global", "prepare")
    fetched(pl, tmp_path)
    (tmp_path / "state").mkdir(exist_ok=True)
    (tmp_path / "state" / "build-tiny-1b.json").write_text("{}")
    mark_done(pl, "tiny", "build")
    finished_run(tmp_path, "tiny-sft50k", "ex50000")
    ev = tmp_path / "runs" / "tiny-sft50k" / "eval"
    ev.mkdir(parents=True)
    (ev / "test-ex50000-n4.json").write_text("{}")
    pl.cmd_next(args(), pl.SITE, pl.Queue(""))
    assert ("grader-qwen38-27b", "qwen38-27b") in sent
    assert sum(1 for s in sent if s[0].startswith("grader-")) == 1
    sent.clear()
    pl.cmd_next(args(), pl.SITE, pl.Queue("9|sum-grader-qwen38-27b|PENDING|0:00|(Priority)\n"))
    assert not any(s[0].startswith("grader-") for s in sent)


def test_missing_site_values_refuse_to_submit(pl):
    site = FakeSite(Path("/nonexistent"))
    site.flags["cpu"] = "--account=MISSING_SUM_ACCOUNT --partition=small"
    with pytest.raises(SystemExit, match="missing value"):
        pl.submit(site, pl.BY_NAME["prepare"], "_global", dry=True)


# ---- inside a job: the task runner ---------------------------------------------------------------------------
def test_tasks_resume_and_report_incomplete_or_failed(pl, tmp_path, monkeypatch):
    st = pl.BY_NAME["sft-grade"]
    calls = []

    def fake_tasks(stage, tag):
        return [pl.Task("a", "echo '{\"remaining\": 0}'", "remaining0"),
                pl.Task("b", f"echo b >> {tmp_path}/b.log; exit 3")]
    monkeypatch.setattr(pl, "tasks", fake_tasks)
    import time
    assert pl.run_stage_tasks(st, "tiny", time.time() + 3600) == pl.INCOMPLETE
    assert pl.task_marker("tiny", st.name, "a").exists() and not pl.task_marker("tiny", st.name, "b").exists()
    pl.run_stage_tasks(st, "tiny", time.time() + 3600)                 # a is skipped, b runs again
    assert (tmp_path / "b.log").read_text().count("b") == 2
    monkeypatch.setattr(pl, "tasks", lambda stage, tag: [pl.Task("c", "exit 7")])
    assert pl.run_stage_tasks(st, "tiny", time.time() + 3600) == 7
    f = json.loads(pl.marker("tiny", st.name, "failed").read_text())
    assert f["rc"] == 7 and "task c" in f["why"]
    calls.append(1)
    # a task whose output says work remains is not complete, even with rc 0
    monkeypatch.setattr(pl, "tasks", lambda stage, tag: [pl.Task("d", "echo '{\"remaining\": 12}'", "remaining0")])
    assert pl.run_stage_tasks(pl.BY_NAME["dpo-grade-s1"], "tiny", time.time() + 3600) == pl.INCOMPLETE


def test_grading_tasks_use_run_names_and_both_question_sets(pl):
    ts = {t.name: t for t in pl.grade_tasks("tiny-prefcc", "ex1233")}
    assert set(ts) == {"fact-a", "fact-b", "claims", "judge"}
    assert "--systems tiny-prefcc" in ts["fact-a"].cmd and dict(ts["fact-b"].env)["SCISUM_FACT_DIR"] == "factcheck-b"
    assert dict(ts["fact-a"].env)["SCISUM_FACT_DIR"] == "factcheck"
    assert "--n 500" in ts["judge"].cmd and "--run tiny-prefcc --ckpt ex1233" in ts["judge"].cmd


# ---- the site layer -----------------------------------------------------------------------------------------------
def site_flags(name: str, env: dict) -> dict:
    m = load()
    old = dict(os.environ)
    try:
        os.environ.update(env)
        return m.Site(name).flags
    finally:
        os.environ.clear()
        os.environ.update(old)


@pytest.mark.skipif((REPO / "slurm" / "site" / "local.sh").exists(), reason="a local.sh overrides the defaults")
def test_lumi_flags():
    f = site_flags("lumi", {"SUM_ACCOUNT": "project_1"})
    assert "--account=project_1" in f["train"] and "--partition=standard-g" in f["train"]
    assert "--gpus-per-node=8" in f["train"] and "--cpus-per-task=56" in f["train"] and "--mem=480G" in f["train"]
    assert "--partition=small-g" in f["gen"] and "--gpus-per-node=1" in f["gen"]
    assert "--gpus-per-node=4" in f["grader"]
    two = site_flags("lumi", {"SUM_ACCOUNT": "project_1", "SUM_TRAIN_GPUS": "2"})
    assert "--partition=small-g" in two["train"] and "--cpus-per-task=14" in two["train"]
    assert "MISSING_SUM_ACCOUNT" in site_flags("lumi", {"SUM_ACCOUNT": ""})["cpu"]


@pytest.mark.skipif((REPO / "slurm" / "site" / "local.sh").exists(), reason="a local.sh overrides the defaults")
def test_cuda_flags():
    f = site_flags("cuda", {"SUM_GPU_PARTITION": "gpux", "SUM_TRAIN_GPUS": "4", "SUM_GRADER_MODE": "external"})
    assert "--partition=gpux" in f["train"] and "--gres=gpu:4" in f["train"]
    assert "--gres" not in f["grader"], "an external grader endpoint needs a CPU job only"


def test_the_lumi_wrapper_runs_python_directly_inside_the_container(tmp_path):
    stub = tmp_path / "bin"
    stub.mkdir()
    (stub / "python").write_text("#!/usr/bin/env bash\necho \"stub-python $*\"\n")
    (stub / "python").chmod(0o755)
    env = {"PATH": f"{stub}:/usr/bin:/bin", "SINGULARITY_CONTAINER": "/x.sif"}
    r = subprocess.run([str(REPO / "slurm" / "site" / "lumi-python"), "-m", "scisum.cli", "env"],
                       env=env, capture_output=True, text=True, timeout=30)
    assert r.stdout.strip() == "stub-python -m scisum.cli env", r.stderr
    r = subprocess.run([str(REPO / "slurm" / "site" / "lumi-python"), "-c", "1"], env={"PATH": "/usr/bin:/bin"},
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 127 and "SUM_SIF" in r.stderr


# ---- the job scripts ---------------------------------------------------------------------------------------------------
SCRIPTS = sorted((REPO / "slurm").rglob("*.sh")) + sorted((REPO / "slurm").rglob("*.sbatch")) + [REPO / "slurm" / "site" / "lumi-python"]


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: str(p.relative_to(REPO)))
def test_every_script_parses(path):
    r = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_job_bodies_keep_main_on_the_last_line_and_headers_exec_a_body():
    """Bash reads a script while it runs: a body edited in place under a running job breaks it."""
    for body in (REPO / "slurm" / "jobs").glob("*"):
        text = body.read_text().rstrip()
        if body.suffix == ".sbatch" and "exec bash" in text:
            target = text.split("slurm/jobs/")[-1].split('"')[0]
            assert (REPO / "slurm" / "jobs" / target).exists(), f"{body.name} execs a missing {target}"
        else:
            assert text.endswith('main "$@"; exit $?'), f"{body.name} must end by calling main"
    for name in ("train.sbatch", "stage.sbatch", "grader_server.sbatch"):
        assert "#SBATCH --partition" not in (REPO / "slurm" / "jobs" / name).read_text(), "resources come from the site"


def test_pipeline_runs_on_an_old_standard_library_only():
    """The login node of LUMI has no venv: pipeline.py may import nothing outside the standard library at module level."""
    src = (REPO / "scripts" / "pipeline.py").read_text()
    top = [ln for ln in src.splitlines() if ln.startswith(("import ", "from ")) and "__future__" not in ln]
    allowed = {"argparse", "dataclasses", "gzip", "json", "os", "re", "shlex", "shutil", "subprocess", "sys", "time",
               "urllib.request", "pathlib"}
    mods = {ln.split()[1] for ln in top}
    assert mods <= allowed, mods - allowed
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "pipeline.py"), "--help"], capture_output=True, text=True)
    assert r.returncode == 0 and "status" in r.stdout


def test_prepare_seeds_questions_but_no_earlier_grades(pl, tmp_path):
    d = tmp_path / "data"
    (d / "corpus").mkdir(parents=True)
    for f in ("index.jsonl.gz", "splits.json", "bodies.sqlite"):
        (d / "corpus" / f).write_text("x")
    for s in ("factcheck", "factcheck-b"):
        (d / "eval" / s).mkdir(parents=True)
        for f in ("questions.jsonl.gz", "papers.json", "fingerprint", "answers-teacher.jsonl.gz", "grades-teacher.jsonl.gz"):
            (d / "eval" / s / f).write_text("x")
    a = d / "anchors" / "q35prefcc"
    (a / "ckpt" / "ex1233" / "adapter").mkdir(parents=True)
    (a / "eval").mkdir()
    (a / "eval" / "test-ex1233-n4.pred.jsonl.gz").write_text("p")
    (a / "eval" / "test-ex1233-n4.judge.jsonl.gz").write_text("j")
    (d / "anchors" / "improve_c").mkdir()
    assert pl.cmd_prepare_data(args(), None, None) == 0
    fc = tmp_path / "state" / "factcheck"
    assert (fc / "questions.jsonl.gz").exists() and (fc / "fingerprint").exists()
    assert not (fc / "answers-teacher.jsonl.gz").exists() and not (fc / "grades-teacher.jsonl.gz").exists()
    run = tmp_path / "runs" / "q35prefcc"
    assert (run / "ckpt").is_symlink() and (run / "ckpt" / "ex1233" / "adapter").is_dir()
    assert (run / "eval" / "test-ex1233-n4.pred.jsonl.gz").exists()
    assert not (run / "eval" / "test-ex1233-n4.judge.jsonl.gz").exists()          # the new judge writes its own
    assert (run / "eval" / "test-ex1233-n4.judge-hosted.jsonl.gz").exists()
    assert not (tmp_path / "runs" / "improve_c").exists()
    assert pl.cmd_prepare_data(args(), None, None) == 0                          # idempotent


def test_quick_incomplete_jobs_back_off_before_resubmitting():
    """A too-short time limit once produced 20 one-second jobs in a row: a quick stop must wait before the next."""
    body = (REPO / "slurm" / "jobs" / "stage_body.sh").read_text()
    assert "--begin=now+10minutes" in body and "SUM_MAX_RESUBMITS" in body
    assert "--begin=now+15minutes" in (REPO / "slurm" / "jobs" / "grader_body.sh").read_text()
