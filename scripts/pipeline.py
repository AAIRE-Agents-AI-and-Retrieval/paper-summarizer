#!/usr/bin/env python3
"""The pipeline driver: every stage of the C -> prefcc recipe, per student, as Slurm jobs.

    python3 scripts/pipeline.py --site lumi status                  # the dashboard: every student x stage
    python3 scripts/pipeline.py --site lumi status --student k2h-7b # one student, with jobs, times, logs
    python3 scripts/pipeline.py --site lumi next --dry-run          # what would be submitted now
    python3 scripts/pipeline.py --site lumi next                    # submit every runnable stage (idempotent)
    python3 scripts/pipeline.py --site lumi run --student k2h-7b --stage sft   # one stage
    python3 scripts/pipeline.py --site lumi driver                  # a small CPU job that runs `next` every 30 min
    python3 scripts/pipeline.py --site lumi logs --student k2h-7b --stage sft  # path of the newest log
    python3 scripts/pipeline.py --site lumi cancel --student k2h-7b [--stage sft]

Run it on the login node with any Python >= 3.9 and only the standard library (on LUMI: ``module load
cray-python``). Inside jobs it is called as ``pipeline.py exec`` (one stage) and ``pipeline.py work`` (all
pending grading for one grader model); those two run the real work through ``$SCISUM`` / ``$PY``.

State lives in ``state/pipeline/``: ``ledger.jsonl`` (every submission, start, finish, failure, appended),
``<tag>/<stage>.done`` and ``.failed`` (JSON markers), ``<tag>/<stage>.env`` (the exact extra environment a
job sourced), ``<tag>/<stage>/<task>.done`` (the sub-tasks of grading stages). A stage counts as done when
its artifacts say so (an adapter, a score file, a token directory), not only when a marker exists, so a job
that died after writing its outputs is still seen as done, and deleting a marker never loses work.

The stages, in order, for each student <tag> (runs: <tag>-sft50k, <tag>-prefcc, <tag>-prefcc-s2):

    prepare (global) -> fetch -> build -> sft -> sft-eval -> sft-grade
                                 pool --> sample -> cand-grade -> cand-claims -> pairs
                                 pairs -> dpo-s1 -> dpo-eval-s1 -> dpo-grade-s1
                                 pairs -> dpo-s2 -> dpo-eval-s2 -> dpo-grade-s2
                                 sft + dpo-s1 -> bench
    anchor-fetch, anchor-grade, anchor-bench (global): Student C and prefcc under the new graders
    report (global): collect every result into tables and the Pareto plot

Grading stages need a served grader model; they are not jobs of their own but are worked through by one
grader job per model (``slurm/jobs/grader_server.sbatch``), which `next` submits when there is work for it.
"""

from __future__ import annotations

import argparse
import dataclasses
import gzip
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
STATE = REPO / "state" / "pipeline"
LEDGER = STATE / "ledger.jsonl"
GRADER_DIR = REPO / "state" / "graders"
LOGS = REPO / "runs" / "logs"

EVAL_SIDE = "qwen38-27b"       # answers and grades the evaluation's fact test, checks claims, judges
TRAIN_SIDE = "gemma4-31b-it"   # answers and grades the training reward's questions, checks candidates' claims
GRADERS = (EVAL_SIDE, TRAIN_SIDE)
SFT, PREF = "sft50k", "prefcc"
ANCHOR_TAGS = ("qwen35-9b",)   # Student C's base model: a profile for benchmarking, never trained here
ANCHOR_RUNS = {"q35": "ex20000", "q35prefcc": "ex1233", "q35prefccs2": "ex1233"}   # run -> final adapter
ANCHOR_LABELS = {"q35": "C", "q35prefcc": "C_prefcc", "q35prefccs2": "C_prefccs2"}  # fact/claim labels (scisum.factcheck)
INCOMPLETE = 3                 # exit code of a task that stopped on its time limit with work left
NOT_READY = 10                 # exit code of `exec` when a stage's inputs are not done yet

# What `prepare` copies from the packaged data into state/: the two fact-test question sets, and nothing graded.
# The packaged answers and grades of the earlier systems (made by the earlier, hosted graders) stay in data/eval
# for reference; copying them would let the new graders skip the teacher, so the anchors would be graded by a
# mix of grader versions. (source dir under data/, destination under the repo, files)
SEED_FILES = ("questions.jsonl.gz", "questions-failed.jsonl.gz", "papers.json", "fingerprint")
SEED = (("eval/factcheck", "state/factcheck"),        # set A (299 papers)
        ("eval/factcheck-b", "state/factcheck-b"))    # set B (600 papers)


# ---- paths and environment -----------------------------------------------------------------------
def data_root() -> Path:
    return Path(os.environ.get("SCISUM_DATA_ROOT") or REPO / "data")


def eval_ids_path() -> Path:
    return Path(os.environ.get("SUM_EVAL_IDS") or data_root() / "eval" / "eval_ids.json")


def n_eval_papers() -> int:
    p = eval_ids_path()
    return len(json.loads(p.read_text())) if p.exists() else 0


def read_env(path: Path, extra: dict | None = None) -> dict:
    """The variables a bash env file exports, as bash itself sees them (variables may refer to each other)."""
    if not path.exists():
        return {}
    base = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "REPO": str(REPO), **(extra or {})}
    r = subprocess.run(["bash", "-c", 'set -a; source "$1" >/dev/null 2>&1; env -0', "_", str(path)],
                       env=base, capture_output=True, text=True, timeout=60)
    out = {}
    for item in r.stdout.split("\0"):
        if "=" in item:
            k, v = item.split("=", 1)
            if k not in base or base.get(k) != v:
                out[k] = v
    out.pop("_", None)
    out.pop("SHLVL", None)
    out.pop("PWD", None)
    return out


def students(only: list[str] | None = None, with_anchor: bool = False) -> list[str]:
    tags = sorted(p.stem for p in (REPO / "configs" / "students").glob("*.env"))
    if not with_anchor:
        tags = [t for t in tags if t not in ANCHOR_TAGS]
    if only:
        unknown = [t for t in only if t not in tags and t not in ANCHOR_TAGS]
        if unknown:
            raise SystemExit(f"unknown student(s) {unknown}; profiles: configs/students/*.env -> {tags}")
        tags = [t for t in tags if t in only]
    return tags


class Site:
    """What slurm/site/<name>.sh says: sbatch flags per job kind, and a few paths."""

    KINDS = ("train", "gen", "grader", "cpu", "cpu-big", "driver")

    def __init__(self, name: str):
        self.name = name
        f = REPO / "slurm" / "site" / f"{name}.sh"
        if not f.exists():
            raise SystemExit(f"no site file {f.relative_to(REPO)}; sites: "
                             f"{[p.stem for p in (REPO / 'slurm' / 'site').glob('*.sh') if p.stem != 'local']}")
        script = ('set -e; [ -f slurm/site/local.sh ] && source slurm/site/local.sh; source "slurm/site/$1.sh"; '
                  'printf "DATA_ROOT=%s\\n" "${SCISUM_DATA_ROOT:-}"; printf "HF_HUB_CACHE=%s\\n" "${HF_HUB_CACHE:-}"; '
                  'printf "GRADER_MODE=%s\\n" "${SUM_GRADER_MODE:-serve}"; '
                  'for k in ' + " ".join(self.KINDS) + '; do printf "FLAGS_%s=%s\\n" "$k" "$(site_sbatch_flags "$k")"; done')
        r = subprocess.run(["bash", "-c", script, "_", name], cwd=REPO, capture_output=True, text=True,
                           env={**os.environ, "REPO": str(REPO)}, timeout=60)
        if r.returncode != 0:
            raise SystemExit(f"slurm/site/{name}.sh failed:\n{r.stderr.strip()}")
        vals = dict(line.split("=", 1) for line in r.stdout.splitlines() if "=" in line)
        self.flags = {k: vals.get(f"FLAGS_{k}", "") for k in self.KINDS}
        self.grader_mode = vals.get("GRADER_MODE") or "serve"
        if vals.get("DATA_ROOT"):
            os.environ.setdefault("SCISUM_DATA_ROOT", vals["DATA_ROOT"])
        self.hf_hub_cache = Path(vals.get("HF_HUB_CACHE") or REPO / ".cache" / "hf" / "hub")


# ---- the stage graph -------------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class Stage:
    name: str
    kind: str                     # sbatch resource kind (Site.KINDS); "grader" stages run inside a grader job
    needs: tuple = ()
    scope: str = "student"        # "student" or "global"
    grader: str = ""              # the served model a grading stage needs
    recipe: str = ""              # configs/recipes/<recipe>.env sourced by the job
    chain: bool = False           # a training chain: self-resubmitting, job ids change
    abbr: str = ""


GLOBAL_STAGES = (
    Stage("prepare", "cpu", scope="global", abbr="prep"),
    Stage("anchor-fetch", "cpu", scope="global", abbr="afetch"),
    Stage("anchor-grade", "grader", ("prepare",), scope="global", grader=EVAL_SIDE, abbr="agrade"),
    Stage("anchor-bench", "gen", ("prepare", "anchor-fetch"), scope="global", abbr="abench"),
    Stage("report", "cpu", ("@students",), scope="global", abbr="report"),
)
STUDENT_STAGES = (
    Stage("fetch", "cpu", abbr="fetch"),
    Stage("build", "cpu-big", ("prepare", "fetch"), recipe=SFT, abbr="build"),
    Stage("sft", "train", ("build",), recipe=SFT, chain=True, abbr="sft"),
    Stage("sft-eval", "gen", ("sft",), recipe=SFT, abbr="s.eval"),
    Stage("sft-grade", "grader", ("sft-eval",), grader=EVAL_SIDE, recipe=SFT, abbr="s.grade"),
    Stage("pool", "cpu", ("prepare", "fetch"), recipe=PREF, abbr="pool"),
    Stage("sample", "gen", ("sft", "pool"), recipe=PREF, abbr="sample"),
    Stage("cand-grade", "grader", ("sample",), grader=TRAIN_SIDE, recipe=PREF, abbr="c.grade"),
    Stage("cand-claims", "grader", ("cand-grade",), grader=TRAIN_SIDE, recipe=PREF, abbr="c.claim"),
    Stage("pairs", "cpu", ("cand-claims",), recipe=PREF, abbr="pairs"),
    Stage("dpo-s1", "train", ("pairs",), recipe=PREF, chain=True, abbr="dpo1"),
    Stage("dpo-s2", "train", ("pairs",), recipe=PREF, chain=True, abbr="dpo2"),
    Stage("dpo-eval-s1", "gen", ("dpo-s1",), recipe=PREF, abbr="d.eval1"),
    Stage("dpo-eval-s2", "gen", ("dpo-s2",), recipe=PREF, abbr="d.eval2"),
    Stage("dpo-grade-s1", "grader", ("dpo-eval-s1",), grader=EVAL_SIDE, recipe=PREF, abbr="d.grd1"),
    Stage("dpo-grade-s2", "grader", ("dpo-eval-s2",), grader=EVAL_SIDE, recipe=PREF, abbr="d.grd2"),
    Stage("bench", "gen", ("sft", "dpo-s1"), abbr="bench"),
)
BY_NAME = {s.name: s for s in (*GLOBAL_STAGES, *STUDENT_STAGES)}
REPORT_NEEDS = ("sft-grade", "dpo-grade-s1", "dpo-grade-s2", "bench")


def stage(name: str) -> Stage:
    if name not in BY_NAME:
        raise SystemExit(f"unknown stage {name!r}; stages: {list(BY_NAME)}")
    return BY_NAME[name]


# Every student run that exists (SFT and both preference seeds), for the cross-student reports.
REPORT_RUNS = "$(cd runs && ls -d *-sft50k *-prefcc *-prefcc-s2 2>/dev/null)"


@dataclasses.dataclass(frozen=True)
class Task:
    """One command of a stage. ``complete`` decides from its output whether the work is finished."""
    name: str
    cmd: str
    complete: str = "rc0"         # "rc0" | "remaining0" (JSON '"remaining": 0' in the output) | "check"
    env: tuple = ()               # extra (key, value) pairs


class Student:
    """One student's profile and the facts the stages need (model tag, step counts, run names)."""

    def __init__(self, tag: str):
        self.tag = tag
        self.env = read_env(REPO / "configs" / "students" / f"{tag}.env")
        self.sft_env = read_env(REPO / "configs" / "recipes" / f"{SFT}.env")
        self.base_model = self.env.get("SCISUM_BASE_MODEL", "")
        self.revision = self.env.get("SCISUM_BASE_MODEL_REVISION", "")
        self.model_tag = self.base_model.split("/")[-1].lower() if self.base_model else tag
        self.n_sft = int(self.sft_env.get("SCISUM_N_TRAIN", "50000"))
        self.serve_model = self.env.get("SCISUM_SERVE_MODEL", "")   # a text-only copy vLLM serves (Qwen3.5)

    def fetched(self, site: Site | None) -> bool:
        if not hf_snapshot(site, self.base_model, self.revision):
            return False
        return not self.serve_model or (REPO / self.serve_model / "config.json").exists() \
            or Path(self.serve_model, "config.json").exists()

    # run names (contract 3)
    @property
    def sft_run(self) -> str:
        return f"{self.tag}-sft50k"

    def pref_run(self, seed: int) -> str:
        return f"{self.tag}-prefcc" + ("-s2" if seed == 2 else "")

    @property
    def sft_ckpt(self) -> str:
        return f"ex{self.n_sft}"

    @property
    def sft_adapter(self) -> str:
        return f"runs/{self.sft_run}/ckpt/{self.sft_ckpt}/adapter"

    @property
    def imp_dir(self) -> str:
        return f"improve-{self.tag}"

    @property
    def pair_split(self) -> str:
        return f"{self.imp_dir}-{PREF}"

    def pairs(self) -> int:
        """The number of preference pairs the `pairs` stage built (0 before)."""
        p = data_root() / self.model_tag / "tokens" / self.pair_split / "build.json"
        try:
            return int(json.loads(p.read_text()).get("pairs") or 0)
        except (OSError, ValueError):
            return 0

    def pref_ckpt(self) -> str:
        return f"ex{self.pairs()}"


# ---- markers and the ledger ------------------------------------------------------------------------
def marker(tag: str, name: str, suffix: str) -> Path:
    return STATE / tag / f"{name}.{suffix}"


def task_marker(tag: str, name: str, task: str) -> Path:
    return STATE / tag / name / f"{task}.done"


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


def ledger(event: str, tag: str = "", stage_name: str = "", **kw) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    rec = {"t": now(), "event": event, "student": tag, "stage": stage_name,
           "job": os.environ.get("SLURM_JOB_ID", ""), **kw}
    with LEDGER.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def last_submission(tag: str, name: str) -> dict:
    if not LEDGER.exists():
        return {}
    last = {}
    for line in LEDGER.read_text().splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("student") == tag and r.get("stage") == name and r.get("event") == "submitted":
            last = r
    return last


def gz_lines(path: Path) -> int:
    try:
        with gzip.open(path, "rt") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


# ---- what "done" means, stage by stage -------------------------------------------------------------
def scope_tag(st: Stage, tag: str) -> str:
    return "_global" if st.scope == "global" else tag


def eval_file(run: str, ckpt: str) -> Path:
    return REPO / "runs" / run / "eval" / f"test-{ckpt}-n{n_eval_papers()}.json"


def run_finished(run: str, ckpt: str) -> bool:
    p = REPO / "runs" / run / "state.json"
    try:
        fin = bool(json.loads(p.read_text()).get("finished"))
    except (OSError, ValueError):
        fin = False
    return fin and (REPO / "runs" / run / "ckpt" / ckpt / "adapter").is_dir()


def hf_snapshot(site: Site | None, repo_id: str, revision: str) -> bool:
    if not repo_id:
        return False
    root = (site.hf_hub_cache if site else REPO / ".cache" / "hf" / "hub") / ("models--" + repo_id.replace("/", "--"))
    if revision:
        return (root / "snapshots" / revision / "config.json").exists()
    return any((root / "snapshots").glob("*/config.json"))


def check(st: Stage, tag: str, site: Site | None = None) -> bool:
    """Artifact-based: is this stage's output there? Markers only for work whose output is a ledger."""
    t = scope_tag(st, tag)
    if marker(t, st.name, "done").exists() and st.name in ("prepare", "pool", "cand-grade", "cand-claims", "report",
                                                           "anchor-grade", "sft-grade", "dpo-grade-s1",
                                                           "dpo-grade-s2", "fetch", "anchor-fetch"):
        return True
    if st.scope == "global":
        if st.name == "prepare":
            return marker(t, st.name, "done").exists()
        if st.name == "anchor-fetch":
            return Student(ANCHOR_TAGS[0]).fetched(site)
        if st.name == "anchor-bench":
            return all((REPO / "runs" / r / "bench" / f"{c}.json").exists() for r, c in ANCHOR_RUNS.items()
                       if r != "q35prefccs2")
        return False
    s = Student(tag)
    n = st.name
    if n == "fetch":
        return s.fetched(site)
    if n == "build":
        return (REPO / "state" / f"build-{s.model_tag}.json").exists() and marker(tag, n, "done").exists()
    if n == "sft":
        return run_finished(s.sft_run, s.sft_ckpt)
    if n == "sft-eval":
        return eval_file(s.sft_run, s.sft_ckpt).exists()
    if n == "pool":
        return (REPO / "state" / s.imp_dir / "pool.jsonl.gz").exists()
    if n == "sample":
        d = REPO / "state" / s.imp_dir
        want = gz_lines(d / "pool.jsonl.gz")      # the student's own pool: SCISUM_IMP_POOL papers
        return want > 0 and gz_lines(d / "samples.jsonl.gz") >= want
    if n == "pairs":
        return s.pairs() > 0
    if n in ("dpo-s1", "dpo-s2"):
        return s.pairs() > 0 and run_finished(s.pref_run(int(n[-1])), s.pref_ckpt())
    if n in ("dpo-eval-s1", "dpo-eval-s2"):
        return s.pairs() > 0 and eval_file(s.pref_run(int(n[-1])), s.pref_ckpt()).exists()
    if n == "bench":
        return ((REPO / "runs" / s.sft_run / "bench" / f"{s.sft_ckpt}.json").exists()
                and s.pairs() > 0 and (REPO / "runs" / s.pref_run(1) / "bench" / f"{s.pref_ckpt()}.json").exists())
    return False   # grading stages: their markers, checked above


# ---- the commands of each stage ----------------------------------------------------------------------
# Qwen3.5 ships as a vision-language checkpoint; vLLM must serve a text-only copy or the adapter matches nothing
# (configs/students/qwen35-*.env set SCISUM_SERVE_MODEL). The anchor stage is global, so it sources the profile.
TEXT_ONLY_CMD = ('if [ -n "${SUM_PROFILE:-}" ]; then source "configs/students/$SUM_PROFILE.env"; fi; '
                 'if [ -n "${SCISUM_SERVE_MODEL:-}" ] && [ ! -e "$SCISUM_SERVE_MODEL/config.json" ]; then '
                 '$PY scripts/make_text_only.py; else echo "no text-only copy needed or present already"; fi')


def grade_tasks(run: str, ckpt: str, label: str = "") -> list[Task]:
    """The evaluation-side grading of one system: fact test A and B, claim check, judge. Labels are run names
    (scisum.factcheck resolves them), except the anchors of the earlier work, which keep their old labels."""
    ids = eval_ids_path()
    label = label or run
    return [
        Task("fact-a", f"$SCISUM -v factcheck run --systems {label} --max-seconds {{T}}", "remaining0",
             (("SCISUM_FACT_DIR", "factcheck"),)),
        Task("fact-b", f"$SCISUM -v factcheck run --systems {label} --max-seconds {{T}}", "remaining0",
             (("SCISUM_FACT_DIR", "factcheck-b"), ("SCISUM_FACT_N_PAPERS", "600"),
              ("SCISUM_FACT_EXCLUDE_DIR", "factcheck"))),
        Task("claims", f"$SCISUM -v claimcheck run --systems {label} --n 300 --max-seconds {{T}}", "remaining0"),
        Task("judge", f"$SCISUM -v eval judge --run {run} --ckpt {ckpt} --split test --subset ids:{ids} --n 500 "
                      f"--max-seconds {{T}}", "check"),
    ]


def eval_tasks(run: str, ckpt: str) -> list[Task]:
    ids = eval_ids_path()
    sub = f"--run {run} --ckpt {ckpt} --split test --subset ids:{ids}"
    return [Task("generate", f"$SCISUM -v eval generate {sub} --max-seconds {{T}}"),
            Task("score", f"$SCISUM -v eval score {sub}")]


def bench_task(name: str, tag: str, run: str, ckpt: str) -> Task:
    return Task(name, f"$PY scripts/analysis/bench_inference.py --student-env configs/students/{tag}.env "
                      f"--adapter runs/{run}/ckpt/{ckpt}/adapter --out runs/{run}/bench/{ckpt}.json")


def tasks(st: Stage, tag: str) -> list[Task]:
    n = st.name
    if n == "prepare":
        return [Task("seed", "$PY scripts/pipeline.py prepare-data")]
    if n == "anchor-fetch":
        return [Task("download", f"$PY scripts/pipeline.py fetch-model --student {ANCHOR_TAGS[0]}"),
                Task("text-only", TEXT_ONLY_CMD, env=(("SUM_PROFILE", ANCHOR_TAGS[0]),))]
    if n == "anchor-grade":
        # the teacher, Student C and prefcc's two seeds under the new graders. C was generated on the whole
        # 15,000-paper test split (which contains the fact-test papers) but not on the 1,246-paper evaluation
        # list, so it has no judge task; prefcc's runs have both.
        out = [Task("teacher-" + t.name, t.cmd, t.complete, t.env)
               for t in grade_tasks("q35", ANCHOR_RUNS["q35"], label="teacher") if t.name != "judge"]
        for run, ckpt in ANCHOR_RUNS.items():
            out += [Task(f"{run}-{t.name}", t.cmd, t.complete, t.env) for t in grade_tasks(run, ckpt, ANCHOR_LABELS[run])
                    if t.name != "judge" or run != "q35"]
        return out
    if n == "anchor-bench":
        a = ANCHOR_TAGS[0]
        return [bench_task("bench-c", a, "q35", ANCHOR_RUNS["q35"]),
                bench_task("bench-prefcc", a, "q35prefcc", ANCHOR_RUNS["q35prefcc"])]
    if n == "report":
        return [Task("collect", "$PY scripts/analysis/collect_results.py --out-dir reports/analysis"),
                Task("pareto", "$PY scripts/analysis/pareto.py reports/analysis/results.json --out-dir reports/analysis/pareto"),
                Task("resources", f"$PY scripts/analysis/train_resources.py {REPORT_RUNS} "
                                  "--out-json reports/analysis/train_resources.json --out-md reports/analysis/train_resources.md"),
                Task("curves", f"$PY scripts/analysis/learning_curves.py {REPORT_RUNS} --out-dir reports/analysis/curves")]
    s = Student(tag)
    if n == "fetch":
        return [Task("download", f"$PY scripts/pipeline.py fetch-model --student {tag}"),
                Task("text-only", TEXT_ONLY_CMD)]
    if n == "build":
        return [Task(f"build-{sp}", f"$SCISUM -v data build --split {sp} --nproc ${{SLURM_CPUS_PER_TASK:-8}}")
                for sp in ("train", "val", "test")] + [
                Task("stats", f"$SCISUM data stats --out state/build-{s.model_tag}.json"),
                Task("verify", "$SCISUM data verify")]
    if n == "sft-eval":
        return eval_tasks(s.sft_run, s.sft_ckpt)
    if n == "sft-grade":
        return grade_tasks(s.sft_run, s.sft_ckpt)
    if n == "pool":
        # the student's own pool (prompts in its chat template and budget, same papers in the same order);
        # the shared questions and reference verdicts are copied in from data/improve_pool by scisum itself
        return [Task("pool", "$SCISUM -v improve pool")]
    if n == "sample":
        return [Task("sample", "$SCISUM -v improve sample --max-seconds {T}")]
    if n == "cand-grade":
        return [Task("reward", "$SCISUM -v improve reward --max-seconds {T} --concurrency ${SUM_IMP_CONCURRENCY:-24}")]
    if n == "cand-claims":
        return [Task("select", "$SCISUM improve select"),
                Task("claims", "$SCISUM -v improve claims --max-seconds {T}")]
    if n == "pairs":
        return [Task("build", f"$SCISUM -v improve build --arm {PREF}")]
    if n in ("dpo-eval-s1", "dpo-eval-s2"):
        return eval_tasks(s.pref_run(int(n[-1])), s.pref_ckpt())
    if n in ("dpo-grade-s1", "dpo-grade-s2"):
        return grade_tasks(s.pref_run(int(n[-1])), s.pref_ckpt())
    if n == "bench":
        return [bench_task("bench-sft", tag, s.sft_run, s.sft_ckpt),
                bench_task("bench-prefcc", tag, s.pref_run(1), s.pref_ckpt())]
    return []   # training stages run slurm/jobs/train_body.sh, not tasks


def task_done(st: Stage, tag: str, t: Task) -> bool:
    if task_marker(scope_tag(st, tag), st.name, t.name).exists():
        return True
    if t.name.endswith("judge") and t.complete == "check":
        run, ckpt = judge_target(st, tag, t)
        f = REPO / "runs" / run / "eval" / f"test-{ckpt}-n{n_eval_papers()}.judge.jsonl.gz"
        return gz_lines(f) >= min(500, n_eval_papers() or 500)
    return False


def judge_target(st: Stage, tag: str, t: Task) -> tuple[str, str]:
    m = re.search(r"--run (\S+) --ckpt (\S+)", t.cmd)
    return (m.group(1), m.group(2)) if m else ("", "")


def stage_env(st: Stage, tag: str) -> dict:
    """The extra environment of one stage's job, on top of the student's profile and the recipe."""
    if st.scope == "global":
        return {}
    s = Student(tag)
    n = st.name
    env: dict = {}
    if n in ("sft", "sft-eval", "sft-grade"):
        env["SUM_RUN"] = s.sft_run
    if n in ("pool", "sample", "cand-grade", "cand-claims", "pairs", "dpo-s1", "dpo-s2"):
        env["SCISUM_IMP_DIR"] = s.imp_dir
    if n in ("sample", "dpo-s1", "dpo-s2"):
        env["SCISUM_INIT_ADAPTER"] = s.sft_adapter
    if n in ("dpo-s1", "dpo-s2", "dpo-eval-s1", "dpo-eval-s2", "dpo-grade-s1", "dpo-grade-s2"):
        seed = int(n[-1])
        env["SUM_RUN"] = s.pref_run(seed)
    if n in ("dpo-s1", "dpo-s2"):
        pairs = s.pairs()
        env.update({"SCISUM_OBJECTIVE": "preference", "SCISUM_TRAIN_SPLIT": s.pair_split, "SUM_IMP_N": str(pairs),
                    "SCISUM_N_TRAIN": str(pairs), "SCISUM_MILESTONES": f"[{pairs}]",
                    "SCISUM_PREF_ORDER_SEED": "0" if n == "dpo-s1" else "1", "SUM_NO_EVAL": "1"})
    if n == "sft":
        env["SUM_MILESTONE_EVAL"] = "1"
    return env


# ---- Slurm -------------------------------------------------------------------------------------------
def job_name(st: Stage, tag: str) -> str:
    if st.kind == "grader":
        return f"sum-grader-{st.grader}"
    return f"sum-{st.name}" if st.scope == "global" else f"sum-{tag}-{st.name}"


def run_cmd(cmd: list[str], timeout: int = 60) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def parse_squeue(text: str) -> dict:
    """``squeue -h -o '%i|%j|%T|%M|%R'`` -> {job name: [{"id", "state", "time", "reason"}, ...]}."""
    jobs: dict = {}
    for line in text.splitlines():
        parts = line.strip().split("|")
        if len(parts) < 3:
            continue
        jid, name, state = parts[0], parts[1], parts[2]
        jobs.setdefault(name, []).append({"id": jid, "state": state, "time": parts[3] if len(parts) > 3 else "",
                                          "reason": parts[4] if len(parts) > 4 else ""})
    return jobs


def parse_sacct(text: str) -> dict:
    """``sacct -n -P -X -o JobID,State,ExitCode,End`` -> {job id: {"state", "exit", "end"}}."""
    out = {}
    for line in text.splitlines():
        p = line.strip().split("|")
        if len(p) >= 2 and p[0]:
            out[p[0]] = {"state": p[1].split()[0] if p[1] else "", "exit": p[2] if len(p) > 2 else "",
                         "end": p[3] if len(p) > 3 else ""}
    return out


class Queue:
    """A snapshot of the user's jobs (squeue), and sacct on demand. Injectable for tests."""

    def __init__(self, squeue_text: str | None = None, sacct=None):
        if squeue_text is None:
            squeue_text = run_cmd(["squeue", "--me", "-h", "-o", "%i|%j|%T|%M|%R"])
        self.jobs = parse_squeue(squeue_text)
        self._sacct = sacct

    def active(self, name: str) -> list[dict]:
        return self.jobs.get(name, [])

    def sacct(self, job_id: str) -> dict:
        if not job_id:
            return {}
        if self._sacct is not None:
            return self._sacct(job_id)
        return parse_sacct(run_cmd(["sacct", "-n", "-P", "-X", "-j", job_id, "-o", "JobID,State,ExitCode,End"])).get(job_id, {})


# ---- status --------------------------------------------------------------------------------------------
def needs_of(st: Stage, tag: str, tags: list[str]) -> list[tuple[Stage, str]]:
    out = []
    for need in st.needs:
        if need == "@students":
            out += [(BY_NAME[r], t) for t in tags for r in REPORT_NEEDS]
        else:
            ns = BY_NAME[need]
            out.append((ns, "_global" if ns.scope == "global" else tag))
    return out


def state_of(st: Stage, tag: str, q: Queue, site: Site | None, tags: list[str], memo: dict | None = None) -> dict:
    """{"state": done|running|pending|failed|ready|waiting, "job", "since", "detail"} for one stage."""
    memo = {} if memo is None else memo
    key = (st.name, tag)
    if key in memo:
        return memo[key]
    t = scope_tag(st, tag)
    res: dict = {"state": "waiting", "job": "", "since": "", "detail": ""}
    if check(st, tag, site):
        m = marker(t, st.name, "done")
        res.update(state="done", since=json.loads(m.read_text()).get("t", "") if m.exists() else "")
        memo[key] = res
        return res
    deps = [state_of(ns, nt, q, site, tags, memo)["state"] for ns, nt in needs_of(st, tag, tags)]
    ready = all(d == "done" for d in deps)
    active = q.active(job_name(st, tag))
    if active and (st.kind != "grader" or ready):
        j = active[0]
        res.update(state="running" if j["state"] == "RUNNING" else "pending", job=j["id"],
                   detail=(j["time"] if j["state"] == "RUNNING" else j["reason"]))
        if st.kind == "grader":
            res["detail"] = f"in grader job ({st.grader})"
        memo[key] = res
        return res
    fail = marker(t, st.name, "failed")
    sub = last_submission(t, st.name)
    if fail.exists():
        f = json.loads(fail.read_text())
        if not sub or f.get("t", "") >= sub.get("t", ""):
            res.update(state="failed", job=f.get("job", ""), since=f.get("t", ""), detail=f.get("why", ""))
            memo[key] = res
            return res
    if sub and st.kind != "grader":
        acct = q.sacct(sub.get("job_id", ""))
        if acct.get("state") in ("FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "CANCELLED", "PREEMPTED") and not active:
            res.update(state="failed", job=sub.get("job_id", ""), since=acct.get("end", ""),
                       detail=f"last job {acct.get('state')} {acct.get('exit', '')}".strip())
            memo[key] = res
            return res
    if ready:
        res["state"] = "ready"
    memo[key] = res
    return res


def progress(st: Stage, tag: str) -> str:
    """A short progress note for the detail view (papers trained, tasks done, samples drawn)."""
    if st.scope == "global" and st.name != "anchor-grade":
        return ""
    if st.kind == "grader" or st.name in ("sft-eval", "dpo-eval-s1", "dpo-eval-s2", "build", "cand-claims"):
        ts = tasks(st, tag)
        return f"{sum(task_done(st, tag, x) for x in ts)}/{len(ts)} tasks" if ts else ""
    if st.scope == "global":
        return ""
    s = Student(tag)
    if st.name in ("sft", "dpo-s1", "dpo-s2"):
        run = s.sft_run if st.name == "sft" else s.pref_run(int(st.name[-1]))
        p = REPO / "runs" / run / "state.json"
        try:
            ex = json.loads(p.read_text()).get("examples", 0)
        except (OSError, ValueError):
            return ""
        total = s.n_sft if st.name == "sft" else s.pairs()
        return f"{ex:,}/{total:,} papers" if st.name == "sft" else f"{ex:,}/{total:,} pairs"
    if st.name == "sample":
        d = REPO / "state" / s.imp_dir
        return f"{gz_lines(d / 'samples.jsonl.gz')}/{gz_lines(d / 'pool.jsonl.gz')} papers"
    if st.name == "pairs" and s.pairs():
        return f"{s.pairs()} pairs"
    return ""


SYMBOL = {"done": "done", "running": "RUN", "pending": "PEND", "failed": "FAIL", "ready": "ready", "waiting": "."}


def cmd_status(a, site: Site | None, q: Queue) -> int:
    tags = students(a.student)
    memo: dict = {}
    if a.student and len(a.student) == 1:
        tag = a.student[0]
        rows = [(st, tag) for st in STUDENT_STAGES]
    else:
        rows = []
    if rows:
        print(f"student {tag}  (runs {Student(tag).sft_run}, {Student(tag).pref_run(1)}, {Student(tag).pref_run(2)})")
        print(f"{'stage':14s} {'state':8s} {'job':>10s}  {'since/elapsed':22s} detail")
        for st, t in rows:
            r = state_of(st, t, q, site, tags, memo)
            note = " ".join(x for x in (r["detail"], progress(st, t)) if x)
            print(f"{st.name:14s} {r['state']:8s} {r['job']:>10s}  {r['since'][:22]:22s} {note}")
        return 0
    if a.json:
        out = {"global": {st.name: state_of(st, "_global", q, site, tags, memo) for st in GLOBAL_STAGES}}
        for t in tags:
            out[t] = {st.name: state_of(st, t, q, site, tags, memo) for st in STUDENT_STAGES}
        print(json.dumps(out, indent=1))
        return 0
    w = max([len(t) for t in tags] + [8])
    print("global: " + "  ".join(f"{st.name}={SYMBOL[state_of(st, '_global', q, site, tags, memo)['state']]}"
                                  for st in GLOBAL_STAGES))
    head = " ".join(f"{st.abbr:>7s}" for st in STUDENT_STAGES)
    print(f"{'student':{w}s} {head}")
    for t in tags:
        cells = []
        for st in STUDENT_STAGES:
            r = state_of(st, t, q, site, tags, memo)
            c = SYMBOL[r["state"]]
            if r["state"] in ("running", "pending") and r["job"] and st.kind != "grader":
                c = f"{c}{r['job'][-4:]}" if len(c) + 4 <= 7 else c
            cells.append(f"{c:>7s}")
        print(f"{t:{w}s} {' '.join(cells)}")
    print("legend: done | RUN/PEND (job) | FAIL | ready = inputs done, not submitted | . = waiting for inputs")
    for model in GRADERS:
        g = q.active(f"sum-grader-{model}")
        info = grader_info(model)
        print(f"grader {model}: " + (", ".join(f"{j['state']} {j['id']}" for j in g) or "no job")
              + (f"; serving at {info.get('base_url')} since {info.get('started', '')}" if info else ""))
    return 0


# ---- submission ------------------------------------------------------------------------------------------
_DRY_N = 0
JOB_SCRIPTS = {"train": "slurm/jobs/train.sbatch", "grader": "slurm/jobs/grader_server.sbatch",
               "driver": "slurm/jobs/driver.sbatch"}


def write_stage_env(st: Stage, tag: str) -> Path:
    path = STATE / scope_tag(st, tag) / f"{st.name}.env"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# written by scripts/pipeline.py at {now()}: the extra environment of stage {st.name}"]
    lines += [f"export {k}={shlex.quote(v)}" for k, v in sorted(stage_env(st, tag).items())]
    path.write_text("\n".join(lines) + "\n")
    return path


CONTEXT_VARS = ("SUM_STUDENT", "SUM_STAGE", "SUM_RECIPE", "SUM_RUN", "SUM_IMP_N", "SUM_STAGE_ENV", "SUM_GRADER", "SUM_SBATCH_FLAGS",
                "SUM_JOB_KIND", "SUM_NO_EVAL", "SUM_MILESTONE_EVAL", "SUM_DRIVER_EVERY_MIN", "SUM_DRIVER_ARGS",
                "SUM_NO_CHAIN", "SUM_HF_OFFLINE")
KEEP_SCISUM = ("SCISUM_DATA_ROOT", "SCISUM_GRADER_ENDPOINTS", "SCISUM_JUDGE_BASE")


def clean_env() -> dict:
    """The submitting environment without leftovers of another student or stage.

    Student and recipe settings (SCISUM_*) come from the profile files inside the job, never from the shell
    that submitted it; site overrides (SUM_ACCOUNT, SUM_TRAIN_GPUS, ...) are kept, though slurm/site/local.sh
    is the better place for them."""
    return {k: v for k, v in os.environ.items()
            if not (k.startswith("SCISUM_") and k not in KEEP_SCISUM) and k not in CONTEXT_VARS}


def submit(site: Site, st: Stage, tag: str, dry: bool, after: list[str] | None = None,
           extra_env: dict | None = None) -> str:
    kind = "grader" if st.kind == "grader" else st.kind
    script = JOB_SCRIPTS.get(st.kind, "slurm/jobs/stage.sbatch")
    flags = site.flags[kind] if kind in site.flags else site.flags["cpu"]
    name = job_name(st, tag)
    if "MISSING_" in flags:
        raise SystemExit(f"slurm/site/{site.name}.sh: flags for {kind!r} contain a missing value: {flags}\n"
                         "Set it in slurm/site/local.sh (see slurm/site/local.sh.example).")
    env = clean_env()
    env.update({"SUM_REPO": str(REPO), "SUM_SITE": site.name, "SUM_SBATCH_FLAGS": flags,
                "SUM_JOB_KIND": kind})
    if st.kind == "grader":
        env["SUM_GRADER"] = st.grader
    else:
        env.update({"SUM_STUDENT": "" if st.scope == "global" else tag, "SUM_STAGE": st.name,
                    "SUM_RECIPE": st.recipe})
        env["SUM_STAGE_ENV"] = str(write_stage_env(st, tag).relative_to(REPO)) if not dry else ""
        # in the submission environment too: the recipe (sourced before the stage env) reads SUM_IMP_N
        env.update({k: v for k, v in stage_env(st, tag).items() if k in ("SUM_RUN", "SUM_IMP_N")})
    env.update(extra_env or {})
    cmd = ["sbatch", "--parsable", "-J", name, "--output=runs/logs/%x_%j.log", *shlex.split(flags)]
    if after:
        # a failed input cancels this job instead of leaving it pending forever (DependencyNeverSatisfied)
        cmd += ["--dependency=afterok:" + ":".join(after), "--kill-on-invalid-dep=yes"]
    cmd += ["--export=ALL", script]
    shown = " ".join(shlex.quote(c) for c in cmd)
    ctx = " ".join(f"{k}={env[k]}" for k in ("SUM_STUDENT", "SUM_STAGE", "SUM_RECIPE", "SUM_RUN", "SUM_GRADER")
                   if env.get(k))
    if dry:
        global _DRY_N
        _DRY_N += 1
        print(f"DRY  {name}: {ctx}\n     {shown}")
        return f"DRY{_DRY_N}"
    LOGS.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        print(f"FAILED to submit {name}: {r.stderr.strip()}")
        return ""
    jid = r.stdout.strip().split(";")[0]
    ledger("submitted", scope_tag(st, tag), st.name if st.kind != "grader" else f"grader:{st.grader}",
           job_id=jid, name=name, flags=flags, after=after or [])
    print(f"submitted {name}: job {jid} ({ctx})")
    return jid


def cmd_next(a, site: Site, q: Queue) -> int:
    tags = students(a.student)
    memo: dict = {}
    want_graders: dict = {}
    submitted = 0
    rows = [(st, "_global") for st in GLOBAL_STAGES if st.name != "report" or a.with_report]
    rows += [(st, t) for t in tags for st in STUDENT_STAGES]
    if a.no_anchors:
        rows = [(st, t) for st, t in rows if not st.name.startswith("anchor-")]
    pending_ids: dict = {}
    for st, t in rows:
        r = state_of(st, t, q, site, tags, memo)
        if r["state"] in ("done", "running", "pending"):
            if r["job"] and r["state"] in ("running", "pending") and not st.chain and st.kind != "grader":
                pending_ids[(st.name, t)] = r["job"]
            continue
        if r["state"] == "failed" and not a.retry_failed:
            print(f"skip {job_name(st, t)}: failed ({r['detail']}); fix it, then `next --retry-failed` "
                  f"or `run --student {t} --stage {st.name}`")
            continue
        if st.kind == "grader":
            if r["state"] in ("ready", "failed"):
                want_graders[st.grader] = want_graders.get(st.grader, 0) + 1
            continue
        if r["state"] in ("ready", "failed"):
            jid = submit(site, st, t, a.dry_run)
            if jid:
                submitted += 1
                memo[(st.name, t)] = {"state": "pending", "job": jid, "since": "", "detail": "submitted now"}
                if not st.chain:
                    pending_ids[(st.name, t)] = jid
            continue
        # waiting: submit behind queued inputs only when every unfinished input is a plain (non-chain) stage job
        deps = needs_of(st, t, tags)
        after, ok = [], True
        for ns, nt in deps:
            ds = state_of(ns, nt, q, site, tags, memo)
            if ds["state"] == "done":
                continue
            jid = pending_ids.get((ns.name, nt))
            if ds["state"] in ("running", "pending") and jid:
                after.append(jid)
            else:
                ok = False
        if ok and after and not a.no_deps:
            jid = submit(site, st, t, a.dry_run, after=after)
            if jid:
                submitted += 1
                memo[(st.name, t)] = {"state": "pending", "job": jid, "since": "", "detail": "submitted now"}
                if not st.chain:
                    pending_ids[(st.name, t)] = jid
    for model, n in want_graders.items():
        if q.active(f"sum-grader-{model}"):
            print(f"grader {model}: {n} stage(s) ready; a grader job is queued already")
            continue
        g = Stage(f"grader-{model}", "grader", scope="global", grader=model)
        if submit(site, g, "_global", a.dry_run):
            submitted += 1
    print(f"== {submitted} job(s) {'would be ' if a.dry_run else ''}submitted")
    return 0


def cmd_run(a, site: Site, q: Queue) -> int:
    st = stage(a.stage)
    tag = "_global" if st.scope == "global" else (a.student[0] if a.student else "")
    if not tag:
        raise SystemExit("--student is required for a student stage")
    tags = students(None)
    r = state_of(st, tag, q, site, tags)
    if r["state"] in ("running", "pending"):
        print(f"{job_name(st, tag)} is {r['state']} already (job {r['job']})")
        return 0
    if r["state"] == "done" and not a.force:
        print(f"{st.name} of {tag} is done already (--force to run it again)")
        return 0
    if r["state"] == "waiting" and not a.force:
        missing = [f"{ns.name}({nt})" for ns, nt in needs_of(st, tag, tags)
                   if state_of(ns, nt, q, site, tags)["state"] != "done"]
        raise SystemExit(f"inputs not done: {missing} (--force to submit anyway)")
    if st.kind == "grader":
        st = Stage(f"grader-{st.grader}", "grader", scope="global", grader=st.grader)
    return 0 if submit(site, st, tag, a.dry_run) else 1


def cmd_driver(a, site: Site, q: Queue) -> int:
    if q.active("sum-pipeline-driver"):
        print("a driver job is queued already")
        return 0
    st = Stage("pipeline-driver", "driver", scope="global")
    jid = submit(site, dataclasses.replace(st, kind="driver"), "_global", a.dry_run,
                 extra_env={"SUM_DRIVER_EVERY_MIN": str(a.every), "SUM_DRIVER_ARGS": " ".join(a.driver_args or [])})
    return 0 if jid else 1


def cmd_logs(a, site: Site | None, q: Queue) -> int:
    st = stage(a.stage)
    tag = "_global" if st.scope == "global" else a.student[0]
    name = job_name(st, tag)
    logs = sorted(LOGS.glob(f"{name}_*.log"), key=lambda p: p.stat().st_mtime)
    if not logs:
        print(f"no log yet: runs/logs/{name}_<job>.log")
        return 1
    print(logs[-1].relative_to(REPO))
    return 0


def cmd_cancel(a, site: Site | None, q: Queue) -> int:
    """Stop a student's jobs safely: training chains get their stop file first, so no successor starts."""
    tags = a.student or []
    names = []
    for t in tags:
        for st in STUDENT_STAGES:
            if a.stage and st.name != a.stage:
                continue
            if st.chain:
                s = Student(t)
                run = s.sft_run if st.name == "sft" else s.pref_run(int(st.name[-1]))
                stop = REPO / "state" / f"train-{run}.stop"
                if not a.dry_run:
                    stop.parent.mkdir(parents=True, exist_ok=True)
                    stop.write_text(f"cancelled by pipeline.py at {now()}\n")
                print(f"wrote {stop.relative_to(REPO)} (remove it before resuming)")
                names.append(f"{job_name(st, t)}-evalw")
            if st.kind != "grader":
                names.append(job_name(st, t))
    for name in names:
        for j in q.active(name):
            print(f"scancel {j['id']} ({name}, {j['state']})")
            if not a.dry_run:
                run_cmd(["scancel", j["id"]])
    return 0


# ---- inside jobs: exec (one stage) and work (a grader's queue) ---------------------------------------------
def run_task(t: Task, budget: float) -> tuple[int, str]:
    cmd = t.cmd.replace("{T}", str(int(max(60, budget))))
    env = {**os.environ, **dict(t.env)}
    print(f"== {now()} task {t.name}: {cmd}" + (f"  [{' '.join(f'{k}={v}' for k, v in t.env)}]" if t.env else ""),
          flush=True)
    p = subprocess.Popen(["bash", "-c", cmd], cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1)
    tail: list[str] = []
    assert p.stdout is not None
    for line in p.stdout:
        sys.stdout.write(line)
        tail.append(line)
        if len(tail) > 4000:
            tail = tail[-2000:]
    rc = p.wait()
    return rc, "".join(tail)


def finished(t: Task, rc: int, out: str, st: Stage, tag: str) -> bool:
    if t.complete == "remaining0":
        return rc == 0 and re.search(r'"remaining":\s*0\b', out) is not None
    if t.complete == "check":
        return rc == 0 and task_done(st, tag, t)
    return rc == 0


def run_stage_tasks(st: Stage, tag: str, deadline: float) -> int:
    """Run a stage's unfinished tasks in order. 0 all done, INCOMPLETE out of time, other: failed."""
    t_scope = scope_tag(st, tag)
    for t in tasks(st, tag):
        if task_done(st, tag, t):
            print(f"== task {t.name}: done already")
            continue
        left = deadline - time.time()
        if left < 120:
            print("== out of time before the next task; the successor continues")
            return INCOMPLETE
        rc, out = run_task(t, max(60.0, left - 60))
        if finished(t, rc, out, st, tag):
            write_json(task_marker(t_scope, st.name, t.name), {"t": now(), "job": os.environ.get("SLURM_JOB_ID", "")})
            continue
        if rc in (0, INCOMPLETE, 4, 5):     # 4/5: improve reward has more to do; 0 without completion: more to do
            print(f"== task {t.name}: not finished yet (rc={rc}); the successor continues")
            return INCOMPLETE
        why = f"task {t.name} failed with rc={rc}"
        write_json(marker(t_scope, st.name, "failed"), {"t": now(), "job": os.environ.get("SLURM_JOB_ID", ""),
                                                        "rc": rc, "why": why, "tail": out[-3000:]})
        ledger("failed", t_scope, st.name, rc=rc, why=why)
        return rc or 1
    return 0


def cmd_exec(a, site: Site | None, q: Queue | None) -> int:
    st = stage(a.stage)
    tag = "_global" if st.scope == "global" else a.student[0]
    t_scope = scope_tag(st, tag)
    tags = students(None)
    deadline = time.time() + (a.max_seconds or 10 ** 7)
    missing = [f"{ns.name}({nt})" for ns, nt in needs_of(st, tag, tags) if not check(ns, nt, site)]
    if missing and not a.force:
        print(f"== {st.name} of {tag}: inputs not done {missing}; nothing to do (next/driver resubmits later)")
        return NOT_READY
    ledger("started", t_scope, st.name)
    rc = run_stage_tasks(st, tag, deadline)
    if rc == 0:
        if st.name in ("build", "prepare", "pool", "cand-grade", "cand-claims", "report", "fetch", "anchor-fetch") \
                or st.kind == "grader" or check(st, tag, site):
            write_json(marker(t_scope, st.name, "done"), {"t": now(), "job": os.environ.get("SLURM_JOB_ID", "")})
            marker(t_scope, st.name, "failed").unlink(missing_ok=True)
            ledger("done", t_scope, st.name)
            print(f"== {st.name} of {tag}: done")
            return 0
        why = "every task finished but the stage's output check fails"
        write_json(marker(t_scope, st.name, "failed"), {"t": now(), "job": os.environ.get("SLURM_JOB_ID", ""),
                                                        "why": why})
        ledger("failed", t_scope, st.name, why=why)
        print(f"== {st.name} of {tag}: {why}")
        return 1
    if rc == INCOMPLETE:
        ledger("incomplete", t_scope, st.name)
    return rc


def grader_queue(model: str, tags: list[str], site: Site | None) -> list[tuple[Stage, str]]:
    """Ready grading stages for one model, in the order they unblock the most work."""
    order = []
    if model == EVAL_SIDE:
        order.append((BY_NAME["anchor-grade"], "_global"))
        for name in ("sft-grade", "dpo-grade-s1", "dpo-grade-s2"):
            order += [(BY_NAME[name], t) for t in tags]
    else:
        for name in ("cand-grade", "cand-claims"):
            order += [(BY_NAME[name], t) for t in tags]
    out = []
    for st, t in order:
        if st.grader != model or check(st, t, site):
            continue
        if all(check(ns, nt, site) for ns, nt in needs_of(st, t, tags)):
            out.append((st, t))
    return out


def cmd_work(a, site: Site | None, q: Queue | None) -> int:
    """Inside a grader job: work through every ready grading stage of this model until none is left or time is up."""
    deadline = time.time() + (a.max_seconds or 10 ** 7)
    tags = students(None)
    failed: set = set()
    while True:
        todo = [(st, t) for st, t in grader_queue(a.grader, tags, site) if (st.name, t) not in failed]
        if a.no_anchors:
            todo = [(st, t) for st, t in todo if not st.name.startswith("anchor-")]
        if not todo:
            print(f"== {now()} grader {a.grader}: nothing ready")
            return 0
        st, t = todo[0]
        print(f"== {now()} grader {a.grader}: {st.name} of {t} ({len(todo)} ready)", flush=True)
        ledger("started", t, st.name, grader=a.grader)
        rc = run_stage_tasks(st, t, deadline)
        if rc == 0:
            write_json(marker(t, st.name, "done"), {"t": now(), "job": os.environ.get("SLURM_JOB_ID", "")})
            marker(t, st.name, "failed").unlink(missing_ok=True)
            ledger("done", t, st.name)
        elif rc == INCOMPLETE:
            if deadline - time.time() < 600:
                return INCOMPLETE
            failed.add((st.name, t))      # not finished but not out of time: e.g. the endpoint stalled; move on
        else:
            failed.add((st.name, t))


# ---- data helpers called by stages ---------------------------------------------------------------------------
def cmd_prepare_data(a, site, q) -> int:
    root = data_root()
    corpus = root / "corpus"
    need = [corpus / f for f in ("index.jsonl.gz", "splits.json", "bodies.sqlite")] + [eval_ids_path()]
    missing = [str(p) for p in need if not p.exists()]
    if missing:
        print(f"missing packaged data: {missing}\nUnpack the data archive into {root} (docs/DATA.md).")
        return 1
    for src, dst in SEED:
        s, d = root / src, REPO / dst
        if not (s / "questions.jsonl.gz").exists():
            print(f"missing packaged question set: {s / 'questions.jsonl.gz'}")
            return 1
        d.mkdir(parents=True, exist_ok=True)
        for f in SEED_FILES:
            if (s / f).exists() and not (d / f).exists():
                shutil.copy2(s / f, d / f)
                print(f"{src}/{f} -> {dst}/{f}")
    # the earlier runs (data/anchors/<run>/: eval/, ckpt/ for C and prefcc, metrics.jsonl): copied into runs/ so the
    # new judge can write next to them and the benchmark finds the adapters where it expects them
    anchors = root / "anchors"
    runs = [p for p in sorted(anchors.iterdir()) if p.is_dir() and ((p / "eval").is_dir() or (p / "ckpt").is_dir())] \
        if anchors.exists() else []
    if not runs:
        print("(no runs under data/anchors; the anchor stages will have nothing to grade)")
        return 0
    for run in runs:
        dst = REPO / "runs" / run.name
        dst.mkdir(parents=True, exist_ok=True)
        if (run / "ckpt").is_dir() and not (dst / "ckpt").exists():
            os.symlink((run / "ckpt").resolve(), dst / "ckpt")     # adapters: read-only, not copied
        n = 0
        for f in run.rglob("*"):
            rel = f.relative_to(run)
            if not f.is_file() or rel.parts[0] == "ckpt":
                continue
            # the earlier, hosted judge's scores are kept under another name: the new judge writes its own
            out = dst / (str(rel).replace(".judge.jsonl.gz", ".judge-hosted.jsonl.gz"))
            if not out.exists():
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, out)
                n += 1
        print(f"runs/{run.name}: {n} file(s) copied from {run}" + (", ckpt linked" if (dst / "ckpt").is_symlink() else ""))
    return 0


def cmd_fetch_model(a, site, q) -> int:
    """Download one student's weights at the pinned revision into the HF cache (the job has the site's HF_HOME)."""
    s = Student(a.student[0])
    if not s.base_model:
        print(f"configs/students/{s.tag}.env has no SCISUM_BASE_MODEL")
        return 1
    from huggingface_hub import snapshot_download  # inside the job's Python (venv or container)

    path = snapshot_download(s.base_model, revision=s.revision or None,
                             ignore_patterns=["*.pth", "*.onnx", "*.msgpack", "*.h5", "*.gguf", "original/*"])
    print(f"{s.base_model}@{s.revision or 'main'} -> {path}")
    return 0


# ---- graders ---------------------------------------------------------------------------------------------------
def grader_info(model: str) -> dict:
    p = GRADER_DIR / f"{model}.json"
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {}


def healthy(base_url: str, timeout: float = 10.0) -> bool:
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=timeout) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001 - any failure means "not serving"
        return False


def cmd_wait_grader(a, site, q) -> int:
    """Block until a grader is served (state/graders/<model>.json and a 200 from /v1/models), or time out."""
    end = time.time() + a.timeout
    while time.time() < end:
        info = grader_info(a.model)
        if info.get("base_url") and healthy(info["base_url"]):
            print(json.dumps(info))
            return 0
        time.sleep(30)
    print(f"grader {a.model} not served within {a.timeout}s")
    return 1


# ---- CLI ---------------------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", default=os.environ.get("SUM_SITE", ""), help="lumi | cuda (slurm/site/<name>.sh)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def p(name, **kw):
        x = sub.add_parser(name, **kw)
        x.add_argument("--student", action="append", help="student tag (configs/students/<tag>.env); repeatable")
        x.add_argument("--dry-run", action="store_true")
        return x

    x = p("status", help="the dashboard")
    x.add_argument("--json", action="store_true")
    x = p("next", help="submit every runnable stage")
    x.add_argument("--no-deps", action="store_true", help="only stages whose inputs are done")
    x.add_argument("--retry-failed", action="store_true")
    x.add_argument("--with-report", action="store_true", help="also the global report")
    x.add_argument("--no-anchors", action="store_true", help="skip re-grading and benchmarking C/prefcc")
    x = p("run", help="submit one stage")
    x.add_argument("--stage", required=True)
    x.add_argument("--force", action="store_true")
    x = p("driver", help="a CPU job that runs `next` every N minutes")
    x.add_argument("--every", type=int, default=30)
    x.add_argument("driver_args", nargs="*", help="extra arguments for `next`")
    x = p("logs", help="path of a stage's newest log")
    x.add_argument("--stage", required=True)
    x = p("cancel", help="cancel a student's jobs safely")
    x.add_argument("--stage")
    x = p("exec", help="(inside a job) run one stage")
    x.add_argument("--stage", required=True)
    x.add_argument("--max-seconds", type=float, default=0)
    x.add_argument("--force", action="store_true")
    x = p("work", help="(inside a grader job) every ready grading stage of one model")
    x.add_argument("--grader", required=True, choices=GRADERS)
    x.add_argument("--max-seconds", type=float, default=0)
    x.add_argument("--no-anchors", action="store_true")
    p("prepare-data", help="(stage prepare) seed state/ from the packaged data")
    p("fetch-model", help="(stage fetch) download a student's weights")
    x = p("wait-grader", help="block until a grader model is served")
    x.add_argument("--model", required=True, choices=GRADERS)
    x.add_argument("--timeout", type=int, default=3600)
    a = ap.parse_args(argv)

    inside = a.cmd in ("exec", "work", "prepare-data", "fetch-model", "wait-grader")
    if a.cmd in ("fetch-model", "logs") and not a.student:
        raise SystemExit("--student is required")
    site = None
    if not inside or a.site:
        if not a.site:
            raise SystemExit("--site lumi|cuda is required (or export SUM_SITE)")
        site = Site(a.site)
    q = None if inside else Queue()
    fn = {"status": cmd_status, "next": cmd_next, "run": cmd_run, "driver": cmd_driver, "logs": cmd_logs,
          "cancel": cmd_cancel, "exec": cmd_exec, "work": cmd_work, "prepare-data": cmd_prepare_data,
          "fetch-model": cmd_fetch_model, "wait-grader": cmd_wait_grader}[a.cmd]
    return fn(a, site, q)


if __name__ == "__main__":
    sys.exit(main())
