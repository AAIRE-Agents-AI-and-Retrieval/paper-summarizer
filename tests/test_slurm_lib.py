"""The shell layer (slurm/lib.sh), tested without a GPU or Slurm.

Every failure that has cost a GPU allocation lived in these few lines, not in the model code: `torchrun` missing
from PATH (rc 127), `timeout` asked to run a shell function (rc 127 again), `scontrol show job 0` dumping every
job so the walltime came back as zero, and successors queued without their parent's card, time or memory.
They are all reproducible on a login node in a second.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LIB = REPO / "slurm" / "lib.sh"
PY = sys.executable     # run the suite with the project's interpreter (PYTHONPATH=<repo> if not installed)


def sh(snippet: str, env: dict | None = None) -> subprocess.CompletedProcess:
    script = f'set -uo pipefail\ncd "{REPO}"\nPY="{PY}"\nsource slurm/lib.sh\n{snippet}\n'
    # The suite itself may run inside an allocation (`srun ... pytest`), which exports SLURM_JOB_ID.
    # Tests that need "outside a job" would then read the *test job's* walltime.
    base = {k: v for k, v in os.environ.items() if k not in ("SLURM_JOB_ID", "SLURM_MEM_PER_NODE", "SUM_SBATCH_FLAGS")}
    base["PYTHONPATH"] = str(REPO) + os.pathsep + base.get("PYTHONPATH", "")
    return subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=300,
                          env={**base, **(env or {})})


def test_launch_runs_the_cli_in_one_process():
    r = sh("launch 1 env")
    assert r.returncode == 0, r.stderr[-400:]
    assert "base_model" in json.loads(r.stdout)


def test_launch_accepts_a_timeout():
    """`timeout` is a binary and cannot run a shell function; LAUNCH_TIMEOUT must wrap python."""
    r = sh("LAUNCH_TIMEOUT=120 launch 1 env")
    assert r.returncode == 0, r.stderr[-400:]
    assert "No such file or directory" not in r.stderr
    assert "seed" in json.loads(r.stdout)


def test_the_multi_process_launcher_is_torchrun_by_module():
    body = LIB.read_text()
    assert "-m torch.distributed.run" in body, "torchrun from PATH is not reliable in a job"
    assert "timeout 1200 launch" not in body, "timeout cannot run a shell function"


def test_seconds_left_falls_back_instead_of_guessing():
    r = sh('echo "left=$(seconds_left 1500)"')
    assert "left=38700" in r.stdout, r.stdout      # no SLURM_JOB_ID: the conservative fallback
    r = sh('scontrol() { return 1; }; echo "left=$(seconds_left 1500)"', env={"SLURM_JOB_ID": "999999999"})
    assert "left=38700" in r.stdout, r.stdout      # unknown job: the same fallback, never 0


def test_candidates_order_depends_on_the_memory_marker(tmp_path):
    """With the marker, checkpointing is tried first; without it, the faster setting leads."""
    missing = tmp_path / "absent"
    present = tmp_path / "present"
    present.write_text("")
    out = sh(f'NEED_CKPT={missing} candidates 2').stdout.split()
    assert [a + " " + b for a, b in zip(out[::2], out[1::2], strict=True)] == ["2 false", "2 true", "1 true"]
    out = sh(f'NEED_CKPT={present} candidates 2').stdout.split()
    assert [a + " " + b for a, b in zip(out[::2], out[1::2], strict=True)] == ["2 true", "1 true"]
    out = sh(f'NEED_CKPT={present} candidates 1').stdout.split()
    assert [a + " " + b for a, b in zip(out[::2], out[1::2], strict=True)] == ["1 true"]


def test_progress_helpers_are_safe_on_a_missing_run():
    r = sh('echo "done=$(examples_done nosuchrun) finished=$(run_finished nosuchrun)"')
    assert "done=0 finished=0" in r.stdout, r.stdout


def test_a_resubmission_keeps_the_site_flags_the_time_and_the_memory():
    """A successor that fell back to defaults once waited 6.5 h longer; one without its card could land on a small GPU."""
    stubs = 'sbatch() { printf "%s\\n" "$@"; }; squeue() { echo 8:00:00; }; '
    r = sh(stubs + "submit_like_me sum-x-sft x.sbatch",
           env={"SLURM_JOB_ID": "1", "SLURM_MEM_PER_NODE": "131072",
                "SUM_SBATCH_FLAGS": "--account=p1 --partition=small-g --gpus-per-node=2 --time=24:00:00"})
    args = r.stdout.split()
    assert args[:3] == ["-J", "sum-x-sft", "--export=ALL"], args
    assert "--account=p1" in args and "--gpus-per-node=2" in args and "--partition=small-g" in args
    # this job's own limit comes after the site's, so it wins
    assert args.index("--time=8:00:00") > args.index("--time=24:00:00") and "--mem=131072M" in args
    assert args[-1] == "x.sbatch"
    r = sh(stubs + "submit_like_me sum-x x.sbatch")                     # outside a job, no flags: nothing added
    assert not any(a.startswith(("--mem", "--time", "--account")) for a in r.stdout.split()), r.stdout


def test_each_job_has_its_own_selfcheck_directory_and_a_key_per_allocation():
    """Jobs that start together must not delete each other's self-check; a check on 8 GPUs says nothing about 2."""
    assert sh("SLURM_JOB_ID=123 selfcheck_run").stdout.strip() == "selfcheck-123"
    assert sh("unset SLURM_JOB_ID; selfcheck_run").stdout.strip() == "selfcheck"
    k8 = sh("MODEL_TAG=m SUM_RECIPE=sft50k SUM_SITE=lumi selfcheck_key 8").stdout.strip()
    k2 = sh("MODEL_TAG=m SUM_RECIPE=sft50k SUM_SITE=lumi selfcheck_key 2").stdout.strip()
    kp = sh("MODEL_TAG=m SUM_RECIPE=prefcc SUM_SITE=lumi selfcheck_key 8").stdout.strip()
    assert len({k8, k2, kp}) == 3 and k8.startswith("state/selfcheck/m-sft50k-lumi")


def test_the_selfcheck_keeps_the_total_batch():
    body = LIB.read_text()
    assert "SCISUM_GRAD_ACCUM=$(( ${SCISUM_TOTAL_BATCH:-8} / n ))" in body
    train = (REPO / "slurm" / "jobs" / "train_body.sh").read_text()
    assert "SCISUM_GRAD_ACCUM=$(( ${SCISUM_TOTAL_BATCH:-8} / NGPU ))" in train


def test_ledger_events_are_json_lines(tmp_path):
    r = subprocess.run(["bash", "-c", f'cd "{tmp_path}"; source "{LIB}"; SUM_STUDENT=s SUM_STAGE=sft SLURM_JOB_ID=5 '
                                      'ledger_event retry "\\"rc\\": 1"; ledger_event started'],
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr
    lines = [json.loads(x) for x in (tmp_path / "state" / "pipeline" / "ledger.jsonl").read_text().splitlines()]
    assert lines[0]["event"] == "retry" and lines[0]["rc"] == 1 and lines[0]["student"] == "s" and lines[0]["job"] == "5"
    assert lines[1]["event"] == "started"


def test_grader_names_map_to_weights_and_can_be_overridden():
    assert sh("grader_hf_id qwen38-27b").stdout.strip() == "Qwen/Qwen3.8-27B"
    assert sh("grader_hf_id gemma4-31b-it").stdout.strip() == "google/gemma-4-31B-it"
    assert sh("grader_hf_id gemma4-31b-it", env={"SUM_GRADER_HF_GEMMA": "x/y"}).stdout.strip() == "x/y"
    assert "--reasoning-parser" in sh("grader_vllm_args qwen38-27b").stdout
    assert sh("grader_vllm_args qwen38-27b", env={"SUM_GRADER_ARGS_QWEN": ""}).stdout.strip() == ""
    assert sh("grader_hf_id nope").returncode != 0
