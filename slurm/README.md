# slurm/

Job scripts and the site layer. You normally do not call these directly: `scripts/pipeline.py` submits them
(docs/PIPELINE.md). Read this when something needs fixing.

| file | role |
|---|---|
| `common.sh` | sourced by every job: repository root, the site, caches, offline mode, then the student profile (`configs/students/$SUM_STUDENT.env`), the recipe (`configs/recipes/$SUM_RECIPE.env`) and the stage's extra environment (`$SUM_STAGE_ENV`, written by pipeline.py under `state/pipeline/`); sets `$PY` and `$SCISUM` |
| `lib.sh` | helpers: `submit_like_me` (resubmit with the job's own site flags, time and memory), `seconds_left`, `launch` (python or torchrun by module), `examples_done`/`run_finished`, the training self-check, `ledger_event`, the grader name → weights map |
| `site/cuda.sh`, `site/lumi.sh` | how Python runs (venv, or `site/lumi-python` = the container), sbatch flags per job kind (`train`, `gen`, `grader`, `cpu`, `cpu-big`, `driver`), GPU count, caches (`site/_caches.sh`) |
| `site/local.sh` | **yours, git-ignored**: account, container paths, partitions, overrides (start from `site/local.sh.example`) |
| `jobs/train.sbatch` → `jobs/train_body.sh` | a training chain (SFT or DPO): successor queued at start, self-check, retries in the allocation, stall watchdog, milestone worker for SFT |
| `jobs/eval_worker.sbatch` | milestone evaluations of an SFT run on 1,000 validation papers; comes back while training runs |
| `jobs/stage.sbatch` → `jobs/stage_body.sh` | any other stage: `pipeline.py exec`; resubmits itself when it stops on the time limit with work left |
| `jobs/grader_server.sbatch` → `jobs/grader_body.sh` | serves one grader model with vLLM (or uses an external endpoint), health-checks it (`scisum graders check`), then `pipeline.py work` grades everything ready for that model |
| `jobs/driver.sbatch` | runs `pipeline.py next` every 30 minutes until everything is done |
| `jobs/smoke.sbatch` | the runtime smoke test (`containers/smoke_test.py`) |
| `submit.sh` | submit any script with a site's flags for a job kind: `bash slurm/submit.sh --site lumi gen slurm/jobs/smoke.sbatch` |

Rules that keep running jobs safe:

1. **Never edit a job body in place while a job runs it** (bash reads scripts as it executes). Every body keeps
   its code in `main()`, called on the last line; replace a body by writing a new file and `mv` it over the old one.
2. The `.sbatch` files are headers that `exec` their body from shared storage, so a job that waited in the queue
   runs today's body. They carry no resources: those come from the site at submission.
3. Every job is safe to run twice: training resumes from its last checkpoint, generation and grading append
   and skip what is done, markers are written last.
