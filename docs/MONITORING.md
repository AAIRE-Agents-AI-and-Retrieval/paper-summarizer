# Monitoring, recovering, stopping

One dashboard (`scripts/pipeline.py status`), one ledger of events (`state/pipeline/ledger.jsonl`), one log per
job with a predictable name, and the training run's own `metrics.jsonl`. This page says where to look, what
normal looks like, and what to do when it is not.

## Where to look

| question | command or file |
|---|---|
| Where does everything stand? | `python3 scripts/pipeline.py --site lumi status` — every student × stage: `done`, `RUN<job>`, `PEND<job>`, `FAIL`, `ready` (inputs done, not submitted), `.` (waiting for inputs); then the two graders |
| One student in detail | `status --student k2h-7b` — per stage: state, job id, since, and progress (papers trained, tasks done, samples drawn) |
| Machine-readable | `status --json` |
| The log of a stage | `tail -f $(python3 scripts/pipeline.py --site lumi logs --student k2h-7b --stage sft)`; all logs are `runs/logs/<job name>_<job id>.log` |
| What happened, in order | `state/pipeline/ledger.jsonl`: submitted, started, retry, watchdog, incomplete, done, failed, grader_up, ... with job ids and times |
| Why a stage failed | `state/pipeline/<tag>/<stage>.failed` (JSON: rc, the failing task, the last 3,000 characters of its output) |
| Training progress | `runs/<run>/metrics.jsonl`, one JSON line every few steps (fields below); `scisum status --run <run>` |
| Learning curves | `reports/curves/<run>/{loss,throughput,quality,schema}.{png,svg}`, redrawn after every training chunk and every milestone evaluation (`scisum curves --run <run>` by hand) |
| The vLLM server of a grader | `runs/logs/sum-grader-<model>_<job>.vllm.log`; `state/graders/<model>.json` while it serves |
| Is a grader usable? | `scisum graders check --models qwen38-27b` (configured / reachable / serves the name) |
| The queue | `squeue --me -o "%.10i %.32j %.8T %.10M %R"` |

### Fields of `metrics.jsonl`

| `kind` | fields | read it as |
|---|---|---|
| `chunk_start` | `cap`, `world` (GPUs), `total_steps`, `examples`, `host`, `job` | a job started or resumed at `examples` |
| `train` | `step`, `examples`, `tokens`, `loss`, `lr`, `grad_norm`, `tokens_per_s`, `eta_s`, `seconds`, `vram_gb`; DPO also `dpo`, `nll`, `margin`, `pref_acc`, `len_diff` | the training signal, every few steps |
| `val_loss` | `examples`, `loss`, `ppl`, `papers` (200) | held-out loss on validation papers, every 2,500 papers (DPO: every 500 pairs) |
| `chunk_end` | `exit_code`, `reason`, `skipped_steps`, `vram_peak_gb` | why a job's training ended (finished, time limit, signal) |

## What normal looks like

| signal | normal | not normal → do |
|---|---|---|
| SFT `loss` | falls fast in the first 1,000 papers (the student learns the format), then slowly; Student C ended near 0.50 train / 0.45 held-out | NaN or jumps up and stays: stop the chain (`touch state/train-<run>.stop`), look at `grad_norm` and the lr schedule |
| `val_loss` | flat or slowly falling, a small drop during the final decay (last 10% of papers) | rising for several points: over-fitting or a data problem; compare the milestone evaluations |
| `tokens_per_s`, `eta_s` | stable within ±20% across steps (long and short papers alternate) | a lasting drop: another job on the node, a slow file system, thermal throttling; the watchdog acts only on a full stall |
| `vram_gb` | below the GPU's memory (C: 38 GB at 32k tokens with checkpointing) | a self-check `rc=3` (out of memory) marks `state/selfcheck/<model>-<recipe>.need_checkpointing` and retries with checkpointing on, then with fewer GPUs |
| DPO `pref_acc`, `margin` | `pref_acc` rises from ~0.5 to 0.7–0.9, `margin` from 0 to 2–3 | `pref_acc` stuck at 0.5: the pairs or the reference adapter are wrong (`SCISUM_INIT_ADAPTER`) |
| milestone quality (`quality.png`) | JSON validity near 1.0 after the first milestone; ROUGE-L and BERTScore rise, then flatten | JSON validity low after 5,000 papers: the chat template or assistant wrapper of this student is wrong (docs/STUDENTS.md) |
| grader jobs | `grader_up` in the ledger within ~15 min, then tasks done one after another | `grader_failed`: read the `.vllm.log` (out of memory → more GPUs `SUM_GRADER_GPUS` or a shorter `SUM_GRADER_MAX_LEN`; model not cached and no network → fetch it on the login node) |
| fact test / claim check tasks | `"remaining": 0` at the end of each task | a task that stays incomplete after several jobs: some papers fail every time; read the task's output in the log |

## Recovering

Every step appends and skips what is done, so recovering is normally "fix the cause, submit again":

| situation | what happens by itself | what you do |
|---|---|---|
| a training job is killed (node failure, scancel, time limit) | its successor (queued at the start) resumes from the last checkpoint (≤ 2,500 papers back) | nothing |
| the trainer crashes | the job keeps the allocation and retries; a fix to `scisum/*.py` made meanwhile is used | fix the code if needed; `touch state/train-<run>.stop` to give up instead |
| the trainer hangs | the watchdog kills it after 20 min without metrics; the retry resumes | nothing (look for `watchdog` in the ledger) |
| three training chunks without progress | the chain stops and writes `state/train-<run>.stop` | fix the cause, `rm state/train-<run>.stop state/train-<run>.noprogress`, then `pipeline.py run --student <tag> --stage sft` |
| a stage job stops on its time limit | it resubmits itself (at most 20 times in a row) | nothing |
| a stage failed (`FAIL` in status) | nothing more is submitted for it | read `state/pipeline/<tag>/<stage>.failed` and the log, fix, then `next --retry-failed` or `run --student <tag> --stage <stage>` |
| a job waits with `DependencyNeverSatisfied` | it is cancelled automatically (`--kill-on-invalid-dep=yes`) | run `next` again once the input is fixed |
| a grader job ends with work left | it resubmits itself | nothing |
| a grading stage keeps failing | the grader job moves on to the other stages | read its task output in the grader log; `run --stage <stage> --student <tag>` resubmits a grader job |
| a done stage must be redone (e.g. a fixed scorer) | — | delete its output (the file the "done" column checks, see docs/PIPELINE.md) and its marker `state/pipeline/<tag>/<stage>.done`, then `run --stage ... --force` |

Never edit a job script while a job runs it: bash reads scripts as it goes. The job bodies keep their code in
`main()` called on the last line, and are replaced with `mv` of a new file, never patched in place. The `.sbatch`
files are headers that `exec` the body from shared storage, so a queued job always runs the current body.

## Stopping

| to stop | do |
|---|---|
| one student, now | `python3 scripts/pipeline.py --site lumi cancel --student <tag>` (writes the training chains' stop files first, so no successor starts, then cancels the jobs); remove `state/train-<run>.stop` before resuming |
| one stage | `cancel --student <tag> --stage <stage>` |
| a training chain after its current chunk | `touch state/train-<run>.stop` (every chain: `state/train.stop`) |
| all new stage and grader jobs | `touch state/pipeline/stop` (running jobs finish; queued ones exit at start) |
| the driver | `touch state/pipeline/driver.stop` |
| milestone evaluations | `touch state/eval.stop` |

Resuming is always the same: remove the stop file and run `next` (or `run` for one stage). Nothing is lost
except the work since the last checkpoint of a training run.
