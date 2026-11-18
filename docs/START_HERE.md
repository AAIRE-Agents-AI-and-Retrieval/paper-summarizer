# Start here

This guide is for the developer who will train the five students and help choose the production one. Read
it top to bottom once. Each step names the document with the details.

## 1. What you are building, and why

The original project trained Qwen3.5-9B (Student C) as a summarizer for scientific papers. It writes an
eight-field JSON summary of a whole paper. Two stages made it as factual as its much larger teacher, at the
same length:

1. **SFT (distillation)** on teacher summaries of full papers.
2. **prefcc.** The student writes several summaries of unseen papers. Each is scored by how many facts about
   the paper it gets right, minus penalties for unsupported content and excess length. The student is then
   trained with DPO to prefer the better summary of each length-matched pair.

Read `docs/WHAT_WORKED_ON_C.md` first. It is self-contained and explains every setting.

Your job is to apply the same recipe to five smaller or newer students, with **50,000 SFT papers** instead of
20,000. Then measure quality, latency and resource use, so the team can pick the production model from a
Pareto front of quality against cost.

## 2. The documents

| Read | When | What it gives you |
|:--|:--|:--|
| `docs/WHAT_WORKED_ON_C.md` | First | The method and the original results |
| `docs/DATA.md` | Before unpacking | Every file in `data/`, sizes, checksums, provenance, the internal-use rule |
| `docs/STUDENTS.md` | Before training | Per student: architecture, LoRA targets, chat template, risks, estimates |
| `docs/GRADERS.md` | Before grading | The two grader LLMs, their roles, how to self-host them, the comparability caveat |
| `docs/LUMI.md` | Before your first job on LUMI | Account, storage, containers, Slurm flags, ROCm risks, your first day |
| `docs/PIPELINE.md` | Daily | Every stage with its exact command, outputs and done criteria; time and GPU estimates |
| `docs/MONITORING.md` | Daily | What normal looks like, how to spot stalls and failures, how to resume or cancel |
| `docs/ANALYSIS.md` | When results arrive | The benchmark protocol, the results table, learning curves, Pareto plots, the decision report |
| `docs/EVAL.md` | Reference | Evaluation details from the original project |
| `paper_docs/` | Reference | The full research record of the original project: every number, finding and caveat |

## 3. Setup (day 1)

1. **Unpack** the zip on the cluster's large storage (on LUMI, scratch, not home). Check the data against
   `data/MANIFEST.json`, as described in `docs/DATA.md`.
2. **Runtime.**
   - On LUMI, build or obtain the containers described in `containers/lumi/README.md`, then run its smoke test.
   - On a CUDA cluster, create a fresh virtual environment from `requirements.lock.txt` and run
     `pip install -e .`. Do not reuse another project's environment: an installed copy of `scisum` would
     shadow modules.
3. **Site settings.** Copy `slurm/site/local.sh.example` to `slurm/site/local.sh`. Fill in the project
   account, paths and container images. This file is git-ignored.
4. **Tests.** Run `ruff check scisum tests scripts && python -m pytest`. They need no GPU and no network, and
   they include a byte-for-byte check of the prompt against the original project.
5. **Status.** Run `python3 scripts/pipeline.py --site lumi status`. It prints every student × stage as
   *waiting*.

## 4. The run, in order

The exact commands are in `docs/PIPELINE.md`. In short:

| Order | Stage | Notes |
|:--|:--|:--|
| 1 | `prepare` | Once: seeds the fact-test questions and the anchors into `state/` and `runs/` |
| 2 | Anchors | Serve the two graders; re-grade the teacher, C and prefcc summaries under them; benchmark C and prefcc. **Do this first.** It shows whether your graders reproduce the original numbers, and every later comparison uses these re-graded anchors. |
| 3 | Smallest student first (`minicpm5-2b`) | `fetch` → `build` → `sft` → evaluation and grading → `pool` → `sample` → candidate grading → `pairs` → `dpo-s1`/`dpo-s2` → evaluation and grading → `bench`. It shakes out problems cheaply. |
| 4 | The other four students | `python3 scripts/pipeline.py --site lumi next` submits every stage whose inputs are ready. Repeat it, or run the `driver` job, which repeats it every 30 minutes. |
| 5 | `report` | `scripts/analysis/collect_results.py`, then `scripts/analysis/pareto.py`, then the decision report (`docs/ANALYSIS.md`) |

Every stage is resumable:
- training restarts from the last checkpoint;
- generation and grading skip finished papers;
- `next` never submits a duplicate.

After a failure, read the log, fix the cause, and run `next` again. `docs/MONITORING.md` lists the usual
failures.

## 5. What to deliver

For each student, the SFT and both prefcc seeds need:

- the test evaluation;
- fact recall on both question sets;
- the claim check;
- the judge;
- the four criteria against its own SFT model;
- training resources (GPU-hours, peak memory, tokens/s);
- the inference benchmark (latency p50/p95, throughput, memory).

Then, across students: the results table, the learning curves, the Pareto plots, and a filled-in
`docs/DECISION_REPORT_TEMPLATE.md`. `collect_results.py` and `pareto.py` produce all of it from the run
artifacts.

## 6. Rules that keep the results trustworthy

- **Numbers come from artifacts** (`runs/*/metrics.jsonl`, `runs/*/eval/*.json`, the grading files), never from
  memory or messages. The analysis scripts read only those.
- **Fix the criteria before you look at test results.** Use the same four criteria as the original project
  (`docs/WHAT_WORKED_ON_C.md`, "How to measure").
- **Two seeds for prefcc**, judged on their mean. The pair order alone moved recall by 0.013–0.020 on C.
- **Compare on the same hardware.** Latency and throughput only compare within one GPU type and one software
  stack (`docs/ANALYSIS.md`).
- **Do not change the prompt or the schema.** The 100,826 teacher summaries were written for exactly this
  prompt, and `tests/test_prompt_parity.py` guards it.
- **Data stays internal** (`docs/DATA.md`).

## 7. Known open points

| Point | Where |
|:--|:--|
| Nothing has been run on LUMI. The LUMI facts are from documentation and are marked "(check)". | `docs/LUMI.md`, "What is verified" |
| ROCm kernels: attention at 32k tokens, the Triton kernels of Qwen3.5's linear attention, vLLM on MI250X | `docs/LUMI.md`, `docs/STUDENTS.md` |
| Self-hosted graders may score differently from the original endpoint, whose exact weights were not exposed | `docs/GRADERS.md`: re-grade the anchors first |
| Qwen3.5-4B ties its embeddings; its text-only serving copy is untested | `docs/STUDENTS.md` |
| K2-Horizon-7B's current revision differs from the one the original project trained (Run B) | `configs/students/k2h-7b.env` pins the current one |
