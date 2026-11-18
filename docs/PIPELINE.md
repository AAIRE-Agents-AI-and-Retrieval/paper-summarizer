# The pipeline: from a base model to an SFT student and its `prefcc` continuation

This is the recipe that turned Qwen3.5-9B into Student C and then into `prefcc` (docs/WHAT_WORKED_ON_C.md),
made into Slurm stages that run the same way for every student in `configs/students/` and on two kinds of
cluster (`--site lumi` or `--site cuda`). One command shows where everything stands, one command submits
whatever can run next, and every stage can be stopped, killed or repeated without losing work.

## Before the first stage (once)

| step | command | done when |
|---|---|---|
| 1. Runtime | CUDA: create `.venv` (containers/README.md). LUMI: `containers/lumi/setup_runtime.sh` (docs/LUMI.md) | `slurm/site/local.sh` names the account and, on LUMI, `SUM_SIF`/`SUM_VENV` |
| 2. Site settings | `cp slurm/site/local.sh.example slurm/site/local.sh` and fill it in | `python3 scripts/pipeline.py --site lumi status` prints the table without errors |
| 3. Data | unpack the data archive into `data/` (or point `SCISUM_DATA_ROOT` at it) — docs/DATA.md | `data/corpus/`, `data/eval/`, `data/improve_pool/`, `data/anchors/` exist |
| 4. Smoke test | `bash slurm/submit.sh --site lumi gen slurm/jobs/smoke.sbatch` | its log ends with `"ok": true` |
| 5. Graders | nothing to do on LUMI (served by the grader jobs). On a cluster with an endpoint: `.env` with `SCISUM_JUDGE_BASE`/`JUDGE_API_KEY` or `SCISUM_GRADER_ENDPOINTS` (docs/GRADERS.md) | `scisum graders check` reports `serves_model: true` once a grader runs |

On LUMI, run `scripts/pipeline.py` with LUMI's Python module: `module load cray-python` (the system
`python3` on the login nodes is too old; the script needs only the standard library of Python >= 3.9).

## Running it

```bash
python3 scripts/pipeline.py --site lumi status                 # every student x stage
python3 scripts/pipeline.py --site lumi next --dry-run         # what would be submitted, with the exact sbatch lines
python3 scripts/pipeline.py --site lumi next                   # submit every stage whose inputs are done (or queued)
python3 scripts/pipeline.py --site lumi driver                 # or: a small CPU job that runs `next` every 30 min
python3 scripts/pipeline.py --site lumi next --student minicpm5-2b   # one student only (repeat --student for more)
```

`next` is idempotent: a stage that is done, running or queued is never submitted again, so running it twice,
or from the driver every 30 minutes, is safe. It submits a stage when its inputs are done, and also when its
inputs are plain jobs already in the queue (then with `--dependency=afterok`, cancelled automatically if an
input fails). Stages behind a training chain wait for the chain to finish and are submitted by the next call.
A failed stage is skipped until you fix the cause and run `next --retry-failed` (or `run --stage ...`).

Start with the smallest student (`minicpm5-2b`) end to end, or at least through `sft-eval`, before
submitting the other four: a problem with the runtime or the data then costs one small model's queue time.

## The stages

Runs per student `<tag>`: `<tag>-sft50k` (SFT), `<tag>-prefcc` and `<tag>-prefcc-s2` (DPO, two pair orders).
Job names: `sum-<tag>-<stage>` (logs `runs/logs/sum-<tag>-<stage>_<job id>.log`); global stages `sum-<stage>`;
grader jobs `sum-grader-<model>`.

| stage | job (site kind) | needs | what it runs | output (what "done" checks) |
|---|---|---|---|---|
| `prepare` (global) | stage (cpu) | — | `pipeline.py prepare-data`: checks the packaged corpus; copies the two fact-test question sets into `state/factcheck{,-b}/` (questions only, no earlier grades); copies the earlier runs from `data/anchors/` into `runs/` (adapters linked, the hosted judge's scores renamed `*.judge-hosted.jsonl.gz`) | marker |
| `fetch` | stage (cpu, network) | — | `pipeline.py fetch-model`: the base model at the pinned revision into the HF cache; for Qwen3.5 also the text-only copy vLLM serves (`scripts/make_text_only.py`) | the snapshot (and text-only copy) exist |
| `build` | stage (cpu-big) | prepare, fetch | `scisum data build --split train/val/test`, `data stats`, `data verify`: prompts fitted to 28,672 tokens, token shards of the first 50,000 training papers | `state/build-<model>.json` + marker |
| `sft` | train chain (train) | build | `slurm/jobs/train_body.sh`: self-check, then SFT chunks that resume until 50,000 papers; held-out loss every 2,500 papers; a milestone worker (`sum-<tag>-sft-evalw`, gen) evaluates 1k/5k/10k/20k/35k/50k on 1,000 validation papers | `runs/<tag>-sft50k/state.json` finished and `ckpt/ex50000/adapter` |
| `sft-eval` | stage (gen) | sft | `scisum eval generate` + `eval score` on the 1,246 test papers of `data/eval/eval_ids.json` (greedy; ROUGE, BERTScore, JSON validity, length, NLI on 300) | `runs/<tag>-sft50k/eval/test-ex50000-n1246.json` |
| `sft-grade` | in the qwen38-27b grader job | sft-eval | fact test set A (299 papers) and B (600 papers), claim check (300 papers), judge (500 papers) | all four task markers |
| `pool` | stage (cpu) | prepare, fetch | `scisum improve pool`: the 1,520 pool papers' prompts in this student's template (the shared questions and reference verdicts are copied from `data/improve_pool/` by scisum) | `state/improve-<tag>/pool.jsonl.gz` |
| `sample` | stage (gen) | sft, pool | `scisum improve sample`: 5 samples per pool paper from the SFT adapter (T 0.8, top-p 0.95), resumable | `samples.jsonl.gz` covers the pool |
| `cand-grade` | in the gemma4-31b-it grader job | sample | `scisum improve reward`: the reward's fact test of every candidate | marker |
| `cand-claims` | in the gemma4-31b-it grader job | cand-grade | `scisum improve select` + `improve claims`: the claim check of the candidates (the reward's claim term) | marker |
| `pairs` | stage (cpu) | cand-claims | `scisum improve build --arm prefcc`: reward with claim penalty, chosen = best, rejected = closest-length sample ≥ 0.25 below | `data/<model>/tokens/improve-<tag>-prefcc/build.json` (`pairs`) |
| `dpo-s1`, `dpo-s2` | train chain (train) | pairs | DPO β 0.1 + 1.0 NLL, LR 2e-5, one pass, from the SFT adapter; s2 shuffles the pair order (`SCISUM_PREF_ORDER_SEED=1`) | `runs/<tag>-prefcc[-s2]/ckpt/ex<pairs>/adapter`, finished |
| `dpo-eval-s1/-s2` | stage (gen) | dpo-s1/-s2 | as `sft-eval` | `runs/<tag>-prefcc[-s2]/eval/test-ex<pairs>-n1246.json` |
| `dpo-grade-s1/-s2` | in the qwen38-27b grader job | dpo-eval | as `sft-grade` | task markers |
| `bench` | stage (gen) | sft, dpo-s1 | `scripts/analysis/bench_inference.py` for the SFT and the prefcc adapter (latency, throughput, memory; docs/ANALYSIS.md) | `runs/<run>/bench/<ckpt>.json` for both |
| `anchor-fetch` (global) | stage (cpu, network) | — | Qwen3.5-9B and its text-only copy (Student C's base) | snapshot + copy |
| `anchor-grade` (global) | in the qwen38-27b grader job | prepare | the teacher, C, prefcc and prefcc-s2 (labels `teacher`, `C`, `C_prefcc`, `C_prefccs2`) under the new graders, so every new number has a same-grader reference | task markers |
| `anchor-bench` (global) | stage (gen) | prepare, anchor-fetch | benchmarks of C and prefcc on this hardware | bench files |
| `report` (global) | stage (cpu) | every student's grading and bench | `scripts/analysis/collect_results.py --out-dir reports/analysis`, then `pareto.py reports/analysis/results.json` (docs/ANALYSIS.md) | marker; outputs in `reports/analysis/` |

The training chains, the stage jobs and the grader jobs resubmit themselves when they stop on their time limit
with work left (generation and grading append and skip what is done), so a stage's output grows over several
jobs if needed. A stage job gives up after 20 resubmissions in a row without finishing (`SUM_MAX_RESUBMITS`).

### Grading: two grader jobs, not one job per stage

Grading needs a 27–31 B model served with vLLM. Rather than one server per stage, `next` submits one grader job
per model when any grading stage of that model is ready: `sum-grader-qwen38-27b` (evaluation side: fact test,
claim check, judge) and `sum-grader-gemma4-31b-it` (training side: candidate grading and claim check). The job
starts the server (4 GCDs, tensor parallel, on LUMI), checks it with `scisum graders check`, writes
`state/graders/<model>.json`, and works through every ready grading stage of every student and the anchors
(`pipeline.py work`), then exits. Starting a server takes about 5–15 minutes (estimate), so grading is batched:
run `next` after several evaluations have finished rather than after each one, or let the driver do it.

## Done criteria for a student

A student is finished when `status` shows `done` in every column. Its numbers are then in:

| what | where |
|---|---|
| learning curves (loss, held-out loss, throughput, quality per milestone) | `reports/curves/<tag>-sft50k/*.png|svg`, raw `runs/<tag>-sft50k/metrics.jsonl` |
| test metrics of SFT and both DPO seeds | `runs/<run>/eval/test-<ckpt>-n1246.json` (+ per-paper `.scores.jsonl.gz`) |
| fact recall (sets A and B), unsupported share, judge | `state/factcheck*/grades-<run>.jsonl.gz`, `state/claimcheck/<run>.qwen38-27b.jsonl.gz`, `runs/<run>/eval/*.judge.jsonl.gz` |
| the preference data | `state/improve-<tag>/` (samples, grades, claims, selection) and `data/<model>/tokens/improve-<tag>-prefcc/build.json` |
| speed and memory | `runs/<run>/bench/*.json`, training throughput in `metrics.jsonl` |
| everything in one table and the Pareto plot | `report` stage outputs (docs/ANALYSIS.md) |

The decision rules for "prefcc is better than its SFT model" are the four criteria of the earlier work, applied to
the two-seed mean (docs/WHAT_WORKED_ON_C.md, "How to measure"; implemented in `collect_results.py`).

## Time and GPU estimates (planning numbers, not measurements)

Measured on the earlier hardware: Student C (9 B, Qwen3.5) trained 20,000 papers (271 M tokens) in 61,021 s on
2 × A100 80 GB (4,445 tokens/s in total, peak 38.1 GB per GPU, gradient checkpointing on); `prefcc`'s DPO
(1,233 pairs, 155 steps) took about 9 h on one L40S at 36.9 GB; sampling ran at about 18 s per paper on a 40 GB
A100 slice; grading one system on the hosted endpoint took 10–13 min per task. 50,000 papers are about
680 M tokens with the Qwen tokenizer (±10% for the others; docs/STUDENTS.md).

Everything below assumes one MI250X GCD does 0.35–0.7 of an A100's training throughput on this workload, which
is a guess: **replace these numbers with the trainer's own `eta_s` after the first 1,000 papers** of the first
student (`metrics.jsonl`, or `status --student <tag>`).

| stage | per student (GCD-hours, estimate) | wall time (estimate) |
|---|---|---|
| build | CPU only, 1–2 h on 64 cores | 1–2 h |
| sft, 9 B class (`k2h-7b`) | 120–240 | 15–30 h on 8 GCDs (two 24 h chunks at most) |
| sft, 3.7–5 B (`k2h-3.7b`, `qwen35-4b`, `granite42-3b`) | 90–200 | 12–25 h on 8 GCDs |
| sft, 2.5 B (`minicpm5-2b`) | 80–180 | 10–22 h on 8 GCDs |
| milestone evaluations (6 + base) | 6–15 | alongside training, on 1 GCD |
| sft-eval / dpo-eval (each) | 1–3 | 1–3 h on 1 GCD |
| sample | 2–8 | 2–8 h on 1 GCD |
| dpo, per seed | 3–12 | 3–12 h on 1 GCD (or a quarter of that on 4) |
| bench | 1–2 | under 2 h |
| grading, per system (eval side) | 2–8 (4-GCD server) | 0.5–2 h inside a grader job |
| candidate grading + claims (training side) | 25–80 (4-GCD server) | 6–20 h of grader time |

For the five students together that is roughly 700–1,900 GCD-hours (SFT 470–1,020; everything else 50–160 per
student), plus the anchors (about 20–40). Smaller models are not
proportionally cheaper to train here: with one variable-length sequence per forward pass a small model
underuses the GPU (on the earlier hardware 2.2× the parameters cost 1.24× the time).
