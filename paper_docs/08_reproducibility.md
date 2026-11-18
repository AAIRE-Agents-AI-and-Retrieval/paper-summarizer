# Reproducibility

Everything needed to repeat a run, with the exact versions that produced the numbers in
`05_results.md`.

## Code

| | |
|:--|:--|
| Code | the original project's code; what is needed to repeat the method is in this repository |
| Corpus and teacher summaries | built by the original project's corpus pipeline (GROBID full texts, teacher LLM summaries); the result ships in `data/` |
| Tests | `pytest -q`, all passing at every stage of the work (54 at Run C's preparation; the suite grew with each round) |

## Model weights

| Run | Model | Revision actually used |
|:--|:--|:--|
| A | `Qwen/Qwen3-4B-Instruct-2507` | `cdbee75f17c01a7cc42f958dc650907174af0554` |
| B | `IFM/K2-Horizon-7B` | `4622f94a6632384784471fd674d5b78f8d527798` |
| C | `Qwen/Qwen3.5-9B` | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| Judge | `qwen38-27b` | served by an OpenAI-compatible endpoint (`SCISUM_JUDGE_BASE`); revision not exposed |
| Faithfulness NLI | `microsoft/deberta-large-mnli` | default |
| BERTScore | `roberta-large`, layer 17, baseline 0.8312 | default |
| Reward-side fact test and claim check (rounds 1–4) | `qwen38-27b` writes the questions; `gemma4-31b-it` answers, grades and checks claims | same endpoint; revision not exposed |
| Evaluation-side fact test and claim check | roles swapped: `gemma4-31b-it` and `qwen38-27b` | same endpoint |
| Agents (reactive run; round-4 designer and pair auditor) | a frontier LLM through an agent CLI | not applicable |

Runs B and C pin the revision in their model profile (`SCISUM_BASE_MODEL_REVISION`). **Run A does
not**: it was downloaded from `main` before pinning was added, and the hash above is the snapshot
resolved in the local HF cache, recovered after the fact. A re-run of A should pass that hash
explicitly.

## Randomness

One seed, `20261014`, used for the split assignment, the data order and torch/numpy/python RNGs.
Checkpoints store the RNG state (restored with `map_location="cpu"`), so a resumed run continues the
same data order. Decoding is greedy and therefore seed-independent, but not bit-reproducible
(see T7).

Preference runs add `SCISUM_PREF_ORDER_SEED`: unset (the stored order, itself the validation split's seeded
order) for the first seed of an arm, `1` for its second seed (`q35*s2`). Round 4's validation set V (600
papers, 376 in reserve) is drawn with seed 20261110. Candidate sampling uses temperature 0.8, top-p 0.95.

## Environment

Python 3.13.5 in `.venv` (uv):

| package | version |
|:--|:--|
| torch | 2.13.0+cu130 |
| transformers | 5.17.0 |
| peft | 0.21.2 |
| accelerate | 1.15.0 |
| vllm | 0.31.0 |
| numpy | 2.4.6 |
| nltk | 3.10.3 |
| bert-score | 0.3.13 (baseline table only) |
| rouge-score | 0.1.2 |

`liger-kernel` is deliberately absent (no Python 3.13 wheel); the head-slice loss replaces it.
`bert-score` is installed only for its baseline table — the scorer itself is reimplemented in
`scisum/bertscore.py`, because `bert-score` 0.3.13 calls `build_inputs_with_special_tokens`, removed
in transformers 5.

## Hardware

| Run | Training | Walltime | Peak memory per rank |
|:--|:--|--:|--:|
| A | 2 × NVIDIA H200 (143,771 MiB each), 8 CPUs, 256 GB RAM | 13.75 h | 16.48 GiB |
| B | 2 × NVIDIA H200, 8 CPUs, 256 GB RAM | 16.63 h | 29.63 GiB |

Nodes used: an H200 node, an A100 node (Slurm partition `gpu`). Evaluation and judging ran on single
GPUs in the `gpu`/`short` partitions; the judge and the CPU re-scoring ran without a GPU. Earlier
attempts of Run A on A100-SXM4-80GB nodes are in the log; the reported run is the H200 one.

## Hyperparameters (defaults in `scisum/config.py`, overridden per run in `slurm/models/*.env`)

| | A | B | C |
|:--|--:|--:|--:|
| `max_seq_len` | 16,384 | 16,384 | 32,768 |
| `gen_reserve_tokens` | 3,072 | 3,072 | 4,096 |
| prompt budget | 13,312 | 13,312 | 28,672 |
| `n_train` | 70,000 | 70,000 | 20,000 |
| `n_val` / `n_test` | 15,000 / 15,000 | 15,000 / 15,000 | 15,000 / 15,000 |
| LoRA r / α / dropout | 64 / 128 / 0.0 | 64 / 128 / 0.0 | 64 / 128 / 0.0 |
| LoRA targets | 7 projections | 7 projections | 9 (adds `in_proj_qkv`, `out_proj`) |
| learning rate | 1e-4 | 1e-4 | 1e-4 |
| schedule | WSD, warmup 1%, decay 10%, floor 0.1 | same | same |
| optimiser | AdamW, β = (0.9, 0.95), wd 0, clip 1.0 | same | same |
| `grad_accum` | 8 examples per rank | 8 | 8 |
| precision | bf16 weights, fp32 adapter, autocast | same | same |
| gradient checkpointing | on | on | on |
| milestones | 1k, 5k, 10k, 20k, 50k, 70k | same | 0.5k, 1k, 2.5k, 5k, 10k, 20k |

LoRA target choice for C follows `scripts/inspect_model.py`, which showed that the hybrid
architecture's 24 linear-attention layers carry no `q_proj`/`k_proj`/`v_proj`/`o_proj`; with the
default seven targets the adapter would have reached only 8 of 32 layers.

## Commands

```bash
# one-time: environment and weights
sbatch slurm/build_env.sbatch
sbatch slurm/prep.sbatch                       # downloads the base model for $MODEL_PROFILE

# data (shared index and splits; tokens and prompts per model)
scisum data index && scisum data split
sbatch slurm/prep.sbatch                       # build + verify for the active profile

# a 45-second check that the model loads, trains one step and saves
sbatch slurm/preflight.sbatch

# training (resumable; the job relaunches itself until the run is finished)
MODEL_PROFILE=slurm/models/<tag>.env sbatch slurm/train.sbatch

# evaluation of every milestone, then the test split and the judge
sbatch slurm/eval_worker.sbatch
sbatch slurm/final_eval.sbatch
sbatch slurm/judge.sbatch

# regenerate everything in paper_docs/data
srun -p short -c 4 --mem=16G -t 15 .venv/bin/python scripts/paper_tables.py
```

`scisum status`, `scisum report` and `scisum curves` read the same artifacts and need no GPU.

## Artifacts a reader can check

| Path | Contents |
|:--|:--|
| `runs/<run>/run.json` | the resolved configuration of the run |
| `runs/<run>/state.json` | examples, steps, tokens, skipped steps |
| `runs/<run>/metrics.jsonl` | one line per logging window: loss, lr, tokens/s, memory |
| `runs/<run>/eval/*.json` | every evaluation, one file per checkpoint and split |
| `runs/<run>/judge/*.json` | every judge call's aggregate, per checkpoint |
| `runs/<run>/ckpt/*` | adapter checkpoints (rolling + one per milestone) |
| `state/splits.json` | the split assignment, by paper id |
| `state/build-<tag>.json` | the dataset card of each model's build |
| `runs/logs/*.log` | the Slurm log of every job, including the failures |

## Secrets

The judge key lives in `.env` (mode 600, gitignored) and is masked by `scisum env`. No key appears
in the repository, in any log line, or in this folder.

## Improvement work (rounds 1–4; `11_reward_selected_distillation.md`, `12_reactive_training.md`)

Every stage is a resumable Slurm job; the model profile (`slurm/models/q35-*.env`) holds every setting of an
arm, and `runs/<run>/run.json` records what was used.

| Stage | Command |
|:--|:--|
| candidates: samples of the policy on the 3,000-paper pool | `SUM_MODEL_ENV=slurm/models/q35-rft.env sbatch slurm/improve_sample.sbatch` |
| reward: fact test, ROUGE-L, numbers, length, claim check | `sbatch slurm/improve_reward.sbatch`, `sbatch slurm/improve_claims.sbatch` (same profile) |
| pairs or targets, then training | `sbatch slurm/improve_close.sbatch`, then `slurm/train.sbatch` with `SUM_RUN=<run>`, `SUM_IMP_N=<pairs>` |
| test evaluation (1,246 papers; fact test, claim check, NLI, judge) | `slurm/improve_eval_gpu.sbatch` with `SUM_RUN`, `SUM_CKPT`, `SUM_FACT_LABEL` (example in its header) |
| comparisons and verdicts | `scripts/improve_compare.py --round N`, `improve_length.py`, `improve_seeds.py`, `improve_r4.py` → `paper_docs/data/improve_*.json` |
| weight blends | `scripts/blend_adapters.py --a <C adapter> --b <adapter> --weight w --out <dir>` |
| reactive run (agent steers training) | `slurm/reactive_train.sbatch` with `scripts/reactive_agent.py`; decisions in `runs/q35prefccrx/agent/` |
| round 4 | `scripts/pair_audit.py`, `scripts/outer_calibrate.py`, `scripts/outer_agent.py --round r4 --phase propose\|select\|review`, `scripts/outer_launch.py profiles\|build\|submit\|blend\|submit-blends\|finalists`, `python -m scisum.outer_eval` (V scores) |

New labels for the fact test and claim check must be added to `scisum.factcheck.SYSTEMS`;
`outer_launch.py` refuses one that is missing.

Artifacts: `state/improve/` (pool, candidates, rewards, pairs, test ids), `state/outer/` (V, its scores, the
designer's files), `runs/<run>/` (adapters, metrics, evaluations); aggregates copied to `paper_docs/data/`
(`improve_*.json`, `outer_calibration.json`, `pair_audit.json`, `outer_r4/`). `state/` and `runs/` are not
distributed.
